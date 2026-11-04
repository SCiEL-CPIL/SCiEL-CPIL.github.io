import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # 1. Generate all 4-subsets of {0, ..., 10}
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)

    # 2. Build conflict graph: conflict if intersection size >= 3
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter >= 3:
                adj[i].add(j)
                adj[j].add(i)

    adj_list = [list(s) for s in adj]

    # Vectorized adjacency for fast blocked-set computation
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in adj_list[i]:
            adj_bool[i, j] = True
            adj_bool[j, i] = True

    # 3. Greedy coloring for upper bound (used in B&B)
    def greedy_color_bound(nodes, adj_bool):
        """Greedy coloring of subgraph induced by nodes. Returns color count (upper bound on clique size)."""
        if not nodes:
            return 0
        n = len(nodes)
        if n <= 1:
            return n
        # Order by degree in subgraph descending
        node_arr = np.array(nodes, dtype=np.int64)
        sub = adj_bool[np.ix_(node_arr, node_arr)]
        degrees = sub.sum(axis=1)
        order = node_arr[np.argsort(-degrees)]
        colors = np.full(n, -1, dtype=np.int32)
        color_count = 0
        for pos, node in enumerate(order):
            used = set()
            for other_pos in range(pos):
                other = order[other_pos]
                if adj_bool[node, other]:
                    used.add(colors[other_pos])
            c = 0
            while c in used:
                c += 1
            colors[pos] = c
            color_count = max(color_count, c + 1)
        return color_count

    # 4. Branch-and-bound for Max Clique (compatibility graph = non-conflict)
    #    We work on the complement: compatible if NOT in adj
    best_clique = []
    best_size = 0

    def branch_bound(candidates, current_clique):
        nonlocal best_clique, best_size

        if time.time() > deadline - 30.0:  # leave time for local search
            return

        if not candidates:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return

        # Upper bound via greedy coloring on the compatibility subgraph
        # Compatibility: two nodes are compatible if NOT adjacent in conflict graph
        bound = greedy_color_bound(candidates, ~adj_bool)
        if len(current_clique) + bound <= best_size:
            return

        # Choose pivot: node with max compatibility degree in candidates
        cand_arr = np.array(candidates, dtype=np.int64)
        compat_sub = ~adj_bool[np.ix_(cand_arr, cand_arr)]
        np.fill_diagonal(compat_sub, False)
        compat_degrees = compat_sub.sum(axis=1)
        pivot_idx = int(np.argmax(compat_degrees))
        pivot = candidates[pivot_idx]

        # Include pivot
        new_candidates = [c for c in candidates if c != pivot and not adj_bool[pivot, c]]
        current_clique.append(pivot)
        branch_bound(new_candidates, current_clique)
        current_clique.pop()

        # Exclude pivot
        candidates.remove(pivot)
        branch_bound(candidates, current_clique)

    # Run B&B with a time cap
    bb_deadline = start_time + 20.0
    all_nodes = list(range(N_S))

    # Temporarily tighten deadline for B&B
    global_deadline = deadline
    deadline = bb_deadline
    branch_bound(all_nodes, [])
    deadline = global_deadline

    # 5. Vectorized local search / simulated annealing on the independent set
    #    State: boolean mask of selected supports
    current_mask = np.zeros(N_S, dtype=bool)
    for idx in best_clique:
        current_mask[idx] = True

    best_mask = current_mask.copy()
    best_size = int(current_mask.sum())

    rng = np.random.RandomState(42)

    def compute_candidates(mask):
        """Return array of indices that can be added (not in set, not blocked)."""
        if mask.any():
            blocked = np.any(adj_bool[mask], axis=0)
        else:
            blocked = np.zeros(N_S, dtype=bool)
        candidates = ~(mask | blocked)
        return np.where(candidates)[0]

    # Local search with add, swap, and multi-swap moves
    max_ls_time = deadline - 5.0
    temp = 1.0
    cooling = 0.9995

    while time.time() < max_ls_time:
        # Try random add
        cands = compute_candidates(current_mask)
        if len(cands) > 0:
            v_add = int(cands[rng.randint(len(cands))])
            current_mask[v_add] = True
            new_size = int(current_mask.sum())
            if new_size > best_size:
                best_size = new_size
                best_mask = current_mask.copy()
            temp *= cooling
            continue

        # Stuck: try 1-swap (remove one, add one)
        selected_indices = np.where(current_mask)[0]
        if len(selected_indices) == 0:
            break

        v_remove = int(selected_indices[rng.randint(len(selected_indices))])
        current_mask[v_remove] = False

        cands2 = compute_candidates(current_mask)
        if len(cands2) > 0:
            v_add = int(cands2[rng.randint(len(cands2))])
            current_mask[v_add] = True
            new_size = int(current_mask.sum())
            if new_size > best_size:
                best_size = new_size
                best_mask = current_mask.copy()
            temp *= cooling
            continue

        # Try 2-swap (remove two, add up to two)
        if len(selected_indices) >= 2:
            perm = rng.permutation(len(selected_indices))
            v_rem1 = int(selected_indices[perm[0]])
            v_rem2 = int(selected_indices[perm[1]])
            current_mask[v_rem1] = False
            current_mask[v_rem2] = False

            cands3 = compute_candidates(current_mask)
            added = 0
            for _ in range(2):
                if len(cands3) > 0:
                    v_a = int(cands3[rng.randint(len(cands3))])
                    current_mask[v_a] = True
                    added += 1
                    cands3 = compute_candidates(current_mask)
                else:
                    break

            new_size = int(current_mask.sum())
            if new_size > best_size:
                best_size = new_size
                best_mask = current_mask.copy()
            elif new_size < best_size - 2:
                # Revert: restore best
                current_mask = best_mask.copy()
            temp *= cooling
            continue

        # Try 3-swap
        if len(selected_indices) >= 3:
            perm = rng.permutation(len(selected_indices))
            v_rem1 = int(selected_indices[perm[0]])
            v_rem2 = int(selected_indices[perm[1]])
            v_rem3 = int(selected_indices[perm[2]])
            current_mask[v_rem1] = False
            current_mask[v_rem2] = False
            current_mask[v_rem3] = False

            cands4 = compute_candidates(current_mask)
            for _ in range(3):
                if len(cands4) > 0:
                    v_a = int(cands4[rng.randint(len(cands4))])
                    current_mask[v_a] = True
                    cands4 = compute_candidates(current_mask)
                else:
                    break

            new_size = int(current_mask.sum())
            if new_size > best_size:
                best_size = new_size
                best_mask = current_mask.copy()
            elif new_size < best_size - 3:
                current_mask = best_mask.copy()
            temp *= cooling
            continue

        # Try 4-swap
        if len(selected_indices) >= 4:
            perm = rng.permutation(len(selected_indices))
            for k in range(4):
                current_mask[int(selected_indices[perm[k]])] = False

            cands5 = compute_candidates(current_mask)
            for _ in range(4):
                if len(cands5) > 0:
                    v_a = int(cands5[rng.randint(len(cands5))])
                    current_mask[v_a] = True
                    cands5 = compute_candidates(current_mask)
                else:
                    break

            new_size = int(current_mask.sum())
            if new_size > best_size:
                best_size = new_size
                best_mask = current_mask.copy()
            elif new_size < best_size - 4:
                current_mask = best_mask.copy()
            temp *= cooling
            continue

        # No moves possible, restart from best
        current_mask = best_mask.copy()
        temp = 1.0

    # 6. Final polish: deterministic greedy add pass
    for _ in range(3):
        cands = compute_candidates(current_mask)
        while len(cands) > 0:
            # Pick the candidate with fewest conflicts with remaining candidates (min-degree)
            best_cand = cands[0]
            min_conf = float('inf')
            for c in cands:
                # Count how many other candidates this one blocks
                conf = int(adj_bool[c, cands].sum())
                if conf < min_conf:
                    min_conf = conf
                    best_cand = c
            current_mask[best_cand] = True
            cands = compute_candidates(current_mask)

        new_size = int(current_mask.sum())
        if new_size > best_size:
            best_size = new_size
            best_mask = current_mask.copy()
        else:
            break

    # 7. Construct vectors from best_mask
    vectors = []
    for s_idx in np.where(best_mask)[0]:
        support = supports[int(s_idx)]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)

    if not vectors:
        # Fallback: 2-subset vectors
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int8)
                    v[i], v[j] = si, sj
                    vectors.append(v)

    return np.array(vectors, dtype=np.int64)
