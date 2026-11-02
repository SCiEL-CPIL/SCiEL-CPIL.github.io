import numpy as np
import itertools
import time

D = 11

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0

    # --- Step 1: Generate candidate vectors ---
    # W1: weight-1 vectors with entry ±2 (norm² = 4)
    w1 = []
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            w1.append(v)

    # W4: weight-4 vectors with entries ±1 (norm² = 4)
    w4 = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            w4.append(v)

    A = np.array(w4, dtype=np.int64)
    N = len(A)

    # Compute pairwise dot products and conflict graph
    # Two vectors conflict if |dot| > 2 (i.e., dot > 2 or dot < -2)
    # For our candidates, dot values are even integers in {-4,-2,0,2,4}
    # Conflict condition: dot == 4 or dot == -4 (i.e., |dot| = 4)
    # Actually, the kissing number condition requires distance >= max norm.
    # Max norm² = 4, so min distance² >= 4, meaning |v_i - v_j|² >= 4
    # |v_i - v_j|² = |v_i|² + |v_j|² - 2*v_i·v_j = 4 + 4 - 2*dot = 8 - 2*dot
    # Need 8 - 2*dot >= 4 => dot <= 2
    # And we also need the "max norm" condition: each vector has norm <= sqrt(max norm²)
    # Since all have norm² = 4, max norm = 2.
    # So the condition is: for all pairs, dot(v_i, v_j) <= 2.
    # Conflict: dot > 2.

    # Actually wait - let me re-read the task: "minimum pairwise distance is at least the maximum vector norm"
    # max vector norm = 2 (since all vectors have norm 2 after normalization, or norm² = 4 before)
    # min pairwise distance >= 2
    # |v_i - v_j| >= 2
    # |v_i - v_j|² >= 4
    # |v_i|² + |v_j|² - 2 v_i·v_j >= 4
    # 4 + 4 - 2*dot >= 4
    # dot <= 2

    # So conflict means dot > 2.

    dots = A @ A.T
    # Conflict if dot > 2
    conflict = (dots > 2)
    # Remove self-conflicts
    np.fill_diagonal(conflict, False)

    # Precompute adjacency lists as sorted arrays for faster iteration
    adj = [np.where(conflict[i])[0] for i in range(N)]
    deg = np.array([len(adj[i]) for i in range(N)])

    # --- Step 2: Greedy initialization with randomized ordering ---
    rng = np.random.default_rng(12345)

    def greedy_random_order():
        """Greedy independent set: process vertices in random order."""
        in_set = np.zeros(N, dtype=bool)
        removed = np.zeros(N, dtype=bool)
        order = rng.permutation(N)
        for i in order:
            if removed[i]:
                continue
            in_set[i] = True
            removed[i] = True
            for j in adj[i]:
                removed[j] = True
        return in_set

    def greedy_min_degree():
        """Greedy independent set: always pick min-degree available vertex."""
        in_set = np.zeros(N, dtype=bool)
        removed = np.zeros(N, dtype=bool)
        # Use a simple approach: repeatedly find min-degree non-removed vertex
        # This is O(N^2) in worst case but N is manageable
        remaining = set(range(N))
        # Compute degrees dynamically
        cur_deg = deg.copy()
        while remaining and time.time() < DEADLINE - 5:
            # Find min degree among remaining
            min_deg = min(cur_deg[i] for i in remaining)
            # Pick a random vertex with that degree
            cands = [i for i in remaining if cur_deg[i] == min_deg]
            v = cands[rng.integers(len(cands))]
            in_set[v] = True
            removed[v] = True
            remaining.discard(v)
            # Remove neighbors and update degrees
            for j in adj[v]:
                if not removed[j]:
                    removed[j] = True
                    remaining.discard(j)
            for j in adj[v]:
                if not removed[j]:
                    # This won't happen since we just marked them removed
                    pass
            # Update degrees of remaining vertices that are neighbors of removed vertices
            # Actually, we need to recompute or track degrees of remaining vertices
            # Simpler: just recompute degrees of remaining vertices periodically
            # For efficiency, let's just track it
            pass
        return in_set

    def greedy_max_degree():
        """Greedy independent set: always pick max-degree available vertex (greedy coloring heuristic)."""
        in_set = np.zeros(N, dtype=bool)
        removed = np.zeros(N, dtype=bool)
        remaining = set(range(N))
        cur_deg = deg.copy()
        while remaining and time.time() < DEADLINE - 5:
            max_deg = max(cur_deg[i] for i in remaining)
            cands = [i for i in remaining if cur_deg[i] == max_deg]
            v = cands[rng.integers(len(cands))]
            in_set[v] = True
            removed[v] = True
            remaining.discard(v)
            for j in adj[v]:
                if not removed[j]:
                    removed[j] = True
                    remaining.discard(j)
                    cur_deg[j] = 0  # mark as removed
        return in_set

    # --- Step 3: Simulated Annealing ---
    def sa_search(init_set, max_time):
        """Simulated annealing to improve independent set size."""
        in_set = init_set.copy()
        cur_size = int(in_set.sum())
        best_size = cur_size
        best_set = in_set.copy()

        # Track for each vertex how many of its neighbors are in the set
        nbr_in_set = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            nbr_in_set[adj[i]] += 1

        T = 5.0
        cooling = 0.9999
        min_T = 0.1
        sa_end = time.time() + max_time

        not_in = list(np.where(~in_set)[0])
        # Build a position map for O(1) removal
        pos_map = {v: i for i, v in enumerate(not_in)}

        def remove_v(v):
            if v in pos_map:
                idx = pos_map[v]
                last = not_in[-1]
                not_in[idx] = last
                pos_map[last] = idx
                not_in.pop()
                del pos_map[v]

        def add_v(v):
            if v not in pos_map:
                pos_map[v] = len(not_in)
                not_in.append(v)

        stuck = 0
        while time.time() < sa_end:
            if not not_in:
                break

            # Choose a move: add a vertex or swap
            # Strategy: prefer adding 0-conflict vertices, then SA swaps
            # Sample candidates
            if len(not_in) > 500:
                sample_idx = rng.integers(0, len(not_in), size=min(500, len(not_in)))
                sample = np.array([not_in[i] for i in sample_idx])
            else:
                sample = np.array(not_in)

            # Find min conflict count in sample
            conflicts = nbr_in_set[sample]
            min_conf = int(np.min(conflicts))

            if min_conf == 0:
                # Greedy addition
                cands = sample[conflicts == 0]
                v = int(cands[rng.integers(len(cands))])
                in_set[v] = True
                cur_size += 1
                nbr_in_set[adj[v]] += 1
                remove_v(v)
                stuck = 0
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = in_set.copy()
                continue

            # SA move: pick a vertex with min conflicts, try to add it
            cands = sample[conflicts == min_conf]
            v = int(cands[rng.integers(len(cands))])
            k = int(nbr_in_set[v])

            # To add v, we must remove all its neighbors that are in the set
            neighbors_in_set = np.intersect1d(adj[v], np.where(in_set)[0])
            if len(neighbors_in_set) > 0:
                delta = 1 - len(neighbors_in_set)
            else:
                delta = 1

            # Acceptance probability
            if delta >= 0:
                accept = True
            else:
                if delta / T < -50:
                    accept = False
                else:
                    accept = rng.random() < np.exp(delta / T)

            if accept:
                # Execute the move
                for u in neighbors_in_set:
                    in_set[u] = False
                    cur_size -= 1
                    nbr_in_set[adj[u]] -= 1
                    add_v(int(u))
                in_set[v] = True
                cur_size += 1
                nbr_in_set[adj[v]] += 1
                remove_v(v)

                if cur_size > best_size:
                    best_size = cur_size
                    best_set = in_set.copy()
                    stuck = 0
                else:
                    stuck += 1
            else:
                stuck += 1

            # Cooling and restart
            if stuck > 500:
                T = max(T * 1.5, 3.0)
                stuck = 0
            T *= cooling
            if T < min_T:
                T = min_T

        return best_set, best_size

    # --- Step 4: Exact 1-move polish ---
    def exact_polish(init_set, max_time):
        """Try all single removal + multiple additions."""
        in_set = init_set.copy()
        cur_size = int(in_set.sum())
        best_size = cur_size
        best_set = in_set.copy()
        polish_end = time.time() + max_time

        nbr_in_set = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            nbr_in_set[adj[i]] += 1

        # First: add all 0-conflict vertices
        not_in = np.where(~in_set)[0]
        if len(not_in) > 0:
            zero_cands = not_in[nbr_in_set[not_in] == 0]
            for v in zero_cands:
                in_set[v] = True
                cur_size += 1
                nbr_in_set[adj[v]] += 1
            if cur_size > best_size:
                best_size = cur_size
                best_set = in_set.copy()

        # Try single swaps: remove one vertex, add all that become 0-conflict
        in_set_list = list(np.where(in_set)[0])
        for i in in_set_list:
            if time.time() >= polish_end:
                break
            # Temporarily remove i
            in_set[i] = False
            cur_size -= 1
            nbr_in_set[adj[i]] -= 1

            # Find all 0-conflict vertices not in set
            not_in_now = np.where(~in_set)[0]
            if len(not_in_now) > 0:
                addable = not_in_now[nbr_in_set[not_in_now] == 0]
                gain = len(addable) - 1  # net gain
                if gain >= 1:
                    for v in addable:
                        in_set[v] = True
                        cur_size += 1
                        nbr_in_set[adj[v]] += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_set = in_set.copy()
                else:
                    # Revert
                    for v in addable:
                        in_set[v] = False
                        cur_size -= 1
                        nbr_in_set[adj[v]] -= 1
                    in_set[i] = True
                    cur_size += 1
                    nbr_in_set[adj[i]] += 1
            else:
                in_set[i] = True
                cur_size += 1
                nbr_in_set[adj[i]] += 1

        return best_set, best_size

    # --- Step 5: Run multiple restarts ---
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)

    max_restarts = 60
    for r in range(max_restarts):
        if time.time() > DEADLINE - 15:
            break

        # Greedy initialization (alternate strategies)
        if r % 3 == 0:
            init = greedy_random_order()
        elif r % 3 == 1:
            init = greedy_max_degree()
        else:
            # Seeded random
            in_set = np.zeros(N, dtype=bool)
            removed = np.zeros(N, dtype=bool)
            order = rng.permutation(N)
            for i in order:
                if removed[i]:
                    continue
                in_set[i] = True
                removed[i] = True
                for j in adj[i]:
                    removed[j] = True
            init = in_set

        sz = int(init.sum())
        if sz > best_size:
            best_size = sz
            best_in_set = init.copy()

        # SA improvement
        remaining = DEADLINE - 15 - time.time()
        if remaining < 3:
            break
        sa_time = min(remaining * 0.4, 12.0)
        if sa_time < 2:
            sa_time = 2.0

        sa_result, sa_sz = sa_search(init, sa_time)
        if sa_sz > best_size:
            best_size = sa_sz
            best_in_set = sa_result.copy()

        # Polish
        remaining = DEADLINE - 10 - time.time()
        if remaining > 2 and sa_sz >= sz:
            pol_time = min(remaining * 0.3, 6.0)
            if pol_time > 1:
                pol_result, pol_sz = exact_polish(sa_result, pol_time)
                if pol_sz > best_size:
                    best_size = pol_sz
                    best_in_set = pol_result.copy()

    # Final polish on best
    remaining = DEADLINE - 2 - time.time()
    if remaining > 1 and best_size > 0:
        final_time = min(remaining * 0.8, 5.0)
        if final_time > 0.5:
            fr, fs = exact_polish(best_in_set, final_time)
            if fs > best_size:
                best_size = fs
                best_in_set = fr

    # --- Step 6: Combine W4 and W1 vectors ---
    final_w4 = A[np.where(best_in_set)[0]]
    w1_arr = np.array(w1, dtype=np.int64)

    # Verify W1 vectors don't conflict with each other or with selected W4
    # W1 vectors: only one nonzero entry of ±2. Two W1 vectors are orthogonal unless same position.
    # W1 at position i with value s1 and W1 at position j with value s2: dot = 0 if i≠j, 4 if i=j and s1=s2
    # So W1 vectors at different positions are fine. Same position with opposite signs: dot = -4, which is < 2, so OK.
    # Same position with same sign: dot = 4 > 2, CONFLICT!
    # So we can only pick at most one W1 per position (or two with opposite signs).
    # Actually, dot = 4 means conflict. dot = -4 is fine (<= 2).
    # So for each position, we can pick both +2 and -2? dot = (2)(−2) = −4 ≤ 2. Yes!
    # So all W1 vectors are mutually non-conflicting? No:
    # W1[i,+2] and W1[i,-2]: dot = 2*(-2) = -4. OK.
    # W1[i,+2] and W1[j,+2] (i≠j): dot = 0. OK.
    # So ALL W1 vectors are pairwise non-conflicting! Great.

    # Check W1 vs W4: dot(W1[i,s*2], W4) = 2*s*W4[i]
    # W4[i] ∈ {-1, 0, 1}, so dot ∈ {-2, 0, 2}
    # All ≤ 2. So no conflicts between W1 and W4!

    # Therefore, we can always include all W1 vectors.
    result = np.vstack([final_w4, w1_arr])

    return result

# EVOLVE-BLOCK-END
