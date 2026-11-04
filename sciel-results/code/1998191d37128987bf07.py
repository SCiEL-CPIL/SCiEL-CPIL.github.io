import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)

    # 2. Build conflict graph: conflict if intersection size is exactly 3
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        mi = masks[i]
        for j in range(i + 1, N_S):
            if (mi & masks[j]).bit_count() == 3:
                adj[i].add(j)
                adj[j].add(i)

    adj_list = [list(s) for s in adj]

    # Vectorized adjacency (boolean matrix) for fast blocked-set / energy ops
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        if adj_list[i]:
            adj_bool[i, adj_list[i]] = True

    # 3. Search machinery -------------------------------------------------

    def greedy_randomized():
        """Vectorized greedy with randomized tie-breaking.
        At each step, pick a node with minimum degree among non-blocked nodes,
        breaking ties randomly."""
        in_set = np.zeros(N_S, dtype=bool)
        s = set()
        for _ in range(N_S):
            if time.time() > deadline - 10.0:
                break
            blocked = np.any(adj_bool[in_set], axis=0) if s else np.zeros(N_S, dtype=bool)
            available = ~(in_set | blocked)
            if not available.any():
                break
            # Compute degree among available nodes
            avail_idx = np.where(available)[0]
            if len(avail_idx) == 1:
                chosen = avail_idx[0]
            else:
                # Degree of each available node within the available set
                degs = adj_bool[avail_idx][:, avail_idx].sum(axis=1)
                min_deg = degs.min()
                min_candidates = avail_idx[degs == min_deg]
                if len(min_candidates) == 1:
                    chosen = min_candidates[0]
                else:
                    chosen = min_candidates[np.random.randint(len(min_candidates))]
            s.add(int(chosen))
            in_set[chosen] = True
        return s

    def local_search_1swap(init_set, max_iter=500, seed=None):
        """Iterative 1-swap local search: repeatedly find best (remove, add) pair
        that increases set size or maintains size while reducing conflicts.
        Uses vectorized operations for speed."""
        if seed is not None:
            np.random.seed(seed)
        current = set(init_set)
        best = set(current)
        best_size = len(current)
        cur_in = np.zeros(N_S, dtype=bool)
        for v in current:
            cur_in[v] = True

        for it in range(max_iter):
            if time.time() > deadline - 8.0:
                break
            # Try to add: find non-blocked candidates
            blocked = np.any(adj_bool[cur_in], axis=0) if current else np.zeros(N_S, dtype=bool)
            add_cand = np.where(~(cur_in | blocked))[0]
            if len(add_cand) > 0:
                # Add a random candidate
                chosen = int(add_cand[np.random.randint(len(add_cand))])
                current.add(chosen)
                cur_in[chosen] = True
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
                continue

            # Stuck: try 1-swap (remove one, add one)
            if not current:
                break
            cur_nodes = np.where(cur_in)[0]
            improved = False
            # Try each removal candidate
            for rm_idx in np.random.permutation(len(cur_nodes)):
                if time.time() > deadline - 8.0:
                    break
                rm_node = cur_nodes[rm_idx]
                # Remove rm_node
                cur_in[rm_node] = False
                current.discard(int(rm_node))

                # Find best add candidate
                blocked2 = np.any(adj_bool[cur_in], axis=0) if current else np.zeros(N_S, dtype=bool)
                add_cand2 = np.where(~(cur_in | blocked2))[0]
                if len(add_cand2) > 0:
                    # Pick the one that maximizes future potential (min degree)
                    if len(add_cand2) > 1:
                        degs2 = adj_bool[add_cand2][:, add_cand2].sum(axis=1)
                        min_d = degs2.min()
                        best_cands = add_cand2[degs2 == min_d]
                        add_node = int(best_cands[np.random.randint(len(best_cands))])
                    else:
                        add_node = int(add_cand2[0])
                    cur_in[add_node] = True
                    current.add(add_node)
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
                    improved = True
                else:
                    # Can't add, restore
                    cur_in[rm_node] = True
                    current.add(int(rm_node))

                if improved:
                    break

            if not improved:
                # Try 2-swap: remove two, add two
                if len(current) >= 2:
                    cur_nodes2 = np.where(cur_in)[0]
                    perm = np.random.permutation(len(cur_nodes2))
                    found_swap = False
                    for i in range(min(5, len(perm))):
                        if time.time() > deadline - 8.0:
                            break
                        j = (i + 1) % len(perm)
                        rm1, rm2 = int(cur_nodes2[perm[i]]), int(cur_nodes2[perm[j]])
                        cur_in[rm1] = False
                        cur_in[rm2] = False
                        current.discard(rm1)
                        current.discard(rm2)

                        blocked3 = np.any(adj_bool[cur_in], axis=0) if current else np.zeros(N_S, dtype=bool)
                        add_cand3 = np.where(~(cur_in | blocked3))[0]
                        if len(add_cand3) >= 2:
                            # Add two best candidates
                            degs3 = adj_bool[add_cand3][:, add_cand3].sum(axis=1)
                            min_d3 = degs3.min()
                            best3 = add_cand3[degs3 == min_d3]
                            if len(best3) > 2:
                                idx3 = np.random.choice(len(best3), 2, replace=False)
                            else:
                                idx3 = np.arange(len(best3))
                            add1, add2 = int(best3[idx3[0]]), int(best3[idx3[1]])
                            cur_in[add1] = True
                            cur_in[add2] = True
                            current.add(add1)
                            current.add(add2)
                            if len(current) > best_size:
                                best_size = len(current)
                                best = set(current)
                            found_swap = True
                        else:
                            # Restore
                            cur_in[rm1] = True
                            cur_in[rm2] = True
                            current.add(rm1)
                            current.add(rm2)

                        if found_swap:
                            break

                    if not found_swap:
                        # No improvement in 2-swap either; try random perturbation
                        if len(current) > 2:
                            rm_node = int(np.random.choice(list(current)))
                            cur_in[rm_node] = False
                            current.discard(rm_node)
                            blocked4 = np.any(adj_bool[cur_in], axis=0) if current else np.zeros(N_S, dtype=bool)
                            add_cand4 = np.where(~(cur_in | blocked4))[0]
                            if len(add_cand4) > 0:
                                add_node = int(add_cand4[np.random.randint(len(add_cand4))])
                                cur_in[add_node] = True
                                current.add(add_node)
                            else:
                                cur_in[rm_node] = True
                                current.add(rm_node)
                        else:
                            break
                else:
                    break

        return best

    def simulated_annealing(init_set, max_iter=600, seed=None):
        """Simulated annealing with vectorized energy and 1-swap/2-swap moves."""
        if seed is not None:
            np.random.seed(seed)
        current = set(init_set)
        cur_in = np.zeros(N_S, dtype=bool)
        for v in current:
            cur_in[v] = True
        cur_energy = int((adj_bool[cur_in][:, cur_in].sum() // 2))

        best = set(current)
        best_size = len(current) if cur_energy == 0 else 0
        T0 = 3.0
        Tend = 0.005
        cool = (Tend / T0) ** (1.0 / max(1, max_iter - 1))

        for it in range(max_iter):
            if time.time() > deadline - 6.0:
                break
            T = T0 * (cool ** it)

            blocked = np.any(adj_bool[cur_in], axis=0) if current else np.zeros(N_S, dtype=bool)
            add_cand = np.where(~(cur_in | blocked))[0]
            remove_cand = np.where(cur_in)[0]

            r = np.random.random()
            new_in = cur_in.copy()
            new_set = set(current)

            if len(add_cand) > 0 and r < 0.4:
                chosen = int(add_cand[np.random.randint(len(add_cand))])
                new_set.add(chosen)
                new_in[chosen] = True
            elif len(remove_cand) > 0 and r < 0.6:
                chosen = int(remove_cand[np.random.randint(len(remove_cand))])
                new_set.discard(chosen)
                new_in[chosen] = False
            elif len(add_cand) > 0 and len(remove_cand) > 0 and r < 0.9:
                rm = int(remove_cand[np.random.randint(len(remove_cand))])
                add = int(add_cand[np.random.randint(len(add_cand))])
                new_set.discard(rm)
                new_set.add(add)
                new_in[rm] = False
                new_in[add] = True
            elif len(add_cand) > 0:
                chosen = int(add_cand[np.random.randint(len(add_cand))])
                new_set.add(chosen)
                new_in[chosen] = True
            elif len(remove_cand) > 0:
                chosen = int(remove_cand[np.random.randint(len(remove_cand))])
                new_set.discard(chosen)
                new_in[chosen] = False
            else:
                continue

            new_energy = int((adj_bool[new_in][:, new_in].sum() // 2))
            delta_e = new_energy - cur_energy
            delta_size = len(new_set) - len(current)

            # Accept if: reduces energy, or increases size with same energy,
            # or with probability based on temperature
            if delta_e < 0 or (delta_size > 0 and delta_e == 0) or \
               (T > Tend and np.random.random() < np.exp(-(delta_e - delta_size) / T)):
                current = new_set
                cur_in = new_in
                cur_energy = new_energy
                if cur_energy == 0 and len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
        return best

    # 4. Run search -------------------------------------------------------
    np.random.seed(42)
    best_size = 0
    best_set = set()

    # Phase A: randomized greedy + 1-swap local search restarts
    while time.time() < deadline - 50.0:
        init_set = greedy_randomized()
        final_set = local_search_1swap(init_set, max_iter=300, seed=np.random.randint(10**9))
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = set(final_set)

    # Phase B: simulated annealing from the best set and fresh restarts
    for r in range(60):
        if time.time() > deadline - 12.0:
            break
        if r == 0:
            init_set = best_set
        elif r % 3 == 1:
            init_set = greedy_randomized()
        else:
            # Perturb the best set
            init_set = set(best_set)
            if init_set:
                # Remove 1-3 random elements
                elems = list(init_set)
                n_remove = min(len(elems), np.random.randint(1, 4))
                for _ in range(n_remove):
                    if elems:
                        idx = np.random.randint(len(elems))
                        init_set.discard(elems.pop(idx))
        final_set = simulated_annealing(init_set, max_iter=350, seed=np.random.randint(10**9))
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = set(final_set)

    # Phase C: final polish - local search from best
    if time.time() < deadline - 8.0:
        final_set = local_search_1swap(best_set, max_iter=200, seed=np.random.randint(10**9))
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = set(final_set)

    # 5. Construct vectors: all sign patterns on each selected support.
    #    For supports with pairwise intersection <= 2, all +/- patterns are
    #    mutually distance 4 (>= max norm 2), so the whole cube is valid.
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)

    return np.array(vectors, dtype=np.int8)
