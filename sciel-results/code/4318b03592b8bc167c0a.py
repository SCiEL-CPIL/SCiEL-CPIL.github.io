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

    def greedy_min_degree():
        """Greedy min-degree heuristic for an independent set."""
        remaining = set(range(N_S))
        s = set()
        while remaining:
            min_deg = float('inf')
            min_node = None
            for v in remaining:
                deg = sum(1 for nb in adj_list[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            s.add(min_node)
            to_remove = {min_node} | {nb for nb in adj_list[min_node] if nb in remaining}
            remaining -= to_remove
        return s

    def local_search_fast(init_set, max_iter=300, seed=None):
        """Vectorized add / 1-swap / 2-swap local search. Returns best set seen."""
        if seed is not None:
            np.random.seed(seed)
        current = set(init_set)
        best = set(current)
        best_size = len(current)
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            in_set = np.zeros(N_S, dtype=bool)
            for v in current:
                in_set[v] = True
            blocked = np.any(adj_bool[in_set], axis=0) if current else np.zeros(N_S, dtype=bool)
            cand = np.where(~(in_set | blocked))[0]
            if len(cand) > 0:
                current.add(int(cand[np.random.randint(len(cand))]))
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
                continue
            # Stuck: 1-swap
            if current:
                v_remove = next(iter(current))
                current.discard(v_remove)
                in2 = np.zeros(N_S, dtype=bool)
                for v in current:
                    in2[v] = True
                bl2 = np.any(adj_bool[in2], axis=0) if current else np.zeros(N_S, dtype=bool)
                cand2 = np.where(~(in2 | bl2))[0]
                if len(cand2) > 0:
                    current.add(int(cand2[np.random.randint(len(cand2))]))
                    if len(current) > best_size:
                        best_size = len(current)
                        best = set(current)
                else:
                    current.add(v_remove)
        return best

    def simulated_annealing(init_set, max_iter=600, seed=None):
        """BQP-style simulated annealing with vectorized energy + multi-swap.
        Energy = number of conflicts within the set; objective = maximize |set| - energy.
        Returns best (conflict-free) set seen."""
        if seed is not None:
            np.random.seed(seed)
        current = set(init_set)
        cur_in = np.zeros(N_S, dtype=bool)
        for v in current:
            cur_in[v] = True
        cur_energy = int((adj_bool[cur_in][:, cur_in].sum() // 2))

        best = set(current)
        best_size = len(current) if cur_energy == 0 else 0
        T0 = 4.0
        Tend = 0.001
        cool = (Tend / T0) ** (1.0 / max(1, max_iter - 1))

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            T = T0 * (cool ** it)

            # pick a move: add / remove / 2-swap
            in_set = cur_in
            blocked = np.any(adj_bool[in_set], axis=0) if current else np.zeros(N_S, dtype=bool)
            add_cand = np.where(~(in_set | blocked))[0]
            remove_cand = np.where(in_set)[0]

            r = np.random.random()
            new_set = set(current)
            if len(add_cand) > 0 and r < 0.5:
                new_set.add(int(add_cand[np.random.randint(len(add_cand))]))
            elif len(remove_cand) > 0 and r < 0.75:
                new_set.discard(int(remove_cand[np.random.randint(len(remove_cand))]))
            elif len(add_cand) > 0 and len(remove_cand) > 0:
                new_set.discard(int(remove_cand[np.random.randint(len(remove_cand))]))
                new_set.add(int(add_cand[np.random.randint(len(add_cand))]))
            elif len(add_cand) > 0:
                new_set.add(int(add_cand[np.random.randint(len(add_cand))]))
            elif len(remove_cand) > 0:
                new_set.discard(int(remove_cand[np.random.randint(len(remove_cand))]))
            else:
                continue

            if new_set == current:
                continue

            new_in = np.zeros(N_S, dtype=bool)
            for v in new_set:
                new_in[v] = True
            new_energy = int((adj_bool[new_in][:, new_in].sum() // 2))
            delta_e = new_energy - cur_energy
            delta_size = len(new_set) - len(current)

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

    # Phase A: greedy + vectorized local search restarts
    while time.time() < deadline - 45.0:
        init_set = greedy_min_degree()
        final_set = local_search_fast(init_set, max_iter=200, seed=np.random.randint(10**9))
        if len(final_set) > best_size:
            best_size = len(final_set)
            best_set = set(final_set)

    # Phase B: simulated annealing from the best set (and fresh restarts)
    for r in range(40):
        if time.time() > deadline - 15.0:
            break
        if r == 0:
            init_set = best_set
        else:
            init_set = greedy_min_degree()
        final_set = simulated_annealing(init_set, max_iter=400, seed=np.random.randint(10**9))
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
