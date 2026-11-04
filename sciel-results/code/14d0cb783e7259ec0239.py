import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11

    # Supports: all 4-element subsets of {0,...,10}
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)

    # Precompute support masks
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)

    # Two supports conflict if they share >= 3 coordinates
    # (then no sign assignment makes them have distance >= 2*sqrt(4)=4)
    conflict = np.zeros((N_S, N_S), dtype=bool)
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            if inter.bit_count() >= 3:
                conflict[i, j] = True
                conflict[j, i] = True
                adj[i].add(j)
                adj[j].add(i)

    adj_arrays = [np.array(list(adj[i]), dtype=np.int64) for i in range(N_S)]
    degrees = np.array([len(adj[i]) for i in range(N_S)])

    rng = np.random.RandomState(12345)

    def greedy_fill(x, blocked):
        """Add all non-blocked, not-yet-in-set nodes."""
        changed = True
        while changed:
            changed = False
            candidates = np.where(~x & ~blocked)[0]
            for v in candidates:
                if not x[v] and not blocked[v]:
                    x[v] = True
                    neighbors = adj_arrays[v]
                    if len(neighbors) > 0:
                        blocked[neighbors] = True
                    changed = True
        return x, blocked

    def compute_blocked(x):
        blocked = np.zeros(N_S, dtype=bool)
        in_set = np.where(x)[0]
        for v in in_set:
            neighbors = adj_arrays[v]
            if len(neighbors) > 0:
                blocked[neighbors] = True
        return blocked

    def local_search(x0, max_iter, time_limit):
        current_x = x0.copy()
        blocked = compute_blocked(current_x)
        current_x, blocked = greedy_fill(current_x, blocked)
        current_score = int(np.sum(current_x))

        best_x = current_x.copy()
        best_score = current_score

        for it in range(max_iter):
            if time.time() > time_limit:
                break

            r = rng.random()

            if r < 0.30:
                # Remove 1-3 nodes, refill
                in_set = np.where(current_x)[0]
                if len(in_set) < 1:
                    continue
                n_remove = 1 + rng.randint(0, 2)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    current_x[v] = False
                blocked = compute_blocked(current_x)
                current_x, blocked = greedy_fill(current_x, blocked)
                current_score = int(np.sum(current_x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

            elif r < 0.55:
                # Swap: remove one, add one
                in_set = np.where(current_x)[0]
                if len(in_set) == 0:
                    continue
                u = in_set[rng.randint(len(in_set))]
                temp_x = current_x.copy()
                temp_x[u] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) > 0:
                    v = candidates[rng.randint(len(candidates))]
                    current_x[u] = False
                    current_x[v] = True
                    blocked = compute_blocked(current_x)
                    current_x, blocked = greedy_fill(current_x, blocked)
                    current_score = int(np.sum(current_x))
                    if current_score > best_score:
                        best_score = current_score
                        best_x = current_x.copy()

            elif r < 0.80:
                # Double swap
                in_set = np.where(current_x)[0]
                if len(in_set) < 2:
                    continue
                u1_idx, u2_idx = rng.choice(len(in_set), size=2, replace=False)
                u1, u2 = in_set[u1_idx], in_set[u2_idx]
                temp_x = current_x.copy()
                temp_x[u1] = False
                temp_x[u2] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) >= 2:
                    v1_idx, v2_idx = rng.choice(len(candidates), size=2, replace=False)
                    v1, v2 = candidates[v1_idx], candidates[v2_idx]
                    if not conflict[v1, v2]:
                        current_x[u1] = False
                        current_x[u2] = False
                        current_x[v1] = True
                        current_x[v2] = True
                        blocked = compute_blocked(current_x)
                        current_x, blocked = greedy_fill(current_x, blocked)
                        current_score = int(np.sum(current_x))
                        if current_score > best_score:
                            best_score = current_score
                            best_x = current_x.copy()

            elif r < 0.92:
                # Triple swap
                in_set = np.where(current_x)[0]
                if len(in_set) < 3:
                    continue
                u_indices = rng.choice(len(in_set), size=3, replace=False)
                u_list = in_set[u_indices]
                temp_x = current_x.copy()
                for u in u_list:
                    temp_x[u] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) >= 3:
                    v_indices = rng.choice(len(candidates), size=3, replace=False)
                    v_list = candidates[v_indices]
                    # Check all pairs among v_list are non-conflicting
                    ok = True
                    for a in range(3):
                        for b in range(a + 1, 3):
                            if conflict[v_list[a], v_list[b]]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        for u in u_list:
                            current_x[u] = False
                        for v in v_list:
                            current_x[v] = True
                        blocked = compute_blocked(current_x)
                        current_x, blocked = greedy_fill(current_x, blocked)
                        current_score = int(np.sum(current_x))
                        if current_score > best_score:
                            best_score = current_score
                            best_x = current_x.copy()

            else:
                # Kick: remove 4-6 nodes
                in_set = np.where(current_x)[0]
                if len(in_set) < 4:
                    continue
                n_remove = 4 + rng.randint(0, 2)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    current_x[v] = False
                blocked = compute_blocked(current_x)
                current_x, blocked = greedy_fill(current_x, blocked)
                current_score = int(np.sum(current_x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

        return best_x, best_score

    best_support_set = set()
    best_size = 0

    # Multiple restarts with varied initialization
    num_restarts = 120
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break

        if r < 30:
            # Greedy with random ordering
            x0 = np.zeros(N_S, dtype=bool)
            order = rng.permutation(N_S)
            blocked = np.zeros(N_S, dtype=bool)
            for v in order:
                if not x0[v] and not blocked[v]:
                    x0[v] = True
                    neighbors = adj_arrays[v]
                    if len(neighbors) > 0:
                        blocked[neighbors] = True

        elif r < 60:
            # Greedy prioritizing low-degree nodes with noise
            x0 = np.zeros(N_S, dtype=bool)
            noisy_deg = degrees.astype(float) + rng.random(N_S) * 5
            order = np.argsort(noisy_deg)
            blocked = np.zeros(N_S, dtype=bool)
            for v in order:
                if not x0[v] and not blocked[v]:
                    x0[v] = True
                    neighbors = adj_arrays[v]
                    if len(neighbors) > 0:
                        blocked[neighbors] = True

        elif r < 90:
            # Random seed then greedy fill
            x0 = np.zeros(N_S, dtype=bool)
            n_random = rng.randint(3, 15)
            random_nodes = rng.choice(N_S, size=n_random, replace=False)
            blocked = np.zeros(N_S, dtype=bool)
            for v in random_nodes:
                if not blocked[v]:
                    x0[v] = True
                    neighbors = adj_arrays[v]
                    if len(neighbors) > 0:
                        blocked[neighbors] = True
            x0, blocked = greedy_fill(x0, blocked)

        else:
            # Perturbed best
            if best_size > 0:
                x0 = np.zeros(N_S, dtype=bool)
                best_nodes = list(best_support_set)
                for v in best_nodes:
                    x0[v] = True
                n_remove = rng.randint(2, 6)
                n_remove = min(n_remove, len(best_nodes))
                to_remove = rng.choice(len(best_nodes), size=n_remove, replace=False)
                for idx in to_remove:
                    x0[best_nodes[idx]] = False
                blocked = compute_blocked(x0)
                x0, blocked = greedy_fill(x0, blocked)
            else:
                x0 = np.zeros(N_S, dtype=bool)
                order = rng.permutation(N_S)
                blocked = np.zeros(N_S, dtype=bool)
                for v in order:
                    if not x0[v] and not blocked[v]:
                        x0[v] = True
                        neighbors = adj_arrays[v]
                        if len(neighbors) > 0:
                            blocked[neighbors] = True

        remaining_time = deadline - time.time()
        max_iter = min(30000, max(5000, int(remaining_time * 400)))
        x_best, size = local_search(x0, max_iter, deadline - 5.0)

        if size > best_size:
            best_size = size
            best_support_set = set(np.where(x_best)[0].tolist())

    # Final polish
    if time.time() < deadline - 2.0 and best_support_set:
        polish_x = np.zeros(N_S, dtype=bool)
        for v in best_support_set:
            polish_x[v] = True
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            blocked = compute_blocked(polish_x)
            candidates = np.where(~polish_x & ~blocked)[0]
            for v in candidates:
                if not polish_x[v] and not blocked[v]:
                    polish_x[v] = True
                    changed = True
        final_size = int(np.sum(polish_x))
        if final_size > best_size:
            best_size = final_size
            best_support_set = set(np.where(polish_x)[0].tolist())

    # Build vectors: Type A (±2e_i) + Type D (sign vectors on supports)
    vectors = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors.append(v2)

    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)

    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)

    return np.array(vectors, dtype=np.int8)
