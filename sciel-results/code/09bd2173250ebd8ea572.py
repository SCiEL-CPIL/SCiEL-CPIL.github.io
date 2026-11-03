<<<<<<< SEARCH
    # Multiple restarts with different seeds
    for restart in range(15):
        if time.time() > deadline - 20.0:
            break
        np.random.seed(100 + restart)
        random.seed(100 + restart)

        # Randomized greedy
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if adj_mask[v, u])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove

        S = current_set.copy()
        S_arr = np.zeros(n_sup, dtype=np.uint8)
        for v in S:
            S_arr[v] = 1
        current_size = len(S)
        T = 1.0
        cooling = 0.9995

        for step2 in range(80000):
            if time.time() > deadline - 20.0:
                break
            v = random.randint(0, n_sup - 1)

            if S_arr[v]:
                if random.random() < 0.05:
                    S_arr[v] = 0
                    S.discard(v)
                    current_size -= 1
                continue

            N = np.where(adj_mask[v] & S_arr)[0]
            n_N = len(N)

            if n_N == 0:
                S_arr[v] = 1
                S.add(v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                delta = 1 - n_N
                if delta >= 0:
                    S_arr[N] = 0
                    S.difference_update(N.tolist())
                    S_arr[v] = 1
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                else:
                    if T > 0.01:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S_arr[N] = 0
                            S.difference_update(N.tolist())
                            S_arr[v] = 1
                            S.add(v)
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_set = S.copy()

            T = max(0.01, T * cooling)
=======
    # Multiple restarts with different seeds
    for restart in range(20):
        if time.time() > deadline - 20.0:
            break
        np.random.seed(100 + restart)
        random.seed(100 + restart)

        # Randomized greedy
        remaining = set(range(n_sup))
        current_set = set()
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if adj_mask[v, u])
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            current_set.add(best_v)
            to_remove = {best_v} | (adj[best_v] & remaining)
            remaining -= to_remove

        S = current_set.copy()
        S_arr = np.zeros(n_sup, dtype=np.uint8)
        for v in S:
            S_arr[v] = 1
        current_size = len(S)
        T = 1.0
        cooling = 0.9997

        for step2 in range(100000):
            if time.time() > deadline - 20.0:
                break
            move_type = random.random()

            if move_type < 0.15 and len(S) >= 2:
                # 2-for-2 swap
                S_list = list(S)
                r1, r2 = random.sample(S_list, 2)
                remaining_S = S - {r1, r2}
                rem_mask = np.zeros(n_sup, dtype=np.uint8)
                for v in remaining_S:
                    rem_mask[v] = 1
                non_S_indices = np.where(~S_arr)[0]
                if len(non_S_indices) < 2:
                    T = max(0.01, T * cooling)
                    continue
                order = non_S_indices[np.argsort(np.sum(adj_mask[non_S_indices] & rem_mask, axis=1))]
                found = False
                for i in range(min(40, len(order))):
                    a = order[i]
                    if S_arr[a]:
                        continue
                    if adj_mask[a, r1] or adj_mask[a, r2]:
                        continue
                    if np.any(adj_mask[a] & rem_mask):
                        continue
                    compat_b = np.where((~S_arr) & (~adj_mask[a]) & (~rem_mask))[0]
                    if len(compat_b) == 0:
                        continue
                    conf_b = np.sum(adj_mask[compat_b] & rem_mask, axis=1)
                    b = compat_b[np.argmin(conf_b)]
                    if np.any(adj_mask[b] & rem_mask):
                        continue
                    S.discard(r1)
                    S.discard(r2)
                    S_arr[r1] = 0
                    S_arr[r2] = 0
                    S.add(a)
                    S.add(b)
                    S_arr[a] = 1
                    S_arr[b] = 1
                    found = True
                    break
                T = max(0.01, T * cooling)
                continue

            v = random.randint(0, n_sup - 1)

            if S_arr[v]:
                if random.random() < 0.04:
                    S_arr[v] = 0
                    S.discard(v)
                    current_size -= 1
                continue

            N = np.where(adj_mask[v] & S_arr)[0]
            n_N = len(N)

            if n_N == 0:
                S_arr[v] = 1
                S.add(v)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = S.copy()
            else:
                delta = 1 - n_N
                if delta >= 0:
                    S_arr[N] = 0
                    S.difference_update(N.tolist())
                    S_arr[v] = 1
                    S.add(v)
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S.copy()
                else:
                    if T > 0.01:
                        p = np.exp(delta / T)
                        if random.random() < p:
                            S_arr[N] = 0
                            S.difference_update(N.tolist())
                            S_arr[v] = 1
                            S.add(v)
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_set = S.copy()

            T = max(0.01, T * cooling)
>>>>>>> REPLACE

