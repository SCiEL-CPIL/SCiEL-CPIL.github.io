<<<<<<< SEARCH
        # Local search with multiple restarts
        best_val = float('inf')
        best_S = frozenset()
        best_ccliques = []
        visited = set()

        def _local_search(S_init, max_iters=200):
            nonlocal best_val, best_S, best_ccliques
            S_cur = S_init
            val_cur, ccl_cur = _eval(S_cur)
            if val_cur < best_val:
                best_val = val_cur
                best_S = S_cur
                best_ccliques = ccl_cur
            visited.add(S_cur)
            for _ in range(max_iters):
                if _time_up():
                    break
                improved = False
                # Try adding one vertex
                for i in range(m):
                    if i in S_cur or d_v[i] < 2:
                        continue
                    S_new = S_cur | frozenset([i])
                    if S_new in visited:
                        continue
                    val_new, ccl_new = _eval(S_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_S = S_new
                        best_ccliques = ccl_new
                        S_cur = S_new
                        visited.add(S_new)
                        improved = True
                        break
                if improved:
                    continue
                # Try removing one vertex
                for i in list(S_cur):
                    S_new = S_cur - frozenset([i])
                    if S_new in visited:
                        continue
                    val_new, ccl_new = _eval(S_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_S = S_new
                        best_ccliques = ccl_new
                        S_cur = S_new
                        visited.add(S_new)
                        improved = True
                        break
                if improved:
                    continue
                # Try swap
                for i_add in range(m):
                    if i_add in S_cur or d_v[i_add] < 2:
                        continue
                    for i_rem in list(S_cur):
                        S_new = (S_cur - frozenset([i_rem])) | frozenset([i_add])
                        if S_new in visited:
                            continue
                        val_new, ccl_new = _eval(S_new)
                        if val_new < best_val:
                            best_val = val_new
                            best_S = S_new
                            best_ccliques = ccl_new
                            S_cur = S_new
                            visited.add(S_new)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            return S_cur, best_val

        # Initial candidates
        candidates_S = []
        S_all = frozenset(i for i in range(m) if d_v[i] >= 2)
        candidates_S.append(S_all)
        candidates_S.append(frozenset())
        for i in range(m):
            if d_v[i] >= 2:
                candidates_S.append(frozenset([i]))
        # Greedy build by degree
        S_greedy = frozenset()
        for i in sorted(range(m), key=lambda x: d_v[x], reverse=True):
            if d_v[i] >= 2:
                S_greedy = S_greedy | frozenset([i])
        candidates_S.append(S_greedy)
        # Random samples
        import random as _rnd
        for _ in range(20):
            if _time_up():
                break
            S_rand = frozenset(i for i in range(m) if d_v[i] >= 2 and _rnd.random() < 0.5)
            candidates_S.append(S_rand)

        for S_init in candidates_S:
            if _time_up():
                break
            if S_init in visited:
                continue
            _local_search(S_init)
=======
        # Simulated annealing / tabu search
        best_val = float('inf')
        best_S = frozenset()
        best_ccliques = []
        
        import random as _rnd
        _rng_sa = Random(42)
        
        def _sa_search(S_init, n_iters=500):
            nonlocal best_val, best_S, best_ccliques
            S_cur = S_init
            val_cur, ccl_cur = _eval(S_cur)
            if val_cur < best_val:
                best_val = val_cur
                best_S = S_cur
                best_ccliques = ccl_cur
            
            # Tabu list: (vertex, action) -> iteration
            tabu = {}
            tabu_tenure = 10
            temp = 5.0
            cool_rate = 0.95
            
            for it in range(n_iters):
                if _time_up():
                    break
                temp *= cool_rate
                
                # Generate random neighbor
                action = _rng_sa.choice(['add', 'remove', 'swap'])
                S_new = S_cur
                
                if action == 'add':
                    candidates = [i for i in range(m) if i not in S_cur and d_v[i] >= 2]
                    if not candidates:
                        continue
                    i = _rng_sa.choice(candidates)
                    S_new = S_cur | frozenset([i])
                    tabu_key = (i, 'add')
                elif action == 'remove':
                    if not S_cur:
                        continue
                    i = _rng_sa.choice(list(S_cur))
                    S_new = S_cur - frozenset([i])
                    tabu_key = (i, 'remove')
                else:  # swap
                    candidates_add = [i for i in range(m) if i not in S_cur and d_v[i] >= 2]
                    if not candidates_add or not S_cur:
                        continue
                    i_add = _rng_sa.choice(candidates_add)
                    i_rem = _rng_sa.choice(list(S_cur))
                    S_new = (S_cur - frozenset([i_rem])) | frozenset([i_add])
                    tabu_key = (i_add, 'swap')
                
                # Check tabu
                if tabu_key in tabu and it - tabu[tabu_key] < tabu_tenure:
                    continue
                
                val_new, ccl_new = _eval(S_new)
                
                # Accept if better or with probability exp(-delta/temp)
                if val_new < val_cur or (_rng_sa.random() < __import__('math').exp(-(val_new - val_cur) / max(temp, 0.01))):
                    S_cur = S_new
                    val_cur = val_new
                    tabu[tabu_key] = it
                    if val_cur < best_val:
                        best_val = val_cur
                        best_S = S_cur
                        best_ccliques = ccl_new
        
        # Initial candidates
        candidates_S = []
        S_all = frozenset(i for i in range(m) if d_v[i] >= 2)
        candidates_S.append(S_all)
        candidates_S.append(frozenset())
        for i in range(m):
            if d_v[i] >= 2:
                candidates_S.append(frozenset([i]))
        # Greedy build by degree
        S_greedy = frozenset()
        for i in sorted(range(m), key=lambda x: d_v[x], reverse=True):
            if d_v[i] >= 2:
                S_greedy = S_greedy | frozenset([i])
        candidates_S.append(S_greedy)
        # Random samples
        for _ in range(15):
            if _time_up():
                break
            S_rand = frozenset(i for i in range(m) if d_v[i] >= 2 and _rnd_sa.random() < 0.5)
            candidates_S.append(S_rand)

        for S_init in candidates_S:
            if _time_up():
                break
            _sa_search(S_init)
>>>>>>> REPLACE
