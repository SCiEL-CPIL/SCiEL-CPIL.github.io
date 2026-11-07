            if stagnation_counter > max_stagnation:
                worst_half = np.argsort(fpop)[npop//2:]
                for idx in worst_half:
                    if calls >= budget: break
                    pop[idx] = rng.uniform(-5, 5, dim)
                    fpop[idx] = func(pop[idx])
                    calls += 1
                    if fpop[idx] < best_f:
                        best_f = fpop[idx]
                        best_x = pop[idx].copy()
                stagnation_counter = 0
                F = 1.0
