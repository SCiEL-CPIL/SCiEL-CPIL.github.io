            if fronts_all:
                pareto_idx_all = fronts_all[0]
                # Polish top 5 Pareto points
                dists_all = crowding_distance(F_all, pareto_idx_all)
                sorted_pareto_all = sorted(pareto_idx_all, key=lambda i: dists_all.get(i, 0.0), reverse=True)
                
                n_polish = min(5, len(sorted_pareto_all))
                # Calculate bounds for weights
                f_min = np.min(F_all[pareto_idx_all], axis=0)
                f_max = np.max(F_all[pareto_idx_all], axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                
                for k in range(n_polish):
                    if used >= budget - 5 or time.time() > deadline - 5:
                        break
                    idx = sorted_pareto_all[k]
                    f_curr = F_all[idx]
                    # Weights: higher weight on objectives where current point is worse (higher)
                    # Normalized position: 0 is best, 1 is worst
                    pos = (f_curr - f_min) / f_range
                    weights = 1.0 - pos
                    # Ensure weights are positive
                    weights = np.maximum(weights, 0.1)
                    x_polished, f_polished = local_polish(all_X[idx], all_F[idx], budget - used, weights)
