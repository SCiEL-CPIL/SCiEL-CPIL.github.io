<<<<<<< SEARCH
        # Polish the best solution found
        if best_solution:
            polished = local_search_polish(best_solution)
            if len(polished) < best_size:
                best_size = len(polished)
                best_solution = set(polished)

        # Multiple restarts with different greedy initializations
        if time.time() - start_time < DEADLINE - 10:
            for restart in range(3):
                # Perturb the greedy choice
                def greedy_cover_perturbed(perturb=0):
                    selected = set()
                    unc = (1 << m) - 1
                    while unc:
                        best_v = -1
                        best_cov = 0
                        best_covs = []
                        for v in candidates:
                            if v in selected:
                                continue
                            cov = bin(unc & vertex_clique_mask[v]).count('1')
                            best_covs.append((cov, v))
                        best_covs.sort(reverse=True)
                        # Pick from top few with perturbation
                        top_k = min(3, len(best_covs))
                        if perturb > 0 and top_k > 1:
                            idx = perturb % top_k
                        else:
                            idx = 0
                        best_cov, best_v = best_covs[idx]
                        if best_cov == 0:
                            break
                        selected.add(best_v)
                        unc &= ~vertex_clique_mask[best_v]
                    return selected

                init_sol = greedy_cover_perturbed(restart)
                if len(init_sol) < best_size:
                    best_size = len(init_sol)
                    best_solution = set(init_sol)

                # Quick B&B from this start
                branch_and_bound(set(init_sol), 0, (1 << m) - 1, 0)
                # But we need to track which are selected... let's fix
                # Actually branch_and_bound expects empty start, so let's just polish
                if best_solution:
                    polished = local_search_polish(best_solution)
                    if len(polished) < best_size:
                        best_size = len(polished)
                        best_solution = set(polished)

                if time.time() - start_time > DEADLINE - 5:
                    break

        if best_solution is None:
            return sorted(init_sol)

        # Final verification: ensure all cliques are covered
        final_mask = 0
        for v in best_solution:
            final_mask |= vertex_clique_mask[v]
        if final_mask != (1 << m) - 1:
            # Fallback to greedy
            return sorted(init_sol)

        return sorted(best_solution)
=======
        # Polish the best solution found
        if best_solution:
            polished = local_search_polish(best_solution)
            if len(polished) < best_size:
                best_size = len(polished)
                best_solution = set(polished)

        # Multiple restarts with different greedy initializations
        if time.time() - start_time < DEADLINE - 10:
            for restart in range(5):
                # Perturb the greedy choice
                def greedy_cover_perturbed(perturb=0):
                    selected = set()
                    unc = (1 << m) - 1
                    while unc:
                        best_v = -1
                        best_cov = 0
                        best_covs = []
                        for v in candidates:
                            if v in selected:
                                continue
                            cov = bin(unc & vertex_clique_mask[v]).count('1')
                            best_covs.append((cov, v))
                        best_covs.sort(reverse=True)
                        # Pick from top few with perturbation
                        top_k = min(3, len(best_covs))
                        if perturb > 0 and top_k > 1:
                            idx = perturb % top_k
                        else:
                            idx = 0
                        best_cov, best_v = best_covs[idx]
                        if best_cov == 0:
                            break
                        selected.add(best_v)
                        unc &= ~vertex_clique_mask[best_v]
                    return selected

                init_sol = greedy_cover_perturbed(restart)
                if len(init_sol) < best_size:
                    best_size = len(init_sol)
                    best_solution = set(init_sol)

                # Polish from this start
                if best_solution:
                    polished = local_search_polish(best_solution)
                    if len(polished) < best_size:
                        best_size = len(polished)
                        best_solution = set(polished)

                if time.time() - start_time > DEADLINE - 5:
                    break

        if best_solution is None:
            return sorted(init_sol)

        # Final verification: ensure all cliques are covered
        final_mask = 0
        for v in best_solution:
            final_mask |= vertex_clique_mask[v]
        if final_mask != (1 << m) - 1:
            # Fallback to greedy
            return sorted(init_sol)

        return sorted(best_solution)
>>>>>>> REPLACE
