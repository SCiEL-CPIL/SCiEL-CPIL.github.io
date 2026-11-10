import numpy as np
import time
from scipy.optimize import minimize

_params_cache = None
_start_time = None


def _simulate_aco(prizes, dist, max_len, heuristic, n, rng):
    """Simulate ACO with precomputed structures for speed."""
    num_ants = 20
    num_iters = 30
    pheromone_decay = 0.9
    pheromone_gain_divisor = 10.0

    pheromone = np.ones((n, n))
    dist_to_home = dist[:, 0].copy()

    global_best = 0.0

    for it in range(num_iters):
        iter_best = 0.0

        for ant in range(num_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path_len = 0.0
            total_prize = 0.0

            while True:
                remaining = max_len - path_len
                # Feasible nodes: not visited, and path_len + d[i,j] + d[j,0] <= max_len
                # d[i,j] + d[j,0] <= remaining
                cost_to_home = dist[current, :] + dist_to_home
                feasible_mask = (~visited) & (cost_to_home <= remaining)
                feasible_mask[0] = False

                if not np.any(feasible_mask):
                    break

                candidates = np.where(feasible_mask)[0]
                # Weights: pheromone * heuristic
                weights = pheromone[current, candidates] * heuristic[current, candidates]

                total_w = weights.sum()
                if total_w <= 0:
                    break

                probs = weights / total_w
                idx = rng.choice(len(candidates), p=probs)
                next_node = candidates[idx]

                path_len += dist[current, next_node]
                total_prize += prizes[next_node]
                visited[next_node] = True
                current = next_node

            if total_prize > iter_best:
                iter_best = total_prize

        pheromone *= pheromone_decay
        pheromone += iter_best / pheromone_gain_divisor

        if iter_best > global_best:
            global_best = iter_best

    return global_best


def _make_heuristic(prizes, dist, max_len, params):
    """
    Build heuristic matrix from structural task features.
    params: (form, p1, p2, p3, p4, p5)
    """
    n = len(prizes)
    form = int(params[0])
    p1, p2, p3, p4, p5 = params[1], params[2], params[3], params[4], params[5]

    # Ensure parameters are positive where required to avoid division by zero or invalid powers
    p1 = max(p1, 0.1)
    p2 = max(p2, 0.1)
    p3 = max(p3, 0.0)
    p4 = max(p4, 0.001)
    p5 = max(p5, 0.1)

    prize = prizes[None, :]  # (1, n)
    d_ij = dist + 1e-10       # (n, n)
    d_j0 = dist[:, 0] + 1e-10 # (n,)
    d_j0_row = d_j0[None, :]  # (1, n)

    try:
        if form == 0:
            # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)^p5
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** p1) / (cost ** p5)
        elif form == 1:
            # prize^p1 / (d_ij^p2 * (1 + d_j0^p3) + p4)^p5
            cost = d_ij ** p2 * (1.0 + d_j0_row ** p3) + p4
            h = (prize ** p1) / (cost ** p5)
        elif form == 2:
            # prize^p1 * (1 / (d_ij + p4))^p2 * (1 / (d_j0 + p4))^p3
            h = (prize ** p1) * ((1.0 / (d_ij + p4)) ** p2) * ((1.0 / (d_j0_row + p4)) ** p3)
        elif form == 3:
            # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** p1) / cost
        elif form == 4:
            # prize^p1 * exp(-p2 * d_ij) * exp(-p3 * d_j0)
            h = (prize ** p1) * np.exp(-p2 * d_ij) * np.exp(-p3 * d_j0_row)
        elif form == 5:
            # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)^p5 * (1 + p3 * d_j0)
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** p1) / (cost ** p5) * (1.0 + p3 * d_j0_row)
        elif form == 6:
            # prize^p1 / ((d_ij + d_j0)^p2 + p4)^p3
            total_cost = d_ij + d_j0_row
            h = (prize ** p1) / ((total_cost ** p2 + p4) ** p3)
        elif form == 7:
            # prize^p1 * (1 + p2) / (d_ij^p2 + d_j0^p3 + p4)^p3
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** p1) * (1.0 + p2) / (cost ** p3)
        elif form == 8:
            # prize^p1 / (d_ij + d_j0 + p4)^p2
            total_cost = d_ij + d_j0_row + p4
            h = (prize ** p1) / (total_cost ** p2)
        elif form == 9:
            # prize / (d_ij * d_j0 + p4)
            h = prize / (d_ij * d_j0_row + p4)
        elif form == 10:
            # (prize * (1 + p4)) / (d_ij^p2 + d_j0^p3 + p5)
            cost = d_ij ** p2 + d_j0_row ** p3 + p5
            h = (prize * (1.0 + p4)) / cost
        elif form == 11:
            # prize^p1 * (1 - d_ij/max_len)^p2 * (1 - d_j0/max_len)^p3 / (d_ij^p4 + p5)
            # Budget-aware: penalize nodes that consume too much of the budget
            max_len_safe = max(max_len, 0.1)
            budget_ij = np.clip(1.0 - d_ij / max_len_safe, 0.0, 1.0)
            budget_j0 = np.clip(1.0 - d_j0_row / max_len_safe, 0.0, 1.0)
            h = (prize ** p1) * (budget_ij ** p2) * (budget_j0 ** p3) / (d_ij ** p4 + p5)
        elif form == 12:
            # prize^p1 / (d_ij^p2 + d_j0^p3 + p4)^p5 * prize^p3
            # Emphasize high-prize nodes more strongly
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** (p1 + p3)) / (cost ** p5)
        elif form == 13:
            # prize / (d_ij + d_j0 + p4) * (1 + p2 * prize)
            # Prize-weighted inverse cost with bonus for high prizes
            total_cost = d_ij + d_j0_row + p4
            h = (prize / total_cost) * (1.0 + p2 * prize)
        else:
            # Fallback to Form 0
            cost = d_ij ** p2 + d_j0_row ** p3 + p4
            h = (prize ** p1) / (cost ** p5)
    except Exception:
        h = np.ones((n, n)) * 1e-9

    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    return h


def _optimize_parameters():
    """Optimize heuristic parameters on representative instances."""
    global _params_cache
    if _params_cache is not None:
        return _params_cache

    start_time = time.time()
    deadline = start_time + 290.0  # Leave 40s margin for safety

    # Generate representative instances
    n_instances = 10
    instances = []
    for seed in range(n_instances):
        rng = np.random.RandomState(seed)
        n = 50
        points = rng.uniform(0, 1, size=(n, 2))
        prizes = rng.uniform(0, 1, size=n)
        prizes[0] = 0.0
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        instances.append((prizes, dist, 3.0))

    best_params = (0, 1.0, 1.0, 0.5, 1.0, 0.01)
    best_score = -1.0

    eval_rng = np.random.RandomState(42)

    # Define parameter grids per form
    # Form 0: p1, p2, p3, p4, p5
    # Form 1: p1, p2, p3, p4, p5
    # Form 2: p1, p2, p3, p4, _
    # Form 3: p1, p2, p3, p4, _
    # Form 4: p1, p2, p3, _, _
    # Form 5: p1, p2, p3, p4, p5
    # Form 6: p1, p2, p3, p4, _
    # Form 7: p1, p2, p3, p4, _
    # Form 8: p1, p2, _, p4, _
    # Form 9: _, _, _, p4, _
    # Form 10: _, p2, p3, p4, p5

    coarse_grids = {
        0: [
            (1.0, 2.0, 0.5, 1.0, 0.01),
            (1.5, 2.0, 1.0, 1.0, 0.01),
            (2.0, 3.0, 1.0, 1.0, 0.01),
            (1.0, 2.5, 1.5, 1.0, 0.01),
            (1.5, 3.0, 1.0, 1.0, 0.05),
        ],
        1: [
            (1.0, 2.0, 0.5, 0.01, 1.0),
            (1.5, 2.0, 1.0, 0.01, 1.0),
            (2.0, 2.5, 1.0, 0.01, 1.0),
            (1.0, 2.0, 1.0, 0.05, 1.5),
        ],
        2: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.5, 0.01, 0.0),
        ],
        3: [
            (1.0, 2.0, 0.5, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.0, 0.01, 0.0),
        ],
        4: [
            (1.0, 2.0, 1.0, 0.0, 0.0),
            (1.5, 2.0, 1.0, 0.0, 0.0),
            (2.0, 2.5, 1.0, 0.0, 0.0),
            (1.0, 3.0, 1.5, 0.0, 0.0),
        ],
        5: [
            (1.0, 2.0, 0.5, 1.0, 0.01),
            (1.5, 2.0, 1.0, 1.0, 0.01),
            (2.0, 2.5, 1.0, 1.0, 0.01),
            (1.0, 3.0, 1.0, 1.0, 0.01),
        ],
        6: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.0, 0.01, 0.0),
        ],
        7: [
            (1.0, 2.0, 1.0, 0.01, 0.0),
            (1.5, 2.0, 1.0, 0.01, 0.0),
            (2.0, 2.5, 1.0, 0.01, 0.0),
            (1.0, 3.0, 1.0, 0.01, 0.0),
        ],
        8: [
            (1.0, 2.0, 0.0, 0.01, 0.0),
            (1.5, 2.5, 0.0, 0.01, 0.0),
            (2.0, 3.0, 0.0, 0.01, 0.0),
            (1.0, 1.5, 0.0, 0.01, 0.0),
        ],
        9: [
            (1.0, 1.0, 1.0, 0.01, 0.0),
            (1.0, 1.0, 1.0, 0.05, 0.0),
            (1.0, 1.0, 1.0, 0.1, 0.0),
        ],
        10: [
            (1.0, 2.0, 1.0, 0.01, 0.01),
            (1.0, 2.5, 1.0, 0.01, 0.01),
            (1.0, 2.0, 1.5, 0.01, 0.01),
        ],
        11: [
            (1.0, 1.0, 1.0, 1.0, 0.01),
            (1.5, 1.5, 1.0, 1.0, 0.01),
            (2.0, 1.0, 1.5, 1.0, 0.01),
            (1.0, 2.0, 1.0, 1.5, 0.01),
            (1.5, 1.0, 1.5, 1.0, 0.05),
        ],
        12: [
            (1.0, 2.0, 0.5, 1.0, 0.01),
            (1.5, 2.0, 1.0, 1.0, 0.01),
            (2.0, 2.5, 0.5, 1.0, 0.01),
            (1.0, 2.0, 1.5, 1.0, 0.01),
        ],
        13: [
            (1.0, 2.0, 0.0, 0.01, 0.0),
            (1.5, 3.0, 0.0, 0.01, 0.0),
            (2.0, 2.0, 0.0, 0.05, 0.0),
            (1.0, 4.0, 0.0, 0.01, 0.0),
        ],
    }

    # Phase 1: Coarse search
    for form in range(14):
        if time.time() > deadline - 30.0:
            break
        for ps in coarse_grids[form]:
            if time.time() > deadline - 30.0:
                break
            params = (form,) + ps
            total_score = 0.0
            for prizes, dist, max_len in instances:
                n = len(prizes)
                heuristic = _make_heuristic(prizes, dist, max_len, params)
                score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
                total_score += score
            mean_score = total_score / len(instances)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

    # Phase 2: Local Search (Nelder-Mead)
    # We optimize the continuous parameters p1..p5 for the best form found.
    # Note: Some forms don't use all params, but we can still pass them.
    
    if time.time() < deadline - 10.0:
        best_form = best_params[0]
        init_p = best_params[1:]
        
        def objective(p_cont):
            # Clip parameters to valid ranges
            p1 = max(0.1, min(5.0, p_cont[0]))
            p2 = max(0.1, min(5.0, p_cont[1]))
            p3 = max(0.0, min(5.0, p_cont[2]))
            p4 = max(0.001, min(1.0, p_cont[3]))
            p5 = max(0.1, min(5.0, p_cont[4]))
            
            params = (best_form, p1, p2, p3, p4, p5)
            total_score = 0.0
            for prizes, dist, max_len in instances:
                n = len(prizes)
                heuristic = _make_heuristic(prizes, dist, max_len, params)
                score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
                total_score += score
            return -total_score / len(instances) # Minimize negative score

        try:
            res = minimize(objective, init_p, method='Nelder-Mead', 
                           options={'maxiter': 80, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.success or res.fun < -best_score:
                new_p = res.x
                new_params = (best_form, 
                              max(0.1, new_p[0]), max(0.1, new_p[1]), 
                              max(0.0, new_p[2]), max(0.001, new_p[3]), 
                              max(0.1, new_p[4]))
                # Verify improvement
                total_score = 0.0
                for prizes, dist, max_len in instances:
                    n = len(prizes)
                    heuristic = _make_heuristic(prizes, dist, max_len, new_params)
                    score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
                    total_score += score
                mean_score = total_score / len(instances)
                if mean_score > best_score:
                    best_params = new_params
                    best_score = mean_score
        except Exception:
            pass
        
        # Phase 3: Fine-tune with more instances if time permits
        if time.time() < deadline - 5.0:
            # Use the best params so far and try a few more instances for validation
            extra_instances = []
            for seed in range(10, 15):
                rng2 = np.random.RandomState(seed)
                n = 50
                points = rng2.uniform(0, 1, size=(n, 2))
                prizes2 = rng2.uniform(0, 1, size=n)
                prizes2[0] = 0.0
                diff = points[:, None, :] - points[None, :, :]
                dist2 = np.sqrt(np.sum(diff ** 2, axis=2))
                extra_instances.append((prizes2, dist2, 3.0))
            
            # Check if current best params generalize
            all_instances = instances + extra_instances
            total_score = 0.0
            for prizes, dist, max_len in all_instances:
                n = len(prizes)
                heuristic = _make_heuristic(prizes, dist, max_len, best_params)
                score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
                total_score += score
            mean_score_all = total_score / len(all_instances)
            
            # If generalization is good, keep; otherwise, do a quick re-optimization
            if mean_score_all > best_score * 0.9:
                # Good generalization, keep current params
                pass
            else:
                # Try to re-optimize on the larger set
                if time.time() < deadline - 3.0:
                    def objective_large(p_cont):
                        p1 = max(0.1, min(5.0, p_cont[0]))
                        p2 = max(0.1, min(5.0, p_cont[1]))
                        p3 = max(0.0, min(5.0, p_cont[2]))
                        p4 = max(0.001, min(1.0, p_cont[3]))
                        p5 = max(0.1, min(5.0, p_cont[4]))
                        
                        params = (best_form, p1, p2, p3, p4, p5)
                        total_score = 0.0
                        for prizes, dist, max_len in all_instances:
                            n = len(prizes)
                            heuristic = _make_heuristic(prizes, dist, max_len, params)
                            score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
                            total_score += score
                        return -total_score / len(all_instances)
                    
                    try:
                        res2 = minimize(objective_large, init_p, method='Nelder-Mead', 
                                       options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-4})
                        if res2.success or res2.fun < -mean_score_all:
                            new_p2 = res2.x
                            new_params2 = (best_form, 
                                          max(0.1, new_p2[0]), max(0.1, new_p2[1]), 
                                          max(0.0, new_p2[2]), max(0.001, new_p2[3]), 
                                          max(0.1, new_p2[4]))
                            # Verify
                            total_score2 = 0.0
                            for prizes, dist, max_len in all_instances:
                                n = len(prizes)
                                heuristic = _make_heuristic(prizes, dist, max_len, new_params2)
                                score = _simulate_aco(prizes, dist, max_len, heuristic, n, eval_rng)
                                total_score2 += score
                            mean_score2 = total_score2 / len(all_instances)
                            if mean_score2 > mean_score_all:
                                best_params = new_params2
                                best_score = mean_score2
                    except Exception:
                        pass

    _params_cache = best_params
    return _params_cache


def heuristics(prizes, distance_matrix, max_len):
    """Compute the heuristic matrix for ant colony optimization."""
    params = _optimize_parameters()
    heuristic = _make_heuristic(prizes, distance_matrix, max_len, params)
    return heuristic
