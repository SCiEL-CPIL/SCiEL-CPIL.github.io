import numpy as np
import time

def _run_aco(prizes, dist, max_len, heuristic_matrix, seed):
    """
    Simulates the ACO process to evaluate a heuristic matrix.
    Returns the best total prize found by any ant in any iteration.
    """
    n = len(prizes)
    np.random.seed(seed)
    pheromone = np.ones((n, n))
    best_global = 0.0
    
    # Precompute d_j0 for faster access
    d_j0 = dist[:, 0]
    
    for _iter in range(30):
        best_iter = 0.0
        # 20 ants
        for _ant in range(20):
            current = 0
            visited = [False] * n
            visited[0] = True
            length = 0.0
            prize_sum = prizes[0] # 0
            
            while True:
                candidates = []
                probs = []
                curr = current
                curr_d = dist[curr]
                curr_p = pheromone[curr]
                curr_h = heuristic_matrix[curr]
                
                for j in range(1, n):
                    if visited[j]:
                        continue
                    if length + curr_d[j] + d_j0[j] <= max_len:
                        candidates.append(j)
                        h = curr_h[j]
                        if h <= 0 or not np.isfinite(h):
                            h = 1e-9
                        p = curr_p[j] * h
                        probs.append(p)
                
                if not candidates:
                    break
                
                total_p = sum(probs)
                if total_p == 0:
                    break
                probs = [p / total_p for p in probs]
                
                r = np.random.rand()
                cumsum = 0.0
                chosen_idx = 0
                for idx, p in enumerate(probs):
                    cumsum += p
                    if r <= cumsum:
                        chosen_idx = idx
                        break
                next_node = candidates[chosen_idx]
                
                length += dist[current, next_node]
                visited[next_node] = True
                prize_sum += prizes[next_node]
                current = next_node
            
            best_iter = max(best_iter, prize_sum)
        
        # Pheromone update
        pheromone *= 0.9
        pheromone += best_iter / 10.0
        
        if best_iter > best_global:
            best_global = best_iter
            
    return best_global

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    start_time = time.time()
    timeout = 280  # Safety margin

    p = prizes
    d = distance_matrix
    ml = max_len
    d_j0 = d[:, 0]

    # Candidate heuristic generators
    def gen_1(p, d, d_j0, ml):
        denom = d + d_j0[None, :]
        return (p[None, :] + 1e-9) / (denom + 1e-9)

    def gen_2(p, d, d_j0, ml):
        return (p[None, :] + 1e-9) / (d + 1e-9)

    def gen_3(p, d, d_j0, ml):
        denom = d + d_j0[None, :]
        return (p[None, :] + 1e-9) / (denom + 1e-9) ** 2

    def gen_4(p, d, d_j0, ml):
        denom = d + d_j0[None, :]
        return (p[None, :] + 1e-9) * np.exp(-(denom + 1e-9))

    def gen_5(p, d, d_j0, ml):
        denom = d * (d + d_j0[None, :])
        return (p[None, :] + 1e-9) / (denom + 1e-9)

    def gen_6(p, d, d_j0, ml):
        base = (p[None, :] + 1e-9) / (d + 1e-9)
        penalty = 1.0 / (d_j0[None, :] + 1.0)
        return base * penalty

    def gen_7(p, d, d_j0, ml):
        denom = d + 0.5 * d_j0[None, :]
        return (p[None, :] + 1e-9) / (denom + 1e-9)

    def gen_8(p, d, d_j0, ml):
        # Prize density: p_j / (d_ij + d_j0) * (ml / (d_ij + d_j0 + 1e-9))
        # Rewards nodes that are both high-prize and feasible
        denom = d + d_j0[None, :]
        base = (p[None, :] + 1e-9) / (denom + 1e-9)
        feasibility = ml / (denom + 1e-9)
        return base * feasibility

    def gen_9(p, d, d_j0, ml):
        # Geometric mean: sqrt(p_j / (d_ij + d_j0))
        denom = d + d_j0[None, :]
        return np.sqrt((p[None, :] + 1e-9) / (denom + 1e-9))

    def gen_10(p, d, d_j0, ml):
        # p_j^2 / (d_ij + d_j0) — emphasize high-prize nodes more
        denom = d + d_j0[None, :]
        return (p[None, :] ** 2 + 1e-9) / (denom + 1e-9)

    def gen_11(p, d, d_j0, ml):
        # p_j / (d_ij + d_j0) * exp(-d_ij / ml) — decay with step distance
        denom = d + d_j0[None, :]
        base = (p[None, :] + 1e-9) / (denom + 1e-9)
        decay = np.exp(-d / max(ml, 1e-9))
        return base * decay

    def gen_12(p, d, d_j0, ml):
        # p_j / (d_ij + d_j0 + 0.1) — regularized version of gen_1
        denom = d + d_j0[None, :] + 0.1
        return (p[None, :] + 1e-9) / denom

    def gen_13(p, d, d_j0, ml):
        # Combined: (p_j / (d_ij + d_j0)) * (1 / (1 + d_j0))
        denom = d + d_j0[None, :]
        base = (p[None, :] + 1e-9) / (denom + 1e-9)
        penalty = 1.0 / (1.0 + d_j0[None, :])
        return base * penalty

    def gen_14(p, d, d_j0, ml):
        # p_j / (d_ij + d_j0) * (p_j / (d_j0 + 0.5)) — prize-weighted return cost
        denom = d + d_j0[None, :]
        base = (p[None, :] + 1e-9) / (denom + 1e-9)
        prize_return = p[None, :] / (d_j0[None, :] + 0.5)
        return base * prize_return

    candidates = [gen_1, gen_2, gen_3, gen_4, gen_5, gen_6, gen_7,
                  gen_8, gen_9, gen_10, gen_11, gen_12, gen_13, gen_14]

    best_score = -np.inf
    best_matrix = None

    # Use multiple seeds for robustness
    test_seeds = [42, 123, 999, 7, 314, 555, 2024]

    for gen in candidates:
        if time.time() - start_time > timeout:
            break

        try:
            h_mat = gen(prizes, distance_matrix, d_j0, max_len)
            h_mat = np.where(np.isfinite(h_mat) & (h_mat > 0), h_mat, 1e-9)

            scores = []
            for seed in test_seeds:
                if time.time() - start_time > timeout:
                    break
                score = _run_aco(prizes, distance_matrix, max_len, h_mat, seed)
                scores.append(score)

            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_matrix = h_mat
        except Exception:
            continue

    if best_matrix is None:
        best_matrix = gen_1(prizes, distance_matrix, d_j0, max_len)
        best_matrix = np.where(np.isfinite(best_matrix) & (best_matrix > 0), best_matrix, 1e-9)

    return best_matrix
