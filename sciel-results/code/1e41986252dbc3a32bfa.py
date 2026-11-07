import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)

    if time.time() > deadline:
        return np.ones((n, n)) * 1e-9

    # Precompute
    dist_to_origin = distance_matrix[:, 0]
    total_cost = distance_matrix + dist_to_origin[None, :]  # (n, n)
    max_cost = np.max(total_cost)
    if max_cost <= 0:
        max_cost = 1.0

    # Greedy evaluator: simulates one ant using the heuristic as a deterministic guide
    def get_prize(eta):
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        curr = 0
        total_len = 0.0
        total_prize = 0.0

        for _ in range(n):
            best_val = -1.0
            best_j = -1
            for j in range(1, n):
                if visited[j]:
                    continue
                new_len = total_len + distance_matrix[curr, j] + distance_matrix[j, 0]
                if new_len <= max_len:
                    val = eta[curr, j]
                    if val > best_val:
                        best_val = val
                        best_j = j
            if best_j == -1:
                break
            total_len += distance_matrix[curr, best_j]
            total_prize += prizes[best_j]
            visited[best_j] = True
            curr = best_j
        return total_prize

    eps = 1e-9

    # Heuristic family: prize[j]^a / (d[i,j]^b * d[j,0]^g) * slack_factor^s
    # slack_factor = 1 - s * (d[i,j] + d[j,0]) / max_cost, clipped to positive
    def compute_heuristic(a, b, g, s):
        prize_term = (prizes + eps) ** a
        dist_ij_term = (distance_matrix + eps) ** b
        dist_j0_term = (dist_to_origin + eps) ** g
        eta = prize_term[None, :] / (dist_ij_term * dist_j0_term[None, :])

        if s > 1e-6:
            slack_factor = 1.0 - s * (total_cost / max_cost)
            slack_factor = np.clip(slack_factor, 0.1, None)
            eta = eta * slack_factor

        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        return eta

    # Phase 1: Coarse grid search over (a, b, g, s)
    alphas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    gammas = [0.5, 1.0, 1.5, 2.0, 2.5]
    slacks = [0.0, 0.2, 0.4, 0.6, 0.8]

    best_prize = -1.0
    best_params = (3.0, 1.5, 1.0, 0.0)

    for a in alphas:
        prize_term = (prizes + eps) ** a
        for b in betas:
            dist_ij_term = (distance_matrix + eps) ** b
            for g in gammas:
                dist_j0_term = (dist_to_origin + eps) ** g
                for s in slacks:
                    if time.time() > deadline - 5:
                        break
                    eta = prize_term[None, :] / (dist_ij_term * dist_j0_term[None, :])
                    if s > 1e-6:
                        slack_factor = 1.0 - s * (total_cost / max_cost)
                        slack_factor = np.clip(slack_factor, 0.1, None)
                        eta = eta * slack_factor
                    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)

                    p = get_prize(eta)
                    if p > best_prize:
                        best_prize = p
                        best_params = (a, b, g, s)

                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break

    # Phase 2: Local refinement around best parameters
    if time.time() < deadline - 30:
        a0, b0, g0, s0 = best_params
        step_sizes = [0.25, 0.1, 0.05, 0.02, 0.01]

        for step in step_sizes:
            improved = False
            for da in [-step, 0, step]:
                for db in [-step, 0, step]:
                    for dg in [-step, 0, step]:
                        for ds in [-step, 0, step]:
                            if da == 0 and db == 0 and dg == 0 and ds == 0:
                                continue
                            a = a0 + da
                            b = b0 + db
                            g = g0 + dg
                            s = s0 + ds

                            if a < 0.1 or b < 0.1 or g < 0.1 or s < 0.0:
                                continue
                            if a > 10 or b > 5 or g > 5 or s > 1.5:
                                continue

                            if time.time() > deadline - 15:
                                break

                            eta = compute_heuristic(a, b, g, s)
                            p = get_prize(eta)
                            if p > best_prize:
                                best_prize = p
                                best_params = (a, b, g, s)
                                a0, b0, g0, s0 = a, b, g, s
                                improved = True

                            if time.time() > deadline - 15:
                                break
                        if time.time() > deadline - 15:
                            break
                    if time.time() > deadline - 15:
                        break
                if time.time() > deadline - 15:
                    break

            if not improved and step < 0.05:
                break

    # Final computation
    a, b, g, s = best_params
    heuristic = compute_heuristic(a, b, g, s)

    return heuristic
