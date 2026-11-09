import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Structural change: Tour-level greedy heuristic.
    For each node j, precompute the total prize of a greedy tour that must include j.
    eta[i,j] = tour_prize[j] / (d[i,j] + d[j,0])
    This captures synergistic node combinations rather than individual node value.
    """
    start_time = time.time()
    deadline = start_time + 280

    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    eps = 1e-9

    # For each node j (j != 0), compute greedy tour prize that must visit j
    tour_prizes = np.zeros(n)
    for j in range(1, n):
        d_0j = distance_matrix[0, j]
        if d_0j > max_len:
            continue
        path_len = d_0j
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        visited[j] = True
        total_prize = prizes[j]
        current = j
        while True:
            best_node = -1
            best_score = -1.0
            for k in range(1, n):
                if visited[k]:
                    continue
                new_len = path_len + distance_matrix[current, k] + distance_matrix[k, 0]
                if new_len <= max_len:
                    score = prizes[k] / (distance_matrix[current, k] + eps)
                    if score > best_score:
                        best_score = score
                        best_node = k
            if best_node == -1:
                break
            path_len += distance_matrix[current, best_node] + distance_matrix[best_node, 0]
            visited[best_node] = True
            total_prize += prizes[best_node]
            current = best_node
            if time.time() > deadline:
                break
        tour_prizes[j] = total_prize

    # Build heuristic matrix
    eta = np.full((n, n), 1e-9)
    for i in range(n):
        for j in range(1, n):
            d_ij = distance_matrix[i, j]
            d_j0 = distance_matrix[j, 0]
            if d_ij + d_j0 > max_len:
                continue
            eta[i, j] = tour_prizes[j] / (d_ij + d_j0 + eps)

    # Normalize so max is 1
    max_eta = np.max(eta)
    if max_eta > eps:
        eta = eta / max_eta

    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta
