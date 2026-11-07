"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_path(prizes, dist, max_len, n_nodes, rng):
    """
    Generates a single path using a randomized greedy heuristic.
    """
    visited = np.zeros(n_nodes, dtype=bool)
    visited[0] = True
    path = [0]
    current = 0
    current_len = 0.0
    current_prize = prizes[0]

    for _ in range(n_nodes - 1):
        best_j = -1
        best_score = -1.0
        candidates = []
        scores = []
        
        for j in range(n_nodes):
            if visited[j]:
                continue
            # Check feasibility: current_len + d[i,j] + d[j,0] <= max_len
            if current_len + dist[current, j] + dist[j, 0] <= max_len:
                # Score: prize / distance, with some randomness
                # We use a power to control exploration vs exploitation
                score = (prizes[j] / dist[current, j]) ** 1.5
                # Add small random noise to break ties and encourage diversity
                score += rng.random() * 0.01 * (prizes[j] / dist[current, j])
                candidates.append(j)
                scores.append(score)
        
        if not candidates:
            break
            
        # Select next node based on scores (roulette wheel or top-k)
        # Simple: pick the max score, or pick randomly from top 3
        # Let's pick randomly from the top 3 to allow some exploration
        scores = np.array(scores)
        candidates = np.array(candidates)
        
        if len(candidates) <= 3:
            idx = np.argmax(scores)
        else:
            # Pick from top 3
            top_3_indices = np.argsort(scores)[-3:]
            idx = top_3_indices[rng.integers(0, 3)]
            
        next_node = candidates[idx]
        current_len += dist[current, next_node]
        current_prize += prizes[next_node]
        visited[next_node] = True
        path.append(next_node)
        current = next_node
        
    return path, current_prize

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix tailored to the specific instance.
    """
    n_nodes = len(prizes)
    start_time = time.time()
    deadline = 330.0  # Safety margin
    
    # 1. Generate a set of diverse paths using randomized greedy
    n_paths = 50
    paths = []
    path_scores = []
    
    rng = np.random.default_rng(42) # Fixed seed for reproducibility in heuristic generation
    
    for _ in range(n_paths):
        if time.time() > deadline - 5.0:
            break
        p, score = _generate_path(prizes, distance_matrix, max_len, n_nodes, rng)
        paths.append(p)
        path_scores.append(score)
    
    # 2. Identify the best paths
    if not paths:
        # Fallback to simple heuristic if no paths generated
        return (prizes[None, :] + 1e-9) / (distance_matrix + 1e-9)
    
    # Sort paths by score
    sorted_indices = np.argsort(path_scores)[::-1]
    top_k = min(5, len(paths))
    best_paths = [paths[i] for i in sorted_indices[:top_k]]
    
    # 3. Build heuristic matrix based on edge usage in best paths
    # Base heuristic: prize_j / d_ij
    base_heuristic = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                base_heuristic[i, j] = 1e-9
            else:
                # Avoid division by zero
                d = distance_matrix[i, j]
                if d < 1e-9:
                    base_heuristic[i, j] = 1e-9
                else:
                    base_heuristic[i, j] = prizes[j] / d
    
    # Edge frequency matrix
    edge_freq = np.zeros((n_nodes, n_nodes))
    for p in best_paths:
        for k in range(len(p) - 1):
            i = p[k]
            j = p[k+1]
            edge_freq[i, j] += 1
    
    # Combine base heuristic and edge frequency
    # Boost edges that are in the best paths
    # heuristic[i,j] = base[i,j] * (1 + alpha * freq[i,j])
    alpha = 5.0
    final_heuristic = base_heuristic * (1.0 + alpha * edge_freq)
    
    # Ensure positive values
    final_heuristic = np.where(final_heuristic > 0, final_heuristic, 1e-9)
    
    # Normalize to avoid numerical issues? The task says "larger is more attractive".
    # ACO usually uses eta^beta. If values are too large, it might dominate pheromone.
    # Let's scale it to be comparable to pheromone (which starts at 1).
    # We can divide by the max value to keep it in [0, 1] or similar, or just leave it.
    # The prompt says "Heuristic entries that are not positive become 1e-9".
    # It doesn't specify scaling. However, if heuristic is huge, pheromone is ignored.
    # If heuristic is tiny, pheromone dominates.
    # A good heuristic should be of similar magnitude to pheromone.
    # Let's normalize by the max entry.
    max_val = np.max(final_heuristic)
    if max_val > 0:
        final_heuristic = final_heuristic / max_val
        # Ensure minimum 1e-9
        final_heuristic = np.where(final_heuristic > 0, final_heuristic, 1e-9)
    
    return final_heuristic

# EVOLVE-BLOCK-END
