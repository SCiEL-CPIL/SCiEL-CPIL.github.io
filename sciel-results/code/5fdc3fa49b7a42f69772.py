import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Avoid division by zero
    d = distance_matrix + 1e-9
    p = prizes[None, :] + 1e-9
    
    # Heuristic: Prize over square root of distance
    # This reduces the dominance of very close nodes, allowing
    # exploration of moderately distant high-prize nodes.
    return p / np.sqrt(d)
