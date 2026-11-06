import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices for all K4s
    def get_idx(u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    k4_edges = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [
            get_idx(a, b), get_idx(a, c), get_idx(a, d),
            get_idx(b, c), get_idx(b, d), get_idx(c, d)
        ]
        k4_edges.append(edges)
        
    num_k4 = len(k4_edges)
    
    # Convert to numpy for vectorized operations
    k4_edges_np = np.array(k4_edges, dtype=np.int32)

    def calculate_penalty(coloring):
        """
        Returns the total penalty using vectorized operations.
        """
        colors = np.array(coloring, dtype=np.int32)
        # Get all colors for all K4s at once: shape (num_k4, 6)
        all_colors = colors[k4_edges_np]  # shape (num_k4, 6)
        
        # For each K4, count distinct colors
        # Use a trick: sort and count unique
        sorted_colors = np.sort(all_colors, axis=1)
        # Count distinct by checking where values change
        distinct_mask = np.concatenate([np.ones((num_k4, 1), dtype=bool), 
                                         sorted_colors[:, 1:] != sorted_colors[:, :-1]], axis=1)
        num_distinct = distinct_mask.sum(axis=1)
        
        # Penalty is max(0, 5 - num_distinct)
        penalty = np.maximum(0, 5 - num_distinct)
        return int(penalty.sum())

    def calculate_penalty_for_k4(coloring, k4_idx):
        """Calculate penalty for a specific K4."""
        edges = k4_edges[k4_idx]
        colors = [coloring[e] for e in edges]
        distinct = len(set(colors))
        return max(0, 5 - distinct)

    def get_violating_k4s(coloring):
        """Return list of K4 indices that violate the constraint."""
        colors = np.array(coloring, dtype=np.int32)
        all_colors = colors[k4_edges_np]
        sorted_colors = np.sort(all_colors, axis=1)
        distinct_mask = np.concatenate([np.ones((num_k4, 1), dtype=bool), 
                                         sorted_colors[:, 1:] != sorted_colors[:, :-1]], axis=1)
        num_distinct = distinct_mask.sum(axis=1)
        violating = np.where(num_distinct < 5)[0]
        return violating.tolist()

    def is_valid(coloring):
        return calculate_penalty(coloring) == 0

    # Initial random coloring
    current_max_colors = 15
    coloring = [random.randint(0, current_max_colors - 1) for _ in range(num_edges)]
    
    best_penalty = calculate_penalty(coloring)
    best_coloring = coloring.copy()
    
    # Simulated Annealing with targeted moves
    T = 100.0
    cooling_rate = 0.9995
    min_T = 0.01
    steps = 0
    
    while T > min_T and time.time() < deadline - 10.0:
        # Pick a random edge to change
        e_idx = random.randint(0, num_edges - 1)
        old_color = coloring[e_idx]
        new_color = random.randint(0, current_max_colors - 1)
        
        if old_color == new_color:
            continue
            
        coloring[e_idx] = new_color
        new_penalty = calculate_penalty(coloring)
        steps += 1
        
        delta = new_penalty - best_penalty
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            best_penalty = new_penalty
            best_coloring = coloring.copy()
        else:
            coloring[e_idx] = old_color
            
        T *= cooling_rate
        
        # If we found a valid solution, try to reduce colors immediately
        if best_penalty == 0:
            # Try to reduce colors by eliminating the highest color
            color_to_eliminate = current_max_colors - 1
            edges_to_change = [i for i, c in enumerate(best_coloring) if c == color_to_eliminate]
            
            if edges_to_change:
                # Try to recolor these edges to lower colors
                success = False
                for _ in range(100):
                    trial = best_coloring.copy()
                    for e in edges_to_change:
                        trial[e] = random.randint(0, current_max_colors - 2)
                    if is_valid(trial):
                        success = True
                        best_coloring = trial
                        current_max_colors -= 1
                        break
                
                if success:
                    # Reset SA to continue optimizing or reducing further
                    coloring = best_coloring.copy()
                    best_penalty = 0
                    T = 10.0
                    continue
            
            # If we can't reduce, keep searching with current max
            if current_max_colors <= 13:
                break
    
    # If we found a valid coloring, return it
    if best_penalty == 0:
        return best_coloring
    
    # Fallback: Try more aggressive random restarts
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        coloring = [random.randint(0, 14) for _ in range(num_edges)]
        T = 50.0
        for _ in range(2000):
            if time.time() > deadline - 5.0:
                break
            e_idx = random.randint(0, num_edges - 1)
            old_color = coloring[e_idx]
            new_color = random.randint(0, 14)
            if old_color == new_color: continue
            coloring[e_idx] = new_color
            new_penalty = calculate_penalty(coloring)
            delta = new_penalty - best_penalty
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                best_penalty = new_penalty
                best_coloring = coloring.copy()
            else:
                coloring[e_idx] = old_color
            T *= 0.999
        if best_penalty == 0:
            return best_coloring
            
    # If still invalid, return a hardcoded valid solution
    # (i+j)%17 uses 17 colors and is valid
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
