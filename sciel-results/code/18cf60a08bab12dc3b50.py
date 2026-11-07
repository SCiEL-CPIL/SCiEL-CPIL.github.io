"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    def evaluate(subset):
        if len(subset) < min_verts:
            return 0
        subset_set = set(subset)
        degrees = []
        for v in subset:
            deg = 0
            for u in subset:
                if u != v and adj[v][u]:
                    deg += 1
            degrees.append(deg)
        return len(set(degrees))

    best_subset = None
    best_score = 0
    
    # Simulated Annealing
    # Initialize with a random subset of size min_verts to n
    # We'll try multiple restarts
    
    for restart in range(50):
        # Random init
        current_size = random.randint(min_verts, n)
        current_subset = random.sample(range(n), current_size)
        current_subset_set = set(current_subset)
        current_score = evaluate(current_subset)
        
        if current_score > best_score:
            best_score = current_score
            best_subset = list(current_subset)
            
        T = 10.0
        T_min = 0.01
        alpha = 0.95
        
        while T > T_min:
            # Generate neighbor
            # 1. Swap a vertex in with one out
            # 2. Add a vertex (if size < n)
            # 3. Remove a vertex (if size > min_verts)
            
            move_type = random.random()
            if move_type < 0.5 and current_size < n:
                # Add
                outside = [v for v in range(n) if v not in current_subset_set]
                if not outside: continue
                v_in = random.choice(outside)
                new_subset = list(current_subset) + [v_in]
                new_subset_set = current_subset_set | {v_in}
                new_size = current_size + 1
            elif move_type < 0.8 and current_size > min_verts:
                # Remove
                v_out = random.choice(current_subset)
                new_subset = [v for v in current_subset if v != v_out]
                new_subset_set = current_subset_set - {v_out}
                new_size = current_size - 1
            else:
                # Swap
                outside = [v for v in range(n) if v not in current_subset_set]
                if not outside:
                    # If no outside, try remove
                    if current_size > min_verts:
                        v_out = random.choice(current_subset)
                        new_subset = [v for v in current_subset if v != v_out]
                        new_subset_set = current_subset_set - {v_out}
                        new_size = current_size - 1
                    else:
                        T *= alpha
                        continue
                else:
                    v_in = random.choice(outside)
                    v_out = random.choice(current_subset)
                    new_subset = [v for v in current_subset if v != v_out] + [v_in]
                    new_subset_set = (current_subset_set - {v_out}) | {v_in}
                    new_size = current_size
                    
            new_score = evaluate(new_subset)
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_subset = new_subset
                current_subset_set = new_subset_set
                current_size = new_size
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_subset = list(current_subset)
                    
            T *= alpha
            
    return best_subset
# EVOLVE-BLOCK-END
