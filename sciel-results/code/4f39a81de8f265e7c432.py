import time
import random

def construct():
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # We aim for n=26
    # We will try to construct n=26, then 25, etc.
    # Actually, if we can construct 26, we stop.
    # If not, we try 25, etc.
    # But 26 is the max known.
    
    for target_n in range(26, 11, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a valid coloring for target_n
        # We use randomized backtracking
        
        success = False
        # Number of restarts depends on time
        max_restarts = 500
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            random.seed(restart) # Deterministic restarts for reproducibility in dev, but random in prod?
            # Let's use random.seed(None) or just rely on random state
            
            neighbors = [[0] * target_n for _ in range(5)]
            
            # Heuristic: Order vertices? 
            # The standard order 0..n-1 is fine.
            
            def search(v, neighbors):
                if time.time() - start_time > time_limit:
                    return False
                
                if v == target_n:
                    return True
                
                # Assign colors to edges (v, u) for u in 0..v-1
                # We process u in random order to break symmetry?
                # Or standard order?
                # Let's use random order for u as well to explore different branches
                us = list(range(v))
                random.shuffle(us)
                
                S = [[] for _ in range(5)]
                
                def assign_edge(u_idx, S, neighbors):
                    if time.time() - start_time > time_limit:
                        return False
                    
                    if u_idx == len(us):
                        return search(v + 1, neighbors)
                    
                    u = us[u_idx]
                    
                    # Shuffle colors for this edge
                    colors = [0, 1, 2, 3, 4]
                    random.shuffle(colors)
                    
                    for c in colors:
                        # Check validity
                        valid = True
                        for w in S[c]:
                            if neighbors[c][u] & neighbors[c][w]:
                                valid = False
                                break
                        
                        if valid:
                            neighbors[c][u] |= (1 << v)
                            neighbors[c][v] |= (1 << u)
                            S[c].append(u)
                            
                            if assign_edge(u_idx + 1, S, neighbors):
                                return True
                            
                            S[c].pop()
                            neighbors[c][u] &= ~(1 << v)
                            neighbors[c][v] &= ~(1 << u)
                        
                    return False
                
                return assign_edge(0, S, neighbors)
            
            if search(0, neighbors):
                # Extract colors
                colors = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        for c in range(5):
                            if neighbors[c][i] & (1 << j):
                                colors.append(c)
                                break
                best_n = target_n
                best_colors = colors
                success = True
                break
        
        if success:
            break # Found a valid coloring for this target_n, which is the max we try
            
    return {'n': best_n, 'colors': best_colors}
