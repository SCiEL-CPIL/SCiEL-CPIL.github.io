import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edges)  # 120
    
    # Precompute K4 edge indices
    k4s = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        k4_edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
        k4s.append(k4_edges)
    
    # Edge index map
    edge_idx = {}
    for idx, e in enumerate(edges):
        edge_idx[e] = idx
    
    # Precompute which K4s each edge belongs to
    edge_to_k4s = [[] for _ in range(m)]
    for k4_i, k4e in enumerate(k4s):
        for e in k4e:
            edge_to_k4s[edge_idx[e]].append(k4_i)
    
    # Precompute K4 edge index arrays for fast access
    k4_edge_idxs = []
    for k4e in k4s:
        k4_edge_idxs.append([edge_idx[e] for e in k4e])
    
    # Precompute K4 edge index arrays as numpy arrays for vectorized ops
    k4_edge_idxs_np = np.array(k4_edge_idxs, dtype=np.int32)
    
    def count_colors(coloring):
        return len(np.unique(coloring))
    
    def check_violations_subset(coloring, k4_indices):
        """Check violations only for a subset of K4s."""
        if len(k4_indices) == 0:
            return 0
        k4_idxs = k4_edge_idxs_np[k4_indices]
        k4_colors = coloring[k4_idxs]
        sorted_k4 = np.sort(k4_colors, axis=1)
        diffs = np.diff(sorted_k4, axis=1)
        distinct_counts = 1 + np.sum(diffs != 0, axis=1)
        return int(np.sum(distinct_counts < 5))
    
    def check_valid(coloring):
        return check_violations_subset(coloring, list(range(len(k4s)))) == 0

    # --- Initialization Strategies ---
    
    candidates = []
    
    # 1. Try the (i+j) % mod seed
    for mod in range(13, 20):
        coloring = np.array([(i + j) % mod for i in range(n) for j in range(i + 1, n)], dtype=int)
        if check_valid(coloring):
            candidates.append(coloring)
            
    # 2. Random initializations
    for _ in range(50):
        if time.time() > deadline - 60: break
        coloring = np.random.randint(0, 15, size=m, dtype=int)
        if check_valid(coloring):
            candidates.append(coloring)
            
    # 3. Greedy construction with limited palette
    # Try to construct a valid coloring using exactly k colors
    for k in range(13, 18):
        if time.time() > deadline - 50: break
        for trial in range(20):
            if time.time() > deadline - 50: break
            coloring = np.full(m, -1, dtype=int)
            # Randomize edge order
            edge_order = np.random.permutation(m)
            success = True
            for idx in edge_order:
                # Try colors 0..k-1
                found = False
                for c in range(k):
                    coloring[idx] = c
                    # Check only affected K4s
                    if check_violations_subset(coloring, edge_to_k4s[idx]) == 0:
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                # Final full check
                if check_valid(coloring):
                    candidates.append(coloring)
                    
    if not candidates:
        # Fallback: high color count
        coloring = np.array([i for i in range(m)], dtype=int)
        candidates.append(coloring)
        
    # Sort by number of colors
    candidates.sort(key=lambda c: count_colors(c))
    
    best_coloring = candidates[0].copy()
    best_colors = count_colors(best_coloring)
    
    # --- Local Search ---
    
    # We will run local search on the top few candidates
    for start_coloring in candidates[:5]:
        if time.time() > deadline - 10: break
        
        coloring = start_coloring.copy()
        current_colors = count_colors(coloring)
        
        no_improve = 0
        max_no_improve = 2000
        
        while no_improve < max_no_improve:
            if time.time() > deadline - 5: break
            
            # Get unique colors and counts
            unique, counts = np.unique(coloring, return_counts=True)
            if len(unique) == 0: break
            
            # Strategy: Try to remove the least frequent color
            # Find the color with the minimum count
            min_idx = np.argmin(counts)
            target_color = unique[min_idx]
            target_edges = np.where(coloring == target_color)[0]
            
            if len(target_edges) == 0:
                break
                
            # Pick a random edge with the target color
            idx = int(random.choice(target_edges))
            
            # Try to recolor it to another existing color
            other_colors = [c for c in unique if c != target_color]
            random.shuffle(other_colors)
            
            improved = False
            for new_color in other_colors:
                old_color = int(coloring[idx])
                coloring[idx] = new_color
                
                # Check only affected K4s
                if check_violations_subset(coloring, edge_to_k4s[idx]) == 0:
                    # Valid change
                    new_count = count_colors(coloring)
                    if new_count < current_colors:
                        current_colors = new_count
                        if new_count < best_colors:
                            best_colors = new_count
                            best_coloring = coloring.copy()
                        improved = True
                        break
                    else:
                        # No improvement in color count, revert to continue searching
                        coloring[idx] = old_color
                        break
                else:
                    coloring[idx] = old_color
                    
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
                
            # If we reached the theoretical lower bound, stop
            if best_colors <= 13:
                break

    # --- Final Polish: Random Recoloring to merge colors ---
    if best_colors > 13:
        coloring = best_coloring.copy()
        current_colors = best_colors
        start_polish = time.time()
        
        while time.time() - start_polish < 5:
            unique, counts = np.unique(coloring, return_counts=True)
            if len(unique) <= 13: break
            
            # Pick a random edge
            idx = random.randint(0, m-1)
            old_color = int(coloring[idx])
            
            # Try to change it to a random other color
            other_colors = [c for c in unique if c != old_color]
            if not other_colors: continue
            new_color = random.choice(other_colors)
            
            coloring[idx] = new_color
            if check_violations_subset(coloring, edge_to_k4s[idx]) == 0:
                new_count = count_colors(coloring)
                if new_count < current_colors:
                    current_colors = new_count
                    if new_count < best_colors:
                        best_colors = new_count
                        best_coloring = coloring.copy()
            else:
                coloring[idx] = old_color

    # Return the best coloring found
    if best_coloring is None:
        result = list(range(m))
    else:
        result = [int(c) for c in best_coloring]
    
    # Ensure colors are in valid range and remap to minimize
    unique_colors = sorted(set(result))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    result = [color_map[c] for c in result]
    
    return result
