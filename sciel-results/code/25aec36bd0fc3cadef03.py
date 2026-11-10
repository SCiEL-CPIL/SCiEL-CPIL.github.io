import time
import random
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Map (i, j) to index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    # Precompute K4s
    # Each K4 is a tuple of 6 edge indices
    k4s = []
    k4_edge_map = {} # edge_idx -> list of k4 indices
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    verts = [a, b, c, d]
                    edges = [
                        edge_idx(verts[0], verts[1]),
                        edge_idx(verts[0], verts[2]),
                        edge_idx(verts[0], verts[3]),
                        edge_idx(verts[1], verts[2]),
                        edge_idx(verts[1], verts[3]),
                        edge_idx(verts[2], verts[3])
                    ]
                    idx = len(k4s)
                    k4s.append(edges)
                    for e in edges:
                        if e not in k4_edge_map:
                            k4_edge_map[e] = []
                        k4_edge_map[e].append(idx)
    
    def count_colors(k4_edges, colors):
        return len(set(colors[e] for e in k4_edges))
    
    def is_valid_k4(k4_edges, colors):
        return count_colors(k4_edges, colors) >= 5
    
    def check_all_valid(colors):
        for edges in k4s:
            if not is_valid_k4(edges, colors):
                return False
        return True
    
    # Try to find a solution with K colors
    # We try K = 13, 14, 15
    # If we find one, return it.
    # Since 13 might be hard, we start with 15 and try to reduce?
    # Or just search for 15 first as it's more likely to be solvable quickly.
    # Actually, let's try 14 first. If it fails, 15.
    
    best_solution = None
    
    # Helper to run local search for a specific K
    def local_search(K, max_iters=500000):
        # Initialize randomly
        colors = np.random.randint(0, K, size=num_edges)
        
        # Precompute validity status
        # valid[k4_idx] = bool
        valid_status = np.array([is_valid_k4(edges, colors) for edges in k4s], dtype=bool)
        num_invalid = np.sum(~valid_status)
        
        if num_invalid == 0:
            return colors
        
        best_invalid_count = num_invalid
        best_colors = colors.copy()
        
        iter_count = 0
        while iter_count < max_iters:
            if time.time() > deadline:
                break
                
            # Pick a random invalid K4
            if num_invalid == 0:
                break
            
            invalid_indices = np.where(~valid_status)[0]
            k4_idx = invalid_indices[random.randint(0, len(invalid_indices) - 1)]
            edges = k4s[k4_idx]
            
            # Pick a random edge in this K4
            e_idx = edges[random.randint(0, 5)]
            
            # Current color of this edge
            old_c = colors[e_idx]
            
            # Find colors NOT in this K4
            current_colors_in_k4 = set(colors[e] for e in edges)
            available_colors = [c for c in range(K) if c not in current_colors_in_k4]
            
            if not available_colors:
                # Cannot fix this K4 by changing one edge to a new color
                # This K4 has K distinct colors already? No, it has < 5.
                # If K is small, maybe all colors are present?
                # If K=5, and it has 4 colors, there is 1 available.
                # If K=4, it has 4 colors, no available. But we need >=5, so K must be >=5.
                # If K=13, there are always available colors.
                continue
            
            new_c = random.choice(available_colors)
            
            # Check impact on other K4s containing e_idx
            affected_k4s = k4_edge_map[e_idx]
            
            # Calculate delta in validity
            # We need to check if any valid K4 becomes invalid
            # And if the target K4 becomes valid
            
            # Optimization: Only check affected K4s
            # We assume changing e_idx only affects K4s containing e_idx
            
            # Check if target K4 becomes valid
            # It will have colors: current_colors_in_k4 - {old_c} + {new_c}
            # Since new_c was not in current_colors_in_k4, and old_c is removed (or count reduced)
            # Actually, we just need to count.
            
            # For speed, we can just apply the change, check the specific K4s, and revert if bad.
            
            colors[e_idx] = new_c
            
            # Check target K4
            target_valid = is_valid_k4(edges, colors)
            
            # Check other affected K4s
            other_invalid = 0
            for other_k4_idx in affected_k4s:
                if other_k4_idx == k4_idx:
                    continue
                if valid_status[other_k4_idx]:
                    # It was valid, check if it remains valid
                    if not is_valid_k4(k4s[other_k4_idx], colors):
                        other_invalid += 1
                        break # Early exit
            
            if target_valid and other_invalid == 0:
                # Accept change
                # Update valid_status
                valid_status[k4_idx] = True
                num_invalid -= 1
                
                # Update status of other affected K4s (they might have changed status? 
                # No, we only checked if valid ones became invalid. 
                # What if invalid ones became valid? We don't track that immediately, 
                # but num_invalid is only decreased if target becomes valid.
                # We should update valid_status for all affected to keep it accurate.
                for other_k4_idx in affected_k4s:
                    if other_k4_idx == k4_idx:
                        continue
                    old_stat = valid_status[other_k4_idx]
                    new_stat = is_valid_k4(k4s[other_k4_idx], colors)
                    if old_stat != new_stat:
                        valid_status[other_k4_idx] = new_stat
                        if old_stat and not new_stat:
                            num_invalid += 1
                        elif not old_stat and new_stat:
                            num_invalid -= 1
                
                if num_invalid < best_invalid_count:
                    best_invalid_count = num_invalid
                    best_colors = colors.copy()
                    
                if num_invalid == 0:
                    return colors
            else:
                # Revert
                colors[e_idx] = old_c
                
            iter_count += 1
            
        return best_colors if best_invalid_count < num_invalid else colors

    # Try K = 14
    for K in [14, 15]:
        if time.time() > deadline - 10:
            break
        sol = local_search(K, max_iters=200000)
        if sol is not None and check_all_valid(sol):
            # Compact colors to 0..max
            used_colors = set(sol)
            color_map = {c: i for i, c in enumerate(sorted(used_colors))}
            compacted = np.array([color_map[c] for c in sol])
            # The task asks for 120 integer colours in 0..119
            # Minimize number of colors used.
            # The evaluator measures the raw value (number of colors used).
            # So we return the compacted list.
            return compacted.tolist()
            
    # Fallback: return the 17-color solution if we fail to find a better one
    # But we should try to return the best found even if not perfectly valid?
    # No, invalid scores 0.
    # So we must return a valid one.
    # The 17-color cyclic one is always valid.
    
    # Generate the 17-color solution
    res = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    return res
