import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    # Fallback: n=11 construction
    n_fb = 11
    colors_fb = [min(j - i, n_fb - j + i) - 1 for i in range(n_fb) for j in range(i + 1, n_fb)]
    best_n = n_fb
    best_colors = colors_fb
    
    # Try to find valid colorings for larger n
    # Prioritize 27 (new result), then 26 (known Lazebnik-Woldar), then others
    targets = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    for n in targets:
        if time.time() > deadline - 3:
            break
            
        # Allocate time budget
        if n == 27:
            t_limit = 25
        elif n == 26:
            t_limit = 35
        elif n == 25:
            t_limit = 15
        elif n <= 23:
            t_limit = 10
        else:
            t_limit = 5
            
        actual_limit = min(deadline - time.time() - 3, t_limit)
        if actual_limit < 2:
            continue
            
        success, colors = _find_coloring(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break  # Found the largest n that works
            
    return {'n': best_n, 'colors': best_colors}

def _find_coloring(n, deadline):
    """
    Find a valid 5-coloring of K_n with no monochromatic C4 using
    efficient local search with bitmask-based common neighbor counting.
    """
    if n < 3:
        return True, []
        
    m = n * (n - 1) // 2
    
    # Precompute edge endpoints
    edge_u = [0] * m
    edge_v = [0] * m
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            idx += 1
            
    # Use numpy for fast bit counting
    # adj_mask[c][v] = bitmask of neighbors of v in color c
    adj_mask = [[0] * n for _ in range(5)]
    
    def build_masks(colors):
        for c in range(5):
            for v in range(n):
                adj_mask[c][v] = 0
        for k in range(m):
            u = edge_u[k]
            v = edge_v[k]
            c = colors[k]
            adj_mask[c][u] |= (1 << v)
            adj_mask[c][v] |= (1 << u)
            
    def count_violations():
        """Count total C4 violations using bitmask intersections."""
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    # Count set bits
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        total += cnt - 1
        return total
        
    # Multi-restart local search with simulated annealing
    best_violations = float('inf')
    best_colors = None
    
    num_restarts = 10
    restart_time = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 1:
            break
            
        # Random initial coloring
        colors = [random.randint(0, 4) for _ in range(m)]
        build_masks(colors)
        
        current_violations = count_violations()
        
        if current_violations == 0:
            return True, colors
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # Simulated annealing
        T = 1.0
        alpha = 0.9999
        min_T = 0.01
        start_time = time.time()
        max_steps = int(500000)  # Cap steps per restart
        
        step = 0
        while step < max_steps and time.time() < deadline - 1:
            step += 1
            
            # Pick random edge
            k = random.randint(0, m - 1)
            u = edge_u[k]
            v = edge_v[k]
            old_c = colors[k]
            new_c = random.randint(0, 4)
            
            if new_c == old_c:
                continue
                
            # Calculate delta in violations
            delta = 0
            
            # For pair (u, w) where w in adj[old_c][v]:
            # Removing (u,v) from old_c reduces common neighbors of (u,w) in old_c by 1
            # if (w,v) is old_c. This decreases violations if count was >= 2.
            mask_w = adj_mask[old_c][v]
            while mask_w:
                lsb = mask_w & -mask_w
                w = lsb.bit_length() - 1
                mask_w -= lsb
                if w == u:
                    continue
                cnt = bin(adj_mask[old_c][u] & adj_mask[old_c][w]).count('1')
                if cnt >= 2:
                    delta -= 1
                    
            # For pair (u, w) where w in adj[new_c][v]:
            # Adding (u,v) to new_c increases common neighbors of (u,w) in new_c by 1
            # if (w,v) is new_c. This increases violations if count was >= 1.
            mask_w = adj_mask[new_c][v]
            while mask_w:
                lsb = mask_w & -mask_w
                w = lsb.bit_length() - 1
                mask_w -= lsb
                if w == u:
                    continue
                cnt = bin(adj_mask[new_c][u] & adj_mask[new_c][w]).count('1')
                if cnt >= 1:
                    delta += 1
                    
            # For pair (v, w) where w in adj[old_c][u]:
            mask_w = adj_mask[old_c][u]
            while mask_w:
                lsb = mask_w & -mask_w
                w = lsb.bit_length() - 1
                mask_w -= lsb
                if w == v:
                    continue
                cnt = bin(adj_mask[old_c][v] & adj_mask[old_c][w]).count('1')
                if cnt >= 2:
                    delta -= 1
                    
            # For pair (v, w) where w in adj[new_c][u]:
            mask_w = adj_mask[new_c][u]
            while mask_w:
                lsb = mask_w & -mask_w
                w = lsb.bit_length() - 1
                mask_w -= lsb
                if w == v:
                    continue
                cnt = bin(adj_mask[new_c][v] & adj_mask[new_c][w]).count('1')
                if cnt >= 1:
                    delta += 1
                    
            # Apply or revert
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                adj_mask[old_c][u] &= ~(1 << v)
                adj_mask[old_c][v] &= ~(1 << u)
                adj_mask[new_c][u] |= (1 << v)
                adj_mask[new_c][v] |= (1 << u)
                colors[k] = new_c
                current_violations += delta
                
                if current_violations == 0:
                    return True, colors
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                # Reject (no change needed since we didn't apply)
                pass
                
            T *= alpha
            if T < min_T:
                T = min_T
                
    if best_violations == 0 and best_colors is not None:
        return True, best_colors
    return False, best_colors if best_colors else []
