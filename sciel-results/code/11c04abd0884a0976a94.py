import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 115.0  # 5s safety margin
    
    # Lazebnik-Woldar construction for n=26
    def _build_lw(n):
        q = 5
        colors = []
        for i in range(n):
            ci = None if i == 25 else (i % 5, i // 5)
            for j in range(i + 1, n):
                cj = None if j == 25 else (j % 5, j // 5)
                if ci is None or cj is None:
                    colors.append(4)
                    continue
                dx = (cj[0] - ci[0]) % 5
                dy = (cj[1] - ci[1]) % 5
                if dx == 0:
                    c = 1
                elif dy == 0:
                    c = 0
                elif dx == dy:
                    c = 2
                elif (dx + dy) % 5 == 0:
                    c = 3
                else:
                    c = 4
                colors.append(c)
        return {'n': n, 'colors': colors}
    
    # Get baseline n=26
    base = _build_lw(26)
    best_n = 26
    best_colors = base['colors'].copy()
    
    # Fast bitset-based conflict counting
    def count_conflicts_bitset(colors, n):
        # Build bitsets for each color
        # adj[c] is list of n integers, each is a bitmask of neighbors
        adj = [0] * (5 * n)  # adj[c*n + i] = bitmask of neighbors of i in color c
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c * n + i] |= (1 << j)
                adj[c * n + j] |= (1 << i)
                idx += 1
        
        conflicts = 0
        for c in range(5):
            base_idx = c * n
            for i in range(n):
                bi = adj[base_idx + i]
                # For each neighbor j of i in color c, count common neighbors
                # Common neighbors of i and j in color c = adj[i] & adj[j]
                # But we need to count pairs (i,j) where |common| >= 2
                # Alternative: for each pair (i,j) in color c, count common neighbors
                # This is O(n^2) per color, which is fine for n<=29
                pass
        
        # More efficient: for each color, for each pair (i,j), count common neighbors
        # Use the bitsets
        for c in range(5):
            base_idx = c * n
            for i in range(n):
                bi = adj[base_idx + i]
                for j in range(i + 1, n):
                    if (bi >> j) & 1:  # i and j are connected in color c
                        bj = adj[base_idx + j]
                        common = bi & bj
                        # Count bits in common
                        # If >= 2, we have a C4
                        cnt = 0
                        temp = common
                        while temp:
                            temp &= temp - 1
                            cnt += 1
                        if cnt >= 2:
                            conflicts += (cnt - 1)  # excess common neighbors
        return conflicts
    
    # Faster conflict count using numpy
    def count_conflicts_np(colors, n):
        adjs = [np.zeros((n, n), dtype=np.uint8) for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adjs[c][i][j] = 1
                adjs[c][j][i] = 1
                idx += 1
        
        conflicts = 0
        for c in range(5):
            A = adjs[c]
            A2 = A @ A
            # Off-diagonal entries > 1 indicate C4s
            # We want sum of (A2[i][j] - 1) for A2[i][j] > 1, i != j
            # But this counts each C4 twice (once for (i,j) and once for (j,i))
            # So divide by 2
            np.fill_diagonal(A2, 0)
            excess = np.maximum(A2 - 1, 0)
            conflicts += np.sum(excess) // 2
        return conflicts
    
    # Verify validity
    def is_valid(colors, n):
        return count_conflicts_np(colors, n) == 0
    
    # Verify baseline
    if not is_valid(best_colors, best_n):
        # Fallback to K11
        n = 11
        best_n = n
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        if not is_valid(best_colors, best_n):
            return {'n': best_n, 'colors': best_colors}
    
    # Try to extend to larger n using SA
    for target_n in range(best_n + 1, 29):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Try multiple restarts with SA
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize: take the current best coloring and extend it
            # For edges involving the new vertex, use random colors
            if target_n > len(best_colors) // (2 * (len(best_colors) // (2 * (len(best_colors) // 2)))):
                pass
            
            # Simpler: create a new coloring for target_n
            # Start with the structure of the best coloring if possible
            c_list = [random.randint(0, 4) for _ in range(target_n * (target_n - 1) // 2)]
            
            # Warm start: if we have a valid coloring for smaller n, use it
            # For now, use random init
            
            current_conf = count_conflicts_np(c_list, target_n)
            
            # SA
            temp = 1.0
            cooling = 0.995
            steps_per_restart = 500
            
            for step in range(steps_per_restart):
                if time.time() > deadline - 5:
                    break
                if current_conf == 0:
                    found = True
                    break
                
                # Pick random edge
                i = random.randint(0, target_n - 1)
                j = random.randint(0, target_n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                # Index formula
                idx = i * (target_n - 1) - (i * (i - 1)) // 2 + (j - i - 1)
                
                old_c = c_list[idx]
                new_c = random.randint(0, 4)
                if new_c == old_c:
                    continue
                
                c_list[idx] = new_c
                new_conf = count_conflicts_np(c_list, target_n)
                delta_conf = new_conf - current_conf
                
                if delta_conf < 0 or random.random() < np.exp(-delta_conf / max(temp, 0.01)):
                    current_conf = new_conf
                else:
                    c_list[idx] = old_c
                
                temp *= cooling
            
            if found:
                best_n = target_n
                best_colors = c_list
                break
        
        if not found and time.time() > deadline - 10:
            break
    
    # Final verification
    if not is_valid(best_colors, best_n):
        # Fallback
        n = 11
        best_n = n
        best_colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    return {'n': best_n, 'colors': best_colors}
