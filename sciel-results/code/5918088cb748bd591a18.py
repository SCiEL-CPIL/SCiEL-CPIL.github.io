import numpy as np
import time
import random

def construct():
    n = 20
    u_nodes = list(range(10))
    v_nodes = list(range(10, 20))
    
    # Matrix representation: M[i, j] is color of edge (u_nodes[i], v_nodes[j])
    # Size 10x10
    M = np.random.randint(0, 20, size=(10, 10))
    
    # Precompute cycles
    # Cycle (u1, v1, u2, v2) corresponds to edges:
    # (u1, v1) -> M[i1, j1]
    # (u2, v1) -> M[i2, j1]
    # (u2, v2) -> M[i2, j2]
    # (u1, v2) -> M[i1, j2]
    
    cycle_indices = []
    for i1 in range(10):
        for i2 in range(i1 + 1, 10):
            for j1 in range(10):
                for j2 in range(j1 + 1, 10):
                    # Edges in M:
                    # (i1, j1), (i2, j1), (i2, j2), (i1, j2)
                    cycle_indices.append(((i1, j1), (i2, j1), (i2, j2), (i1, j2)))
                    
    num_cycles = len(cycle_indices)
    
    # Flatten indices for vectorized access
    # We want an array of shape (num_cycles, 8) containing indices (r1, c1, r2, c2, r3, c3, r4, c4)
    idx_list = np.array([(c[0][0], c[0][1], c[1][0], c[1][1], c[2][0], c[2][1], c[3][0], c[3][1]) for c in cycle_indices])
    
    def get_colors_for_cycles(M):
        # M is 10x10
        r1, c1, r2, c2, r3, c3, r4, c4 = idx_list.T
        cols = np.column_stack((M[r1, c1], M[r2, c2], M[r3, c3], M[r4, c4]))
        return cols

    def count_bad(M):
        cols = get_colors_for_cycles(M)
        c0, c1, c2, c3 = cols.T
        bad = (c0 == c1) | (c0 == c2) | (c0 == c3) | (c1 == c2) | (c1 == c3) | (c2 == c3)
        return np.sum(bad), bad

    start_time = time.time()
    timeout = 115.0
    
    current_bad_count, current_bad_mask = count_bad(M)
    best_M = M.copy()
    best_bad_count = current_bad_count
    
    while time.time() - start_time < timeout:
        if current_bad_count == 0:
            break
            
        # Pick a random bad cycle
        bad_indices = np.where(current_bad_mask)[0]
        if len(bad_indices) == 0:
            break
        target_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
        
        # Get the 4 edges in this cycle
        r1, c1, r2, c2, r3, c3, r4, c4 = idx_list[target_idx]
        edges_in_cycle = [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
        
        # Pick a random edge from the cycle to change
        r, c = edges_in_cycle[random.randint(0, 3)]
        old_color = M[r, c]
        
        improved = False
        # Try a few random new colors
        for _ in range(10):
            new_color = random.randint(0, 19)
            if new_color == old_color:
                continue
            M[r, c] = new_color
            new_bad_count, new_bad_mask = count_bad(M)
            
            if new_bad_count < current_bad_count:
                current_bad_count = new_bad_count
                current_bad_mask = new_bad_mask
                if current_bad_count < best_bad_count:
                    best_bad_count = current_bad_count
                    best_M = M.copy()
                improved = True
                break
            else:
                # Revert
                M[r, c] = old_color
                
        if not improved:
            # If stuck, do a random move
            rr = random.randint(0, 9)
            cc = random.randint(0, 9)
            M[rr, cc] = random.randint(0, 19)
            current_bad_count, current_bad_mask = count_bad(M)
            if current_bad_count < best_bad_count:
                best_bad_count = current_bad_count
                best_M = M.copy()

    # Final check
    final_bad, _ = count_bad(best_M)
    if final_bad > 0:
        # Fallback: Return a valid sparse graph (Star)
        result = []
        for i in range(1, 20):
            result.append([0, i, i % 20])
        return result
    
    # Construct result from best_M
    result = []
    for i in range(10):
        for j in range(10):
            u = u_nodes[i]
            v = v_nodes[j]
            c = best_M[i, j]
            result.append([u, v, c])
            
    return result
