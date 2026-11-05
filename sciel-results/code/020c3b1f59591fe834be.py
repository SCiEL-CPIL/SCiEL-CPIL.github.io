import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    num_colors = 5
    
    # Try n from high to low
    for n in range(28, 10, -1):
        if time.time() > deadline - 10:
            break
        
        # Algebraic-inspired initialization
        # Use a structured coloring based on vertex differences
        # For 5 colors, use modular arithmetic
        def init_structured():
            col = [[0]*n for _ in range(n)]
            # Use a hash-like function based on vertex indices
            # Try multiple strategies
            for i in range(n):
                for j in range(i+1, n):
                    # Combine multiple modular hashes
                    d = j - i
                    h = (i * 7 + j * 13 + d * 11) % 5
                    col[i][j] = h
                    col[j][i] = h
            return col
        
        # Incremental C4 counting with bitmasks
        # adj_mask[c][u] = bitmask of neighbors of u in color c
        def build_adj_masks(col):
            adj = [[0]*n for _ in range(num_colors)]
            for i in range(n):
                for j in range(i+1, n):
                    c = col[i][j]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
            return adj
        
        def count_c4s_for_pair(u, v, adj):
            """Count total C4s involving pair (u,v) across all colors."""
            total = 0
            for c in range(num_colors):
                common = adj[c][u] & adj[c][v]
                k = bin(common).count('1')
                if k >= 2:
                    total += k * (k - 1) // 2
            return total
        
        def total_penalty(adj):
            total = 0
            for c in range(num_colors):
                for u in range(n):
                    for v in range(u+1, n):
                        common = adj[c][u] & adj[c][v]
                        k = bin(common).count('1')
                        if k >= 2:
                            total += k * (k - 1) // 2
            return total
        
        def compute_delta(i, j, old_c, new_c, col, adj):
            """Compute change in total C4 count when edge (i,j) changes from old_c to new_c."""
            delta = 0
            
            # For old_c: removing edge (i,j)
            # Affected pairs: (i,k) and (j,k) for k != i,j, and (i,j)
            # Before removal, N_old_c(i) contains j, N_old_c(j) contains i
            
            # Pair (i, j): 
            # Before: common = adj[old_c][i] & adj[old_c][j], which includes each other
            # After: neither i in N(j) nor j in N(i)
            # The common neighbors before: adj[old_c][i] & adj[old_c][j]
            # After removing: (adj[old_c][i] & ~(1<<j)) & (adj[old_c][j] & ~(1<<i))
            # = adj[old_c][i] & adj[old_c][j] & ~(1<<i) & ~(1<<j)
            # Since i not in adj[old_c][i] and j not in adj[old_c][j], 
            # the common set doesn't include i or j anyway (no self-loops)
            # So common_before = adj[old_c][i] & adj[old_c][j]
            # common_after = same (since i,j not in their own neighbor sets)
            # Wait, but the edge (i,j) means j in N(i) and i in N(j)
            # So adj[old_c][i] has bit j, adj[old_c][j] has bit i
            # common_before includes... j is in adj[i], is j in adj[j]? No. i is in adj[j], is i in adj[i]? No.
            # So common_before = adj[old_c][i] & adj[old_c][j] does NOT include i or j
            # common_after = (adj[old_c][i] & ~(1<<j)) & (adj[old_c][j] & ~(1<<i))
            # Since j not in adj[old_c][j] and i not in adj[old_c][i], this equals common_before
            # So pair (i,j) is unaffected!
            
            # Pair (i, k) for k != i, j:
            # Before: common = adj[old_c][i] & adj[old_c][k], includes j if j in adj[old_c][k]
            # After: common' = (adj[old_c][i] & ~(1<<j)) & adj[old_c][k]
            # Change: if j in adj[old_c][k], then common loses j, so k_common decreases by 1
            # C(k,2) changes from C(k,2) to C(k-1,2) = C(k,2) - (k-1)
            # So delta for this pair = -(k-1) where k = |common_before|
            
            # Similarly for (j, k)
            
            # Let's compute this properly
            # For each k != i, j:
            #   delta_{i,k} = C(k_after, 2) - C(k_before, 2) where k_before = |N(i) ∩ N(k)|, k_after = k_before - [j in N(k)]
            #   delta_{j,k} = C(k_after, 2) - C(k_before, 2) where k_before = |N(j) ∩ N(k)|, k_after = k_before - [i in N(k)]
            
            # This is O(n) per color, so O(5n) total. Good.
            
            for c in [old_c, new_c]:
                sign = -1 if c == old_c else 1  # -1 for removal, +1 for addition
                
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    # Pair (i, k)
                    # Before change: common_ik = adj[c][i] & adj[c][k]
                    # The edge (i,j) contributes j to N(i) if c is the color
                    # j is in N(k) iff adj[c][k] has bit j
                    j_in_k = (adj[c][k] >> j) & 1
                    k_before_ik = bin(adj[c][i] & adj[c][k]).count('1')
                    k_after_ik = k_before_ik - sign * j_in_k
                    if k_after_ik < 0:
                        k_after_ik = 0
                    delta_ik = k_after_ik * (k_after_ik - 1) // 2 - k_before_ik * (k_before_ik - 1) // 2
                    delta += delta_ik
                    
                    # Pair (j, k)
                    i_in_k = (adj[c][k] >> i) & 1
                    k_before_jk = bin(adj[c][j] & adj[c][k]).count('1')
                    k_after_jk = k_before_jk - sign * i_in_k
                    if k_after_jk < 0:
                        k_after_jk = 0
                    delta_jk = k_after_jk * (k_after_jk - 1) // 2 - k_before_jk * (k_before_jk - 1) // 2
                    delta += delta_jk
            
            return delta
        
        def apply_change(i, j, old_c, new_c, col, adj):
            col[i][j] = new_c
            col[j][i] = new_c
            adj[old_c][i] &= ~(1 << j)
            adj[old_c][j] &= ~(1 << i)
            adj[new_c][i] |= (1 << j)
            adj[new_c][j] |= (1 << i)
        
        def undo_change(i, j, old_c, new_c, col, adj):
            col[i][j] = old_c
            col[j][i] = old_c
            adj[new_c][i] &= ~(1 << j)
            adj[new_c][j] &= ~(1 << i)
            adj[old_c][i] |= (1 << j)
            adj[old_c][j] |= (1 << i)
        
        def try_solve():
            # Try multiple initializations
            best_col = None
            best_penalty = float('inf')
            
            for init_method in range(3):
                if time.time() > deadline:
                    break
                
                if init_method == 0:
                    col = init_structured()
                elif init_method == 1:
                    # Random
                    col = [[0]*n for _ in range(n)]
                    for i in range(n):
                        for j in range(i+1, n):
                            c = random.randint(0, num_colors-1)
                            col[i][j] = c
                            col[j][i] = c
                else:
                    # Another structured init
                    col = [[0]*n for _ in range(n)]
                    for i in range(n):
                        for j in range(i+1, n):
                            c = (i + 2*j) % 5
                            col[i][j] = c
                            col[j][i] = c
                
                adj = build_adj_masks(col)
                penalty = total_penalty(adj)
                
                if penalty == 0:
                    return col
                
                best_local_penalty = penalty
                best_local_col = [row[:] for row in col]
                best_local_adj = [[row[:] for row in adj[c]] for c in range(num_colors)]
                
                no_improve = 0
                temp = 2.0
                
                while time.time() < deadline:
                    if penalty == 0:
                        return col
                    
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i == j:
                        continue
                    if i > j:
                        i, j = j, i
                    
                    old_c = col[i][j]
                    new_c = random.randint(0, num_colors-1)
                    if new_c == old_c:
                        continue
                    
                    delta = compute_delta(i, j, old_c, new_c, col, adj)
                    
                    if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / max(temp, 0.1))):
                        apply_change(i, j, old_c, new_c, col, adj)
                        penalty += delta
                        no_improve = 0
                        if penalty < best_local_penalty:
                            best_local_penalty = penalty
                            best_local_col = [row[:] for row in col]
                            best_local_adj = [[row[:] for row in adj[c]] for c in range(num_colors)]
                    else:
                        no_improve += 1
                    
                    if no_improve > 300:
                        break
                    temp *= 0.997
                
                if best_local_penalty < best_penalty:
                    best_penalty = best_local_penalty
                    best_col = best_local_col
            
            if best_col is not None and best_penalty == 0:
                return best_col
            return None
        
        sol = try_solve()
        if sol is not None:
            # Verify
            adj = build_adj_masks(sol)
            if total_penalty(adj) == 0:
                res_colors = []
                for i in range(n):
                    for j in range(i+1, n):
                        res_colors.append(sol[i][j])
                return {'n': n, 'colors': res_colors}
    
    # Fallback: try n=11 with simple construction
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
