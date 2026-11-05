"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # 1. Generate all 4-term arithmetic progressions
    # a, a+d, a+2d, a+3d with 1 <= a, a+3d <= n
    # 0-indexed: a0, a0+d, a0+2d, a0+3d
    ap_list = []
    for d in range(1, n):
        max_a0 = n - 3 * d
        if max_a0 <= 0:
            break
        for a0 in range(max_a0):
            ap_list.append((a0, a0 + d, a0 + 2 * d, a0 + 3 * d))
    
    ap_list = np.array(ap_list, dtype=np.int32)
    num_aps = len(ap_list)
    
    # 2. Build Pair Weight Matrix W
    # The energy term for an AP is 1/8 * (1 + sum_pairs + prod_quad)
    # Pairs are (a,b), (b,c), (c,d), (a,c), (b,d), (a,d)
    W = np.zeros((n, n), dtype=np.float64)
    
    # Extract indices for vectorized update of W
    a_idx = ap_list[:, 0]
    b_idx = ap_list[:, 1]
    c_idx = ap_list[:, 2]
    d_idx = ap_list[:, 3]
    
    # Update W for each pair type
    # (a,b)
    W[a_idx, b_idx] += 1
    W[b_idx, a_idx] += 1
    # (b,c)
    W[b_idx, c_idx] += 1
    W[c_idx, b_idx] += 1
    # (c,d)
    W[c_idx, d_idx] += 1
    W[d_idx, c_idx] += 1
    # (a,c)
    W[a_idx, c_idx] += 1
    W[c_idx, a_idx] += 1
    # (b,d)
    W[b_idx, d_idx] += 1
    W[d_idx, b_idx] += 1
    # (a,d)
    W[a_idx, d_idx] += 1
    W[d_idx, a_idx] += 1
    
    W *= 0.125  # 1/8
    
    # 3. Initialize y in {-1, 1}
    # Start with a random initialization
    rng = np.random.default_rng(42)
    y = rng.choice([-1, 1], size=n).astype(np.float64)
    
    def count_mono(y):
        # Convert y to colors 0,1 for counting?
        # Or just use the energy formula?
        # Energy H = sum over APs of I(mono).
        # We can compute H directly from y.
        # I(mono) = 0.125 * (1 + y_a y_b + y_b y_c + y_c y_d + y_a y_c + y_b y_d + y_a y_d + y_a y_b y_c y_d)
        # Let's just count directly for verification/best tracking, it's fast enough.
        # colors = (y + 1) / 2
        # quad_colors = colors[ap_list]
        # return np.sum((quad_colors.min(axis=1) == quad_colors.max(axis=1)))
        
        # Faster:
        y_aps = y[ap_list]
        # Compute indicator for each AP
        # term = 1 + y0y1 + y1y2 + y2y3 + y0y2 + y1y3 + y0y3 + y0y1y2y3
        term = (1 + 
                y_aps[:, 0] * y_aps[:, 1] + 
                y_aps[:, 1] * y_aps[:, 2] + 
                y_aps[:, 2] * y_aps[:, 3] + 
                y_aps[:, 0] * y_aps[:, 2] + 
                y_aps[:, 1] * y_aps[:, 3] + 
                y_aps[:, 0] * y_aps[:, 3] + 
                y_aps[:, 0] * y_aps[:, 1] * y_aps[:, 2] * y_aps[:, 3])
        indicators = (term * 0.125)
        # Due to floating point, indicators should be 0 or 1
        return np.sum(indicators > 0.5)
    
    best_y = y.copy()
    best_score = count_mono(y)
    
    # 4. Local Search Loop
    # We perform batched flips.
    for it in range(100):
        # Compute Force F
        # F_k = sum_j W[k,j] y_j + sum_{Q in k} 0.125 * prod_{j in Q\k} y_j
        
        # Pair part
        F_pair = W @ y
        
        # Quad part
        # We need to compute for each AP the product of the other 3 y's
        y_aps = y[ap_list] # shape (num_aps, 4)
        
        P0 = y_aps[:, 1] * y_aps[:, 2] * y_aps[:, 3]
        P1 = y_aps[:, 0] * y_aps[:, 2] * y_aps[:, 3]
        P2 = y_aps[:, 0] * y_aps[:, 1] * y_aps[:, 3]
        P3 = y_aps[:, 0] * y_aps[:, 1] * y_aps[:, 2]
        
        F_quad = np.zeros(n, dtype=np.float64)
        np.add.at(F_quad, a_idx, P0 * 0.125)
        np.add.at(F_quad, b_idx, P1 * 0.125)
        np.add.at(F_quad, c_idx, P2 * 0.125)
        np.add.at(F_quad, d_idx, P3 * 0.125)
        
        F = F_pair + F_quad
        
        # Flip bits where F > 0
        # Condition: if sum > 0, then Delta H < 0, so flip improves energy
        flip_mask = F > 0
        
        if not np.any(flip_mask):
            break
        
        y[flip_mask] *= -1
        
        current_score = count_mono(y)
        if current_score < best_score:
            best_score = current_score
            best_y = y.copy()
            
    # Convert best_y back to 0/1 colors
    # y = 1 -> color 0? Or 1?
    # In count_mono, we used y directly.
    # The task requires returning colors in {0,1}.
    # Let's map y=1 to 0 and y=-1 to 1? Or y=1 to 1?
    # It doesn't matter for the count, just consistency.
    # Let's say color = 0 if y == 1, color = 1 if y == -1.
    colors = np.where(best_y == 1, 0, 1).astype(int)
    
    return [int(c) for c in colors]
# EVOLVE-BLOCK-END
