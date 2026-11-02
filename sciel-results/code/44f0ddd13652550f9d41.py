"""Matrix multiplication 3x3x7 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition."""
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        if len(U) != nm or len(V) != mp or len(W) != pn:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p  # 63
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Start with schoolbook
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try Strassen-based block decomposition
    # Strategy: Split the 3x3 into 2x2 + 1x1 blocks
    # Use Strassen's 7-multiplication algorithm for the 2x2 x 2x2 part
    
    # For the full 3x3 * 3x7 problem, we decompose as:
    # Let A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # Let B = [[B1], [B2]] where B1 is 2x7, B2 is 1x7
    # C11 = A11*B1 (2x7), C12 = A11*B2 (2x1)
    # C21 = A21*B1 (1x7), C22 = A21*B2 + A22*B2 (1x7)
    
    # For A11*B1 (2x2 * 2x7): Use Strassen-like, rank = 7*7 = 49? No.
    # Actually, 2x2 * 2x7 can be done with 7 scalar products where each involves
    # linear combinations. The tensor rank of 2x2 x 2x7 is 14 (since it's 7 copies of 2x2 x 2x1, each rank 4, but they can be merged).
    
    # Let's try a different approach: use the known rank-59 decomposition
    # based on combining Strassen for 2x2x2 with schoolbook for the rest.
    
    # Construction:
    # Partition indices:
    # i: {0,1} top, {2} bottom
    # j: {0,1} left, {2} right  
    # k: 0..6
    
    # Top-left 2x2 block multiply with all 7 columns: use 7 Strassen multiplications
    # Each Strassen multiplication handles one column's worth of 2x2 * 2x1
    # But Strassen for 2x2 * 2x1 has rank 4, so 7*4 = 28 for top-left
    
    # This is getting complex. Let me just do a computational search.
    
    # Search for rank reductions by trying to eliminate dependencies
    # Use numpy for vectorized operations
    
    def try_random_search(current_rank, max_trials=200):
        """Try to find a lower rank decomposition via random perturbation and refinement."""
        U = np.array(current_U, dtype=float)[:, :current_rank]
        V = np.array(current_V, dtype=float)[:, :current_rank]
        W = np.array(current_W, dtype=float)[:, :current_rank]
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 12345 + 67890)
            noise_scale = 0.5
            U_p = U + noise_scale * np.random.randn(*U.shape)
            V_p = V + noise_scale * np.random.randn(*V.shape)
            W_p = W + noise_scale * np.random.randn(*W.shape)
            
            # Iteratively refine
            for it in range(200):
                if time.time() > deadline:
                    break
                
                # Compute all errors
                errors = {}
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i*m+j
                            idx_v = j*p+k
                            idx_w = k*n+i
                            cur = np.dot(U_p[idx_u], V_p[idx_v]) * 0  # placeholder
                            # Actually: sum_r U[idx_u,r] * V[idx_v,r] * W[idx_w,r]
                            cur = 0.0
                            for r in range(current_rank):
                                cur += U_p[idx_u, r] * V_p[idx_v, r] * W_p[idx_w, r]
                            err = 1.0 - cur
                            errors[(i,j,k)] = err
                            if abs(err) > max_err:
                                max_err = abs(err)
                
                if max_err < 1e-10:
                    # Found a solution!
                    if verify(np.array(U_p).tolist(), np.array(V_p).tolist(), np.array(W_p).tolist()):
                        return U_p, V_p, W_p, current_rank
                
                # Update: adjust the component with largest magnitude
                for (i,j,k), err in errors.items():
                    if abs(err) < 1e-12:
                        continue
                    idx_u = i*m+j
                    idx_v = j*p+k
                    idx_w = k*n+i
                    
                    # Find best r to adjust
                    best_r = -1
                    best_gain = 0
                    for r in range(current_rank):
                        uv = U_p[idx_u, r] * V_p[idx_v, r]
                        vw = V_p[idx_v, r] * W_p[idx_w, r]
                        wu = W_p[idx_w, r] * U_p[idx_u, r]
                        # d(U[r])/dr = V*W, d(V[r])/dr = U*W, d(W[r])/dr = U*V
                        gain_u = abs(V_p[idx_v, r] * W_p[idx_w, r])
                        gain_v = abs(U_p[idx_u, r] * W_p[idx_w, r])
                        gain_w = abs(U_p[idx_u, r] * V_p[idx_v, r])
                        gain = max(gain_u, gain_v, gain_w)
                        if gain > best_gain:
                            best_gain = gain
                            best_r = r
                    
                    if best_r >= 0 and best_gain > 1e-6:
                        lr = 0.05  # learning rate
                        if gain_u >= gain_v and gain_u >= gain_w:
                            U_p[idx_u, best_r] -= lr * err / (V_p[idx_v, best_r] * W_p[idx_w, best_r]) if abs(V_p[idx_v, best_r] * W_p[idx_w, best_r]) > 1e-8 else 0
                        elif gain_v >= gain_u and gain_v >= gain_w:
                            V_p[idx_v, best_r] -= lr * err / (U_p[idx_u, best_r] * W_p[idx_w, best_r]) if abs(U_p[idx_u, best_r] * W_p[idx_w, best_r]) > 1e-8 else 0
                        else:
                            W_p[idx_w, best_r] -= lr * err / (U_p[idx_u, best_r] * V_p[idx_v, best_r]) if abs(U_p[idx_u, best_r] * V_p[idx_v, best_r]) > 1e-8 else 0
        
        return None
    
    # Search from rank 62 down to 50
    for target_rank in range(best_rank - 1, max(50, best_rank - 15), -1):
        if time.time() > deadline - 5:
            break
        
        current_U = best_U
        current_V = best_V
        current_W = best_W
        
        result = try_random_search(target_rank, max_trials=100)
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            best_U = U_new.tolist()
            best_V = V_new.tolist()
            best_W = W_new.tolist()
            best_rank = new_rank
            if verify(best_U, best_V, best_W):
                break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W, best_rank = schoolbook()
    
    # Convert to clean format (integers or half-integers)
    def clean_entries(arr):
        """Round entries to nearest half-integer if close."""
        result = []
        for row in arr:
            new_row = []
            for x in row:
                x2 = round(x * 2)
                if abs(x * 2 - x2) < 1e-6:
                    val = x2 / 2.0
                    if abs(val - round(val)) < 1e-6:
                        new_row.append(int(round(val)))
                    else:
                        new_row.append(val)
                else:
                    new_row.append(x)
            result.append(new_row)
        return result
    
    U_out = clean_entries(best_U)
    V_out = clean_entries(best_V)
    W_out = clean_entries(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
