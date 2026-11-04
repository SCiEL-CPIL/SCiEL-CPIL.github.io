import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 145.0  # Safety margin

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    def is_valid_w_val(v):
        """Check if v is integer or half-integer."""
        if abs(v) < 1e-12:
            return True
        if abs(v - round(v)) < 1e-9:
            return True
        if abs(v*2 - round(v*2)) < 1e-9:
            return True
        return False

    def snap_w_vals(w_vec):
        """Snap values to nearest integer or half-integer if close."""
        snapped = np.zeros_like(w_vec)
        for idx, val in enumerate(w_vec):
            if abs(val) < 1e-12:
                snapped[idx] = 0.0
            elif abs(val - round(val)) < 1e-9:
                snapped[idx] = round(val)
            elif abs(val*2 - round(val*2)) < 1e-9:
                snapped[idx] = round(val*2) / 2.0
            else:
                return None
        return snapped

    def find_decomp(n, m, p, target_rank, time_limit):
        """
        Find a rank-minimal exact tensor decomposition for n x m x p.
        Uses random sparse U, V and solves for W.
        """
        best_rank = n * m * p
        best_U = None
        best_V = None
        best_W = None
        
        # Schoolbook baseline
        U_sb = np.zeros((n*m, best_rank))
        V_sb = np.zeros((m*p, best_rank))
        W_sb = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1.0
                    V_sb[j*p + k, col] = 1.0
                    W_sb[k*n + i, col] = 1.0
                    col += 1
        
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        
        # Try to find better rank
        for rank in range(max(10, int(target_rank*0.8)), best_rank):
            if time.time() - start_time > time_limit:
                break
            
            # Try a few random instances for this rank
            for trial in range(100):
                if time.time() - start_time > time_limit:
                    break
                
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                
                # Generate sparse U, V with entries in {-1, 1}
                # Heuristic: Each column should have some structure
                # Random sparse
                for r in range(rank):
                    # Pick 2-4 non-zeros for U
                    nz_u = random.randint(2, min(4, n*m))
                    indices_u = random.sample(range(n*m), nz_u)
                    for idx in indices_u:
                        U[idx, r] = random.choice([-1.0, 1.0])
                        
                    # Pick 2-4 non-zeros for V
                    nz_v = random.randint(2, min(4, m*p))
                    indices_v = random.sample(range(m*p), nz_v)
                    for idx in indices_v:
                        V[idx, r] = random.choice([-1.0, 1.0])
                
                # Check if valid W exists
                W = np.zeros((p*n, rank))
                valid = True
                for i in range(n):
                    for k in range(p):
                        # Construct A_sub of shape (m, rank)
                        A_sub = np.zeros((m, rank))
                        for j in range(m):
                            A_sub[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        
                        # Solve A_sub @ W_row = 1
                        # Use lstsq
                        try:
                            w_sol, residuals, rank_A, sv = np.linalg.lstsq(A_sub, np.ones(m), rcond=None)
                            # Check if solution is close to integer/half-integer
                            w_snapped = snap_w_vals(w_sol)
                            if w_snapped is None:
                                valid = False
                                break
                            
                            # Verify the snapped solution exactly
                            # A_sub @ w_snapped should be 1
                            check = A_sub @ w_snapped
                            if not np.allclose(check, np.ones(m), atol=1e-9):
                                valid = False
                                break
                            
                            W[k*n+i, :] = w_snapped
                        except np.linalg.LinAlgError:
                            valid = False
                            break
                            
                    if not valid:
                        break
                
                if valid:
                    # Final verification
                    # (Optional, but good for safety)
                    is_ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                                if abs(s - 1.0) > 1e-9:
                                    is_ok = False
                                    break
                            if not is_ok: break
                        if not is_ok: break
                    if is_ok:
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W
                        break # Found a valid one for this rank, try next lower rank? 
                        # Actually, we want the minimum rank. So we should continue to lower ranks.
                        # But we are iterating rank upwards? No, we want to find ANY rank < 84.
                        # If we find rank 37, we should try to find 36, 35...
                        # So we should break the trial loop and continue the rank loop downwards.
                        # But my loop is `for rank in range(...)`.
                        # Let's iterate from low to high? No, we want the minimum.
                        # So we should iterate from low to high and stop when found?
                        # No, if we find rank 37, we don't know if 36 exists.
                        # But 37 is better than 84.
                        # To be safe, we can just return the first one found if it's significantly better.
                        # Or continue searching for lower.
                        # Given time constraints, finding one good one is a big win.
                        # Let's break out of the rank loop too if we find a good one.
                        break
            
            if best_rank < n * m * p:
                # Found an improvement, try to improve further if time permits
                if time.time() - start_time > time_limit * 0.8:
                    break
        return best_U, best_V, best_W, best_rank

    # Strategy: Split 3x4x7 into 3x4x4 and 3x4x3
    # 3x4x4 rank ~ 20, 3x4x3 rank ~ 17. Total ~ 37.
    
    # Time allocation
    time_left = DEADLINE - (time.time() - start_time)
    t1 = time_left * 0.5
    t2 = time_left * 0.5
    
    # Find 3x4x4 decomposition
    U1, V1, W1, r1 = find_decomp(3, 4, 4, 25, t1)
    
    # Find 3x4x3 decomposition
    U2, V2, W2, r2 = find_decomp(3, 4, 3, 20, t2)
    
    # Combine if successful
    if r1 < 12*4 and r2 < 12*3: # If improved from schoolbook
        total_rank = r1 + r2
        
        U_total = np.zeros((12, total_rank))
        V_total = np.zeros((28, total_rank))
        W_total = np.zeros((21, total_rank))
        
        # U is same for both (A is 3x4)
        U_total[:, :r1] = U1
        U_total[:, r1:] = U2
        
        # V: B is 4x7. B1 is 4x4 (cols 0-3), B2 is 4x3 (cols 4-6)
        # V1 is for B1 (4x4). V2 is for B2 (4x3).
        # V indices: j (0-3), k (0-6)
        for j in range(4):
            for k in range(4):
                V_total[j*7+k, :r1] = V1[j*4+k, :]
            for k in range(3):
                V_total[j*7+(4+k), r1:] = V2[j*4+k, :]
        
        # W: C is 3x7. C1 is 3x4 (cols 0-3), C2 is 3x3 (cols 4-6)
        # W indices: k (0-6), i (0-2)
        for k in range(4):
            for i in range(3):
                W_total[k*3+i, :r1] = W1[k*3+i, :]
        for k in range(3):
            for i in range(3):
                W_total[(4+k)*3+i, r1:] = W2[k*3+i, :]
        
        # Final check
        valid = True
        for i in range(3):
            for j in range(4):
                for k in range(7):
                    s = np.dot(U_total[i*4+j, :], V_total[j*7+k, :] * W_total[k*3+i, :])
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            return {
                "U": U_total.tolist(),
                "V": V_total.tolist(),
                "W": W_total.tolist()
            }
    
    # Fallback: Direct search for 3x4x7
    U, V, W, r = find_decomp(3, 4, 7, 84, DEADLINE - (time.time() - start_time) - 5)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
