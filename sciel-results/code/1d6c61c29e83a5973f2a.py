import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s limit, leave 10s margin for overhead and output
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    num_eqs = n * m * p  # 84
    
    min_rank = max(dim_U, dim_V, dim_W)  # 28
    
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank))
        V = np.zeros((dim_V, rank))
        W = np.zeros((dim_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def build_system(U, V, r):
        A = np.zeros((num_eqs, dim_W * r))
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_W = k * n + i
                    A[eq_idx, row_W*r : (row_W+1)*r] = U[i*m+j, :] * V[j*p+k, :]
                    eq_idx += 1
        b = np.ones(num_eqs)
        return A, b
    
    def verify_exact(U, V, W2, r, n, m, p):
        """Verify that sum_c U[i,j,c]*V[j,k,c]*W2[k,i,c] == 2 for all (i,j,k)
        W2 contains 2*W values (integers if W is half-integer)
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(r):
                        total += int(U[i*m+j, c]) * int(V[j*p+k, c]) * int(W2[k*n+i, c])
                    if total != 2:
                        return False
        return True
    
    def refine_local(U, V, W2, r, n, m, p):
        """Try to fix small integer errors in W2 via local search.
        W2 should contain integers (2*W). Returns refined W2 or None.
        """
        W2 = W2.copy().astype(int)
        # Check current error
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(r):
                        total += int(U[i*m+j, c]) * int(V[j*p+k, c]) * int(W2[k*n+i, c])
                    if total != 2:
                        errors.append((i, j, k, total))
        
        if not errors:
            return W2
        
        # Try flipping individual entries
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 1:
                break
            progress = False
            for i, j, k, total in errors[:]:
                diff = 2 - total
                if diff == 0:
                    continue
                # Try changing W2[k*n+i, c] by +/-1
                row_W = k * n + i
                for c in range(r):
                    u_val = int(U[i*m+j, c])
                    v_val = int(V[j*p+k, c])
                    if u_val == 0 or v_val == 0:
                        continue
                    # Changing W2[row_W, c] by delta changes total by u_val*v_val*delta
                    change = u_val * v_val
                    if change == 0:
                        continue
                    if diff % change == 0:
                        delta = diff // change
                        if abs(delta) <= 2:  # small adjustment
                            new_val = int(W2[row_W, c]) + delta
                            # Check if this helps other equations too
                            W2[row_W, c] = new_val
                            # Verify all equations
                            all_ok = True
                            for ii in range(n):
                                for jj in range(m):
                                    for kk in range(p):
                                        tot = 0
                                        for cc in range(r):
                                            tot += int(U[ii*m+jj, cc]) * int(V[jj*p+kk, cc]) * int(W2[kk*n+ii, cc])
                                        if tot != 2:
                                            all_ok = False
                                            break
                                    if not all_ok:
                                        break
                                if not all_ok:
                                    break
                            if all_ok:
                                return W2
                            else:
                                W2[row_W, c] = int(W2[row_W, c]) - delta
                                progress = True
            if not progress:
                break
        
        return None
    
    # Search for better rank
    best_result = None
    
    # Try different value sets for random initialization
    value_sets = [
        [-1, 0, 1],
        [-1, 1],
        [-2, -1, 0, 1, 2],
    ]
    
    for r in range(min_rank, 84):
        if time.time() > deadline - 10:
            break
        
        # Adaptive number of tries
        remaining_time = deadline - time.time()
        if r < 35:
            num_tries = min(50, max(10, int(remaining_time / 0.5)))
        elif r < 50:
            num_tries = min(30, max(5, int(remaining_time / 1.0)))
        else:
            num_tries = min(15, max(3, int(remaining_time / 2.0)))
        
        for vs_idx, values in enumerate(value_sets):
            if time.time() > deadline - 5:
                break
            
            tries_this_set = max(1, num_tries // len(value_sets))
            
            for _ in range(tries_this_set):
                if time.time() > deadline - 2:
                    break
                
                U = np.random.choice(values, size=(dim_U, r))
                V = np.random.choice(values, size=(dim_V, r))
                
                A, b = build_system(U, V, r)
                
                try:
                    W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
                except Exception:
                    continue
                    
                if residuals.size > 0 and residuals[0] > 1e-6:
                    continue
                    
                W_sol = W_flat.reshape((dim_W, r))
                
                # Check if close to half-integers
                W_scaled = W_sol * 2
                W_rounded = np.round(W_scaled).astype(int)
                
                max_diff = np.max(np.abs(W_scaled - W_rounded))
                
                if max_diff < 0.01:
                    # Very close to half-integers - verify directly
                    if verify_exact(U, V, W_rounded, r, n, m, p):
                        W_int = W_rounded / 2.0
                        result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W_int.tolist()
                        }
                        return result
                elif max_diff < 0.5:
                    # Close but not exact - try local refinement
                    refined = refine_local(U, V, W_rounded, r, n, m, p)
                    if refined is not None:
                        W_int = refined / 2.0
                        result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W_int.tolist()
                        }
                        return result
    
    # If no better rank found, return schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We need to ensure the interface is correct.
# The prompt says: Interface: construct() -> {'U', 'V', 'W'}
# My construct returns exactly that.
# EVOLVE-BLOCK-END
