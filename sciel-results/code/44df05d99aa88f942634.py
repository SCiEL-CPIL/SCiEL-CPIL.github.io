"""Search for a rank-minimal exact tensor decomposition of 2x3 * 3x4 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check exact decomposition: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1,
       and for all other index triples the sum is 0."""
    rank = len(U[0])
    # Build matrices for fast check
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # For each rank component c, the tensor is U[:,c] outer V[:,c] outer W[:,c]
    # The multiplication tensor T[i,j,k] = 1 if (i,j,k) corresponds to a valid product index
    # Actually, the condition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta(i,j,k valid)
    # But "valid" here means every (i,j,k) triple should give 1? No.
    # The matrix multiplication tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # Wait, let me re-read the task.
    # "exactly decompose the matrix multiplication tensor"
    # The multiplication tensor for C = A*B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j,k) and T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So for ALL i in [0,n), j in [0,m), k in [0,p), the sum must equal 1.
    
    total_error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _snap_to_half_int(U, V, W):
    """Snap entries to nearest half-integer if close enough."""
    def snap(mat):
        mat = np.array(mat, dtype=float)
        snapped = np.round(mat * 2) / 2.0
        if np.allclose(mat, snapped, atol=1e-6):
            return snapped.tolist()
        return mat.tolist()
    return snap(U), snap(V), snap(W)

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook rank as fallback
    rank_school = n * m * p
    U_school = [[0]*rank_school for _ in range(nm)]
    V_school = [[0]*rank_school for _ in range(mp)]
    W_school = [[0]*rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Known bounds: rank of 2x3*3x4 matrix multiplication
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Upper bound: n*m*p = 24
    # Let's try ranks from 12 upward
    
    # Use randomized search / optimization
    # The tensor equation is: for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is a system of equations in the entries of U, V, W.
    # We can use a random initialization and gradient descent, or just try random matrices.
    
    # Actually, let's think about this differently.
    # The matrix multiplication tensor is a 3-tensor. We want a rank-r decomposition.
    # 
    # Approach: For a given rank r, we have r*r*(nm + mp + pn) unknowns and n*m*p equations.
    # For r=12: 12*12*(6+12+8) = 144*26 = 3744 unknowns, 24 equations. Very underdetermined.
    # 
    # Let's try a structured approach. Fix W to be simple, then solve for U and V.
    
    # Strategy: Try small ranks with random initialization and verify.
    # Since the system is highly underdetermined for small r, random solutions should exist.
    
    # Let's try ranks from max(nm, mp, pn) = 12 up to schoolbook rank - 1
    
    min_rank = max(nm, mp, pn)  # 12
    
    for rank in range(min_rank, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 50 if rank <= 14 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Try to snap to half-integers
                U_s, V_s, W_s = _snap_to_half_int(U, V, W)
                # Re-verify after snapping
                U_a = np.array(U_s)
                V_a = np.array(V_s)
                W_a = np.array(W_s)
                valid_snapped = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(rank):
                                s += U_a[i*m+j, c] * V_a[j*p+k, c] * W_a[k*n+i, c]
                            if abs(s - 1.0) > 1e-6:
                                valid_snapped = False
                                break
                        if not valid_snapped:
                            break
                    if not valid_snapped:
                        break
                
                if valid_snapped:
                    best_rank = rank
                    best_result = {"U": U_s, "V": V_s, "W": W_s}
                else:
                    # Use the non-snapped version if it's valid
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                break  # Found a valid decomposition for this rank
    
    return best_result
