"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T[a, b, c]
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i=i', j=j', k=k'
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    
    # Schoolbook solution as fallback
    rank_school = n*m*p
    U_school = np.zeros((nm, rank_school), dtype=np.int64)
    V_school = np.zeros((mp, rank_school), dtype=np.int64)
    W_school = np.zeros((pn, rank_school), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank decompositions
    # Strategy: for a given R, randomly generate U and V, solve for W, check consistency
    
    # Candidate value sets for entries
    # Use small integers and half-integers
    val_sets = [
        [-1, 0, 1],
        [-1, 0, 1, 2],
        [-2, -1, 0, 1, 2],
        [-1, 0, 1, -0.5, 0.5],
        [-2, -1, -0.5, 0, 0.5, 1, 2],
    ]
    
    def verify(U, V, W, tol=1e-9):
        """Verify that sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c] for all a,b,c"""
        # Compute the decomposed tensor
        # Use einsum for efficiency
        try:
            # T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            diff = np.max(np.abs(T_decomp - T.astype(float)))
            return diff < tol
        except:
            return False
    
    def is_rational_half(U, V, W, tol=1e-6):
        """Check if all entries are integer or half-integer"""
        for mat in [U, V, W]:
            for val in mat.flatten():
                # Check if val is close to an integer or half-integer
                nearest_half = round(val * 2) / 2
                if abs(val - nearest_half) > tol:
                    return False
        return True
    
    def try_rank(R, max_iters=200, val_set=None):
        """Try to find a decomposition of rank R"""
        if val_set is None:
            val_set = [-1, 0, 1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random U and V
            U = np.array([[random.choice(val_set) for _ in range(R)] for _ in range(nm)], dtype=float)
            V = np.array([[random.choice(val_set) for _ in range(R)] for _ in range(mp)], dtype=float)
            
            # For each c, solve: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b
            # This is: (U ⊙ V) W[:,c] = T_c where ⊙ means outer product structure
            # More precisely: for each c, the system is:
            # sum_r W[c,r] * (U[:,r] ⊗ V[:,r]) = T[:,:,c].flatten()
            
            # Build the system matrix A_sys of shape (nm*mp, R)
            # A_sys[a*mp+b, r] = U[a,r] * V[b,r]
            A_sys = np.zeros((nm*mp, R), dtype=float)
            for r in range(R):
                outer = np.outer(U[:, r], V[:, r])  # shape (nm, mp)
                A_sys[:, r] = outer.flatten()
            
            # For each c, solve A_sys * W[:,c] = T[:,:,c].flatten()
            W = np.zeros((pn, R), dtype=float)
            consistent = True
            for c in range(pn):
                rhs = T[:,:,c].flatten()
                # Solve least squares
                try:
                    sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, rhs, rcond=None)
                    W[c, :] = sol
                    # Check if solution is consistent (residuals should be ~0)
                    if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-6:
                        consistent = False
                        break
                    # Check if solution is exact
                    check = A_sys @ sol - rhs
                    if np.max(np.abs(check)) > 1e-6:
                        consistent = False
                        break
                except:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Check if W entries are rational (integer or half-integer)
            if not is_rational_half(U, V, W):
                continue
            
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Verify exactness with rounded values
            if verify(U_r, V_r, W_r, tol=1e-9):
                return (U_r, V_r, W_r)
        
        return None
    
    # Try decreasing ranks
    for R in range(55, 20, -1):
        if time.time() > deadline:
            break
        
        # Try with different value sets
        for vs in val_sets[:2]:  # Start with simpler value sets
            result = try_rank(R, max_iters=50, val_set=vs)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = R
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break
        
        if best_rank < rank_school:
            break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
