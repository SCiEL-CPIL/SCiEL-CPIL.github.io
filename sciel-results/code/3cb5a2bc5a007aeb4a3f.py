"""Find rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import numpy as np
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indices: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Number of equations: n*m*p = 36
    num_eqs = n * m * p
    
    # Try ranks from low to high
    best_result = None
    best_rank = nm * mp  # schoolbook as fallback
    
    # Pre-compute the index mapping
    # For equation (i,j,k): U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
    eq_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_indices.append((i * m + j, j * p + k, k * n + i))
    
    def try_rank(rank, max_iters=5000, seed=42):
        """Try to find a decomposition of given rank using iterative refinement."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(nm, rank) * 0.1
        V = rng.randn(mp, rank) * 0.1
        W = rng.randn(pn, rank) * 0.1
        
        # Use a fixed-point / alternating minimization approach
        # For each equation (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # We can alternate: fix two, solve for the third
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update W: for each (i,j,k), we need W[k*n+i, :] such that
            # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is a linear system if we fix U and V
            # For each (i,j,k): W[k*n+i, :] dot (U[i*m+j,:] * V[j*p+k,:]) = 1
            # But W rows are shared across equations with same (k,i)
            
            # Let's use a simpler approach: direct least squares per iteration
            # Build the linear system for W given U,V
            # Actually, let's use a different strategy: random restarts with verification
            
            break
        
        # Alternative: use a structured approach
        # Try to find a decomposition using the fact that we can factor through
        # intermediate dimensions
        
        return None
    
    # Better approach: use known algebraic constructions
    # For (2,3,6), we can try to use a block structure
    
    # Let me try a systematic search with small integer/half-integer entries
    # Start with rank 24, 27, 30, etc.
    
    # First, let's try to use a simpler construction
    # The matrix product C = A*B where A is 2x3, B is 3x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can write this as: C = A * B
    # Using tensor decomposition: C[i,k] = sum_r U[i,?,r] * V[?,k,r] ... 
    # But our convention is different.
    
    # Let me think about this differently. The tensor T has entries T[i,j,k] = 1 for all i,j,k
    # We want T = sum_r U_r ⊗ V_r ⊗ W_r where
    # U_r[i*m+j] = u_r[i,j], V_r[j*p+k] = v_r[j,k], W_r[k*n+i] = w_r[k,i]
    
    # So T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
    
    # This is a 3-tensor decomposition problem.
    
    # Let me try a direct approach: for each rank, set up the equations and solve
    # using a randomized method
    
    def solve_for_rank(rank, num_trials=20):
        """Solve for a rank-rank decomposition using random initialization and Newton's method."""
        if time.time() > deadline:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            rng = np.random.RandomState(12345 + trial)
            
            # Initialize
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
            
            # Scale to get reasonable magnitudes
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Use alternating minimization / Gauss-Seidel style
            for it in range(200):
                if time.time() > deadline:
                    return None
                
                # Check convergence
                residual = 0.0
                for (u_idx, v_idx, w_idx) in eq_indices:
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    residual += (val - 1.0) ** 2
                
                if residual < 1e-15:
                    # Found a solution! Verify exactly
                    exact_ok = True
                    for (u_idx, v_idx, w_idx) in eq_indices:
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        if abs(val - 1.0) > 1e-12:
                            exact_ok = False
                            break
                    if exact_ok:
                        return (U, V, W, rank)
                
                # Update W: for each w_idx, solve linear system
                # Group equations by w_idx
                w_groups = {}
                for (u_idx, v_idx, w_idx) in eq_indices:
                    if w_idx not in w_groups:
                        w_groups[w_idx] = []
                    w_groups[w_idx].append((u_idx, v_idx))
                
                for w_idx, eqs in w_groups.items():
                    # For each equation: sum_r U[u_idx,r]*V[v_idx,r]*W[w_idx,r] = 1
                    # This is: W[w_idx,:] dot (U[u_idx,:] * V[v_idx,:]) = 1
                    # Multiple equations for same w_idx form a linear system
                    A_mat = np.zeros((len(eqs), rank))
                    for eq_i, (u_idx, v_idx) in enumerate(eqs):
                        A_mat[eq_i, :] = U[u_idx, :] * V[v_idx, :]
                    b_vec = np.ones(len(eqs))
                    
                    # Solve least squares
                    try:
                        W[w_idx, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
                
                # Similarly update U and V
                u_groups = {}
                for (u_idx, v_idx, w_idx) in eq_indices:
                    if u_idx not in u_groups:
                        u_groups[u_idx] = []
                    u_groups[u_idx].append((v_idx, w_idx))
                
                for u_idx, eqs in u_groups.items():
                    A_mat = np.zeros((len(eqs), rank))
                    for eq_i, (v_idx, w_idx) in enumerate(eqs):
                        A_mat[eq_i, :] = V[v_idx, :] * W[w_idx, :]
                    b_vec = np.ones(len(eqs))
                    try:
                        U[u_idx, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
                
                v_groups = {}
                for (u_idx, v_idx, w_idx) in eq_indices:
                    if v_idx not in v_groups:
                        v_groups[v_idx] = []
                    v_groups[v_idx].append((u_idx, w_idx))
                
                for v_idx, eqs in v_groups.items():
                    A_mat = np.zeros((len(eqs), rank))
                    for eq_i, (u_idx, w_idx) in enumerate(eqs):
                        A_mat[eq_i, :] = U[u_idx, :] * W[w_idx, :]
                    b_vec = np.ones(len(eqs))
                    try:
                        V[v_idx, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Final check
            residual = 0.0
            for (u_idx, v_idx, w_idx) in eq_indices:
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                residual += (val - 1.0) ** 2
            if residual < 1e-12:
                return (U, V, W, rank)
        
        return None
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # But actually for matrix mult, lower bound is around (n*m*p)^(1/2) or similar
    # Let's try ranks: 18, 20, 22, 24, 27, 30, 32, 36
    
    for rank in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 36]:
        if time.time() > deadline:
            break
        
        result = solve_for_rank(rank, num_trials=5)
        if result is not None:
            U, V, W, r = result
            # Convert to lists of floats, check if they're close to half-integers
            # For the output, we need integer/half-integer entries
            # Let's round to nearest half-integer and verify
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify the rounded version
            ok = True
            for (u_idx, v_idx, w_idx) in eq_indices:
                val = np.dot(U_rounded[u_idx, :], V_rounded[v_idx, :] * W_rounded[w_idx, :])
                if abs(val - 1.0) > 1e-6:
                    ok = False
                    break
            
            if ok:
                # Return the rounded version
                U_list = [[float(x) for x in U_rounded[i]] for i in range(nm)]
                V_list = [[float(x) for x in V_rounded[i]] for i in range(mp)]
                W_list = [[float(x) for x in W_rounded[i]] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we found a numerical solution but rounding failed, try to find exact
        # by solving with rational constraints - this is harder
        # For now, continue
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
