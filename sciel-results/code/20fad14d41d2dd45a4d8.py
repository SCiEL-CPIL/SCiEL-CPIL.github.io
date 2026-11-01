"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U @ V^T reconstructs the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0
    
    # Build tensor T[i,j,k] where i in [0,nm), j in [0,mp), k in [0,pn)
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # The tensor for matrix mult: T[i,j,k] = 1 if the corresponding entries multiply
    
    # For A (n x m) * B (m x p) = C (n x p):
    # i indexes A: i = a*n + b where a in [0,n), b in [0,m)
    # j indexes B: j = b*m + c where b in [0,m), c in [0,p)
    # k indexes C: k = c*n + a where c in [0,p), a in [0,n)
    # T[i,j,k] = 1 if the b indices match, else 0
    
    # Check by building the full tensor
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # Compute for all i, j, k
    for i in range(nm):
        a, b = divmod(i, n)
        for j in range(mp):
            b2, c = divmod(j, m)
            for k in range(pn):
                c2, a2 = divmod(k, p)
                # Expected value
                if b == b2 and a == a2 and c == c2:
                    expected = 1.0
                else:
                    expected = 0.0
                # Computed value
                computed = float(np.dot(U_np[i], V_np[j]) * 0)  # placeholder
                # Actually: sum_r U[i,r] * V[j,r] * W[k,r]
                computed = 0.0
                for r in range(rank):
                    computed += U_np[i, r] * V_np[j, r] * W_np[k, r]
                if abs(computed - expected) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using vectorized operations."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each rank term r, the contribution to T is outer(U[:,r], outer(V[:,r], W[:,r]))
    # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    
    # Build expected tensor
    T_expected = np.zeros((nm, mp, pn), dtype=float)
    for i in range(nm):
        a, b = divmod(i, n)
        for j in range(mp):
            b2, c = divmod(j, m)
            for k in range(pn):
                c2, a2 = divmod(k, p)
                if b == b2 and a == a2 and c == c2:
                    T_expected[i, j, k] = 1.0
    
    # Compute T from decomposition
    T_computed = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        T_computed += np.einsum('i,j,k->ijk', U_np[:, r], V_np[:, r], W_np[:, r])
    
    return np.allclose(T_computed, T_expected, atol=tol)

def _try_rank_r_decomposition(n, m, p, rank, max_tries=100, seed=42, time_budget=30):
    """Try to find a rank-r decomposition via randomized algebraic elimination."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The tensor has nm*mp*pn entries, but many are zero.
    # Nonzero entries: for each b in [0,m), a in [0,n), c in [0,p):
    #   i = a*n + b, j = b*m + c, k = c*n + a
    # So there are m*n*p nonzero entries.
    nz_count = m * n * p
    
    # For rank r, we have r*(nm + mp + pn) unknowns.
    # We have nz_count equations (one per nonzero tensor entry).
    # Plus we need the zero entries to also be satisfied.
    
    # Strategy: Use the known rank-14 solution as a starting point,
    # then try to reduce by solving for which terms can be eliminated.
    
    # Alternative: Direct approach - set up the system for rank r
    # Variables: U (nm x r), V (mp x r), W (pn x r)
    # Constraints: for all (i,j,k), sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    
    # This is a bilinear system. We can use the following approach:
    # Fix U and V, then W is determined linearly.
    # Or use iterative refinement.
    
    # For now, let's try a simpler approach: start from rank 14 and try to merge terms.
    pass

def _construct_rank14():
    """Return the known rank-14 decomposition."""
    n, m, p = DIMENSIONS
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U[0][0] = 1; U[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    U[0][2] = 1
    U[3][3] = 1
    U[0][4] = 1; U[1][4] = 1
    U[2][5] = 1; U[0][5] = -1
    U[1][6] = 1; U[3][6] = -1
    
    U[0][7] = 1; U[3][7] = 1
    U[2][8] = 1; U[3][8] = 1
    U[0][9] = 1
    U[3][10] = 1
    U[0][11] = 1; U[1][11] = 1
    U[2][12] = 1; U[0][12] = -1
    U[1][13] = 1; U[3][13] = -1
    
    V[0][0] = 1; V[5][0] = 1
    V[0][1] = 1
    V[1][2] = 1; V[5][2] = -1
    V[4][3] = 1; V[0][3] = -1
    V[5][4] = 1
    V[0][5] = 1; V[1][5] = 1
    V[4][6] = 1; V[5][6] = 1
    
    V[2][7] = 1; V[7][7] = 1
    V[2][8] = 1
    V[3][9] = 1; V[7][9] = -1
    V[6][10] = 1; V[2][10] = -1
    V[7][11] = 1
    V[2][12] = 1; V[3][12] = 1
    V[6][13] = 1; V[7][13] = 1
    
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    W[1][1] = 1; W[1][3] = 1
    W[2][2] = 1; W[2][4] = 1
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    W[4][7] = 1; W[4][10] = 1; W[4][11] = -1; W[4][13] = 1
    W[5][8] = 1; W[5][10] = 1
    W[6][9] = 1; W[6][11] = 1
    W[7][7] = 1; W[7][8] = -1; W[7][9] = 1; W[7][12] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Leave margin for evaluator overhead
    
    # First, return the known rank-14 solution as fallback
    result_14 = _construct_rank14()
    best_result = result_14
    best_rank = 14
    
    # Try to find a rank-13 decomposition
    # Strategy: Use the rank-14 solution and try to eliminate one term
    # by solving a linear system
    
    # For rank 13: we need to find U (8x13), V (8x13), W (8x13)
    # such that sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
    
    # Approach: Start from rank 14 and try to set one term to zero
    # by adjusting the others. This is a nonlinear problem, but we can
    # try linearization.
    
    # Better approach: Direct search for rank 12 or 13 using structured forms
    
    # Let's try a more systematic approach:
    # The tensor T for 2x2 * 2x4 has structure:
    # A is 2x2: a00, a01, a10, a11 (indices 0,1,2,3)
    # B is 2x4: b00,b01,b02,b03,b10,b11,b12,b13 (indices 0-7)
    # C is 2x4: c00,c01,c02,c03,c10,c11,c12,c13 (indices 0-7, k=c*n+a)
    
    # Nonzero tensor entries (i,j,k):
    # i=0 (a00), j=0 (b00), k=0 (c00): 1
    # i=0, j=1 (b01), k=2 (c01): 1  [k = 1*2+0 = 2]
    # i=0, j=2 (b02), k=4 (c02): 1  [k = 2*2+0 = 4]
    # i=0, j=3 (b03), k=6 (c03): 1  [k = 3*2+0 = 6]
    # i=1 (a01), j=0 (b00), k=1 (c10): 1  [k = 0*2+1 = 1]
    # i=1, j=1 (b01), k=3 (c11): 1  [k = 1*2+1 = 3]
    # i=1, j=2 (b02), k=5 (c12): 1  [k = 2*2+1 = 5]
    # i=1, j=3 (b03), k=7 (c13): 1  [k = 3*2+1 = 7]
    # i=2 (a10), j=4 (b10), k=0: 1
    # i=2, j=5 (b11), k=2: 1
    # i=2, j=6 (b12), k=4: 1
    # i=2, j=7 (b13), k=6: 1
    # i=3 (a11), j=4 (b10), k=1: 1
    # i=3, j=5 (b11), k=3: 1
    # i=3, j=6 (b12), k=5: 1
    # i=3, j=7 (b13), k=7: 1
    
    # Total: 16 nonzero entries
    
    # For rank r, we have 16 equations and r*(8+8+8) = 24r unknowns.
    # For r=12: 288 unknowns, 16 equations - very underdetermined.
    # But the zero entries also impose constraints.
    
    # Total entries: 8*8*8 = 512, with 16 nonzero and 496 zero.
    # So we have 512 constraints on 24r variables.
    # For r=12: 288 variables, 512 constraints - overdetermined but may have solutions.
    
    # Let's try a direct approach for rank 12:
    # Use the structure to reduce the problem.
    
    # Actually, let's try a randomized approach:
    # Generate random U, V of size 8x12, then solve for W.
    # The constraint is: for each (i,j), sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] for all k.
    # This is: (U[i,:] * V[j,:])_r * W[:,r] = T[i,j,:] for each (i,j)
    # Where (U[i,:] * V[j,:])_r = U[i,r]*V[j,r]
    
    # So for each (i,j), we have: sum_r (U[i,r]*V[j,r]) * W[k,r] = T[i,j,k]
    # This is a linear system in W for fixed U, V.
    
    # Let's try this approach with random U, V.
    
    rng = np.random.RandomState(42)
    
    for rank in [12, 13]:
        if time.time() > deadline - 30:
            break
        
        for attempt in range(200):
            if time.time() > deadline - 30:
                break
            
            # Generate random U, V
            U_rand = rng.randn(8, rank)
            V_rand = rng.randn(8, rank)
            
            # For each (i,j), solve for W contributions
            # T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            # Let M[i,j,r] = U[i,r]*V[j,r]
            # Then sum_r M[i,j,r]*W[k,r] = T[i,j,k]
            
            # This means: W^T @ M[i,j,:] = T[i,j,:]
            # Or: for each k, sum_r W[k,r]*M[i,j,r] = T[i,j,k]
            
            # So M[i,j,:] @ W[:,k] = T[i,j,k] for all k
            # This is a linear system: M[i,j,:] is a row vector of length rank
            # We need W such that for all i,j,k: M[i,j,r]*W[k,r] summed over r = T[i,j,k]
            
            # Let's set up the system differently.
            # For fixed U, V, the constraint on W is:
            # For each k: sum_r W[k,r] * (sum over i,j of T[i,j,k] * M[i,j,r] / norm) ...
            
            # Actually, let's just solve the full linear system.
            # Variables: W[0..7, 0..rank-1] = 8*rank unknowns
            # Constraints: for each (i,j,k): sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
            
            # But this is not linear in W if we fix U,V... wait, it IS linear in W for fixed U,V.
            # sum_r (U[i,r]*V[j,r]) * W[k,r] = T[i,j,k]
            # Let A[i,j,k,r] = U[i,r]*V[j,r]
            # Then sum_r A[i,j,k,r]*W[k,r] = T[i,j,k]
            
            # For each k, the equation involves only W[k,:].
            # So for each k: sum_r (sum over i,j of A[i,j,k,r] * T[i,j,k]) ... no.
            
            # Let me re-index. For fixed k, the constraint is:
            # For all i,j: sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
            # This is: (U * V^T)_{i,j} weighted by W[k,r]...
            
            # Actually: Let B_k[r] = W[k,r]. Then:
            # sum_r B_k[r] * U[i,r] * V[j,r] = T[i,j,k]
            # This is: (U * diag(B_k) * V^T)[i,j] = T[i,j,k]
            # Or: U * diag(B_k) * V^T = T[:, :, k]
            
            # So for each k, we need: U * diag(B_k) * V^T = T[:,:,k]
            # This is a linear system in B_k (which is W[k,:]).
            
            # Let's solve: U @ np.diag(B_k) @ V.T = T[:,:,k]
            # => (U * B_k[:,None]) @ V.T = T[:,:,k]
            # => sum_r B_k[r] * (U[:,r] @ V[:,r].T) = T[:,:,k]
            
            # So: sum_r B_k[r] * outer(U[:,r], V[:,r]) = T[:,:,k]
            # This is a linear system in B_k with 8*8=64 equations and rank unknowns.
            
            T = np.zeros((8, 8, 8))
            # Fill in the tensor
            # i = a*n + b, j = b2*m + c, k = c2*p + a2
            for a in range(n):
                for b in range(m):
                    for c in range(p):
                        i = a * n + b
                        j = b * m + c
                        k = c * n + a
                        T[i, j, k] = 1.0
            
            # For each k, solve for B_k = W[k,:]
            W_sol = np.zeros((8, rank))
            valid = True
            for k in range(8):
                # Build the linear system: sum_r B_k[r] * outer(U[:,r], V[:,r]) = T[:,:,k]
                # Vectorize: for each (i,j), sum_r B_k[r] * U[i,r]*V[j,r] = T[i,j,k]
                # A[i,j,r] = U[i,r]*V[j,r]
                A = np.einsum('ir,jr->ijr', U_rand, V_rand)  # shape (8,8,rank)
                rhs = T[:, :, k].flatten()  # shape (64,)
                A_flat = A.reshape(64, rank)  # shape (64, rank)
                
                # Solve A_flat @ B_k = rhs
                if rank <= 64:
                    try:
                        B_k, residuals, rank_sv, sv = np.linalg.lstsq(A_flat, rhs, rcond=None)
                        if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-6:
                            valid = False
                            break
                        W_sol[k, :] = B_k
                    except:
                        valid = False
                        break
                else:
                    valid = False
                    break
            
            if valid:
                # Verify the full decomposition
                U_list = [[float(x) for x in row] for row in U_rand]
                V_list = [[float(x) for x in row] for row in V_rand]
                W_list = [[float(x) for x in row] for row in W_sol]
                
                if _verify_fast(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    break
        
        if best_rank == rank:
            break
    
    if time.time() > deadline - 10:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
