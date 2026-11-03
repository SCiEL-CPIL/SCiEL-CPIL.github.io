"""Tensor decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    # Matrix multiplication: A (n x m) * B (m x p) = C (n x p)
    # Tensor T[i, j, k] = 1 if we compute C[i,k] += A[i,j] * B[j,k]
    # In the trace convention: W[k*n + i, r], U[i*m + j, r], V[j*p + k, r]
    
    # The tensor has entries T[(i,j), (j,k)] = 1 for all valid i,j,k
    # We need rank-r decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    
    # Schoolbook rank = n*m*p = 3*3*7 = 63
    # We want to find a lower rank.
    
    # For 3x3 * 3x3, the known minimal rank is 23 (Strassen-like improvements).
    # For 3x3 * 3x7, we need to search.
    
    # Strategy: Try to find a valid decomposition of rank r by solving
    # the system of polynomial equations. We'll use a randomized approach:
    # 1. Start with a known structure (e.g., block diagonal or Strassen-inspired)
    # 2. Try to reduce rank by merging terms
    
    # First, let's try a direct algebraic approach:
    # The tensor T is a (9, 21, 21) tensor.
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a, b, c] = 1 iff a, b, c are consistent (i.e., the j in a matches the j in b, and the k in b matches the k in c)
    
    # Let's build the tensor explicitly
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Also, the tensor rank is at least the max of the matrix ranks of the three slices
    
    # Let's try to use a structured approach:
    # For the "middle" dimension m=3, we can think of this as 3 separate 3x1 * 1x7 multiplications
    # but that's just the schoolbook.
    
    # Better idea: Use the fact that for fixed j, the sub-tensor is:
    # sum over i,k of E[i,j] ⊗ E[j,k] ⊗ E[k,i]
    # This is like an outer product structure.
    
    # Let's try a randomized search for low-rank decompositions:
    # For a given rank r, we want U, V, W such that:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c
    
    # This is a system of (n*m)*(m*p)*(p*n) = 9*21*21 = 3969 equations
    # with r*(n*m + m*p + p*n) = r*(9+21+21) = 51r unknowns
    
    # For r = 21: 51*21 = 1071 unknowns, 3969 equations -> overdetermined
    # For r = 63: 51*63 = 3213 unknowns, 3969 equations -> still overdetermined but close
    
    # Actually, the tensor rank for matmul(3,3,7) is not known to be much below 63.
    # Let's try a different approach: use the known Strassen algorithm structure.
    
    # Strassen for 2x2x2 uses rank 7 instead of 8.
    # For 3x3x3, there are algorithms with rank 23.
    
    # For 3x3 * 3x7, let's try to construct a decomposition.
    
    # Approach: Try to find a valid decomposition by solving a least-squares problem
    # with a sparsity constraint, then verify exactness.
    
    def try_rank(r, max_time=30):
        """Try to find a rank-r decomposition using randomized initialization and least squares."""
        t0 = time.time()
        if time.time() - t0 > max_time:
            return None
        
        # Number of unknowns
        num_unknowns = r * (n * m + m * p + p * n)
        num_eqs = n * m * m * p * p * n
        
        # Flatten the tensor
        T_flat = T.flatten()
        
        # Build the system: for each (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
        # This is nonlinear. Use iterative approach.
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Use alternating least squares:
        # Fix V, W -> solve for U (linear)
        # Fix U, W -> solve for V (linear)
        # Fix U, V -> solve for W (linear)
        
        for iteration in range(100):
            if time.time() - t0 > max_time:
                return None
            
            # Solve for U: for each a, sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
            # This is: U[a,:] @ (V * W)[:, (b,c)] = T[a,b,c]
            # where (V*W) is the element-wise product reshaped
            
            # For fixed b,c: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # Let M[a, r] = U[a,r], and for each (b,c) define v_wc[r] = V[b,r]*W[c,r]
            # Then sum_r U[a,r] * v_wc[r] = T[a,b,c]
            
            # Build the system for U:
            # For all (a, b, c): sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # Group by a: for each a, we have (m*p)*(p*n) equations in r unknowns U[a,0..r-1]
            
            # Actually, this is a linear system. Let's build it properly.
            # For each a, the equations are:
            # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
            # This is (m*p)*(p*n) equations in r unknowns.
            
            # Build the design matrix for each a
            # D_a is a ((m*p)*(p*n)) x r matrix where D_a[(b,c), r] = V[b,r] * W[c,r]
            # Then U[a,:] @ D_a = T[a,:,:].flatten()
            
            D = np.zeros((n * m, m * p * p * n, r))
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        D[a, b * p * n + c, :] = V[b, :] * W[c, :]
            
            # Solve least squares for each a
            for a in range(n * m):
                Da = D[a]
                Ta = T[a].flatten()
                U[a, :], residuals, rank_d, sv = np.linalg.lstsq(Da, Ta, rcond=None)
            
            # Similarly for V
            D = np.zeros((m * p, n * m * p * n, r))
            for b in range(m * p):
                for a in range(n * m):
                    for c in range(p * n):
                        D[b, a * p * n + c, :] = U[a, :] * W[c, :]
            
            for b in range(m * p):
                Db = D[b]
                Tb = T[:, b, :].reshape(-1)
                V[b, :], residuals, rank_d, sv = np.linalg.lstsq(Db, Tb, rcond=None)
            
            # Similarly for W
            D = np.zeros((p * n, n * m * m * p, r))
            for c in range(p * n):
                for a in range(n * m):
                    for b in range(m * p):
                        D[c, a * m * p + b, :] = U[a, :] * V[b, :]
            
            for c in range(p * n):
                Dc = D[c]
                Tc = T[:, :, c].reshape(-1)
                W[c, :], residuals, rank_d, sv = np.linalg.lstsq(Dc, Tc, rcond=None)
            
            # Check convergence
            error = np.max(np.abs(np.einsum('ar,br,cr->abc', U, V, W) - T))
            if error < 1e-10:
                # Verify exactness with rational reconstruction
                return U, V, W
        
        return None
    
    # Try ranks from low to high
    # Start with a reasonable lower bound
    # For matmul(n,m,p), the tensor rank is at least max(n*m, m*p, p*n)
    # but can be higher.
    
    # Let's first return the schoolbook if we can't find better quickly
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition
    best_result = None
    best_rank = schoolbook_rank
    
    # Try a few ranks
    for r in range(max(n*m, m*p, p*n), schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_time=20)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to list of lists with integer/half-integer entries
        # Round to nearest half-integer
        U_list = [[round(U[i][j] * 2) / 2 for j in range(best_rank)] for i in range(n * m)]
        V_list = [[round(V[i][j] * 2) / 2 for j in range(best_rank)] for i in range(m * p)]
        W_list = [[round(W[i][j] * 2) / 2 for j in range(best_rank)] for i in range(p * n)]
        
        # Verify the decomposition with the rounded values
        U_np = np.array(U_list)
        V_np = np.array(V_list)
        W_np = np.array(W_list)
        reconstructed = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        if np.max(np.abs(reconstructed - T)) < 1e-10:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
