"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find lower-rank decomposition
    # The matmul tensor T[(i,j),(j',k),(i',k')] = 1 iff j=j', i=i', k=k'
    # Rank-r decomposition: sum_c U[(i,j),c] * V[(j',k),c] * W[(i',k'),c] = T[...]
    
    # Key constraint: For each (i,k), and for each c, define:
    #   f_{i,k,c} = sum_j U[(i,j),c] * V[(j,k),c]
    # Then we need: sum_c f_{i,k,c} * W[(i,k),c] = 1 for all i,k
    # And: for any "inconsistent" triple (i,j),(j',k),(i',k') where not (j=j' and i=i' and k=k'),
    #      sum_c U[(i,j),c] * V[(j',k),c] * W[(i',k'),c] = 0
    
    # This is complex. Let's try a different approach:
    # Use the fact that the problem is small and try to find exact decompositions
    # by solving a system of equations.
    
    # For a given rank r, we need to find U (nm x r), V (mp x r), W (pn x r)
    # such that the tensor equation holds.
    
    # The number of constraints: 
    # - For each (i,j,j',k,i',k') with j!=j' or i!=i' or k!=k': one zero constraint
    # - For each (i,j,k): one one-constraint
    # Total: n*m*p one-constraints + (n*m*m*p*p*n - n*m*p) zero-constraints... that's huge.
    
    # Better approach: The tensor has a special structure. Let's exploit it.
    # T[(i,j),(j',k),(i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    # Consider fixing c. Then U[:,c], V[:,c], W[:,c] are vectors.
    # The contribution of term c to T[(i,j),(j',k),(i',k')] is U[(i,j),c]*V[(j',k),c]*W[(i',k'),c]
    
    # For the "diagonal" (j=j', i=i', k=k'): sum_c U[(i,j),c]*V[(j,k),c]*W[(i,k),c] = 1
    # For off-diagonal: sum_c U[(i,j),c]*V[(j',k),c]*W[(i',k'),c] = 0
    
    # This is a system of trilinear equations. Hard to solve directly.
    
    # Alternative: Use the structure that W only depends on (i,k), not on j.
    # Let's define for each c: A_c = U[:,c] reshaped to (n,m), B_c = V[:,c] reshaped to (m,p)
    # Then the c-th term contributes to the "matmul tensor" as: 
    #   (A_c[i,j] * B_c[j,k]) for the (i,k) output, weighted by W[(i,k),c]
    
    # Actually, the c-th term gives a rank-1 contribution to the bilinear map:
    #   (A,B) -> sum_c W[(i,k),c] * (sum_j A_c[i,j]*B_c[j,k])
    
    # Wait, that's not right either. Let me think again.
    
    # The standard formulation: The matmul can be seen as a bilinear map from 
    # (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k].
    # A rank-r bilinear algorithm uses r scalar multiplications:
    #   M_c = sum_{i,j} U[(i,j),c] * A[i,j]  (for each c)
    #   N_c = sum_{j,k} V[(j,k),c] * B[j,k]  (for each c)
    #   C[i,k] = sum_c W[(i,k),c] * M_c * N_c
    #
    # The correctness condition is:
    #   sum_c W[(i,k),c] * (sum_j U[(i,j),c] * A[i,j]) * (sum_j' V[(j',k),c] * B[j',k])
    #   = sum_j A[i,j] * B[j,k]  for all A, B
    #
    # This means: for all i,k, and for all i',j,j',k' (with i'!=i or k'!=k, the cross terms must vanish)
    #   sum_c W[(i,k),c] * U[(i',j),c] * V[(j',k'),c] = delta_{i,i'}*delta_{k,k'}*delta_{j,j'}
    #
    # More precisely, expanding:
    #   sum_c W[(i,k),c] * sum_j U[(i,j),c]*A[i,j] * sum_j' V[(j',k),c]*B[j',k]
    #   = sum_j,j' A[i,j]*B[j',k] * sum_c W[(i,k),c]*U[(i,j),c]*V[(j',k),c]
    #
    # This must equal sum_j A[i,j]*B[j,k] for all A,B.
    # So: sum_c W[(i,k),c]*U[(i,j),c]*V[(j',k),c] = delta_{j,j'}  for all i,k,j,j'
    
    # THAT's the key constraint! For each (i,k,j,j'):
    #   sum_c W[(i,k),c] * U[(i,j),c] * V[(j',k),c] = delta_{j,j'}
    
    # This is a system of quadratic equations in the unknowns U, V, W.
    # Number of unknowns: r*(n*m + m*p + p*n) = r*(6+30+20) = 56r
    # Number of equations: n*p*m*m = 2*10*3*3 = 180
    
    # For r >= 56/180... well, we need enough degrees of freedom.
    # With r=60: 3360 unknowns, 180 equations. Very underdetermined.
    # We want to minimize r.
    
    # Strategy: For a given r, try to find U,V,W satisfying the constraints.
    # The constraints are quadratic, so this is non-convex. But we can try:
    # 1. Fix W randomly, then the constraints become linear in U*V (outer product structure)
    # 2. Or use iterative methods
    
    # Let's try: for each (i,k), define the matrix M_{i,k} where M_{i,k}[j,j'] = delta_{j,j'}
    # We need: sum_c W[(i,k),c] * U[(i,·),c] * V[(·,k),c]^T = I_m  (identity in the j,j' space)
    
    # Let u_{i,c} = U[(i,·),c] (vector of length m)
    # Let v_{k,c} = V[(·,k),c] (vector of length m)
    # Let w_{i,k,c} = W[(i,k),c] (scalar)
    
    # Constraint: sum_c w_{i,k,c} * u_{i,c} * v_{k,c}^T = I_m  for all i,k
    
    # This is: sum_c w_{i,k,c} * (u_{i,c} ⊗ v_{k,c}) = I_m
    
    # So I_m is expressed as a sum of r rank-1 matrices, each weighted by w_{i,k,c}.
    
    # For each (i,k), we need to decompose I_m as a sum of r rank-1 terms.
    # But the u_{i,c} and v_{k,c} are shared across different (i,k) pairs!
    # u_{i,c} is shared across all k for fixed i,c
    # v_{k,c} is shared across all i for fixed k,c
    
    # This is the key coupling that makes it hard.
    
    # Let's try a constructive approach for small m=3:
    # I_3 is 3x3 identity. We want to write it as sum_c w_{i,k,c} * u_{i,c} * v_{k,c}^T
    # 
    # If we could choose u_{i,c} and v_{k,c} independently for each (i,k), we'd need rank(I_3)=3 terms.
    # But they're coupled.
    
    # Let's try: can we find u_{i,c}, v_{k,c}, w_{i,k,c} for r terms?
    # 
    # Simple idea: Let u_{i,c} depend only on c (not i), and v_{k,c} depend only on c (not k).
    # Then: sum_c w_{i,k,c} * u_c * v_c^T = I_3 for all i,k.
    # This requires w_{i,k,c} to be the same for all i,k, say w_c.
    # Then sum_c w_c * u_c * v_c^T = I_3, which is just a rank decomposition of I_3.
    # But then the bilinear algorithm would be:
    #   M_c = sum_{i,j} u_c[j] * A[i,j]  -- this sums over ALL i,j, not just row i!
    # That's wrong. U[(i,j),c] should depend on i.
    
    # So u_{i,c} must depend on i. Let's try:
    # u_{i,c} = e_i * alpha_c  (where e_i is the i-th standard basis vector in R^n... wait, u is length m)
    
    # Hmm, u_{i,c} is a vector of length m (the inner dimension). Let me re-index.
    # u_{i,c}[j] = U[(i,j),c] for j=0..m-1
    # v_{k,c}[j'] = V[(j',k),c] for j'=0..m-1
    
    # Constraint: sum_c w_{i,k,c} * u_{i,c} * v_{k,c}^T = I_m  (m x m identity)
    
    # For m=3, I_3 has rank 3. So we need at least 3 terms in the sum for each (i,k).
    # But the terms are coupled across (i,k).
    
    # Let's try: u_{i,c} = e_{c mod m} * s_{i,c} and v_{k,c} = e_{c mod m} * t_{k,c}
    # where e_j is the j-th standard basis vector in R^m.
    # Then u_{i,c} * v_{k,c}^T = s_{i,c} * t_{k,c} * e_{c mod m} * e_{c mod m}^T
    # Sum_c w_{i,k,c} * s_{i,c} * t_{k,c} * e_{c mod m} e_{c mod m}^T = I_3
    # This gives: for each j, sum_{c: c mod m = j} w_{i,k,c} * s_{i,c} * t_{k,c} = 1, and 0 for off-diagonal.
    # Off-diagonal is automatically 0 since e_j e_j^T has 0 off-diagonal.
    # So we need: for each i,k,j: sum_{c: c mod m = j} w_{i,k,c} * s_{i,c} * t_{k,c} = 1
    
    # If we set s_{i,c} = 1/sqrt(n) and t_{k,c} = 1/sqrt(p) and w_{i,k,c} = sqrt(np)/... 
    # This is getting complicated. Let me just try to solve it numerically.
    
    # Given the complexity and time constraint, let me try a practical search:
    # Start with schoolbook (rank 60) and try to find lower rank using random restarts
    # and the constraint equations.
    
    # Actually, let me try a simpler approach: use the known Strassen-like ideas.
    # For 2x3x10, we can think of it as 5 blocks of 2x3x2 (since 10=5*2).
    # For each 2x3x2 block, what's the minimal rank?
    # 2x2x2 matmul has rank 7 (Strassen). But 2x3x2 is different.
    
    # For 2x3x2: C[i,k] = sum_j A[i,j]*B[j,k] where i in {0,1}, j in {0,1,2}, k in {0,1}
    # This is a 6x6x4 tensor. Schoolbook rank = 12.
    # Can we do better? 
    
    # Let me just try to find a decomposition computationally for the full problem.
    
    def try_rank(r, time_limit=10):
        """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
        start = time.time()
        np.random.seed(42)
        
        # Initialize
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # We need: for all i,k,j,j': sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
        
        # This is a system of quadratic equations. Let's try to solve it iteratively.
        # Fix two of the three matrices, solve for the third.
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(500):
            if time.time() - start > time_limit:
                break
            
            # Update W: for each (i,k), W[k*n+i, :] should satisfy
            # sum_c W[k*n+i, c] * (U[i*m+0:(i+1)*m, c] * V[0:m*p:m, c]... 
            # Actually: for fixed (i,k), define for each c:
            #   s_c = U[i*m:(i+1)*m, c] * V[0:m*p:m, c]... no
            #   For each j,j': U[i*m+j, c] * V[j'*p+k, c]
            #   We need: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] = delta_{j,j'}
            #   
            # Let's define for fixed (i,k): M_{i,k} is an m x m matrix where
            # M_{i,k}[j,j',c] = U[i*m+j, c] * V[j'*p+k, c]
            # We need: W[k*n+i, :] @ M_{i,k} = I_m  (as a linear system)
            # But M_{i,k} is m x m x r, and W[k*n+i, :] is 1 x r.
            # So: sum_c W_c * M_{i,k}[:,:,c] = I_m
            # This is: W[k*n+i, :] @ M_{i,k}[:,:,0] + W[k*n+i, :] @ M_{i,k]... 
            # No: sum_c W_c * (M_{i,k}[:,:,c]) = I_m
            # Where M_{i,k}[:,:,c] is m x m.
            # So we have m^2 scalar equations for r unknowns (W[k*n+i, 0..r-1]).
            # If r >= m^2, we can solve this linear system!
            
            # For each (i,k), solve: find W_vec such that sum_c W_vec[c] * M_{i,k}[:,:,c] = I_m
            # This is a linear system: A_{i,k} @ W_vec = vec(I_m)
            # where A_{i,k} is m^2 x r
            
            for i in range(n):
                for k in range(p):
                    # Build A_{i,k}: m^2 x r
                    A_ik = np.zeros((m*m, r))
                    for j in range(m):
                        for jp in range(m):
                            # A_ik[j*m+jp, c] = U[i*m+j, c] * V[jp*p+k, c]
                            A_ik[j*m+jp, :] = U[i*m+j, :] * V[jp*p+k, :]
                    
                    # Target: vec(I_m)
                    target = np.zeros(m*m)
                    for j in range(m):
                        target[j*m+j] = 1.0
                    
                    # Solve A_ik @ W_vec = target (least squares if r < m^2)
                    if r >= m*m:
                        W_vec, _, _, _ = np.linalg.lstsq(A_ik, target, rcond=None)
                    else:
                        W_vec, _, _, _ = np.linalg.lstsq(A_ik, target, rcond=None)
                    
                    W[k*n+i, :] = W_vec
            
            # Update U: for each (i,j), U[i*m+j, :] should satisfy...
            # The constraint is: sum_c W[k*n+i, c] * U[i*m+j, c] * V[jp*p+k, c] = delta_{j,jp}
            # For fixed (i,j), this must hold for all k and jp.
            # Let's fix (i,j) and consider all (k,jp) pairs.
            # For each (k,jp): sum_c (W[k*n+i, c] * V[jp*p+k, c]) * U[i*m+j, c] = delta_{j,jp}
            # Let B_{i,j}[k,jp, c] = W[k*n+i, c] * V[jp*p+k, c]
            # We need: U[i*m+j, :] @ B_{i,j}[:,:,c] summed over c... 
            # sum_c U_c * B_{i,j}[k,jp,c] = delta_{j,jp} for all k,jp
            # This is: U[i*m+j, :] @ B_{i,j} = vec(delta_{j,jp} for all k,jp)
            # B_{i,j} is (p*m) x r
            # Target: for each (k,jp), delta_{j,jp}
            
            for i in range(n):
                for j in range(m):
                    B_ij = np.zeros((p*m, r))
                    for k in range(p):
                        for jp in range(m):
                            B_ij[k*m+jp, :] = W[k*n+i, :] * V[jp*p+k, :]
                    
                    target = np.zeros(p*m)
                    for k in range(p):
                        for jp in range(m):
                            if jp == j:
                                target[k*m+jp] = 1.0
                    
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(B_ij, target, rcond=None)
            
            # Update V: similar
            for k in range(p):
                for jp in range(m):  # jp is the inner index for V
                    C_kj = np.zeros((n*m, r))
                    for i in range(n):
                        for j in range(m):
                            C_kj[i*m+j, :] = W[k*n+i, :] * U[i*m+j, :]
                    
                    target = np.zeros(n*m)
                    for i in range(n):
                        for j in range(m):
                            if j == jp:
                                target[i*m+j] = 1.0
                    
                    V[jp*p+k, :], _, _, _ = np.linalg.lstsq(C_kj, target, rcond=None)
            
            # Check error
            err = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            val = np.sum(W[k*n+i, :] * U[i*m+j, :] * V[jp*p+k, :])
                            target_val = 1.0 if j == jp else 0.0
                            err += (val - target_val) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-8:
            # Convert to integer/half-integer if possible
            U_int = np.round(best_U * 2) / 2
            V_int = np.round(best_V * 2) / 2
            W_int = np.round(best_W * 2) / 2
            
            # Check if the rounded version still works
            err_rounded = 0.0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for jp in range(m):
                            val = np.sum(W_int[k*n+i, :] * U_int[i*m+j, :] * V_int[jp*p+k, :])
                            target_val = 1.0 if j == jp else 0.0
                            err_rounded += (val - target_val) ** 2
            
            if err_rounded < 1e-8:
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
            else:
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
        else:
            return None
    
    # Try ranks from low to high
    best_result = None
    for r in range(30, 61):
        if time.time() > deadline:
            break
        result = try_rank(r, time_limit=min(10, max(1, deadline - time.time() - 5)))
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break  # Found the lowest rank, stop
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END

