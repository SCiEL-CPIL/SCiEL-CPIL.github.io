"""
Strategy: Rank Reduction via Re-Optimization
1. Start with the known rank-39 decomposition.
2. Attempt to reduce rank to 38 by removing one term and re-optimizing the rest.
3. Use Nelder-Mead optimization to minimize the tensor reconstruction error.
4. If a valid rank-38 solution is found (error ~ 0), return it.
5. Otherwise, return the best valid solution found (rank 39).
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = 2, 3, 7
    deadline = time.time() + 140.0  # Safety margin

    # Helper to build the rank-39 solution
    def get_rank_39_solution():
        U = np.zeros((n * m, 39))
        V = np.zeros((m * p, 39))
        W = np.zeros((p * n, 39))
        
        # Strassen blocks: j={0,1}, k in {0,1}, {2,3}, {4,5}
        # We reuse the logic from the current program but encapsulate it
        def add_strassen(i0, j0, k0, offset):
            # M1
            U[i0*m+j0, offset+0] = 1; U[(i0+1)*m+(j0+1), offset+0] = 1
            V[j0*p+k0, offset+0] = 1; V[(j0+1)*p+(k0+1), offset+0] = 1
            W[k0*n+i0, offset+0] = 1; W[(k0+1)*n+(i0+1), offset+0] = 1
            # M2
            U[(i0+1)*m+j0, offset+1] = 1; U[(i0+1)*m+(j0+1), offset+1] = 1
            V[j0*p+k0, offset+1] = 1
            W[k0*n+(i0+1), offset+1] = 1
            # M3
            U[i0*m+j0, offset+2] = 1
            V[j0*p+(k0+1), offset+2] = 1; V[(j0+1)*p+(k0+1), offset+2] = -1
            W[(k0+1)*n+i0, offset+2] = 1
            # M4
            U[(i0+1)*m+(j0+1), offset+3] = 1
            V[(j0+1)*p+k0, offset+3] = 1; V[j0*p+k0, offset+3] = -1
            W[k0*n+i0, offset+3] = 1; W[k0*n+(i0+1), offset+3] = 1
            # M5
            U[i0*m+j0, offset+4] = 1; U[i0*m+(j0+1), offset+4] = 1
            V[(j0+1)*p+(k0+1), offset+4] = 1
            W[k0*n+i0, offset+4] = -1; W[(k0+1)*n+i0, offset+4] = 1
            # M6
            U[(i0+1)*m+j0, offset+5] = 1; U[i0*m+j0, offset+5] = -1
            V[j0*p+k0, offset+5] = 1; V[j0*p+(k0+1), offset+5] = 1
            W[(k0+1)*n+(i0+1), offset+5] = 1
            # M7
            U[i0*m+(j0+1), offset+6] = 1; U[(i0+1)*m+(j0+1), offset+6] = -1
            V[(j0+1)*p+k0, offset+6] = 1; V[(j0+1)*p+(k0+1), offset+6] = 1
            W[k0*n+i0, offset+6] = 1; W[(k0+1)*n+(i0+1), offset+6] = 1

        offset = 0
        for k0 in [0, 2, 4]:
            add_strassen(0, 0, k0, offset)
            offset += 7
            
        # Schoolbook blocks
        def add_schoolbook(i_list, j_list, k_list, offset):
            for i in i_list:
                for j in j_list:
                    for k in k_list:
                        U[i*m+j, offset] = 1
                        V[j*p+k, offset] = 1
                        W[k*n+i, offset] = 1
                        offset += 1
            return offset

        offset = add_schoolbook([0, 1], [0, 1], [6], offset)
        offset = add_schoolbook([0, 1], [2], [0, 1], offset)
        offset = add_schoolbook([0, 1], [2], [2, 3], offset)
        offset = add_schoolbook([0, 1], [2], [4, 5], offset)
        offset = add_schoolbook([0, 1], [2], [6], offset)
        
        return U, V, W

    # Get the base solution
    U_base, V_base, W_base = get_rank_39_solution()
    
    # Verify base solution
    def verify(U, V, W):
        # Efficient check using einsum
        # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        # We want this to be 1 for all i,j,k
        # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # Sum over rank
        # Result[i, j, k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # This is a contraction of 3 tensors.
        # We can do it in steps.
        # Temp[i, k, c] = sum_j U_r[i,j,c] * V_r[j,k,c] ? No, V_r is j,k,c.
        # Let's just loop over i, j, k for verification since n,m,p are small.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-5:
                        return False
        return True

    if not verify(U_base, V_base, W_base):
        # Fallback to schoolbook if base is broken (should not happen)
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        for c, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = 39

    # Strategy: Try to reduce rank to 38
    # We will try removing each of the 39 terms and re-optimizing the rest.
    # To make it faster, we only try a subset of removals or use a smart heuristic.
    # Actually, let's try random perturbations of the rank-39 solution to see if we can find a rank-38 solution directly.
    # But rank is discrete. We can't just perturb.
    # We must find a set of 38 vectors that span the tensor.
    
    # Alternative: Use scipy.optimize to find a rank-38 solution.
    # Variables: U (42*38), V (21*38), W (14*38). Total 2436 variables.
    # Objective: sum_{i,j,k} (1 - sum_c U[i,j,c]V[j,k,c]W[k,i,c])^2
    # This is a non-convex optimization. Nelder-Mead might be slow.
    
    # Let's try a simpler heuristic: 
    # 1. Identify "redundant" terms in the rank-39 solution.
    # 2. In the schoolbook parts, terms are independent. 
    # 3. In the Strassen parts, terms are coupled.
    # 4. Maybe we can merge two schoolbook terms into one? No, they are rank 1.
    
    # Let's try to optimize the rank-39 solution to see if it can be "compressed" 
    # by allowing entries to be non-integer, then snapping.
    # But we want to reduce RANK.
    
    # Let's try a random search for rank 38.
    # We fix U and V to be sparse random matrices with entries in {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}.
    # Then we solve for W? No, it's not linear in W if U,V are fixed? 
    # sum_c U_c V_c^T W_c^T = T.
    # For each (i,j,k), we have a linear equation in W? 
    # No, W[k*n+i, c] is multiplied by U[i*m+j, c] and V[j*p+k, c].
    # If U and V are fixed, the system is linear in W!
    # Let A_{(i,j,k), (k',i',c)} = U[i*m+j, c] * V[j*p+k, c] if (k',i') == (k,i) else 0?
    # No, the term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # This is linear in W for fixed U, V.
    # The coefficient of W[k*n+i, c] in the equation for (i,j,k) is U[i*m+j, c] * V[j*p+k, c].
    # So we have a system of 42 equations (for each i,j,k) with 14*38 = 532 unknowns (W entries).
    # Wait, 42 equations and 532 unknowns? Underdetermined.
    # We can find a W that satisfies the equations exactly if the system is consistent.
    # Since we want an exact solution, we can:
    # 1. Randomly generate U and V (rank 38) with sparse integer/half-integer entries.
    # 2. Construct the linear system for W.
    # 3. Solve for W using least squares (or exact if possible).
    # 4. Check if W has integer/half-integer entries (or close).
    # 5. If yes, verify.
    
    # This is a viable strategy!
    # Let's implement this.
    
    def random_sparse_matrix(size, rank, density=0.2, scale=1):
        M = np.zeros((size, rank))
        for c in range(rank):
            # Pick a few non-zeros
            num_nz = np.random.randint(1, 4)
            indices = np.random.choice(size, num_nz, replace=False)
            for idx in indices:
                M[idx, c] = np.random.choice([-1, -0.5, 0.5, 1])
        return M

    # System for W:
    # For each (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    # Let's group by c? No, W is the variable.
    # The unknowns are W[k*n+i, c] for all k,i,c.
    # Let's vectorize W into a vector w of length p*n*rank.
    # w[k*n*rank + i*rank + c] ? No, W is (p*n, rank). 
    # Let's flatten W to w.
    # The coefficient of w[k*n*rank + i*rank + c] in eq (i,j,k) is U[i*m+j, c] * V[j*p+k, c].
    
    # Let's build the matrix A for the linear system A @ w = b.
    # b is a vector of 1s of length n*m*p.
    # A has shape (n*m*p, p*n*rank).
    
    def try_random_rank_38():
        if time.time() > deadline:
            return None
        
        rank = 38
        U = random_sparse_matrix(n*m, rank)
        V = random_sparse_matrix(m*p, rank)
        
        # Build A
        # A[eq_idx, var_idx]
        # eq_idx = i*m*p + j*p + k
        # var_idx = (k*n + i)*rank + c  (since W is (p*n, rank), row k*n+i, col c)
        
        A = np.zeros((n*m*p, p*n*rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i*m*p + j*p + k
                    for c in range(rank):
                        coeff = U[i*m+j, c] * V[j*p+k, c]
                        if coeff != 0:
                            var_idx = (k*n + i)*rank + c
                            A[eq_idx, var_idx] = coeff
        
        b = np.ones(n*m*p)
        
        # Solve A @ w = b
        # Use least squares
        try:
            w, residuals, rank_lu, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        
        # Check if the solution is "good" (residuals close to 0)
        if residuals is not None and residuals[0] > 1e-6:
            return None
            
        # Reshape w back to W
        W = w.reshape(p*n, rank)
        
        # Check if W entries are close to half-integers
        # And check if U, V are already half-integers (they are by construction)
        # Snap W to nearest half-integer
        W_snapped = np.round(W * 2) / 2
        
        # Verify with snapped values
        if verify(U, V, W_snapped):
            return U, V, W_snapped
        
        # If snapping failed, maybe the original W was valid but not snapped?
        # The problem requires integer/half-integer entries.
        # If W is not close to half-integers, it's invalid.
        if np.allclose(W, W_snapped, atol=1e-6):
             return U, V, W_snapped
             
        return None

    # Try a few random restarts
    for _ in range(20):
        if time.time() > deadline:
            break
        res = try_random_rank_38()
        if res:
            best_U, best_V, best_W = res
            best_rank = 38
            break

    # If we didn't find rank 38, we stick with 39.
    # We could also try to optimize the rank-39 solution to see if we can find a "tighter" 
    # representation that might allow for a different structure, but rank 39 is likely optimal 
    # or very close.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
