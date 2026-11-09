"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    def schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    # Try to find better decompositions
    # The tensor T[i,j,k] = delta where C[i,k] = sum_j A[i,j]*B[j,k]
    # We need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)
    # where delta = 1 if the indices form a valid multiplication term, else 0
    
    # Actually, the constraint is: for each (i,k), C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition requires: sum_c U[i*m+j,c] * V[j*p+k,c] = 0 for all (i,j,k) not in support
    # and sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for (i,j,k) in support
    
    # Wait, I need to be more careful. The matrix multiplication tensor has entries:
    # T[i,j,k] = 1 if i is row of A, j is col of A / row of B, k is col of B
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let me verify: C[i,k] = sum_j A[i,j] * B[j,k]
    # In the tensor decomposition, we need:
    # For each (i,j,k): the coefficient of A[i,j]*B[j,k] in C[i,k] is 1
    # So T[i,j,k] = 1 for all valid (i,j,k), and the "output" index is (i,k)
    
    # The standard formulation: we have a 3-tensor T with indices (i,j,k) where
    # i in [0,n), j in [0,m), k in [0,p), and T[i,j,k] = 1
    # We decompose as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But this must equal 1 for ALL (i,j,k) since every term contributes.
    
    # Wait no. The tensor for matrix multiplication is:
    # M[i,j,k] = 1 for all i,j,k (it's the "all ones" tensor in this indexing)
    # Because C[i,k] = sum_j A[i,j] B[j,k], so every (i,j,k) triple has coefficient 1.
    
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a bilinear/trilinear constraint. Let me set up the problem.
    
    # For a given rank r, we need to find U (9 x r), V (21 x r), W (21 x r) such that
    # for all i in [0,3), j in [0,3), k in [0,7):
    #   sum_c U[i*3+j, c] * V[j*7+k, c] * W[k*3+i, c] = 1
    
    # This is a system of 3*3*7 = 63 equations.
    
    # Let me try a randomized approach: fix U and V randomly, then solve for W,
    # or use iterative methods.
    
    # Actually, let me try a simpler approach: use the structure.
    # For fixed (i,k), the equation is:
    #   sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    #   W[k*n+i,c] * sum_j sum_c' ... 
    
    # This is getting complex. Let me try a different approach.
    
    # Approach: Use the fact that we can write the multiplication as a sum of
    # rank-1 tensors. A rank-1 tensor is U_col * V_col * W_col (outer product).
    # We need sum_c (U[:,c] ⊗ V[:,c] ⊗ W[:,c])[i,j,k] = 1 for all i,j,k.
    
    # Let me try a greedy/randomized search.
    
    # First, let me try small ranks and see if they're feasible.
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    
    # Let me try to use a linear programming / optimization approach.
    # For fixed U and V, the constraints on W are linear (if we fix U*V products).
    
    # Actually, for fixed U and V, define M[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c]
    # Then we need W[k*n+i,c] such that sum_c M[i,j,k] * W[k*n+i,c] = 1 for all i,j,k.
    # But M depends on both U and V, so it's not linear in W alone.
    
    # Let me try: fix U, then the problem becomes finding V and W.
    # For fixed U, define for each c: u_c = U[:,c] (length 9)
    # We need sum_c u_c[i*m+j] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is still bilinear in V and W.
    
    # Let me try a different strategy: use known results or structured decompositions.
    
    # For 3x3 x 3x3, the best known rank is... I think it's around 19-20.
    # For 3x3 x 3x7, it should be higher.
    
    # Let me try a block decomposition:
    # Split B (3x7) into B1 (3x3) and B2 (3x4)
    # C = A*B1 + A*B2
    # A*B1: 3x3 x 3x3, rank ≤ r333
    # A*B2: 3x3 x 3x4, rank ≤ r334
    # Total rank ≤ r333 + r334
    
    # For 3x3 x 3x3: schoolbook is 27, Strassen-like can do better.
    # For 3x3 x 3x4: schoolbook is 36.
    
    # Let me try to find good decompositions for these sub-problems.
    
    # Actually, let me just try a numerical search with a time limit.
    
    def try_rank(r, max_iters=5000, seed=42):
        """Try to find a decomposition of rank r using random initialization and local search."""
        if time.time() > deadline:
            return None
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W randomly
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # The constraint: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # Let's define the residual and try to minimize it.
        
        def compute_residual(U, V, W):
            res = np.zeros((n, m, p))
            for c in range(r):
                uc = U[:, c]
                vc = V[:, c]
                wc = W[:, c]
                # uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res[i, j, k] += uc[i * m + j] * vc[j * p + k] * wc[k * n + i]
            return res - 1.0
        
        def residual_norm(U, V, W):
            res = compute_residual(U, V, W)
            return np.linalg.norm(res)
        
        # Try gradient-free local search
        best_norm = residual_norm(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            if best_norm < 1e-10:
                # Found a solution!
                # Round to nearest rational/half-integer
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                # Verify
                res = compute_residual(U_round, V_round, W_round)
                if np.linalg.norm(res) < 1e-8:
                    return U_round.tolist(), V_round.tolist(), W_round.tolist()
                # If rounding doesn't work, try without rounding
                return U.tolist(), V.tolist(), W.tolist()
            
            # Pick a random matrix and coordinate to perturb
            mat = rng.randint(0, 3)  # 0=U, 1=V, 2=W
            if mat == 0:
                row = rng.randint(0, n * m)
                col = rng.randint(0, r)
                step = rng.randn() * 0.1
                U[row, col] += step
                new_norm = residual_norm(U, V, W)
                if new_norm < best_norm:
                    best_norm = new_norm
                else:
                    U[row, col] -= step
            elif mat == 1:
                row = rng.randint(0, m * p)
                col = rng.randint(0, r)
                step = rng.randn() * 0.1
                V[row, col] += step
                new_norm = residual_norm(V, U, W) if False else residual_norm(U, V, W)
                if new_norm < best_norm:
                    best_norm = new_norm
                else:
                    V[row, col] -= step
            else:
                row = rng.randint(0, p * n)
                col = rng.randint(0, r)
                step = rng.randn() * 0.1
                W[row, col] += step
                new_norm = residual_norm(U, V, W)
                if new_norm < best_norm:
                    best_norm = new_norm
                else:
                    W[row, col] -= step
        
        return None
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    for r in range(lower_bound, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        result = try_rank(r, max_iters=2000, seed=r)
        if result is not None:
            U_list, V_list, W_list = result
            # Convert to proper format
            U_out = [[float(x) for x in row] for row in U_list]
            V_out = [[float(x) for x in row] for row in V_list]
            W_out = [[float(x) for x in row] for row in W_list]
            # Verify the solution
            # Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U_out[i*m+j][c] * V_out[j*p+k][c] * W_out[k*n+i][c]
                        if abs(s - 1.0) > 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                best_rank = r
                best_result = {"U": U_out, "V": V_out, "W": W_out}
                break  # Found the minimum rank
    
    return best_result
# EVOLVE-BLOCK-END
