import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # 1. Build the target tensor T of shape (n*m, m*p, p*n)
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T."""
        U_f = np.array(U, dtype=float)
        V_f = np.array(V, dtype=float)
        W_f = np.array(W, dtype=float)
        T_approx = np.einsum('ic,jc,kc->ijk', U_f, V_f, W_f)
        return np.allclose(T_approx, T, atol=1e-9)

    def schoolbook():
        """Fallback schoolbook decomposition (rank 18)."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    # Strategy: Structured Block Decomposition
    # We model the 2x3 * 3x3 multiplication as:
    # A = [A11 A12 A13]   B = [B11 B12 B13]
    #     [A21 A22 A23]       [B21 B22 B23]
    #     [A31 A32 A33]       [B31 B32 B33]
    # Wait, A is 2x3, B is 3x3.
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    
    # Output C is 2x3.
    # C[i][k] = sum_j A[i][j] B[j][k]
    
    # The tensor is sparse. We can try to find a low-rank decomposition by 
    # fixing the structure based on Strassen for the 2x2 sub-blocks and 
    # then solving for the rest.
    
    # However, a more robust "structured" approach for small dimensions is 
    # to use the known minimal rank if it's small enough to enumerate, 
    # or use a specific algebraic construction.
    
    # For 2x3 * 3x3, the known minimal rank is 10.
    # Let's try to construct it explicitly.
    
    # Construction Idea:
    # The multiplication can be viewed as 3 separate 2x1 * 1x3 multiplications? No.
    # It is sum over j of A[:,j] * B[j,:].
    # Each term is an outer product of a 2x1 vector and 1x3 vector.
    # This is rank 1 for each term, but we have 3 terms (j=0,1,2).
    # Wait, the tensor decomposition is for the *entire* multiplication tensor.
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # T is 1 at (i*m+j, j*p+k, k*n+i).
    
    # Let's try a rank-10 construction based on literature or derived logic.
    # Since we don't have the explicit formulas hardcoded in the seed, 
    # and random search failed, we will try a **Systematic Enumeration** 
    # of the "support" of the decomposition.
    
    # If the rank is 10, we have 10 "terms".
    # Each term is defined by a set of indices (u_idx, v_idx, w_idx) and coefficients.
    # Actually, the coefficients are not fixed to 1.
    
    # Alternative: Use the fact that the tensor is "small" and try to find 
    # a decomposition with integer coefficients in {-1, 0, 1} or {-2, -1, 0, 1, 2}.
    # We can formulate this as a constraint satisfaction problem.
    
    # Given the time limit and the failure of SA, we will try a **Greedy Heuristic** 
    # that tries to "match" the non-zero entries of T using rank-1 tensors.
    
    def greedy_decomposition(target_T, max_rank=12, coeff_range=[-2, -1, 0, 1, 2]):
        """
        Greedily subtract rank-1 tensors from the target tensor.
        """
        residual = target_T.copy()
        U = []
        V = []
        W = []
        
        for r in range(max_rank):
            if np.allclose(residual, 0, atol=1e-9):
                break
            
            # Find the best rank-1 tensor to subtract
            # We approximate the best rank-1 approximation of the residual.
            # For small tensors, we can try all pairs of vectors?
            # No, that's expensive.
            
            # Instead, let's look for a "corner" or a specific pattern.
            # A simpler heuristic: Find the index (i,j,k) with the largest absolute value in residual.
            # Try to construct a rank-1 tensor that matches this value and minimizes the increase in error elsewhere.
            
            # Let's try a more structured approach:
            # The tensor T has 18 non-zeros.
            # We want to cover them with 10 rank-1 tensors.
            
            # Let's try to find a decomposition by solving a linear system for the coefficients
            # given a fixed support (indices of non-zero entries in U, V, W).
            
            # This is complex. Let's stick to a simpler but potentially effective method:
            # **Local Search with Better Initialization**.
            
            # Initialize U, V, W using the schoolbook solution but with "merged" columns.
            # If we merge columns in schoolbook, we get a lower rank approximation.
            # We can then refine.
            
            # Actually, let's try to hardcode a known rank-10 decomposition if we can derive it.
            # For 2x2 * 2x2, Strassen is rank 7.
            # For 2x3 * 3x3, we can split the 3x3 matrix into a 2x2 block and a 2x1/1x2 block?
            # No, the dimensions are 2x3 and 3x3.
            
            # Let's try to use the "border rank" or "approximate" rank concepts?
            # No, it must be exact.
            
            # Given the constraints, I will implement a **Fast Greedy Matching** algorithm.
            # 1. Start with residual = T.
            # 2. In each step, find a rank-1 tensor R = u v^T w^T (flattened) that best fits the residual.
            #    To speed this up, we restrict u, v, w to have sparsity.
            
            # This is still hard to implement efficiently in Python for 120s.
            
            # Let's try a different angle: **Randomized Restart with Smarter Moves**.
            # The previous SA failed because it moved one variable at a time.
            # Let's try moving a whole row of U, V, or W at once.
            
            return None # Placeholder, we will implement a better search below

    # Improved Search: **Simulated Annealing with Row-wise Moves**
    def try_sa_row_moves(rank, max_time=30):
        if time.time() > deadline - 10:
            return None
            
        U = np.random.randint(-2, 3, size=(n*m, rank)) / 2.0
        V = np.random.randint(-2, 3, size=(m*p, rank)) / 2.0
        W = np.random.randint(-2, 3, size=(p*n, rank)) / 2.0
        
        def calc_error(U, V, W):
            T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
            return np.sum((T - T_approx)**2)
        
        current_error = calc_error(U, V, W)
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        T_start = 1.0
        T_end = 1e-6
        alpha = 0.995
        max_iter = 50000
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            T_curr = T_start * (alpha ** it)
            
            # Move: Pick a matrix and a row, perturb the whole row
            choice = np.random.randint(0, 3)
            old_val = None
            if choice == 0:
                i = np.random.randint(n*m)
                old_val = U[i, :].copy()
                U[i, :] += np.random.choice([-0.5, 0, 0.5], size=rank)
            elif choice == 1:
                j = np.random.randint(m*p)
                old_val = V[j, :].copy()
                V[j, :] += np.random.choice([-0.5, 0, 0.5], size=rank)
            else:
                k = np.random.randint(p*n)
                old_val = W[k, :].copy()
                W[k, :] += np.random.choice([-0.5, 0, 0.5], size=rank)
            
            new_error = calc_error(U, V, W)
            delta_e = new_error - current_error
            
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T_curr):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error < 1e-9:
                        return best_U, best_V, best_W
            else:
                if choice == 0:
                    U[i, :] = old_val
                elif choice == 1:
                    V[j, :] = old_val
                else:
                    W[k, :] = old_val
                    
        if best_error < 1e-9:
            return best_U, best_V, best_W
        return None

    found = False
    # Try rank 10
    for restart in range(3):
        if time.time() > deadline - 15:
            break
        np.random.seed(123 + restart)
        result = try_sa_row_moves(10)
        if result:
            U, V, W = result
            if verify_decomposition(U, V, W):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = 10
                found = True
                break

    if not found and time.time() < deadline - 20:
        # Try rank 11
        for restart in range(2):
            if time.time() > deadline - 10:
                break
            np.random.seed(456 + restart)
            result = try_sa_row_moves(11)
            if result:
                U, V, W = result
                if verify_decomposition(U, V, W):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = 11
                    found = True
                    break

    return best_result
# EVOLVE-BLOCK-END
