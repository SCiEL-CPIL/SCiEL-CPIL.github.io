import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def check_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Target T[a,b,c,d,i,k] = 1 if a==i, b==d, c==k else 0
    """
    r = U.shape[1]
    
    # Full reconstruction check
    T_recon = np.zeros((n, m, m, p, n, p))
    for l in range(r):
        u_l = U[:, l].reshape(n, m)
        v_l = V[:, l].reshape(m, p)
        w_l = W[:, l].reshape(p, n)
        # Tensor term: u_l[a,b] * v_l[c,d] * w_l[k,i]
        term = np.einsum('ab,cd,ki->abcdki', u_l, v_l, w_l)
        T_recon += term
        
    # Target tensor
    T_target = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                T_target[a, b, c, b, a, c] = 1.0
                
    if not np.allclose(T_recon, T_target, atol=1e-9):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130.0 # 20s margin
    
    n, m, p = DIMENSIONS
    
    # Dimensions of factor matrices
    U_dim = n * m  # 6
    V_dim = m * p  # 18
    W_dim = p * n  # 12
    
    # Strategy:
    # 1. Construct a rank 34 solution (current best known in this lineage) as a fallback.
    #    This splits the 3x6 multiplication into two independent 3x3 multiplications.
    #    Rank(3x3) = 7? No, rank(3x3) is 17 for standard, but we found 17 in the previous code.
    #    Wait, the previous code used a rank 17 decomposition for 2x3x3.
    #    Let's re-verify the "splitting" idea.
    #    A 2x3 matrix times a 3x6 matrix.
    #    Split the 3x6 matrix into two 3x3 blocks: B = [B1, B2].
    #    Then A * B = [A * B1, A * B2].
    #    This requires computing A * B1 (2x3 * 3x3) and A * B2 (2x3 * 3x3).
    #    The rank of 2x3 * 3x3 multiplication is 17 (from the previous code's hardcoded solution).
    #    So total rank = 17 + 17 = 34.
    #    
    #    Can we do better?
    #    The tensor for 2x3 * 3x6 is equivalent to two copies of 2x3 * 3x3.
    #    The rank of 2x3 * 3x3 is known to be 17 (Coppersmith-Winograd gives 2.376 for large, but for small it's 17).
    #    Actually, the rank of 2x2 * 2x2 is 7.
    #    The rank of 2x3 * 3x3 is 17.
    #    So 34 is likely the minimum if we just split.
    #    
    #    However, there might be a more efficient decomposition that doesn't strictly split.
    #    But 34 is a strong candidate. The current program returns 34.
    #    The prompt says "current program (score 1.0588)".
    #    Score = seed / value.
    #    If seed is 36 (schoolbook), score 36/34 = 1.0588.
    #    So the current program returns rank 34.
    #    
    #    To improve, we need rank 33 or lower.
    #    Is rank 33 possible?
    #    Lower bound for 2x3 * 3x6:
    #    Rank(2x3 * 3x6) >= Rank(2x2 * 2x2) * something?
    #    Submultiplicativity of rank: R(m1 x n1 * n1 x p1) * R(m2 x n2 * n2 x p2) >= R((m1*m2) x (n1*n2) * (n1*n2) x (p1*p2)).
    #    This doesn't help directly.
    #    
    #    Let's try to search for rank 33.
    #    We will use a random restart approach with a local search (gradient descent or simplex)
    #    on the non-linear equations.
    #    
    #    Equations:
    #    For all a,b,c,d,i,k:
    #    sum_l U_ab,l V_cd,l W_ki,l = delta_ab,cd,ki (where delta is 1 if a=i, b=d, c=k, else 0)
    #    
    #    This is a system of quadratic equations in the entries of U, V, W.
    #    We can fix U and V and solve for W? No, it's trilinear.
    #    We can fix U and solve for V and W? Still non-linear.
    #    
    #    However, we can use the fact that the solution is sparse (many zeros in schoolbook).
    #    We can try to find a sparse solution of rank 33.
    
    def get_rank_34_solution():
        # Reuse the logic from the previous program to ensure we have a valid fallback.
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        # Rank 17 decomposition for 2x3x3
        # (Values from the current program)
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1
        W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1
        W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1
        W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1
        W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1
        W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return U_comb, V_comb, W_comb

    U_fallback, V_fallback, W_fallback = get_rank_34_solution()
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    best_rank = 34
    
    # Attempt to find rank 33
    # We will use a random search with a local refinement step.
    # The local refinement will use gradient descent on the squared error.
    
    def try_find_rank(target_rank, time_budget):
        start = time.time()
        
        # Precompute indices
        non_zero_indices = []
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    non_zero_indices.append((a, b, c, b, a, c))
                    
        zero_indices = []
        all_indices = set(itertools.product(range(n), range(m), range(m), range(p), range(n), range(p)))
        nz_set = set(non_zero_indices)
        zero_indices = list(all_indices - nz_set)
        
        # Sample zeros for faster error computation
        # We will use a fixed subset of zeros to ensure the error landscape is smooth
        # and to speed up the computation.
        # 50 zeros should be enough to capture the main structure.
        zero_sample = zero_indices[:50]
        
        def compute_error(U, V, W):
            err = 0.0
            for (a, b, c, d, i, k) in non_zero_indices:
                idx_u = a * m + b
                idx_v = c * p + d
                idx_w = k * n + i
                s = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                err += (s - 1.0)**2
            
            for (a, b, c, d, i, k) in zero_sample:
                idx_u = a * m + b
                idx_v = c * p + d
                idx_w = k * n + i
                s = np.dot(U[idx_u], V[idx_v] * W[idx_w])
                err += s**2
            return err

        def gradient_step(U, V, W, lr):
            # Compute gradients numerically or analytically?
            # Analytical is better.
            # E = sum (sum_l U_ab,l V_cd,l W_ki,l - target)^2
            # dE/dU_ab,l = 2 * (sum_l ... - target) * V_cd,l W_ki,l
            # This is expensive to compute for all l, a, b, c, d, i, k.
            # We will use a random subset of the equations for the gradient estimate.
            
            # For simplicity and speed, we will use a simple coordinate descent or random perturbation.
            # Given the small dimension, we can try to update one matrix at a time.
            
            # Let's try updating W first, as it has the fewest rows (12).
            # For each column l of W, we can solve for the optimal W[:, l] given U and V.
            # This is a linear least squares problem!
            # 
            # Let T be the target tensor.
            # T_abcdki = sum_l U_ab,l V_cd,l W_ki,l
            # 
            # If we fix U and V, the equation for W is:
            # T_abcdki = sum_l (U_ab,l V_cd,l) W_ki,l
            # 
            # Let M_l be the matrix M_l[ab, cd, ki] = U_ab,l V_cd,l.
            # Then T = sum_l M_l * W_l (where W_l is the vector W[:, l]).
            # 
            # This is not a standard linear system because the indices are mixed.
            # However, we can vectorize it.
            # 
            # Let's reshape U, V, W to 3D arrays.
            # U: (n, m, r)
            # V: (m, p, r)
            # W: (p, n, r)
            # 
            # T: (n, m, m, p, n, p)
            # 
            # T[a, b, c, d, i, k] = sum_l U[a, b, l] V[c, d, l] W[k, i, l]
            # 
            # This is a tensor contraction.
            # 
            # If we fix U and V, we want to find W to minimize the error.
            # This is a linear least squares problem in the entries of W.
            # 
            # Let's construct the matrix A and vector b for the least squares problem.
            # The unknowns are W[k, i, l] for all k, i, l.
            # The equations are T[a, b, c, d, i, k] = sum_l U[a, b, l] V[c, d, l] W[k, i, l]
            # 
            # For a fixed (a, b, c, d, i, k), the equation is:
            # sum_l (U[a, b, l] V[c, d, l]) W[k, i, l] = T[a, b, c, d, i, k]
            # 
            # This is a linear equation in the variables W[k, i, l].
            # 
            # We can set up a large linear system.
            # Number of variables: p * n * r = 12 * r.
            # Number of equations: n * m * m * p * n * p = 1296.
            # 
            # This is solvable with numpy.linalg.lstsq.
            
            U_3d = U.reshape(n, m, -1)
            V_3d = V.reshape(m, p, -1)
            
            # Construct A and b
            # Variables: W[k, i, l] -> index = (k * n + i) * r + l
            # Equations: (a, b, c, d, i, k) -> index = ((a * m + b) * m + c) * p * n * p + (d * n + i) * p + k
            
            num_vars = p * n * r
            num_eqs = n * m * m * p * n * p
            
            # We will solve for W given U and V.
            # A is a sparse matrix, but we can use dense for small sizes.
            # 1296 x 12r. For r=33, 1296 x 396. This is fine.
            
            A = np.zeros((num_eqs, num_vars))
            b = np.zeros(num_eqs)
            
            # Fill A and b
            # This loop is slow in Python. We need to vectorize or use a smarter approach.
            # 
            # Alternative: Use the fact that the tensor is sparse.
            # Only 18 non-zero entries in T.
            # We can solve the system using only the non-zero entries and a few zero entries.
            # 
            # Let's use the non-zero entries (18) and 50 zero entries (50) to form the system.
            # Total 68 equations.
            # Variables: 12 * r.
            # 
            # This is an underdetermined system (68 < 12r for r>5).
            # We want the minimum norm solution.
            # 
            # Let's try this.
            
            eqs = non_zero_indices + zero_sample
            num_active_eqs = len(eqs)
            
            A_active = np.zeros((num_active_eqs, num_vars))
            b_active = np.zeros(num_active_eqs)
            
            for idx, (a, b_idx, c, d, i, k) in enumerate(eqs):
                # Equation: sum_l U[a, b_idx, l] V[c, d, l] W[k, i, l] = T[a, b_idx, c, d, i, k]
                # T is 1 if a==i, b_idx==d, c==k, else 0.
                target = 1.0 if (a == i and b_idx == d and c == k) else 0.0
                b_active[idx] = target
                
                # Coefficients for W[k, i, l]
                # The variable index for W[k, i, l] is (k * n + i) * r + l
                # The coefficient is U[a, b_idx, l] * V[c, d, l]
                
                # We can fill the row of A_active for this equation.
                # The row is sparse, with non-zeros only at indices (k * n + i) * r + l.
                
                row = A_active[idx]
                for l in range(r):
                    coeff = U_3d[a, b_idx, l] * V_3d[c, d, l]
                    var_idx = (k * n + i) * r + l
                    row[var_idx] = coeff
            
            # Solve A_active * W_flat = b_active
            # Use lstsq for minimum norm solution.
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A_active, b_active, rcond=None)
            except np.linalg.LinAlgError:
                return None, None, None
            
            W_new = W_flat.reshape(p, n, r).transpose(2, 0, 1) # (r, p, n) -> (p, n, r) is wrong.
            # W is (p, n, r). W_flat is (p * n * r,).
            # W_flat index: (k * n + i) * r + l.
            # So W_flat.reshape(p, n, r) gives W[k, i, l].
            # We need W as (p, n, r) for the next iteration?
            # The interface expects W as (p*n, r).
            # So W_new = W_flat.reshape(p, n, r).transpose(2, 0, 1) is wrong.
            # W_flat.reshape(p, n, r) is W[k, i, l].
            # We need W as (p*n, r).
            # So we can just keep W_flat and reshape it when needed.
            # But the function signature expects U, V, W as 2D arrays.
            # Let's convert W_flat to the expected shape.
            # W_expected: (p*n, r).
            # W_flat: (p*n*r,).
            # W_flat.reshape(p*n, r) works if the order is (k, i, l) -> (k*n+i)*r + l.
            # Yes, this matches.
            
            W_new_2d = W_flat.reshape(p * n, r)
            
            return U, V, W_new_2d

        # We will alternate between solving for W and solving for U and V.
        # Solving for U and V is more complex because the equations are quadratic in U and V.
        # However, we can fix W and solve for U and V using a similar least squares approach?
        # No, because U and V appear as a product U_ab,l V_cd,l.
        # 
        # Let's try a different approach: Random restarts with a simple local search.
        # We will randomly initialize U, V, W.
        # Then we will iteratively solve for W given U, V, and then solve for U and V given W?
        # Solving for U and V is hard.
        # 
        # Let's try to solve for U and V using gradient descent.
        # Or, we can use the fact that the solution is likely to be sparse.
        
        # Given the time limit, we will just try a few random restarts and solve for W.
        # This might not be enough to find a rank 33 solution.
        
        # Let's try to find a rank 33 solution by modifying the rank 34 solution.
        # We can try to merge two terms of the rank 34 solution into one.
        # 
        # The rank 34 solution is two copies of a rank 17 solution.
        # Can we merge a term from the first copy with a term from the second copy?
        # 
        # Let's try to find a rank 33 solution by random search.
        
        for restart in range(100):
            if time.time() - start > time_budget:
                break
            
            # Initialize U, V, W with random values
            U = np.random.randn(U_dim, target_rank)
            V = np.random.randn(V_dim, target_rank)
            W = np.random.randn(W_dim, target_rank)
            
            # Iterate: Solve for W, then update U and V with a small step?
            # Let's just solve for W and check the error.
            # If the error is small, we have found a solution.
            
            for step in range(10):
                if time.time() - start > time_budget:
                    break
                
                U, V, W_new = gradient_step(U, V, W, 0.01)
                if W_new is None:
                    break
                W = W_new
                
                # Check error
                err = compute_error(U, V, W)
                if err < 1e-6:
                    # Verify the solution
                    if check_decomposition(U, V, W, n, m, p):
                        return U, V, W
            
            # If we didn't find a solution, try a new random initialization.
            
        return None, None, None

    # Attempt to find rank 33
    U_33, V_33, W_33 = try_find_rank(33, 10.0)
    if U_33 is not None:
        best_U, best_V, best_W = U_33, V_33, W_33
        best_rank = 33
        
    # If we have time, try rank 32
    if best_rank == 33 and time.time() < deadline - 20.0:
        U_32, V_32, W_32 = try_find_rank(32, 15.0)
        if U_32 is not None:
            best_U, best_V, best_W = U_32, V_32, W_32
            best_rank = 32
            
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    construct()
