"""Program for 2x3 by 3x3 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The tensor T is of size (n*m) x (m*p) x (p*n) = 6 x 9 x 6.
    We seek rank-r decomposition: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    Returns dict with U (6 x r), V (9 x r), W (6 x r) where r is the rank.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n, m, p = N, M, P
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    
    # Build the full tensor T[i,j,k]
    T = np.zeros((u_dim, v_dim, w_dim))
    for i in range(u_dim):
        i_a = i // m
        j_a = i % m
        for j in range(v_dim):
            j_a2 = j // p
            k_b = j % p
            if j_a != j_a2:
                continue
            for k in range(w_dim):
                k_b2 = k // n
                i_c = k % n
                if i_a != i_c or k_b != k_b2:
                    continue
                T[i, j, k] = 1.0
    
    T_flat = T.flatten()
    
    # Schoolbook decomposition as baseline (rank 18)
    rank_school = n * m * p
    U_s = np.zeros((u_dim, rank_school))
    V_s = np.zeros((v_dim, rank_school))
    W_s = np.zeros((w_dim, rank_school))
    col = 0
    for i_a in range(n):
        for j_a in range(m):
            for k_b in range(p):
                i_idx = i_a * m + j_a
                j_idx = j_a * p + k_b
                k_idx = k_b * n + i_a
                U_s[i_idx, col] = 1.0
                V_s[j_idx, col] = 1.0
                W_s[k_idx, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = rank_school
    
    def check_exact(U_r, V_r, W_r, tol=1e-6):
        """Check if the decomposition is exact."""
        recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
        return np.max(np.abs(recon - T)) < tol
    
    # --- Tensor network contraction: use rank-7 for 3x3, extend to 2x3x3 ---
    # Known rank-7 decomposition for 3x3 matrix multiplication (Coppersmith-Winograd style)
    # We'll construct a rank-14 candidate: 7 terms for output row 0, 7 for output row 1
    # Each term handles one "block" of the multiplication
    
    # For 2x3 * 3x3:
    # Output[i_c, j_a] = sum_{k_b} A[i_c, j_a] * B[j_a, k_b]  ... wait, that's not right
    # Actually: C[i_c, k_b] = sum_{j_a} A[i_c, j_a] * B[j_a, k_b]
    # So for fixed i_c: C[i_c, :] = A[i_c, :] * B  (row vector times matrix)
    # This is a 1x3 * 3x3 = 1x3 multiplication, which has rank 3 (schoolbook) or can be done in rank 2 (Strassen-like for 1x3*3x3?)
    # Actually 1x3 * 3x3: C[k] = sum_j A[j]*B[j,k], this is just 3 scalar products, rank 3.
    # But we can share factors!
    
    # Better approach: use the rank-7 decomposition of 3x3 matrix multiplication.
    # For 3x3 * 3x3 -> 3x3, we have rank 7 (Coppersmith-Winograd).
    # For 2x3 * 3x3 -> 2x3, we can think of it as two independent 1x3 * 3x3 multiplications.
    # Each 1x3 * 3x3 has rank 3, so total rank 6? No, that's the schoolbook.
    # But we can share the B factors!
    
    # Actually, let's use a different approach. The 2x3 * 3x3 multiplication can be seen as:
    # For each output column k_b: C[:, k_b] = A * B[:, k_b]  (2x3 * 3x1 = 2x1)
    # This is 3 independent 2x3 * 3x1 multiplications.
    # Each 2x3 * 3x1: C[i] = sum_j A[i,j] * B[j], rank 2 (since it's bilinear in A and B)
    # Actually rank of 2x3 * 3x1 is 2 (each output is a dot product of 3 elements, but we can use Strassen)
    # Hmm, 2x3 * 3x1: C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    #                C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    # This is equivalent to computing 2 dot products of length 3, which has rank 2 (since each is a single bilinear form).
    # So total rank would be 3 * 2 = 6? That seems too optimistic.
    
    # Let me be more careful. The tensor for 2x3 * 3x1 has:
    # U dimension: 2*3 = 6 (entries of A)
    # V dimension: 3 (entries of B)
    # W dimension: 2 (entries of C)
    # T[i,j,k] = 1 if i corresponds to A[i//3, i%3], j to B[j], k to C[k//3, k%3]... 
    # Actually T[a_idx, b_idx, c_idx] = 1 if c_idx // 1 == a_idx // 3 and ... 
    # Let me just construct it directly.
    
    # For now, let's try a structured rank-14 approach:
    # Use the fact that 3x3 matrix mult has rank 7.
    # For 2x3 * 3x3, we can write:
    # C = A * B where A is 2x3, B is 3x3.
    # Pad A to 3x3 by adding a zero row: A_padded = [[A[0,:]], [A[1,:]], [0,0,0]]
    # Then C_padded = A_padded * B is 3x3, with last row zero.
    # C_padded has rank-7 decomposition using the 3x3 rank-7 factors.
    # But we only need the first 2 rows of C_padded.
    # The rank-7 decomposition of 3x3 gives us 7 terms.
    # Each term contributes to all 3 rows of C_padded.
    # We need to "project" to get only rows 0 and 1.
    # This doesn't directly reduce the rank.
    
    # Alternative: use Strassen's algorithm for 2x2 blocks.
    # Split A into [A1, A2] where A1 is 2x1, A2 is 2x2.
    # Split B into [[B1], [B2]] where B1 is 1x3, B2 is 2x3.
    # C = A1 * B1 + A2 * B2
    # A1*B1: 2x1 * 1x3 = 2x3, rank 2 (each row is a scalar * vector)
    # A2*B2: 2x2 * 2x3 = 2x3, rank 7 (Strassen for 2x2 * 2x2 gives rank 7, but here it's 2x2*2x3)
    # Hmm, this is getting complicated.
    
    # Let's just try to find a good initial rank-14 decomposition and optimize.
    
    def try_rank(rank, num_trials=50, init=None):
        """Try to find a rank-r decomposition using multiple optimization strategies."""
        if init is not None:
            U_init, V_init, W_init = init
            trials = [(U_init, V_init, W_init)]
        else:
            trials = []
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            if init is not None and trial == 0:
                U_init, V_init, W_init = init
            else:
                # Different initialization strategies
                strategy = trial % 8
                
                if strategy == 0:
                    # Pure random
                    np.random.seed(1000 + trial * 7 + rank)
                    U_init = np.random.randn(u_dim, rank)
                    V_init = np.random.randn(v_dim, rank)
                    W_init = np.random.randn(w_dim, rank)
                elif strategy == 1:
                    # Sparse random (many zeros)
                    np.random.seed(2000 + trial * 11 + rank)
                    U_init = np.random.randn(u_dim, rank) * (np.random.rand(u_dim, rank) < 0.3)
                    V_init = np.random.randn(v_dim, rank) * (np.random.rand(v_dim, rank) < 0.3)
                    W_init = np.random.randn(w_dim, rank) * (np.random.rand(w_dim, rank) < 0.3)
                elif strategy == 2:
                    # Based on partial schoolbook structure
                    U_init = np.zeros((u_dim, rank))
                    V_init = np.zeros((v_dim, rank))
                    W_init = np.zeros((w_dim, rank))
                    for c in range(min(rank, 18)):
                        i_a = c // (m * p)
                        j_a = (c // p) % m
                        k_b = c % p
                        if i_a < n and j_a < m and k_b < p:
                            U_init[i_a * m + j_a, c] = 1.0
                            V_init[j_a * p + k_b, c] = 1.0
                            W_init[k_b * n + i_a, c] = 1.0
                    # Add random perturbation
                    np.random.seed(3000 + trial * 13 + rank)
                    U_init += np.random.randn(u_dim, rank) * 0.1
                    V_init += np.random.randn(v_dim, rank) * 0.1
                    W_init += np.random.randn(w_dim, rank) * 0.1
                elif strategy == 3:
                    # Block-structured: group terms by output row (exploit n=2)
                    U_init = np.zeros((u_dim, rank))
                    V_init = np.zeros((v_dim, rank))
                    W_init = np.zeros((w_dim, rank))
                    # Assign roughly half terms to row 0, half to row 1
                    terms_per_row = max(1, rank // n)
                    for i_c in range(n):
                        for t in range(terms_per_row):
                            c = i_c * terms_per_row + t
                            if c >= rank:
                                break
                            np.random.seed(4000 + i_c * 17 + t * 23 + rank)
                            # U: only rows for output row i_c
                            for j_a in range(m):
                                U_init[i_c * m + j_a, c] = np.random.choice([-1, 0, 1])
                            # V: all columns possible
                            for j_a in range(m):
                                for k_b in range(p):
                                    V_init[j_a * p + k_b, c] = np.random.choice([-1, 0, 1]) * 0.5
                            # W: only columns for output row i_c
                            for k_b in range(p):
                                W_init[k_b * n + i_c, c] = np.random.choice([-1, 0, 1])
                elif strategy == 4:
                    # Winograd-like: try to exploit the structure
                    U_init = np.zeros((u_dim, rank))
                    V_init = np.zeros((v_dim, rank))
                    W_init = np.zeros((w_dim, rank))
                    for i_c in range(n):
                        for t in range(rank // n):
                            c = i_c * (rank // n) + t
                            if c >= rank:
                                break
                            np.random.seed(5000 + i_c * 31 + t * 41 + rank)
                            for j_a in range(m):
                                U_init[i_c * m + j_a, c] = np.random.choice([-1, 0, 1])
                            for j_a in range(m):
                                for k_b in range(p):
                                    V_init[j_a * p + k_b, c] = np.random.choice([-1, 0, 1]) * 0.5
                            for k_b in range(p):
                                W_init[k_b * n + i_c, c] = np.random.choice([-1, 0, 1])
                elif strategy == 5:
                    # Winograd-inspired: structured initialization exploiting 2x3x3 symmetry
                    U_init = np.zeros((u_dim, rank))
                    V_init = np.zeros((v_dim, rank))
                    W_init = np.zeros((w_dim, rank))
                    np.random.seed(6000 + trial * 19 + rank)
                    # Try to create terms that combine multiple products
                    for c in range(rank):
                        # Alternate between output rows
                        i_c = c % n
                        # Distribute input columns
                        j_a = (c // n) % m
                        k_b = (c // (n * m)) % p
                        # Core contribution
                        U_init[i_c * m + j_a, c] = 1.0
                        V_init[j_a * p + k_b, c] = 1.0
                        W_init[k_b * n + i_c, c] = 1.0
                        # Add structured cross-terms (key for rank reduction)
                        # Each term can contribute to multiple output entries
                        if c % 2 == 0:
                            # Even terms: add contribution to other row's j_a
                            other_i = 1 - i_c
                            U_init[other_i * m + j_a, c] = np.random.choice([-1, 0, 1])
                            W_init[k_b * n + other_i, c] = np.random.choice([-1, 0, 1])
                        else:
                            # Odd terms: add contribution to adjacent k_b
                            k_b2 = (k_b + 1) % p
                            V_init[j_a * p + k_b2, c] = np.random.choice([-1, 0, 1])
                            W_init[k_b2 * n + i_c, c] = np.random.choice([-1, 0, 1])
                elif strategy == 6:
                    # Symmetric initialization
                    np.random.seed(7000 + trial * 23 + rank)
                    U_init = np.random.randn(u_dim, rank)
                    V_init = np.random.randn(v_dim, rank)
                    W_init = np.random.randn(w_dim, rank)
                    # Scale to have similar magnitudes
                    U_init /= np.linalg.norm(U_init, axis=0, keepdims=True) + 1e-10
                    V_init /= np.linalg.norm(V_init, axis=0, keepdims=True) + 1e-10
                    W_init /= np.linalg.norm(W_init, axis=0, keepdims=True) + 1e-10
                else:
                    # Hybrid: merge schoolbook terms pairwise to reduce rank
                    U_init = np.zeros((u_dim, rank))
                    V_init = np.zeros((v_dim, rank))
                    W_init = np.zeros((w_dim, rank))
                    np.random.seed(8000 + trial * 29 + rank)
                    # Start with schoolbook-like structure but merge terms
                    schoolbook_terms = []
                    for i_a in range(n):
                        for j_a in range(m):
                            for k_b in range(p):
                                schoolbook_terms.append((i_a, j_a, k_b))
                    # Shuffle and assign to rank slots, merging when possible
                    np.random.shuffle(schoolbook_terms)
                    for c in range(rank):
                        if c < len(schoolbook_terms):
                            i_a, j_a, k_b = schoolbook_terms[c]
                            U_init[i_a * m + j_a, c] = 1.0
                            V_init[j_a * p + k_b, c] = 1.0
                            W_init[k_b * n + i_a, c] = 1.0
                        # Add small random components to enable optimization
                        U_init[:, c] += np.random.randn(u_dim) * 0.2
                        V_init[:, c] += np.random.randn(v_dim) * 0.2
                        W_init[:, c] += np.random.randn(w_dim) * 0.2
            
            # Flatten parameters
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(params):
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                diff = recon.flatten() - T_flat
                return np.dot(diff, diff)
            
            def objective_grad(params):
                """Return (objective, gradient) for L-BFGS-B with analytical gradient."""
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                
                recon = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
                diff = recon - T
                obj = np.sum(diff * diff)
                
                # Gradient: d/dU[i,c] = 2 * sum_{j,k} diff[i,j,k] * V[j,c] * W[k,c]
                dU = 2.0 * np.einsum('ijk,jc,kc->ic', diff, V_r, W_r)
                dV = 2.0 * np.einsum('ijk,ic,kc->jc', diff, U_r, W_r)
                dW = 2.0 * np.einsum('ijk,ic,jc->kc', diff, U_r, V_r)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return obj, grad
            
            # Use multiple optimization methods
            if trial % 4 == 0:
                method = 'L-BFGS-B'
                options = {'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13}
                jac = objective_grad
            elif trial % 4 == 1:
                method = 'Nelder-Mead'
                options = {'maxiter': 1200, 'xatol': 1e-12, 'fatol': 1e-14}
                jac = None
            elif trial % 4 == 2:
                method = 'Powell'
                options = {'maxiter': 500, 'ftol': 1e-14, 'xtol': 1e-12}
                jac = None
            else:
                method = 'L-BFGS-B'
                options = {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
                jac = objective_grad
            
            try:
                from scipy.optimize import minimize
                if jac is not None:
                    result = minimize(objective, params_init, method=method, jac=jac, options=options)
                else:
                    result = minimize(objective, params_init, method=method, options=options)
            except Exception:
                continue
            
            if result.fun < 1e-8:
                params = result.x
                U_r = params[:u_dim * rank].reshape(u_dim, rank)
                V_r = params[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                W_r = params[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                
                if check_exact(U_r, V_r, W_r, tol=1e-5):
                    return U_r, V_r, W_r
                
                # Try to polish and round
                if result.fun < 1e-4:
                    # Attempt rounding to half-integers
                    U_round = np.round(U_r * 2) / 2
                    V_round = np.round(V_r * 2) / 2
                    W_round = np.round(W_r * 2) / 2
                    if check_exact(U_round, V_round, W_round, tol=1e-5):
                        return U_round, V_round, W_round
                    
                    # Try integer rounding
                    U_int = np.round(U_r)
                    V_int = np.round(V_r)
                    W_int = np.round(W_r)
                    if check_exact(U_int, V_int, W_int, tol=1e-5):
                        return U_int, V_int, W_int
                    
                    # Final high-precision polish: re-optimize from rounded solution
                    if result.fun < 1e-3:
                        params_polish = np.concatenate([U_round.flatten(), V_round.flatten(), W_round.flatten()])
                        try:
                            result2 = minimize(objective, params_polish, method='L-BFGS-B', jac=objective_grad,
                                              options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14})
                            if result2.fun < 1e-10:
                                U_r2 = params_polish[:u_dim * rank].reshape(u_dim, rank)
                                V_r2 = params_polish[u_dim * rank:u_dim * rank + v_dim * rank].reshape(v_dim, rank)
                                W_r2 = params_polish[u_dim * rank + v_dim * rank:].reshape(w_dim, rank)
                                if check_exact(U_r2, V_r2, W_r2, tol=1e-6):
                                    return U_r2, V_r2, W_r2
                        except Exception:
                            pass
        
        return None
    
    # --- Construct a structured rank-14 initial guess ---
    # Use the idea: for each of the 3 columns of B, compute A * B[:,k] (2x3 * 3x1 = 2x1)
    # Each such product has rank 2 (two bilinear forms)
    # Total: 3 * 2 = 6 terms? That's the schoolbook rank for this subproblem.
    # But we can share structure across columns.
    
    # Let's try a different structured approach:
    # For 2x3 * 3x3, we can use the following:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Use the rank-7 decomposition of 3x3 matrix multiplication as a building block.
    # The 3x3 * 3x3 -> 3x3 tensor has rank 7.
    # If we fix the first row of A to be zero, we get a 2x3 * 3x3 -> 2x3 tensor.
    # The rank-7 decomposition of 3x3 gives us 7 terms, but some terms may only
    # contribute to the (now-zero) first row of the output.
    
    # Actually, let's just try to optimize from a good structured start.
    
    # Structured rank-14 start: 7 terms for row 0, 7 for row 1
    # Each group of 7 terms computes one row of C = A_row * B
    # A_row is 1x3, B is 3x3, C_row is 1x3
    # This is a 1x3 * 3x3 = 1x3 multiplication, which has rank 3 (schoolbook)
    # But we can do better with structured factors.
    
    # For 1x3 * 3x3 -> 1x3: C[k] = sum_j A[j] * B[j,k]
    # This is equivalent to computing 3 dot products of length 3.
    # Rank of this tensor is 3 (each output is a separate bilinear form).
    # So 2 rows * 3 = 6 terms total? That's the schoolbook.
    
    # The key insight: we can share B factors between the two rows!
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    # 
    # If we use the same B factors for both rows, we can potentially reduce rank.
    # But each term still needs to multiply by a specific A[0,j] or A[1,j].
    
    # Let's try: for each j (0,1,2) and k (0,1,2), we need A[i,j]*B[j,k] to contribute to C[i,k].
    # Schoolbook: 2*3*3 = 18 terms.
    # 
    # Strassen-like for 2x2 blocks:
    # Split A = [A1, A2] where A1 is 2x1, A2 is 2x2
    # Split B = [B1; B2] where B1 is 1x3, B2 is 2x3
    # C = A1*B1 + A2*B2
    # A1*B1: 2x1 * 1x3 = 2x3. Each entry C[i,k] gets A1[i]*B1[k]. Rank 2 (one factor per row).
    # A2*B2: 2x2 * 2x3 = 2x3. This is like two independent 1x2 * 2x3 multiplications.
    # 1x2 * 2x3: C[k] = A[0]*B[0,k] + A[1]*B[1,k]. Rank 2.
    # So A2*B2 has rank 2*2 = 4? No, the two rows share B factors.
    # Actually, A2*B2: C[0,k] = A2[0,0]*B2[0,k] + A2[0,1]*B2[1,k]
    #                   C[1,k] = A2[1,0]*B2[0,k] + A2[1,1]*B2[1,k]
    # We can use Strassen's 7-multiplication algorithm for 2x2*2x2, but here it's 2x2*2x3.
    # The rank of 2x2 * 2x3 is at most 7 (by padding to 2x2*2x2 and using Strassen).
    # But actually, 2x2 * 2x3: each of the 2 outputs is a 1x2 * 2x3 product, rank 2 each.
    # Total rank 4 if no sharing, but we can share.
    
    # This is getting too complicated. Let's just use the optimization with good structured starts.
    
    # Structured start 1: Based on splitting by output column
    # For each output column k (0,1,2):
    #   C[:,k] = A * B[:,k]  (2x3 * 3x1 = 2x1)
    #   This has rank 2: C[0,k] = sum_j A[0,j]*B[j,k], C[1,k] = sum_j A[1,j]*B[j,k]
    #   We can use 2 terms per column: term1 for row 0, term2 for row 1
    #   Total: 3*2 = 6 terms. But this is just schoolbook with different grouping.
    
    # Let's try a rank-14 start that's close to schoolbook but with some merging.
    
    # For now, just try ranks from 13 to 17 with the existing strategies.
    
    # Try ranks from 13 up to 17 (schoolbook is 18)
    # Focus on 13-15 first as these are more likely achievable
    candidate_ranks = [13, 14, 15, 16, 17]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Use more trials for smaller ranks where structure matters more
        num_trials = 120 if rank <= 15 else 60
        result = try_rank(rank, num_trials=num_trials)
        if result is not None:
            U_r, V_r, W_r = result
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = rank
            break  # Found the lowest rank, stop
    
    # Convert to lists with integer/half-integer entries if possible
    def to_half_integer(mat):
        """Round values to nearest half-integer and verify."""
        rounded = np.round(mat * 2) / 2
        return rounded
    
    U_out = to_half_integer(best_U)
    V_out = to_half_integer(best_V)
    W_out = to_half_integer(best_W)
    
    # Verify the rounded version still works
    recon = np.einsum('ic,jc,kc->ijk', U_out, V_out, W_out)
    if np.max(np.abs(recon - T)) > 1e-4:
        # Rounding broke it, use original
        U_out = best_U
        V_out = best_V
        W_out = best_W
    
    # Convert to Python lists
    U_list = [[float(x) for x in row] for row in U_out]
    V_list = [[float(x) for x in row] for row in V_out]
    W_list = [[float(x) for x in row] for row in W_out]
    
    return {"U": U_list, "V": V_list, "W": W_list}
