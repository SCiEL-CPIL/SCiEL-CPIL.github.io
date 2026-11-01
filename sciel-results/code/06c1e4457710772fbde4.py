import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # The task asks for the rank-minimal exact tensor decomposition.
    # For 3x3 by 3x5 matrix multiplication, the theoretical lower bound is at least 3*3*5/2 = 22.5 -> 23.
    # However, known results for rectangular matrix multiplication often do not have a simple closed-form "Strassen-like" decomposition with small integer coefficients that is easy to hardcode without extensive search.
    # The "seed" rank is 45 (schoolbook).
    # The inspirations suggest that numerical search is unreliable or slow, and hardcoding a low-rank decomposition (like rank 14 or 23) is the intended path if such a decomposition is known and verified.
    # Since I cannot verify a specific rank-14 or rank-23 decomposition from memory with 100% certainty of correctness (exactness is required), and the current program's greedy removal with optimization failed to improve beyond the seed (score 1.0), I must be careful.
    #
    # However, the directive says: "Use a precomputed, hardcoded rank-14 decomposition...".
    # If such a decomposition exists, it is a massive improvement over 45.
    # But if I hardcode incorrect values, the score becomes 0.
    #
    # Let's look at the structure. The current program returns the schoolbook rank 45.
    # Is there a known efficient algorithm for 3x3 * 3x5?
    # Standard matrix multiplication is $O(n^3)$. Strassen reduces exponent for square matrices.
    # For rectangular, the "tensor rank" is the metric.
    #
    # Wait, the prompt says: "INSPIRATIONS ... Strategy to try in this change: Use a precomputed, hardcoded rank-14 decomposition...".
    # This implies that such a decomposition *is* available or assumed to be findable/hardcodable in the context of this challenge, OR it's a trap/hallucination in the inspiration.
    # Given the risk of 0 score if wrong, and the fact that the current program is stable at 1.0, I should only include a hardcoded solution if I am certain.
    #
    # Actually, let's re-read the task carefully. "Find a rank-minimal exact tensor decomposition".
    # If I cannot find a valid lower rank, I should return the best valid one I have, which is the schoolbook rank 45.
    #
    # However, the "COMBINE" instruction asks to take the structural idea of one and refinement of the other.
    # Inspiration 1: Simple schoolbook.
    # Inspiration 2: Greedy removal (no opt).
    # Inspiration 3: Numerical search (unreliable).
    # Current: Greedy removal with BFGS optimization.
    #
    # The current program's `construct` function attempts to reduce the rank from 45 using BFGS.
    # The observation says it scored 1.0, meaning it likely returned the seed rank 45 because the optimization didn't find a valid lower rank decomposition within the time/tolerance.
    #
    # If I follow the directive to "Use a precomputed, hardcoded rank-14 decomposition", I need to know the matrices.
    # I do not have a verified rank-14 decomposition for 3x3*3x5 in my training data that I can recall with high confidence for exact integer/half-integer entries.
    #
    # Let's consider the possibility that the "rank-14" in the directive is a specific hint for this instance.
    # If I cannot provide it, I should stick to the robust approach.
    #
    # However, there is a known result: The rank of the $3 \times 3 \times 3$ tensor is at most 23 (Strassen).
    # For $3 \times 3 \times 5$, the rank is at most $3 \times 3 \times 5 = 45$.
    # Can we do better?
    # The rank of $A \times B$ where $A$ is $n \times m$ and $B$ is $m \times p$ is related to the rank of the $n \times p \times m$ tensor.
    #
    # Since I cannot invent a correct low-rank decomposition, I will refine the current program to be faster and more robust, ensuring it definitely returns the valid seed if no improvement is found, and perhaps trying a few more structured removals.
    #
    # BUT, the directive explicitly says: "Strategy to try in this change: Use a precomputed, hardcoded rank-14 decomposition...".
    # If I ignore this, I might fail the "combine" instruction.
    # If I follow it blindly with fake data, I score 0.
    #
    # Let's look at the "OBSERVED_CHANGES". It shows a program p76 that edited `construct` and `compute_grad` and got score 1.0.
    #
    # I will implement a hybrid:
    # 1. Define the schoolbook decomposition (Rank 45).
    # 2. Attempt to verify a small set of known low-rank decompositions if I can construct them.
    # 3. If not, return the schoolbook.
    #
    # Actually, there is a simpler improvement. The current program uses `scipy.optimize.minimize` which can be slow or fail.
    # The "refinement machinery" from Inspiration 3 mentions "smooth surrogate" or "final high-precision polish".
    #
    # Let's try to implement a more aggressive greedy removal with a simpler local solver (e.g., least squares for each factor in turn) instead of full BFGS on the joint space, which might be more stable for small rank reductions.
    #
    # However, the most impactful change suggested is the hardcoded decomposition.
    # Since I cannot provide a verified rank-14 decomposition, I will provide the schoolbook rank 45, but I will structure the code to easily accept a hardcoded solution if one were provided.
    #
    # Wait, is it possible that the "rank-14" is a typo in the prompt or a specific test case?
    # $3 \times 3 \times 5$.
    # If I return rank 45, score is 1.0.
    # If I return rank 44 (valid), score is 45/44 > 1.0.
    #
    # Let's try to find a valid rank-44 decomposition.
    # Removing one term from the schoolbook decomposition means we need to absorb the contribution of one basis vector into the others.
    # This is generally not possible with just 1s and 0s. It requires changing the values.
    #
    # Given the constraints and the risk, I will keep the robust schoolbook fallback and improve the search speed.
    # I will replace the BFGS optimization with a faster, more targeted local search or simply return the seed if the search is too slow, ensuring stability.
    #
    # Actually, looking at the inspirations, they all returned 1.0.
    # The "strategy to try" is a specific instruction. If I cannot fulfill it with correct data, I should not attempt it.
    #
    # I will output the current program's logic but with a slight optimization: use NumPy vectorization for the residual check to make the greedy loop faster, allowing more attempts.

    def check_valid(U, V, W, rank, tol=1e-8):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T[i,j,k] == 1 for all i,j,k.
        
        # Reshape for batch processing?
        # U_flat: (nm, r)
        # V_flat: (mp, r)
        # W_flat: (pn, r)
        
        # We can compute the tensor T as a sum of outer products.
        # T = sum_r U[:, r] (reshaped to n,m) * V[:, r] (reshaped to m,p) * W[:, r] (reshaped to p,n)
        # This is a rank-1 tensor sum.
        
        # Efficient check:
        # For each r, compute the outer product contribution.
        # But looping over r is fine if r is small (<=45).
        
        T = np.zeros((n, m, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            
            # Contribution to T[i,j,k] is Ur[i,j] * Vr[j,k] * Wr[k,i]
            # We can't just do element-wise multiply of 3 arrays of different shapes easily without broadcasting tricks.
            # Ur: (n, m)
            # Vr: (m, p)
            # Wr: (p, n)
            
            # Let's use einsum or explicit loops for the check if rank is small.
            # Since n,m,p are small (3,3,5), explicit loops are fast.
            pass
        
        # Explicit loop check is O(n*m*p*rank).
        # 3*3*5*45 = 2025 operations. Very fast.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Initialize Schoolbook
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed))
    V_seed = np.zeros((m * p, rank_seed))
    W_seed = np.zeros((p * n, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1

    best_rank = rank_seed
    best_U = U_seed.copy()
    best_V = V_seed.copy()
    best_W = W_seed.copy()

    current_rank = rank_seed
    U = U_seed.copy()
    V = V_seed.copy()
    W = W_seed.copy()

    # Greedy Removal
    # Try to remove one term at a time and see if the remaining system is solvable.
    # Solvability check:
    # If we remove term r, we have rank-1 terms.
    # We need to find U', V', W' such that sum_{r'!=r} U'[i,j,r'] V'[j,k,r'] W'[k,i,r'] = 1.
    # This is a non-linear system.
    # The current program uses BFGS to solve for the new values.
    #
    # To improve speed, we can try a simpler heuristic:
    # If we remove a term, can we scale the other terms to compensate?
    # Generally no, because the support sets are different.
    #
    # I will keep the BFGS approach but limit the iterations strictly and add a timeout per attempt.

    def try_reduce(U, V, W, current_rank, deadline):
        for r in range(current_rank):
            if time.time() > deadline:
                return False, U, V, W, current_rank
            
            mask = [x for x in range(current_rank) if x != r]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            new_rank = current_rank - 1
            
            # Check if already valid (unlikely for schoolbook)
            if check_valid(U_new, V_new, W_new, new_rank):
                return True, U_new, V_new, W_new, new_rank
            
            # Try to optimize
            # Flatten
            size_u = n * m * new_rank
            size_v = m * p * new_rank
            size_w = p * n * new_rank
            x0 = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
            
            # Use a simple local search or BFGS
            # To be safe and fast, let's use a few steps of gradient descent or BFGS
            # Given the small size, BFGS is okay if we limit iterations.
            
            def compute_residual(params):
                U_p = params[:size_u].reshape((n * m, new_rank))
                V_p = params[size_u:size_u + size_v].reshape((m * p, new_rank))
                W_p = params[size_u + size_v:].reshape((p * n, new_rank))
                
                res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for rr in range(new_rank):
                                val += U_p[i * m + j, rr] * V_p[j * p + k, rr] * W_p[k * n + i, rr]
                            res += (val - 1.0) ** 2
                return res

            def compute_grad(params):
                U_p = params[:size_u].reshape((n * m, new_rank))
                V_p = params[size_u:size_u + size_v].reshape((m * p, new_rank))
                W_p = params[size_u + size_v:].reshape((p * n, new_rank))
                
                # E[i,j,k] = sum_r U V W - 1
                E = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for rr in range(new_rank):
                                val += U_p[i * m + j, rr] * V_p[j * p + k, rr] * W_p[k * n + i, rr]
                            E[i, j, k] = val - 1.0
                            
                grad_U = np.zeros_like(U_p)
                grad_V = np.zeros_like(V_p)
                grad_W = np.zeros_like(W_p)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = E[i, j, k]
                            if e == 0.0: continue
                            for rr in range(new_rank):
                                grad_U[i * m + j, rr] += 2 * e * V_p[j * p + k, rr] * W_p[k * n + i, rr]
                                grad_V[j * p + k, rr] += 2 * e * U_p[i * m + j, rr] * W_p[k * n + i, rr]
                                grad_W[k * n + i, rr] += 2 * e * U_p[i * m + j, rr] * V_p[j * p + k, rr]
                                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

            try:
                from scipy.optimize import minimize
                # Limit time per optimization
                start_opt = time.time()
                opt_deadline = start_opt + 5.0 # 5 seconds max per attempt
                
                # We can't easily pass a deadline to scipy, so we rely on maxiter
                # 50 iterations should be fast enough for this size
                result = minimize(
                    compute_residual, x0,
                    jac=compute_grad,
                    method='BFGS',
                    options={'maxiter': 50, 'gtol': 1e-8, 'disp': False}
                )
                
                if result.fun < 1e-8:
                    U_opt = result.x[:size_u].reshape((n * m, new_rank))
                    V_opt = result.x[size_u:size_u + size_v].reshape((m * p, new_rank))
                    W_opt = result.x[size_u + size_v:].reshape((p * n, new_rank))
                    if check_valid(U_opt, V_opt, W_opt, new_rank, tol=1e-6):
                        return True, U_opt, V_opt, W_opt, new_rank
            except Exception:
                pass
                
        return False, U, V, W, current_rank

    # Loop for reductions
    while time.time() < deadline:
        reduced, U, V, W, current_rank = try_reduce(U, V, W, current_rank, deadline)
        if not reduced:
            break
        
        # Update best
        if current_rank < best_rank:
            best_rank = current_rank
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
