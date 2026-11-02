import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    
    # We need to decompose the tensor for C = A * B
    # A is 2x2, B is 2x15
    # C is 2x15
    # T_{i, j, k} = 1 if i is row of C, j is col of A/row of B, k is col of B
    # Wait, standard index convention for matrix mult A(n x m) * B(m x p) = C(n x p)
    # Tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    # In the factorization: C[i,k] = sum_r U[i,j] * V[j,k] * W[k,i] ? 
    # Let's check the interface:
    # U shape (n*m, rank) -> indexed by (i, j)
    # V shape (m*p, rank) -> indexed by (j, k)
    # W shape (p*n, rank) -> indexed by (k, i)
    # Product: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal 1 if the term is present, 0 otherwise.
    # For standard matrix mult, every term A[i,j]B[j,k] appears exactly once in C[i,k].
    # So T[i, j, k] = 1 for all valid i, j, k.
    
    # Rank of (2, 2, p) tensor.
    # For p=1, rank is 2 (Strassen-like or just 2 terms? 2x2x1 is just dot product of length 2. Rank 2 is trivial: U=[1,0;0,1], V=[1,0;0,1], W=[1,0;0,1] -> sum 1*1*1 + 0... wait.
    # For 2x2x1: C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0].
    # We need 2 terms.
    # Term 1: U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # Term 2: U[1,1]=1, V[1,0]=1, W[0,1]=1? No, W index is (k,i).
    # Let's stick to known ranks.
    # Rank(2,2,2) = 7 (Strassen)
    # Rank(2,2,3) = ?
    # Rank(2,2,4) = ?
    # Rank(2,2,5) = ?
    # Rank(2,2,6) = ?
    # Rank(2,2,7) = ?
    # Rank(2,2,8) = ?
    # Rank(2,2,9) = ?
    # Rank(2,2,10) = ?
    # Rank(2,2,11) = ?
    # Rank(2,2,12) = ?
    # Rank(2,2,13) = ?
    # Rank(2,2,14) = ?
    # Rank(2,2,15) = ?
    
    # The current program tries to solve (2,2,5) rank 14 and (2,2,10) rank 28.
    # Total rank 42.
    # Schoolbook rank is 2*2*15 = 60.
    # 42 is a good improvement.
    
    # However, the numerical solver in the current program is likely to fail or be slow.
    # We should try to find a better, more robust construction or use a known result.
    
    # Is there a known rank for 2x2xp?
    # For p=1, rank=2.
    # For p=2, rank=7.
    # For p=3, rank=10? (2+7+1? No, subadditivity: R(2,2,3) <= R(2,2,1)+R(2,2,2) = 2+7=9? No, R(2,2,1)=2. So R(2,2,3) <= 2+7=9?
    # Actually, R(2,2,p) is roughly 3.5p + c.
    # 2*2*15 = 60.
    # Strassen for 2x2x2 is 7.
    # If we can pack 15 columns into blocks of 2, we have 7 blocks of 2 and 1 block of 1.
    # 7 * 7 + 2 = 51.
    # Can we do better?
    # The current program aims for 42.
    # 42 / 15 = 2.8.
    # 7 / 2 = 3.5.
    # So 42 is significantly better than the block Strassen approach.
    
    # Let's try to implement a solver that is more likely to succeed or use a precomputed solution if possible.
    # Since I cannot precompute and hardcode a 42-rank solution easily without running it, I will try to improve the solver.
    # The previous solver used least_squares with 50 trials.
    # Maybe I can use a different initialization or more trials, or a different optimizer.
    # Or maybe I can split into smaller chunks?
    # (2,2,15) -> (2,2,3) + (2,2,12)?
    # R(2,2,3) <= 9?
    # R(2,2,12) <= 6 * R(2,2,2) = 42? No, R(2,2,2)=7. 6*7=42.
    # Total 9+42=51.
    # The target is 42.
    
    # Let's try to solve for rank 42 directly.
    # The previous code had a bug in the residual calculation or indexing?
    # "val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])"
    # U is (n*m, rank). U[i*m+j] is a vector of length rank.
    # This is correct.
    
    # Let's try to run a solver with more robustness.
    # I will try to use scipy.optimize.minimize with Nelder-Mead or L-BFGS-B, or stick to least_squares but with better init.
    # Random init with 0.5 scale might be too large or small.
    # Also, the problem is non-convex.
    
    # Alternative: Use the fact that for small dimensions, we can use Groebner bases or just brute force search for integer/half-integer solutions?
    # Rank 42 means 42 terms.
    # Variables: 42 * (4 + 30 + 30) = 42 * 64 = 2688 variables.
    # Equations: 2*2*15 = 60 equations.
    # Underdetermined.
    
    # Let's try to improve the numerical solver in the current program.
    # I will increase the number of trials and use a better random seed strategy.
    # I will also add a check for the exactness of the rounded solution.
    
    def solve_decomp(n, m, p, rank, seed=0, trials=200):
        np.random.seed(seed)
        total_U = n * m
        total_V = m * p
        total_W = p * n
        
        def residuals(params):
            U = params[:rank*total_U].reshape(rank, total_U).T
            V = params[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
            W = params[rank*(total_U+total_V):].reshape(rank, total_W).T
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        res.append(val - 1.0)
            return res
        
        best_U, best_V, best_W = None, None, None
        best_cost = float('inf')
        
        for trial in range(trials):
            # Scale initialization
            scale = 0.5 + 0.5 * np.random.rand()
            params0 = np.random.randn(rank * (total_U + total_V + total_W)) * scale
            try:
                result = least_squares(residuals, params0, method='lm', max_nfev=10000)
                cost = result.cost
                if cost < best_cost:
                    best_cost = cost
                    U = result.x[:rank*total_U].reshape(rank, total_U).T
                    V = result.x[rank*total_U:rank*(total_U+total_V)].reshape(rank, total_V).T
                    W = result.x[rank*(total_U+total_V):].reshape(rank, total_W).T
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    # Check if rounded solution is exact
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-9:
                        return U_r, V_r, W_r
                    if cost < best_cost:
                        best_U, best_V, best_W = U_r, V_r, W_r
            except:
                pass
        
        return best_U, best_V, best_W
    
    # Try to solve for (2,2,15) rank 42
    U, V, W = solve_decomp(2, 2, 15, 42, seed=123, trials=100)
    
    if U is not None:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to block decomposition if direct solve fails
    # (2,2,15) = (2,2,5) + (2,2,10)
    U5, V5, W5 = solve_decomp(2, 2, 5, 14, seed=42, trials=100)
    U10, V10, W10 = solve_decomp(2, 2, 10, 28, seed=43, trials=100)
    
    if U5 is not None and U10 is not None:
        U_full = np.hstack([U5, U10])
        V_full = np.zeros((m * p, 42))
        V_full[:m*5, :14] = V5
        V_full[m*5:, 14:] = V10
        W_full = np.zeros((p * n, 42))
        W_full[:5*n, :14] = W5
        W_full[5*n:, 14:] = W10
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    rank = n * m * p
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
