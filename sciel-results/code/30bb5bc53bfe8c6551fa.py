import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    # Construct Schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Target tensor: all ones
    # Dimensions: n, m, p
    # Target flat index: i * m * p + j * p + k ?
    # Wait, the output W index is k * n + i.
    # The tensor T is defined on indices (i, j, k).
    # Value is 1.
    
    # We want to find U, V, W of shape (n*m, r), (m*p, r), (p*n, r)
    # such that for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    def get_residual(x, r):
        U = x[:r * n * m].reshape(n * m, r)
        V = x[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
        W = x[r * n * m + r * m * p :].reshape(p * n, r)
        
        # Compute tensor
        # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # We can do this efficiently with einsum or loops
        # Since n,m,p are small, loops are fine.
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over c
                    T[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
        # Residual is T - 1
        return (T - 1).flatten()
        
    # Try ranks from best_rank - 1 down to 30
    # We prioritize finding a valid lower rank.
    # Start with ranks likely to be solvable.
    # 62, 61, 60, 59...
    
    # To save time, we can try a few specific ranks first.
    # If we find rank 60, score 1.05.
    # If we find rank 55, score 1.14.
    
    ranks_to_try = list(range(rank_schoolbook - 1, 39, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        # Number of restarts decreases as time runs out
        max_restarts = 5 if (deadline - time.time()) > 60 else 2
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with small random numbers
            # Scale: if U~1, V~1, W~1, sum is ~r. Target is 1.
            # So we need small values. 1/sqrt(r) ?
            scale = 1.0 / np.sqrt(r)
            x0 = np.random.randn(r * (n * m + m * p + p * n)) * scale
            
            # Optimize
            try:
                res = opt.least_squares(get_residual, x0, args=(r,), method='trf', max_nfev=1000)
                if res.cost < 1e-8:
                    # Found a solution
                    U = res.x[:r * n * m].reshape(n * m, r)
                    V = res.x[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
                    W = res.x[r * n * m + r * m * p :].reshape(p * n, r)
                    
                    # Convert to list of lists of floats (or ints if close)
                    # The problem asks for integer/half-integer entries.
                    # The solution from least_squares will be floats.
                    # We need to check if they can be rounded.
                    # If not, we might need to enforce integrality or find a rational solution.
                    # However, the evaluator says "integer/half-integer entries".
                    # If we return floats, it might be rejected or score 0?
                    # "score is the rank". "invalid result... scores 0".
                    # So we MUST return valid integer/half-integer entries.
                    
                    # Check if values are close to integers or half-integers
                    def is_valid(x_mat):
                        # Check if x is close to x*2 being integer
                        x2 = x_mat * 2
                        if np.all(np.abs(x2 - np.round(x2)) < 1e-5):
                            return True
                        return False
                    
                    if is_valid(U) and is_valid(V) and is_valid(W):
                        U_int = (U * 2).round() / 2
                        V_int = (V * 2).round() / 2
                        W_int = (W * 2).round() / 2
                        
                        # Verify exact decomposition with rounded values
                        T_check = np.zeros((n, m, p))
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    T_check[i, j, k] = np.dot(U_int[i * m + j, :], V_int[j * p + k, :] * W_int[k * n + i, :])
                        if np.all(np.abs(T_check - 1) < 1e-9):
                            best_rank = r
                            best_result = {
                                "U": U_int.tolist(),
                                "V": V_int.tolist(),
                                "W": W_int.tolist()
                            }
                            # Since we are iterating down, this is the best so far.
                            # We can break or continue to find even lower?
                            # We break to save time for lower ranks?
                            # No, we continue to try lower ranks in the outer loop.
                            pass
                    
            except Exception as e:
                pass
    return best_result
