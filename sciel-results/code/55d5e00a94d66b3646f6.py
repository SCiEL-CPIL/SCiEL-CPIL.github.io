import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Target rank to try
    target_rank = 23
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    # Build the target tensor T
    # T has shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Try to find a decomposition of rank target_rank
    # Variables: U, V, W flattened
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # Total variables: rank * (n*m + m*p + p*n)
    
    def objective(params):
        if time.time() > deadline:
            return np.inf
        U = params[:rank_schoolbook*target_rank].reshape(n*m, target_rank)
        V = params[rank_schoolbook*target_rank:2*rank_schoolbook*target_rank].reshape(m*p, target_rank)
        W = params[2*rank_schoolbook*target_rank:].reshape(p*n, target_rank)
        
        # Compute tensor decomposition
        # T_approx = sum_r U[:,r] * V[:,r] * W[:,r]
        # This is a tensor contraction
        # Efficient way:
        # T_approx[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # We can compute this by iterating over r or using einsum
        # T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # But we need to compare with T
        # T is sparse, so we can just compute the error on the non-zero elements?
        # No, we need to ensure the whole tensor matches.
        # However, the tensor is fixed.
        
        # Compute error
        # T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        # err = np.sum((T_approx - T)**2)
        
        # To save time, we can sample or use a sparse check
        # But for exactness, we need full check.
        # 9x9x9 is small (729 elements).
        T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.sum((T_approx - T)**2)
        return err

    # Initialize with random values
    np.random.seed(42)
    x0 = np.random.randn(target_rank * (n*m + m*p + p*n)) * 0.1
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Try a few restarts
    for restart in range(5):
        if time.time() > deadline:
            break
        
        x0 = np.random.randn(target_rank * (n*m + m*p + p*n)) * 0.1
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 5000})
        
        if res.fun < 1e-6:
            # Check if we can round to integers/half-integers
            U = res.x[:rank_schoolbook*target_rank].reshape(n*m, target_rank)
            V = res.x[rank_schoolbook*target_rank:2*rank_schoolbook*target_rank].reshape(m*p, target_rank)
            W = res.x[2*rank_schoolbook*target_rank:].reshape(p*n, target_rank)
            
            # Round to nearest 0.5
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Check exactness
            T_approx = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
            if np.allclose(T_approx, T, atol=1e-9):
                best_rank = target_rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break
        
    if best_rank < rank_schoolbook:
        # Convert to lists for the interface
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
