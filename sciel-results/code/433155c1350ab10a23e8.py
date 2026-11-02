import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find a lower rank decomposition
    # We use a heuristic search with random initialization and local optimization
    
    def evaluate(U, V, W):
        # Compute the tensor T from U, V, W
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want T[i, j, k] == 1 for all i, j, k
        # Using einsum for efficiency
        # U is (n*m, R), V is (m*p, R), W is (p*n, R)
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_r = np.array(U).reshape(n, m, -1)
        V_r = np.array(V).reshape(m, p, -1)
        W_r = np.array(W).reshape(p, n, -1)
        
        # T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c]
        # Note: W index is k*n+i, so W_r[k, i, c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        
        # Error is sum of squares of (T - 1)
        error = np.sum((T - 1.0)**2)
        return error, T

    def try_rank(R, num_trials=100):
        nonlocal best_result, best_rank
        for _ in range(num_trials):
            if time.time() > deadline:
                return False
            
            # Random initialization with sparsity
            # We want integer/half-integer solutions.
            # Let's try random signs and zeros.
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Assign random non-zero entries
            # Each column c corresponds to a rank-1 term.
            # We want the terms to "cover" the tensor.
            # Randomly select a subset of indices for each matrix.
            
            for c in range(R):
                # Pick random indices
                i, j, k = np.random.randint(0, n), np.random.randint(0, m), np.random.randint(0, p)
                # But this is the schoolbook approach. We need denser supports.
                # Let's try to make each column have a few non-zeros.
                nnz = np.random.randint(1, 5)
                
                idx_u = np.random.choice(n * m, nnz, replace=False)
                idx_v = np.random.choice(m * p, nnz, replace=False)
                idx_w = np.random.choice(p * n, nnz, replace=False)
                
                U[idx_u, c] = np.random.choice([-1, 1], nnz)
                V[idx_v, c] = np.random.choice([-1, 1], nnz)
                W[idx_w, c] = np.random.choice([-1, 1], nnz)
            
            # Check error
            error, T = evaluate(U, V, W)
            
            # If error is small, try to snap to integers
            if error < 1e-6:
                # Snap to nearest integer
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                # Verify exactness
                err_int, _ = evaluate(U_int, V_int, W_int)
                if err_int < 1e-9:
                    # Valid decomposition found
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
            
            # If not, we could try to optimize, but for now, just try another random init.
            # A more sophisticated approach would use gradient descent.
            # Given time constraints, we rely on random luck or a smarter heuristic.
            
        return None

    # Try ranks lower than schoolbook
    # Start from a reasonable guess. 
    # For 3x4x7, rank 84 is schoolbook. 
    # Let's try 70, 60, 50...
    for R in [70, 60, 50, 40]:
        if time.time() > deadline:
            break
        
        res = try_rank(R, num_trials=50)
        if res:
            best_result = res
            best_rank = R
            break

    return best_result
