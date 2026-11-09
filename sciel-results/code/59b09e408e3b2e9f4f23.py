import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def tensor_residual(U, V, W, n, m, p):
    r = U.shape[1]
    # Compute the tensor T = sum_c u_c v_c w_c
    # T[I, J, K] = sum_c U[I, c] V[J, c] W[K, c]
    # We want T to match the target tensor.
    # Target tensor has 1 at valid triples, 0 elsewhere.
    # We can compute the error on the valid triples and the zeros.
    
    # For efficiency, we only compute the sum for the valid triples and a few zeros.
    # But we need to ensure zeros are zero.
    # Let's compute the full tensor error.
    # T = np.einsum('ic,jc,kc->ijk', U, V, W)
    # This is fast for small matrices.
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    
    # Target tensor
    # We can create the target tensor as a sparse array or just compute the error directly.
    # Error = sum (T[I, J, K] - Target[I, J, K])^2
    # Target is 1 if (I, J, K) is valid, 0 otherwise.
    
    # Let's compute the error on the valid triples
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i*m + j
                J = j*p + k
                K = k*n + i
                error += (T[I, J, K] - 1)**2
                
    # Error on zeros
    # There are many zeros. We can sample some or compute all.
    # Total entries: (n*m)*(m*p)*(p*n) = 16*28*28 = 12544
    # This is small, so we can compute all.
    for I in range(n*m):
        for J in range(m*p):
            for K in range(p*n):
                # Check if (I, J, K) is valid
                i = I // m
                j = I % m
                j2 = J // p
                k = J % p
                k2 = K // n
                i2 = K % n
                if i == i2 and j == j2 and k == k2:
                    continue # Already counted
                error += T[I, J, K]**2
    return error

def find_decomp(n, m, p, target_rank, max_time=50):
    start_time = time.time()
    best_error = float('inf')
    best_solution = None
    
    for restart in range(20):
        if time.time() - start_time > max_time:
            break
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # ALS iterations
        for it in range(50):
            if time.time() - start_time > max_time:
                break
            # Update U
            # For each column c, U[:, c] is a vector.
            # We want to minimize sum_{I,J,K} (sum_c U[I,c] V[J,c] W[K,c] - T_target[I,J,K])^2
            # This is not linear in U because V and W are fixed, but the sum is over c.
            # Actually, for a fixed c, the term is U[:, c] * (V[:, c] * W[:, c])? No.
            # The term is sum_c U[I,c] V[J,c] W[K,c].
            # This is linear in U if V and W are fixed? No, it's a sum of products.
            # Wait, if we fix V and W, the residual is linear in U?
            # Residual[I, J, K] = sum_c U[I,c] V[J,c] W[K,c] - T_target
            # This is linear in U.
            # So we can solve for U using least squares.
            
            # Let's vectorize the update.
            # For each I, we want to find U[I, :] that minimizes the error.
            # The error contribution from row I of U is:
            # sum_{J,K} (sum_c U[I,c] V[J,c] W[K,c] - T_target[I,J,K])^2
            # This is a quadratic form in U[I, :].
            # Let x = U[I, :].
            # f(x) = || A x - b ||^2
            # Where A is a matrix of size (m*p*p*n) x r? No.
            # The indices J and K are coupled.
            # Let's just use a simple gradient step or a solver.
            
            # Given the small size, we can use scipy.optimize.least_squares on the flattened variables.
            # But it might be slow.
            
            # Let's try a simple heuristic:
            # Round to half-integers and check.
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_solution(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded
                
        # If not found, try next restart
    return None

def check_solution(U, V, W, n, m, p):
    r = U.shape[1]
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i*m + j
                J = j*p + k
                K = k*n + i
                if abs(T[I, J, K] - 1) > 1e-9:
                    return False
    for I in range(n*m):
        for J in range(m*p):
            for K in range(p*n):
                i = I // m
                j = I % m
                j2 = J // p
                k = J % p
                k2 = K // n
                i2 = K % n
                if i == i2 and j == j2 and k == k2:
                    continue
                if abs(T[I, J, K]) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    # Try to find a better solution
    # Try 4x4x4 rank 49 and 4x4x3 rank 48
    U1, V1, W1 = find_decomp(4, 4, 4, 49, max_time=50)
    if U1 is not None:
        U2, V2, W2 = find_decomp(4, 4, 3, 48, max_time=50)
        if U2 is not None:
            # Combine
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            r = r1 + r2
            U = np.zeros((16, r))
            V = np.zeros((28, r))
            W = np.zeros((28, r))
            
            U[:, :r1] = U1
            U[:, r1:] = U2
            
            # V1 has rows (j, k) for k in [0,4)
            # V2 has rows (j, k) for k in [0,3)
            # We need to map V2 rows to (j, k+4)
            V[:, :r1] = V1
            for j in range(4):
                for k in range(3):
                    V[j*(7) + (k+4), r1:] = V2[j*(3) + k, :]
            
            # W1 has rows (k, i) for k in [0,4)
            # W2 has rows (k, i) for k in [0,3)
            # We need to map W2 rows to (k+4, i)
            W[:, :r1] = W1
            for k in range(3):
                for i in range(4):
                    W[(k+4)*4 + i, r1:] = W2[k*4 + i, :]
            
            # Convert to lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
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
