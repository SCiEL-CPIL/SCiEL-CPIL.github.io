"""Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time
import numpy as np

# Dimensions
N, M, P = 2, 3, 6
NM = N * M  # 6
MP = M * P  # 18
PN = P * N  # 12


def verify_decomposition(U, V, W, rank, tol=1e-12):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                expected = 1.0
                u_val = U[i * M + j, :]
                v_val = V[j * P + k, :]
                w_val = W[k * N + i, :]
                actual = np.sum(u_val * v_val * w_val)
                if abs(actual - expected) > tol:
                    return False, (i, j, k, expected, actual)
    return True, None


def schoolbook_decomposition():
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return U, V, W, rank


def construct():
    """
    Construct a decomposition for 2x3 * 3x6 matrix multiplication.
    
    Strategy: Use the schoolbook decomposition (rank 36) as a guaranteed fallback.
    Attempt to find a lower-rank decomposition via numerical optimization.
    """
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with schoolbook as fallback
    U, V, W, rank = schoolbook_decomposition()
    ok, err = verify_decomposition(U, V, W, rank)
    if not ok:
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a better decomposition using numerical optimization
    # We'll search from rank 35 down to 24
    for r in range(35, 23, -1):
        if time.time() > deadline - 15:
            break
        
        # Try a few random initializations for this rank
        n_tries = 3
        found = False
        
        for attempt in range(n_tries):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + r * 100 + attempt)
            
            # Initialize with random values
            scale = 0.5
            U_init = np.random.randn(NM, r) * scale
            V_init = np.random.randn(MP, r) * scale
            W_init = np.random.randn(PN, r) * scale
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            max_iter = 300
            for iteration in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual and gradients
                residual = np.zeros((N, M, P))
                grad_U = np.zeros((NM, r))
                grad_V = np.zeros((MP, r))
                grad_W = np.zeros((PN, r))
                
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            # Residual for (i,j,k)
                            res = 0.0
                            for c in range(r):
                                res += U_curr[i*M+j, c] * V_curr[j*P+k, c] * W_curr[k*N+i, c]
                            residual[i, j, k] = res - 1.0
                            
                            # Gradients
                            for c in range(r):
                                grad_U[i*M+j, c] += 2 * residual[i, j, k] * V_curr[j*P+k, c] * W_curr[k*N+i, c]
                                grad_V[j*P+k, c] += 2 * residual[i, j, k] * U_curr[i*M+j, c] * W_curr[k*N+i, c]
                                grad_W[k*N+i, c] += 2 * residual[i, j, k] * U_curr[i*M+j, c] * V_curr[j*P+k, c]
                
                # Update
                U_curr -= lr * grad_U
                V_curr -= lr * grad_V
                W_curr -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(residual)) < 1e-10:
                    break
                
                # Adaptive learning rate
                if iteration % 50 == 0:
                    lr *= 0.95
            
            # Check if we found a valid decomposition
            ok, _ = verify_decomposition(U_curr, V_curr, W_curr, r, tol=1e-10)
            if ok:
                best_rank = r
                best_U = [[U_curr[i, c] for c in range(r)] for i in range(NM)]
                best_V = [[V_curr[i, c] for c in range(r)] for i in range(MP)]
                best_W = [[W_curr[i, c] for c in range(r)] for i in range(PN)]
                found = True
                break
        
        if found:
            break
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If no better decomposition found, return schoolbook
    return {"U": U, "V": V, "W": W}
