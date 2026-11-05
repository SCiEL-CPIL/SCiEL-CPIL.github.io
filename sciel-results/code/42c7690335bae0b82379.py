"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    def verify_decomposition(U, V, W, n, m, p, num_trials=10):
        """Verify that the decomposition exactly computes matrix multiplication."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        for _ in range(num_trials):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            
            # Compute using factors
            # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                # u_r is (n, m), v_r is (m, p), w_r is (p, n)
                u_r = U_reshaped[:, :, r]
                v_r = V_reshaped[:, :, r]
                w_r = W_reshaped[:, :, r]
                
                # For each output (i, k):
                # C[i,k] += sum_j A[i,j] * u_r[i,j] * (sum_j' B[j',k] * v_r[j',k]) * w_r[k,i]
                # Actually: C[i,k] = sum_r (sum_j A[i,j]*U[i,j,r]) * (sum_j B[j,k]*V[j,k,r]) * W[k,i,r]
                left = A @ u_r  # (n,) : left[i] = sum_j A[i,j]*U[i,j,r]
                right = v_r @ B  # (p,) : right[k] = sum_j V[j,k,r]*B[j,k]
                C_computed += np.outer(left, right) * w_r.T  # w_r.T[i,k] = w_r[k,i]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p = 45."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    # Try to find a lower-rank decomposition using numerical optimization
    # Strategy: Start with schoolbook and try to reduce rank
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 45
    
    # Try ranks from 44 down to 30
    for target_rank in range(44, 29, -1):
        try:
            # Initialize with random factors
            np.random.seed(42)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Use gradient-free optimization: try multiple random initializations
            found = False
            for init_trial in range(5):
                if init_trial > 0:
                    U = np.random.randn(n * m, target_rank)
                    V = np.random.randn(m * p, target_rank)
                    W = np.random.randn(p * n, target_rank)
                
                # Simple verification with a few test cases
                if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, num_trials=3):
                    found = True
                    break
            
            if found:
                # Full verification
                if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, num_trials=20):
                    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
                    best_rank = target_rank
                    break  # Found the lowest rank, stop searching
        
        except Exception:
            continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
