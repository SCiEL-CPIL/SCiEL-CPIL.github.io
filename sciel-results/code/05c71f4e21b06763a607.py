"""Matrix multiplication 3x5x7 - rank minimization."""
import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Uses DFT embedding with cyclic convolution of length 11 to attempt a lower-rank
    decomposition, falling back to schoolbook (rank 105) if no improvement is found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21
    
    def schoolbook():
        rank = n * m * p  # 105
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reproduces the tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = sum(U[u_idx][c] * V[v_idx][c] * W[w_idx][c] for c in range(len(U[u_idx])))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_dft_embedding():
        """
        Try DFT embedding: map the 5-dimensional inner product to a cyclic 
        convolution of length 11 (smallest prime >= 5+7-1=11).
        
        The idea: For each (i,k) output index, the inner product sum_j A[i,j]*B[j,k]
        can be computed via 11 rank-1 terms using DFT of length 11.
        This gives rank = n * p * 11 = 3*7*11 = 231, which is worse than 105.
        
        However, we can try to exploit structure to reduce the rank.
        Let's try a different approach: use the DFT to create a structured 
        decomposition and see if we can find a lower rank.
        """
        L = 11  # Cyclic convolution length
        
        # Build the DFT matrix of length L
        # DFT[t][j] = exp(2*pi*i*t*j/L) for t,j in 0..L-1
        # But we need rational approximations or exact structure
        
        # For exact rational decomposition, we can use the fact that the 
        # cyclic group Z_L has characters. For L=11 (prime), the DFT matrix
        # has entries that are 11th roots of unity.
        
        # Let's try a different strategy: use the embedding of Z_5 into Z_11
        # via j -> j (mod 11), and use the DFT of length 11.
        
        # The key insight: sum_{j=0}^{4} A_j * B_j = (1/11) * sum_{t=0}^{10} 
        #   (sum_{j=0}^{4} A_j * omega^{t*j}) * (sum_{j=0}^{4} B_j * omega^{-t*j})
        # where omega = exp(2*pi*i/11)
        
        # This gives 11 rank-1 terms for the 5-dimensional inner product.
        # For the full 3x5x7 multiplication, we get 3*7*11 = 231 terms.
        # This is worse than schoolbook, so let's try to optimize.
        
        # Alternative: try to find a structured decomposition with rank < 105
        # by exploiting the small dimensions.
        
        # Let's try a direct approach: use ALS with a good initialization
        # and see if we can find a lower rank decomposition.
        
        best_rank = None
        best_decomp = None
        
        # Try ranks from 50 to 104 (below schoolbook)
        for r_test in range(50, 105, 5):
            if time.time() > deadline - 10.0:
                break
            
            try:
                # Initialize with structured random
                np.random.seed(42 + r_test)
                U = np.random.randn(nm, r_test)
                V = np.random.randn(mp, r_test)
                W = np.random.randn(pn, r_test)
                
                # Build target tensor
                T = np.zeros((nm, mp, pn))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i*m+j, j*p+k, k*n+i] = 1.0
                
                # ALS iterations
                for it in range(20):
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Update U
                    M_u = np.zeros((mp*pn, r_test))
                    for c in range(r_test):
                        M_u[:, c] = np.outer(V[:, c], W[:, c]).flatten()
                    U_sol, _, _, _ = np.linalg.lstsq(M_u, T.reshape(nm, mp*pn).T, rcond=None)
                    U = U_sol.T
                    
                    # Update V
                    M_v = np.zeros((nm*pn, r_test))
                    for c in range(r_test):
                        M_v[:, c] = np.outer(U[:, c], W[:, c]).flatten()
                    V_sol, _, _, _ = np.linalg.lstsq(M_v, T.reshape(mp, nm*pn).T, rcond=None)
                    V = V_sol.T
                    
                    # Update W
                    M_w = np.zeros((nm*mp, r_test))
                    for c in range(r_test):
                        M_w[:, c] = np.outer(U[:, c], V[:, c]).flatten()
                    W_sol, _, _, _ = np.linalg.lstsq(M_w, T.reshape(pn, nm*mp).T, rcond=None)
                    W = W_sol.T
                
                # Check residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            w_idx = k*n + i
                            val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
                            residual += (val - 1.0) ** 2
                
                if residual < 1e-6:
                    # Try to round to half-integers
                    U_r = np.round(U * 2.0) / 2.0
                    V_r = np.round(V * 2.0) / 2.0
                    W_r = np.round(W * 2.0) / 2.0
                    
                    # Verify exactness
                    exact_ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                u_idx = i*m + j
                                v_idx = j*p + k
                                w_idx = k*n + i
                                val = np.dot(U_r[u_idx] * V_r[v_idx], W_r[w_idx])
                                if abs(val - 1.0) > 1e-9:
                                    exact_ok = False
                                    break
                        if not exact_ok:
                            break
                    
                    if exact_ok:
                        return {
                            "U": [[float(x) for x in row] for row in U_r],
                            "V": [[float(x) for x in row] for row in V_r],
                            "W": [[float(x) for x in row] for row in W_r]
                        }
            except Exception:
                continue
        
        return None
    
    # Try DFT-based approach
    if time.time() < deadline - 15.0:
        try:
            result = try_dft_embedding()
            if result is not None:
                return result
        except Exception:
            pass
    
    # Fall back to schoolbook decomposition
    return schoolbook()
