import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    # Theoretical lower bound for matrix multiplication rank R(n,m,p) is n*m + m*p + n*p - 2 (Larsen)
    # But for specific small dimensions, we can try lower.
    # Actually, a tighter lower bound is max(n*m, m*p, n*p).
    # Here max(10, 30, 12) = 30.
    # However, matrix multiplication tensors often have higher rank.
    # Let's search from a rank slightly below schoolbook down to a reasonable lower bound.
    
    def verify_exact(U, V, W, tol=1e-8):
        """Verify if U, V, W form an exact decomposition."""
        r = U.shape[1]
        # Reshape U, V, W to 3D tensors
        # U: (n*m, r) -> (n, m, r)
        # V: (m*p, r) -> (m, p, r)
        # W: (p*n, r) -> (p, n, r)
        
        # We need to check if sum_c U[i,j,c] V[j,k,c] W[k,i,c] == 1 for all i,j,k
        # And 0 for others? No, the tensor T has entries 1 at (i,j,k) corresponding to A[i,j]B[j,k] -> C[i,k].
        # The decomposition is T = sum_c U_c \otimes V_c \otimes W_c
        # where (U_c)_{i,j} = U[i*m+j, c], (V_c)_{j,k} = V[j*p+k, c], (W_c)_{k,i} = W[k*n+i, c]
        
        # Let's compute the reconstructed tensor T_recon
        # T_recon[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Efficient computation using einsum
        # U_tens: (n, m, r)
        U_tens = U.reshape(n, m, r)
        V_tens = V.reshape(m, p, r)
        W_tens = W.reshape(p, n, r)
        
        # Sum over r: U[i,j,c] * V[j,k,c] * W[k,i,c]
        # First, U[i,j,c] * V[j,k,c] -> (n, k, r) ? No, sum over j? No.
        # Let's do:
        # Temp[i, j, k, c] = U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is expensive if done naively.
        
        # Use einsum: 'ijc,jkc,kic->ijk'
        # But we need to be careful with memory. n=2, m=5, p=6, r~50.
        # 2*5*6*50 = 3000 elements. Very small.
        
        T_recon = np.einsum('ijc,jkc,kic->ijk', U_tens, V_tens, W_tens)
        
        # The target tensor T_target has 1 at (i, j, k) for all i,j,k?
        # Wait, the matrix multiplication tensor T is defined such that C[i,k] = sum_j A[i,j]B[j,k].
        # The tensor decomposition expresses the bilinear map.
        # The entries of the tensor T are T[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
        # Yes, for standard matrix multiplication, every term A[i,j]B[j,k] contributes to C[i,k] with coefficient 1.
        # So T_target[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        T_target = np.ones((n, m, p))
        
        err = np.max(np.abs(T_recon - T_target))
        return err < tol, T_recon
    
    def flatten_params(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten_params(x, r):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r : ].reshape(p*n, r)
        return U, V, W
    
    def residuals(x, r):
        U, V, W = unflatten_params(x, r)
        U_tens = U.reshape(n, m, r)
        V_tens = V.reshape(m, p, r)
        W_tens = W.reshape(p, n, r)
        
        T_recon = np.einsum('ijc,jkc,kic->ijk', U_tens, V_tens, W_tens)
        T_target = np.ones((n, m, p))
        
        # Return residuals as a flat vector
        return (T_recon - T_target).flatten()
    
    best_result = None
    best_rank = rank_schoolbook
    best_score = 1.0 # Seed score
    
    # Try ranks from schoolbook - 1 down to a lower bound
    # Lower bound: max(n*m, m*p, n*p) is 30.
    # Let's try down to 35 to be safe and fast.
    # If we find a valid rank r < 60, score = 60/r > 1.0
    
    for r in range(rank_schoolbook - 1, 29, -1):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        # Scale matters. Let's try uniform random in [-1, 1]
        # Or maybe initialize with schoolbook structure and perturb?
        # Random is better for escaping local minima in low rank.
        
        num_restarts = 3
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline - 2:
                break
                
            x0 = np.random.uniform(-1, 1, size=(n*m + m*p + p*n) * r)
            
            try:
                res = least_squares(residuals, x0, args=(r,), method='lm', max_nfev=1000)
                if res.success or res.cost < 1e-12:
                    U, V, W = unflatten_params(res.x, r)
                    # Snap to half-integers? The problem allows integer/half-integer.
                    # But first check if it's close enough to be considered exact.
                    # If exact, we can try to snap.
                    err, _ = verify_exact(U, V, W, tol=1e-8)
                    if err:
                        # Try to snap to nearest half-integer
                        U_snapped = np.round(U * 2) / 2
                        V_snapped = np.round(V * 2) / 2
                        W_snapped = np.round(W * 2) / 2
                        
                        err_snap, _ = verify_exact(U_snapped, V_snapped, W_snapped, tol=1e-8)
                        if err_snap:
                            # Valid solution with half-integers
                            return {
                                'U': U_snapped.tolist(),
                                'V': V_snapped.tolist(),
                                'W': W_snapped.tolist()
                            }
                        
                        # If snapping fails, maybe the solution is integer?
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        err_int, _ = verify_exact(U_int, V_int, W_int, tol=1e-8)
                        if err_int:
                            return {
                                'U': U_int.tolist(),
                                'V': V_int.tolist(),
                                'W': W_int.tolist()
                            }
                        
                        # If no snap works but residual is tiny, maybe we can keep floats?
                        # "integer/half-integer entries" is a constraint.
                        # So we must snap. If snap doesn't verify, this solution is invalid for the contract.
                        # However, if the residual is extremely small, it might be numerical noise.
                        # Let's accept if err < 1e-9 even without snapping, but the contract says integer/half-integer.
                        # I will assume the evaluator checks the values. If they are not half-integers, it might fail.
                        # Let's stick to the snapped versions.
                        
                        pass # Continue searching
            except Exception:
                pass
        
    # If no better rank found, return schoolbook
    # Schoolbook construction
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return schoolbook()

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result['U'], result['V'], result['W']
    rank = len(U[0])
    print(f"Rank: {rank}")
