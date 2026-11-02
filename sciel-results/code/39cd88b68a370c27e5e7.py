# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Schoolbook
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    best_rank = rank_schoolbook
    
    # Target ranks to try
    # Lower bound is 18.
    # Strassen-like for 3x3 might be 19.
    # For 3x3x6, 2*19 = 38.
    # Try 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 19, 18.
    
    target_ranks = [38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 19, 18]
    
    # Precompute tensor T
    T = np.ones((n, m, p))
    
    def verify_and_round(U_r, V_r, W_r, tol=1e-5):
        # Check if close to integer/half-integer
        # Round to nearest half
        U_rounded = np.round(U_r * 2) / 2
        V_rounded = np.round(V_r * 2) / 2
        W_rounded = np.round(W_r * 2) / 2
        
        # Check exactness with rounded values
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(U_r.shape[1]):
                        val += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                    err += (T[i,j,k] - val) ** 2
        if np.sqrt(err) < 1e-9:
            return True, U_rounded, V_rounded, W_rounded
        return False, None, None, None

    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Try a few random initializations
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Initialize
            U_r = np.random.randn(n * m, target_rank) * 0.1
            V_r = np.random.randn(m * p, target_rank) * 0.1
            W_r = np.random.randn(p * n, target_rank) * 0.1
            
            # Reshape for einsum
            U_3d = U_r.reshape(n, m, target_rank)
            V_3d = V_r.reshape(m, p, target_rank)
            W_3d = W_r.reshape(p, n, target_rank)
            
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U_3d)
            vV = np.zeros_like(V_3d)
            vW = np.zeros_like(W_3d)
            
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                    
                # Compute Error
                # E_ijk = T_ijk - sum_c U_ijc V_jkc W_kic
                prod = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
                E = T - prod
                
                # Compute Gradients
                dU = np.einsum('ijk,jkc,kic->ijc', E, V_3d, W_3d)
                dV = np.einsum('ijk,ijc,kic->jkc', E, U_3d, W_3d)
                dW = np.einsum('ijk,ijc,jkc->kic', E, U_3d, V_3d)
                
                # Update with Momentum
                vU = momentum * vU - lr * dU
                vV = momentum * vV - lr * dV
                vW = momentum * vW - lr * dW
                
                U_3d += vU
                V_3d += vV
                W_3d += vW
                
                # Check convergence
                if np.max(np.abs(E)) < 1e-6:
                    break
            
            # Check if solution is valid and can be rounded
            valid, U_rnd, V_rnd, W_rnd = verify_and_round(U_3d.reshape(n*m, target_rank), 
                                                          V_3d.reshape(m*p, target_rank), 
                                                          W_3d.reshape(p*n, target_rank))
            if valid:
                best_result = {'U': U_rnd.tolist(), 'V': V_rnd.tolist(), 'W': W_rnd.tolist()}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
