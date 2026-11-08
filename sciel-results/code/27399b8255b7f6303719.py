import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1 for all valid i,j,k
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # Initialize schoolbook
    U0 = np.zeros((n * m, best_rank))
    V0 = np.zeros((m * p, best_rank))
    W0 = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U0, V0, W0
    
    # Try Kronecker product decomposition:
    # 3x4x8 = (3x2x4) ⊗ (1x2x2)
    # Sub-problem 1: 3x2x4 with Strassen rank 7
    # Sub-problem 2: 1x2x2 with schoolbook rank 2
    # Total rank: 7 * 2 = 14
    
    def strassen_3x2x4():
        """Return U, V, W for 3x2x4 matrix multiplication using Strassen-like rank 7"""
        n1, m1, p1 = 3, 2, 4
        rank1 = 7
        U1 = np.zeros((n1 * m1, rank1))
        V1 = np.zeros((m1 * p1, rank1))
        W1 = np.zeros((p1 * n1, rank1))
        
        # For 3x2x4, we can use a structured approach
        # Let's use the standard Strassen algorithm adapted for this size
        # Actually, for small sizes, let's use a direct construction
        
        # The tensor T1[i*m1+j, j*p1+k, k*n1+i] = 1
        # We need rank-7 decomposition
        
        # Let's try a random search for rank 7
        best_err = np.inf
        best_U1, best_V1, best_W1 = None, None, None
        
        T1 = np.zeros((n1 * m1, m1 * p1, p1 * n1), dtype=float)
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    T1[i * m1 + j, j * p1 + k, k * n1 + i] = 1.0
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() - start_time > 130:
                break
                
            U_rand = np.random.randn(n1 * m1, rank1)
            V_rand = np.random.randn(m1 * p1, rank1)
            W_rand = np.random.randn(p1 * n1, rank1)
            
            # Gradient descent
            lr = 0.05
            for it in range(500):
                if time.time() - start_time > 130:
                    break
                    
                recon = np.zeros_like(T1)
                for r in range(rank1):
                    recon += np.einsum('i,j,k->ijk', U_rand[:, r], V_rand[:, r], W_rand[:, r])
                err = np.sum((T1 - recon) ** 2)
                
                if err < best_err:
                    best_err = err
                    best_U1, best_V1, best_W1 = U_rand.copy(), V_rand.copy(), W_rand.copy()
                    if best_err < 1e-10:
                        return best_U1, best_V1, best_W1
                
                # Compute gradients
                grad_U = np.zeros_like(U_rand)
                grad_V = np.zeros_like(V_rand)
                grad_W = np.zeros_like(W_rand)
                
                res = T1 - recon
                for r in range(rank1):
                    u = U_rand[:, r]
                    v = V_rand[:, r]
                    w = W_rand[:, r]
                    
                    grad_u = np.einsum('ijk,j,k->i', res, v, w)
                    grad_v = np.einsum('ijk,i,k->j', res, u, w)
                    grad_w = np.einsum('ijk,i,j->k', res, u, v)
                    
                    grad_U[:, r] = grad_u
                    grad_V[:, r] = grad_v
                    grad_W[:, r] = grad_w
                
                U_rand -= lr * grad_U
                V_rand -= lr * grad_V
                W_rand -= lr * grad_W
                
                lr *= 0.995
        
        if best_err < 1e-8:
            return best_U1, best_V1, best_W1
        return None
    
    def schoolbook_1x2x2():
        """Return U, V, W for 1x2x2 matrix multiplication (rank 2)"""
        n2, m2, p2 = 1, 2, 2
        rank2 = 2
        U2 = np.zeros((n2 * m2, rank2))
        V2 = np.zeros((m2 * p2, rank2))
        W2 = np.zeros((p2 * n2, rank2))
        
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U2[i * m2 + j, k] = 1.0 if i == 0 else 0.0
                    V2[j * p2 + k, k] = 1.0
                    W2[k * n2 + i, k] = 1.0
        
        return U2, V2, W2
    
    # Try Kronecker decomposition
    result1 = strassen_3x2x4()
    if result1 is not None:
        U1, V1, W1 = result1
        U2, V2, W2 = schoolbook_1x2x2()
        
        rank_kron = U1.shape[1] * U2.shape[1]
        
        # Kronecker product of decompositions
        U_kron = np.zeros((n * m, rank_kron))
        V_kron = np.zeros((m * p, rank_kron))
        W_kron = np.zeros((p * n, rank_kron))
        
        # Map indices: 
        # Original: i in [0,3), j in [0,4), k in [0,8)
        # Factor 1: i1 in [0,3), j1 in [0,2), k1 in [0,4)
        # Factor 2: i2 in [0,1), j2 in [0,2), k2 in [0,2)
        # j = j1 * 2 + j2, k = k1 * 2 + k2
        
        col = 0
        for r1 in range(U1.shape[1]):
            for r2 in range(U2.shape[1]):
                for i in range(n):
                    for j in range(m):
                        j1 = j // 2
                        j2 = j % 2
                        U_kron[i * m + j, col] = U1[i * 2 + j1, r1] * U2[0 * 2 + j2, r2]
                
                for j in range(m):
                    for k in range(p):
                        j1 = j // 2
                        j2 = j % 2
                        k1 = k // 2
                        k2 = k % 2
                        V_kron[j * p + k, col] = V1[j1 * 4 + k1, r1] * V2[j2 * 2 + k2, r2]
                
                for k in range(p):
                    for i in range(n):
                        k1 = k // 2
                        k2 = k % 2
                        W_kron[k * n + i, col] = W1[k1 * 3 + i, r1] * W2[k2 * 1 + i, r2]
                
                col += 1
        
        # Verify
        recon = np.zeros_like(T)
        for r in range(rank_kron):
            recon += np.einsum('i,j,k->ijk', U_kron[:, r], V_kron[:, r], W_kron[:, r])
        
        if np.max(np.abs(T - recon)) < 1e-8:
            best_rank = rank_kron
            best_U, best_V, best_W = U_kron, V_kron, W_kron
    
    # Convert to lists with rounded values
    def to_list(arr):
        # Round to nearest 0.5 or integer if close
        rounded = np.round(arr * 2) / 2
        # Check if it's close to the original
        if np.max(np.abs(arr - rounded)) < 1e-6:
            return rounded.tolist()
        else:
            return arr.tolist()
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
