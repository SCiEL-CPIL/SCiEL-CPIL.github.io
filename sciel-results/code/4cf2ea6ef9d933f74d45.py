"""Find rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
N, M, P = DIMENSIONS

def _matmul_tensor():
    """Return the matrix multiplication tensor T[i, j, k] = 1 if the (i,j) of A and (j,k) of B multiply to C[i,k].
    Actually the tensor for matmul: T[i, j, k] = 1 (for all i,j,k) in the standard basis.
    Wait, the matrix multiplication tensor T has entries T[i, j, k] = 1 for all i,j,k,
    and the multiplication is: C[i,k] = sum_j A[i,j] * B[j,k].
    So the tensor T is the identity in the sense that T[i,j,k] = 1 for all i,j,k.
    Actually no - the tensor is defined such that <T, A ⊗ B> = C where C[i,k] = sum_j A[i,j]B[j,k].
    So T[i,j,k] = 1 for all i,j,k.
    """
    # The matmul tensor has all ones
    return np.ones((N, M, P))

def _verify(U, V, W, rank):
    """Verify that U, V, W form a valid rank-rank decomposition of the matmul tensor.
    U: (N*M, rank), V: (M*P, rank), W: (P*N, rank)
    The decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = np.dot(U[i*M+j], V[j*P+k] * W[k*N+i])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _verify_fast(U, V, W, rank):
    """Faster verification using numpy broadcasting."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # U: (N*M, rank), V: (M*P, rank), W: (P*N, rank)
    # For each c, the contribution to T[i,j,k] is U[i*M+j,c]*V[j*P+k,c]*W[k*N+i,c]
    # T[i,j,k] = sum_c U[i*M+j,c]*V[j*P+k,c]*W[k*N+i,c]
    
    # Reshape U to (N, M, rank), V to (M, P, rank), W to (P, N, rank)
    U_r = U.reshape(N, M, rank)
    V_r = V.reshape(M, P, rank)
    W_r = W.reshape(P, N, rank)
    
    # For each c: U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # Sum over c
    # Use einsum: U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c] summed over c
    # = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    return np.allclose(T, np.ones((N, M, P)), atol=1e-6)

def _schoolbook():
    """Return the schoolbook rank N*M*P decomposition."""
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist(), rank

def _try_lattice_based(rank, max_iters=5000):
    """Try to find a decomposition of given rank using random initialization and local search."""
    if rank < N * M:  # Lower bound
        return None
    
    deadline = time.time() + 120
    best_result = None
    
    # Try several random initializations
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        U = np.random.randn(N * M, rank)
        V = np.random.randn(M * P, rank)
        W = np.random.randn(P * N, rank)
        
        # Normalize
        U_r = U.reshape(N, M, rank)
        V_r = V.reshape(M, P, rank)
        W_r = W.reshape(P, N, rank)
        
        # Gradient descent to minimize ||T_computed - T_target||^2
        T_target = np.ones((N, M, P))
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Compute T
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            err = T - T_target
            loss = np.sum(err ** 2)
            
            if loss < 1e-10:
                # Found a valid decomposition
                return U.tolist(), V.tolist(), W.tolist(), rank
            
            # Compute gradients
            # dL/dU_r[i,j,c] = 2 * err[i,j,k] * V_r[j,k,c] * W_r[k,i,c] summed over k
            # dL/dV_r[j,k,c] = 2 * err[i,j,k] * U_r[i,j,c] * W_r[k,i,c] summed over i
            # dL/dW_r[k,i,c] = 2 * err[i,j,k] * U_r[i,j,c] * V_r[j,k,c] summed over j
            
            dU = np.einsum('ijk,jkc,kic->ijc', err, V_r, W_r) * 2
            dV = np.einsum('ijk,ijc,kic->jkc', err, U_r, W_r) * 2
            dW = np.einsum('ijk,ijc,jkc->kic', err, U_r, V_r) * 2
            
            U_r -= lr * dU
            V_r -= lr * dV
            W_r -= lr * dW
        
        # Check final result
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        if np.allclose(T, T_target, atol=1e-6):
            return U_r.reshape(N*M, rank).tolist(), V_r.reshape(M*P, rank).tolist(), W_r.reshape(P*N, rank).tolist(), rank
    
    return None

def _try_bipartite_like(rank, max_iters=10000):
    """Try a structured decomposition inspired by known low-rank matmul results.
    For 3x4x7, we know the tensor rank is at most 84 (schoolbook) and lower bounds exist.
    Known results: rank(3,4,7) <= ? Let's try various ranks.
    """
    deadline = time.time() + 120
    
    # Try to find decomposition for given rank
    if rank < N * M:
        return None
    
    # Use a different parametrization: think of it as finding
    # rank-rank decomposition where each "slice" is a rank-1 tensor
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    
    # Try random with annealing
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(5):
        if time.time() > deadline - 5:
            break
        
        U = np.random.randn(N, M, rank) * 0.1
        V = np.random.randn(M, P, rank) * 0.1
        W = np.random.randn(P, N, rank) * 0.1
        
        T_target = np.ones((N, M, P))
        lr = 0.05
        patience = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T - T_target
            loss = np.sum(err ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                patience = 0
            else:
                patience += 1
            
            if loss < 1e-10:
                return (U.reshape(N*M, rank).tolist(), V.reshape(M*P, rank).tolist(), 
                        W.reshape(P*N, rank).tolist(), rank)
            
            if patience > 500:
                break
            
            # Adaptive learning rate
            if loss > best_loss * 1.1:
                lr *= 0.95
            
            dU = np.einsum('ijk,jkc,kic->ijc', err, V, W) * 2
            dV = np.einsum('ijk,ijc,kic->jkc', err, U, W) * 2
            dW = np.einsum('ijk,ijc,jkc->kic', err, U, V) * 2
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
    
    # Check best
    if best_U is not None:
        T = np.einsum('ijc,jkc,kic->ijk', best_U, best_V, best_W)
        if np.allclose(T, T_target, atol=1e-6):
            return (best_U.reshape(N*M, rank).tolist(), best_V.reshape(M*P, rank).tolist(),
                    best_W.reshape(P*N, rank).tolist(), rank)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook()
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to improve
    # Known lower bound for matmul(3,4,7): at least 3*4 = 12, at least 4*7=28, at least 3*7=21
    # So rank >= max(12, 28, 21) = 28
    # Actually the lower bound is max(n*m, m*p, n*p) = max(12, 28, 21) = 28
    # But there are better lower bounds. For now, try ranks from 28 upward.
    
    # Try a few promising ranks
    ranks_to_try = list(range(28, min(rank, 50)))
    
    # Sort by likelihood (try lower ranks first)
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        
        result = _try_bipartite_like(r, max_iters=3000)
        if result is not None:
            U, V, W, rank = result
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # Found a valid decomposition, stop
    
    # If we found something better, return it
    if best_rank < N * M * P:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Otherwise return schoolbook
    return {"U": U, "V": V, "W": W}
