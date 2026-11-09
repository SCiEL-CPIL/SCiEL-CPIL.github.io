"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U * V * W exactly decomposes the matrix multiplication tensor."""
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # and for each (i,j,k) where not matching, the sum == 0
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # Efficient check using einsum
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape for einsum
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T = np.einsum('ijc,kjc,klc->ikl', U3, V3, W3)
    
    # Expected: T[i,k,j] should be 1 if we're computing (i,j) dot (j,k)
    # Actually the convention is: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor entry for (i, j, k) should be 1 (it multiplies A[i,j]*B[j,k])
    # So T[i,j,k] should equal 1 for all i,j,k? No wait.
    
    # Let me reconsider. The decomposition gives:
    # C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
    #         = sum_c sum_j sum_l U[i*m+j,c]*A[i,j] * V[l*p+k,c]*B[l,k]
    #         = sum_j sum_l A[i,j] B[l,k] (sum_c U[i*m+j,c]*V[l*p+k,c]*W[k*n+i,c])
    
    # For this to equal C[i,k] = sum_j A[i,j]*B[j,k], we need:
    # sum_c U[i*m+j,c]*V[l*p+k,c]*W[k*n+i,c] = delta_{j,l} for all i,k
    # i.e., it equals 1 if j==l, 0 if j!=l
    
    # So T[i,j,l,k] ... wait, let me re-derive with proper indices.
    # T[i,j,l,k] = sum_c U[i*m+j,c] * V[l*p+k,c] * W[k*n+i,c]
    # should equal delta_{j,l}
    
    # Using einsum: U3[i,j,c] * V3[l,k,c] * W3[k,i,c] -> T[i,j,l,k]
    T4 = np.einsum('ijc,lkc,kic->ijlk', U3, V3, W3)
    
    # Expected: T4[i,j,l,k] = 1 if j==l else 0
    expected = np.zeros((n, m, m, p))
    for j in range(m):
        expected[:, j, j, :] = 1.0
    
    diff = np.abs(T4 - expected)
    return diff.max() < tol

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0  # Safety margin
    
    # Schoolbook decomposition (baseline rank = n*m*p = 84)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find a lower-rank decomposition using random initialization + gradient descent
    # or structured search. For 3x4x7, the tensor rank is unknown but likely < 84.
    # Known bounds: rank(M_{3,4,7}) <= 3*4*7 = 84, and lower bounds exist.
    # Let's try rank values from 60 to 83 and see if we can find valid decompositions.
    
    def try_rank(r):
        """Try to find a rank-r decomposition via random init + local search."""
        if r >= best_rank:
            return None
        # Random initialization with small values
        np.random.seed(42 + r)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Use gradient descent to minimize ||T4 - expected||^2
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        
        expected = np.zeros((n, m, m, p))
        for j in range(m):
            expected[:, j, j, :] = 1.0
        
        lr = 0.01
        for it in range(2000):
            if time.time() > deadline - 10:
                break
            # Compute T4
            T4 = np.einsum('ijc,lkc,kic->ijlk', U3, V3, W3)
            loss = np.sum((T4 - expected) ** 2)
            if loss < 1e-6:
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    return U, V, W, r
            
            # Gradients
            dT4 = 2 * (T4 - expected)
            dU3 = np.einsum('ijlk,lkc,kic->ijc', dT4, V3, W3)
            dV3 = np.einsum('ijlk,ijc,kic->lkc', dT4, U3, W3)
            dW3 = np.einsum('ijlk,ijc,lkc->kic', dT4, U3, V3)
            
            U3 -= lr * dU3
            V3 -= lr * dV3
            W3 -= lr * dW3
            
        return None
    
    # Try a few ranks
    for r in [83, 82, 81, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60]:
        if time.time() > deadline - 20:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
