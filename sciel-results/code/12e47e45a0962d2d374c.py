"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check whether U, V, W exactly decompose the matmul tensor."""
    rank = U.shape[1]
    # The matmul tensor T[i, j, k] = delta (i.e., T[i,j,k]=1 for all i,j,k? No.)
    # Actually, the matrix multiplication tensor is T[i,j,k] such that
    # (A B)[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear map is: (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # Factor form: C[i,k] = sum_c U[i*m+j?, c] ... 
    # Let's use the standard formulation:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # No, that's not right either.
    #
    # Standard tensor decomposition for matmul:
    # C[i,k] = sum_c U[i, c] * V[k, c]  ... no.
    #
    # The correct formulation: The matmul operation is a bilinear map
    # f: R^{n x m} x R^{m x p} -> R^{n x p}
    # f(A, B)[i, k] = sum_j A[i, j] B[j, k]
    #
    # A rank-r factorization is:
    # f(A, B)[i, k] = sum_{c=1}^r (sum_j U[i, j, c] A[i, j]) * (sum_j V[j, k, c] B[j, k])
    # where U is n x m x r, V is m x p x r.
    #
    # In the interface, U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # The output factor W follows the trace convention W[k*n+i, r].
    #
    # So: C[i, k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # But wait, that gives C[i,k] as a product of two sums, which is correct for bilinear.
    # But the W factor? In the standard Strassen-style formulation:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # No, that's not standard either.
    #
    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # The schoolbook has W[k*n+i, col] = 1 for each (i,j,k).
    # And U[i*m+j, col] = 1, V[j*p+k, col] = 1.
    # So the product is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] * A[i,j] B[j,k]
    # = sum_c delta_{c corresponds to (i,j,k)} * 1 * 1 * 1 * A[i,j] B[j,k]
    # = A[i,j] B[j,k] for the specific (i,j,k) term.
    #
    # So the full formula is:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k]) * ... 
    #
    # No. Let me think again. The standard bilinear decomposition is:
    # C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_j V[j,k,c] B[j,k])
    # This is a product of two linear forms, so it's rank 1 in the bilinear sense.
    #
    # But the interface has THREE factors U, V, W. This suggests:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    #
    # For schoolbook: U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for the matching (i,j,k).
    # Then: sum_c 1 * (A[i,j]) * (B[j,k]) = A[i,j]B[j,k] for the specific j.
    # Summing over all c (which correspond to all (i',j',k') triples), we get:
    # sum_{i',j',k'} delta_{i'=i} delta_{j'=j} delta_{k'=k} * A[i,j] B[j,k] = A[i,j]B[j,k]
    # Summed over j: sum_j A[i,j]B[j,k]. Correct!
    #
    # So the formula is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    #
    # Wait, but that has a product of three terms per c, which is trilinear, not bilinear.
    # That can't be right for a bilinear map.
    #
    # I think the correct interpretation is that W is just a scalar weight, and the
    # actual bilinear form is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # This IS bilinear in (A, B) for each c, and the sum over c is also bilinear.
    # So it's a valid bilinear decomposition. The "rank" is the number of such terms.
    
    # Let's verify with random matrices.
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for c in range(rank):
        u_vec = U[:, c]  # (n*m,)
        v_vec = V[:, c]  # (m*p,)
        w_scalar = W[:, c]  # (p*n,)
        
        # sum_j U[i*m+j, c] A[i,j] for each i
        u_mat = u_vec.reshape(n, m)
        v_mat = v_vec.reshape(m, p)
        w_mat = w_scalar.reshape(p, n)
        
        # (sum_j U[i,j,c] A[i,j]) is a scalar per i
        uA = np.einsum('ij,ij->i', u_mat, A)
        vB = np.einsum('jk,jk->k', v_mat, B)
        
        # C[i,k] += W[k*n+i, c] * uA[i] * vB[k]
        # w_mat is (p, n), so w_mat[k, i] = W[k*n+i, c]
        C_comp += np.einsum('ki,i,k->ik', w_mat, uA, vB)
    
    return np.allclose(C_true, C_comp, atol=tol)


def _try_random_rank(n, m, p, target_rank, time_limit=10):
    """Try to find a rank decomposition via random initialization and iterative refinement."""
    start = time.time()
    
    U, V, W = _schoolbook(n, m, p)
    
    # We'll try to find a lower rank decomposition.
    # Start with a random initialization of the target rank.
    for attempt in range(20):
        if time.time() - start > time_limit:
            break
        
        U_r = np.random.randn(n * m, target_rank) * 0.1
        V_r = np.random.randn(m * p, target_rank) * 0.1
        W_r = np.random.randn(p * n, target_rank) * 0.1
        
        # Use a few iterations of gradient-free local search
        best_loss = np.inf
        for it in range(50):
            if time.time() - start > time_limit:
                break
            
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            C_comp = np.zeros((n, p))
            for c in range(target_rank):
                u_mat = U_r[:, c].reshape(n, m)
                v_mat = V_r[:, c].reshape(m, p)
                w_mat = W_r[:, c].reshape(p, n)
                uA = np.einsum('ij,ij->i', u_mat, A)
                vB = np.einsum('jk,jk->k', v_mat, B)
                C_comp += np.einsum('ki,i,k->ik', w_mat, uA, vB)
            
            loss = np.sum((C_true - C_comp) ** 2)
            if loss < best_loss:
                best_loss = loss
                if best_loss < 1e-8:
                    return U_r, V_r, W_r
        
        if best_loss < 1e-8:
            return U_r, V_r, W_r
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_result = (U, V, W)
    
    # Try known bounds for 3x4 x 4x5
    # The tensor rank of 3x4 x 4x5 is at most 60 (schoolbook).
    # Lower bounds: max(n*m, m*p, n*p) = max(12, 20, 15) = 20
    # Actually, the tensor rank is at least the matrix rank of any unfolding.
    # Let's try a few target ranks.
    
    for target_rank in range(20, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = _try_random_rank(n, m, p, target_rank, time_limit=5)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_result = (U_r, V_r, W_r)
                break  # Found a valid decomposition at this rank
    
    U_best, V_best, W_best = best_result
    
    # Convert to lists
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }

# EVOLVE-BLOCK-END
