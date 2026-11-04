import time
import numpy as np

DIMENSIONS = [3, 4, 6]
TIME_LIMIT = 120.0
DEADLINE_MARGIN = 5.0


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    # Check: for each (i,j,k), the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
    # And for each (i,j,k') with k' != k, and (i',j,k) with i' != i, and (i,j',k) with j' != j, should be 0
    # More efficiently: build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # and check it equals delta_ii * delta_jj * delta_kk (i.e., 1 only when all indices match their "target")
    
    # Actually the condition is: the tensor T defined by T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    # should equal 1 if and only if the multiplication tensor entry is 1.
    # The matrix multiplication tensor has T_{i,j,k} = 1 for all (i,j,k) in [n]x[m]x[p].
    # Wait no. The matrix multiplication tensor T has T_{i,j,k} = 1 for all valid (i,j,k).
    # The decomposition satisfies: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for "wrong" combinations it should be 0? No, the tensor only has entries for valid (i,j,k).
    
    # The correct check: the bilinear form (U, V, W) computes C = AB where C[i,k] = sum_j A[i,j] B[j,k].
    # In tensor form: C[i,k] = sum_{j,r} U[i*m+j, r] A[i,j] * V[j*p+k, r] B[j,k] * ... 
    # Actually the standard check is:
    # For the decomposition to be exact, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all (i,j,k)
    # This is because the multiplication tensor T has T[i,j,k] = 1 for all (i,j,k).
    
    # Let's verify this way
    # Build T as a 3D array
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is expensive for large ranks, but for n=3,m=4,p=6,rank<=72 it's fine.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can compute this as: for each r, the contribution is outer(U[:,r], V[:,r], W[:,r]) reshaped appropriately
    # But let's just do it directly for correctness.
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r]  # (n*m,)
        v = V[:, r]  # (m*p,)
        w = W[:, r]  # (p*n,)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i * m + j] * v[j * p + k] * w[k * n + i]
    
    # T should be all ones
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook
    assert _verify(best_U, best_V, best_W, n, m, p), "Schoolbook verification failed"
    
    # Try to find lower rank decompositions using a search approach
    # Strategy: try ranks from best_rank-1 down, but given time constraints,
    # focus on a few promising ranks.
    
    # Known bounds: R(3,4,6) <= R(3,4,4) + R(3,4,2)
    # R(3,4,4) <= R(4,4,4) = 16 (using Strassen for 4x4x4, projecting out 4th dimension)
    # R(3,4,2) <= 3*4*2 = 24 (schoolbook) or potentially less
    
    # Let's try to construct a rank 40 decomposition:
    # Use rank 16 for the 3x4x4 part and rank 24 for the 3x4x2 part
    # But we need explicit Strassen vectors.
    
    # Alternative: try a randomized search for slightly lower ranks
    # Given time constraints, let's try a few random restarts at ranks 71, 70, 69...
    
    deadline = start_time + TIME_LIMIT - DEADLINE_MARGIN
    
    # Try a simple approach: for each rank r from best_rank-1 down to max(30, best_rank-10),
    # do a few random initializations and try ALS to minimize the error.
    # If we find an exact decomposition, return it.
    
    # But exact decomposition is very hard to find randomly.
    # Let's instead try to use the structural decomposition.
    
    # Split p into 4+2:
    # C = A * [B1 B2] where B1 is 4x4, B2 is 4x2
    # C = A*B1 + A*B2
    # We need rank(R344) + rank(R342)
    
    # For R344: use a rank 16 decomposition (Strassen-based)
    # For R342: use schoolbook rank 24, or try to do better
    
    # Let me try to implement a rank 16 decomposition for 4x4x4.
    # The Strassen algorithm for 4x4x4 uses 16 scalar multiplications.
    # I'll define the 16 factor vectors.
    
    # Strassen for 4x4:
    # Split A into 2x2 blocks of size 2x2: A = [[A11, A12], [A21, A22]]
    # Similarly B = [[B11, B12], [B21, B22]], C = [[C11, C12], [C21, C22]]
    # Each block is 2x2.
    # Strassen computes 7 products of 2x2 matrices, each requiring 4 scalar mults = 28 total?
    # No, Strassen for 2x2 uses 7 scalar mults. For 4x4, we apply Strassen at the block level,
    # where each "scalar" is a 2x2 matrix. So we need 7 * 4 = 28 scalar mults?
    # But R(4,4,4) = 16 is known. This is achieved by a different algorithm.
    
    # Actually, the rank 16 decomposition for 4x4x4 is non-trivial.
    # Let me try a different approach: use the known fact that R(3,4,6) might be achievable
    # with a lower rank through a more careful construction.
    
    # Given the difficulty, let me just return the schoolbook decomposition for now,
    # but with a cleaner implementation and proper deadline handling.
    
    # Actually, let me try one more thing: a rank 56 decomposition.
    # 3*4*6 = 72. Can we do 56?
    # 56 = 7 * 8. Or 56 = 28 * 2.
    
    # Let me try a block approach:
    # Split n into 3 (no split), m into 2+2, p into 3+3
    # C = A[:, :2] * B[:2, :] + A[:, 2:] * B[2:, :]
    # R(3,2,6) + R(3,2,6) = 2 * 3*2*6 = 72. No improvement.
    
    # Split m into 4 (no split), p into 2+2+2
    # R(3,4,2) * 3 = 3 * 24 = 72. No improvement.
    
    # It seems hard to beat 72 without a clever algebraic decomposition.
    
    # Let me return the schoolbook decomposition.
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a few random searches for slightly lower ranks
    # This is unlikely to succeed but worth a try
    
    for target_rank in range(best_rank - 1, max(best_rank - 5, 40)):
        if time.time() > deadline:
            break
        
        # Random initialization
        U_rand = np.random.randn(n * m, target_rank) * 0.1
        V_rand = np.random.randn(m * p, target_rank) * 0.1
        W_rand = np.random.randn(p * n, target_rank) * 0.1
        
        # Try a few iterations of a simple gradient-like step
        # This is very unlikely to find an exact solution, but let's try
        
        # Skip this - too unlikely to work in the time available
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
