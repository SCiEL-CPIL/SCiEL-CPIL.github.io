import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = DIMENSIONS
    
    def build_rank50():
        rank = 50
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        col = 0
        
        def emit_strassen(i0, j0, k0):
            nonlocal col
            U[i0*m + j0, col] = 1.0
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[j0*p + k0, col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            U[(i0+1)*m + j0, col] = 1.0
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[j0*p + k0, col] = 1.0
            W[k0*n + (i0+1), col] = 1.0
            W[(k0+1)*n + (i0+1), col] = -1.0
            col += 1
            U[i0*m + j0, col] = 1.0
            V[j0*p + (k0+1), col] = 1.0
            V[(j0+1)*p + (k0+1), col] = -1.0
            W[(k0+1)*n + i0, col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[(j0+1)*p + k0, col] = 1.0
            V[j0*p + k0, col] = -1.0
            W[k0*n + (i0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1
            U[i0*m + j0, col] = 1.0
            U[i0*m + (j0+1), col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = -1.0
            W[(k0+1)*n + i0, col] = 1.0
            col += 1
            U[(i0+1)*m + j0, col] = 1.0
            U[i0*m + j0, col] = -1.0
            V[j0*p + k0, col] = 1.0
            V[j0*p + (k0+1), col] = 1.0
            W[(k0+1)*n + (i0+1), col] = 1.0
            col += 1
            U[i0*m + (j0+1), col] = 1.0
            U[(i0+1)*m + (j0+1), col] = -1.0
            V[(j0+1)*p + k0, col] = 1.0
            V[(j0+1)*p + (k0+1), col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1

        def emit_schoolbook_2x2x1(i0, j0, k0):
            nonlocal col
            U[i0*m + j0, col] = 1.0
            V[j0*p + k0, col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1
            U[i0*m + (j0+1), col] = 1.0
            V[(j0+1)*p + k0, col] = 1.0
            W[k0*n + i0, col] = 1.0
            col += 1
            U[(i0+1)*m + j0, col] = 1.0
            V[j0*p + k0, col] = 1.0
            W[k0*n + (i0+1), col] = 1.0
            col += 1
            U[(i0+1)*m + (j0+1), col] = 1.0
            V[(j0+1)*p + k0, col] = 1.0
            W[k0*n + (i0+1), col] = 1.0
            col += 1

        emit_strassen(0, 0, 0)
        emit_strassen(0, 0, 2)
        emit_strassen(0, 0, 4)
        emit_schoolbook_2x2x1(0, 0, 6)
        emit_strassen(0, 2, 0)
        emit_strassen(0, 2, 2)
        emit_strassen(0, 2, 4)
        emit_schoolbook_2x2x1(0, 2, 6)
        
        return U, V, W

    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def reduce_rank(U, V, W):
        """Try to reduce the rank by finding linear dependencies."""
        nU, nV, nW = n*m, m*p, p*n
        rank = U.shape[1]
        
        # Build the 3136 x rank matrix of flattened rank-1 tensors
        flat = np.zeros((nU * nV * nW, rank))
        for r in range(rank):
            tensor_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            flat[:, r] = tensor_r.flatten()
        
        # Find the rank
        s = np.linalg.svd(flat, compute_uv=False)
        tol = 1e-10
        num_sig = int(np.sum(s > tol))
        
        if num_sig >= rank:
            return U, V, W, rank
        
        # There are linear dependencies
        # Find the null space
        # Use the right singular vectors corresponding to small singular values
        U_svd, s, Vt = np.linalg.svd(flat, full_matrices=False)
        
        # The last (rank - num_sig) columns of Vt.T span the null space
        null_dim = rank - num_sig
        if null_dim > 0:
            null_vecs = Vt[num_sig:, :].T  # rank x null_dim
            
            # For each null vector, we can eliminate one term
            # A null vector c means: sum_r c[r] * T_r = 0
            # We can use this to express one T_r in terms of the others
            # But this doesn't directly give a lower-rank decomposition
            # We need to find a set of rank-1 tensors that sum to T
            # This is a non-trivial optimization problem
            
            # For now, just return the original
            pass
        
        return U, V, W, rank

    U, V, W = build_rank50()
    
    if not verify(U, V, W):
        return None
    
    # Try to reduce rank
    if time.time() < DEADLINE:
        U, V, W, rank = reduce_rank(U, V, W)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
