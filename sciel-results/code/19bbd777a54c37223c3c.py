"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Build the target tensor T[a, b, c] where:
    # a = i*m + j (index into A, i in [0,n), j in [0,m))
    # b = j*p + k (index into B, j in [0,m), k in [0,p))
    # c = k*n + i (index into C, using trace convention W[k*n+i])
    # T[a,b,c] = 1 if the indices are consistent, else 0
    # For a given (i,j,k): a=i*m+j, b=j*p+k, c=k*n+i
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """Verify that sum_r U[:,r] outer V[:,r] outer W[:,r] == T"""
        recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(recon, T, atol=1e-10)
    
    # Trivial decomposition
    U = np.zeros((n*m, trivial_rank), dtype=np.float64)
    V = np.zeros((m*p, trivial_rank), dtype=np.float64)
    W = np.zeros((p*n, trivial_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    best_rank = trivial_rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # For M_{n,m,p}, the rank is at most n*m*p but can be lower.
    # We try to find a lower rank decomposition by exploiting the structure.
    
    # A key insight: For each intermediate index j, we can potentially share
    # computations across different i and k values.
    
    # Let's try a structured decomposition approach:
    # For fixed j, the contribution to C[i,k] is A[i,j] * B[j,k]
    # We can try to group these terms more efficiently.
    
    # One approach: use the fact that for small matrices, we can sometimes
    # find explicit decompositions with lower rank.
    
    # For now, let's try a local search approach that starts from the trivial
    # decomposition and tries to reduce the rank by merging columns.
    
    def can_merge_columns(U, V, W, r1, r2):
        """Check if columns r1 and r2 can be merged into a single column"""
        # The combined tensor is:
        T_combined = np.einsum('a,b,c->abc', U[:, r1], V[:, r1], W[:, r1]) + \
                     np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
        
        # Check if this is a rank-1 tensor
        # A tensor is rank-1 if it can be written as u outer v outer w
        
        # Reshape to a matrix: (n*m) x (m*p * p*n)
        mat = T_combined.reshape(n*m, -1)
        
        # If rank-1, the SVD should have only one non-zero singular value
        try:
            singular_values = np.linalg.svd(mat, compute_uv=False)
            if len(singular_values) > 0 and singular_values[0] > 1e-10:
                if len(singular_values) > 1 and singular_values[1] > 1e-8 * singular_values[0]:
                    return False
            if singular_values[0] < 1e-10:
                return True  # Zero tensor, trivially rank-1
            return True
        except:
            return False
    
    # Try to merge pairs of columns
    current_rank = trivial_rank
    U_cur = best_U.copy()
    V_cur = best_V.copy()
    W_cur = best_W.copy()
    
    merged_any = True
    while merged_any and time.time() < deadline - 10:
        merged_any = False
        for r1 in range(current_rank):
            if time.time() > deadline - 10:
                break
            for r2 in range(r1 + 1, current_rank):
                if time.time() > deadline - 10:
                    break
                if can_merge_columns(U_cur, V_cur, W_cur, r1, r2):
                    # Try to merge them
                    # Remove column r2
                    mask = [c for c in range(current_rank) if c != r2]
                    U_cur = U_cur[:, mask]
                    V_cur = V_cur[:, mask]
                    W_cur = W_cur[:, mask]
                    current_rank -= 1
                    merged_any = True
                    break
            if merged_any:
                break
        
        # Verify the decomposition still works
        if verify_decomposition(U_cur, V_cur, W_cur, current_rank):
            if current_rank < best_rank:
                best_rank = current_rank
                best_U = U_cur.copy()
                best_V = V_cur.copy()
                best_W = W_cur.copy()
        else:
            # Roll back
            U_cur = best_U.copy()
            V_cur = best_V.copy()
            W_cur = best_W.copy()
            current_rank = best_rank
            merged_any = False
    
    # Convert to lists for output
    U_out = best_U[:, :best_rank].tolist()
    V_out = best_V[:, :best_rank].tolist()
    W_out = best_W[:, :best_rank].tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
