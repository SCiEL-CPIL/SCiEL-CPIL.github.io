"""Tensor decomposition for 3x3x7 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Build the matrix multiplication tensor
    # T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    # This means: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    
    # The trivial rank is n*m*p = 63
    # We want to find a lower rank decomposition
    
    # Strategy: Use the structure of matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For each output position (i,k), we need sum_j A[i,j]*B[j,k]
    # This is a bilinear form in (A, B)
    
    # Let's try to find a decomposition using the fact that
    # for fixed j, A[:,j] * B[j,:] gives a rank-1 contribution to C
    
    # Actually, let's think about this more carefully.
    # The tensor M has M[i*m+j, j*p+k, k*n+i] = 1
    
    # We can try to find a low-rank decomposition by solving
    # M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Approach: Use the structure to group terms
    # For 3x3x7, let's try rank = 3*7 + 3*3 = 21+9 = 30? No, that's not right.
    
    # Let me try a different approach: use the known result that for m×n×p
    # matrix multiplication, the rank is at most m*n*p but can be lower.
    
    # For 3x3x3, the rank is 21 (not 27). Let's see if we can do better than 63.
    
    # Actually, let me try to construct a decomposition directly.
    
    # Key insight: C[i,k] = sum_j A[i,j]*B[j,k]
    # We can write this as:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # For each i, we have 3 terms. For 3 values of i and 7 values of k,
    # that's 3*3*7 = 63 terms in the trivial decomposition.
    
    # Can we share terms? If we use the same "multiplication" for different (i,k) pairs...
    
    # Let's try: for each j, define a "multiplication" operation that takes
    # A[:,j] and B[j,:] and produces contributions to all C[i,k].
    
    # Actually, let me try a concrete construction:
    # Use rank-1 tensors that exploit the linear structure.
    
    # For the 3x3x3 case, the rank is known to be 21.
    # For 3x3x7, let's estimate: it should be around 3*3*7 - some savings.
    
    # Let me try to use a bilinear decomposition approach.
    
    # The matrix multiplication tensor can be viewed as:
    # M = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    
    # We want to find U, V, W such that M = sum_r u_r ⊗ v_r ⊗ w_r
    
    # Let's try to find a decomposition with rank < 63.
    
    # Approach: Use the fact that for fixed i, the terms A[i,j]*B[j,k] 
    # for different j can be combined.
    
    # Actually, let me just try to use a numerical approach to find a low-rank decomposition.
    
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition
    # We can reshape T into a matrix and use SVD, but that gives a different kind of decomposition.
    
    # For tensor rank, we need a different approach.
    
    # Let's try a greedy approach: iteratively find rank-1 components.
    
    def find_rank1_component(T):
        """Find the best rank-1 approximation of T."""
        # Try all possible rank-1 tensors and find the one with max inner product
        best_score = -np.inf
        best_u, best_v, best_w = None, None, None
        
        # For efficiency, we'll sample some candidate rank-1 tensors
        # A rank-1 tensor is u ⊗ v ⊗ w
        # The inner product with T is sum_{a,b,c} T[a,b,c] * u[a] * v[b] * w[c]
        
        # We can fix u and v, then the optimal w is proportional to 
        # sum_{a,b} T[a,b,:] * u[a] * v[b]
        
        # Let's try a few random initializations
        np.random.seed(42)
        
        for trial in range(50):
            # Random initialization
            u = np.random.randn(n*m)
            v = np.random.randn(m*p)
            w = np.random.randn(p*n)
            
            # Alternating minimization
            for it in range(20):
                # Update u: u[a] = sum_{b,c} T[a,b,c] * v[b] * w[c] / (sum v^2 * w^2)
                vw = np.outer(v, w)  # (m*p, p*n)
                u_new = np.einsum('abc,bc->a', T, vw)
                norm_u = np.sqrt(np.sum(u_new**2))
                if norm_u > 1e-10:
                    u = u_new / norm_u
                
                # Update v
                uw = np.outer(u, w)  # (n*m, p*n)
                v_new = np.einsum('abc,ac->b', T, uw)
                norm_v = np.sqrt(np.sum(v_new**2))
                if norm_v > 1e-10:
                    v = v_new / norm_v
                
                # Update w
                uv = np.outer(u, v)  # (n*m, m*p)
                w_new = np.einsum('abc,ab->c', T, uv)
                norm_w = np.sqrt(np.sum(w_new**2))
                if norm_w > 1e-10:
                    w = w_new / norm_w
            
            # Compute the inner product
            score = np.sum(T * np.einsum('a,b,c->abc', u, v, w))
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        return best_u, best_v, best_w, best_score
    
    # Greedy rank-1 decomposition
    U_cols = []
    V_cols = []
    W_cols = []
    T_residual = T.copy()
    
    max_rank = 40  # Try to find a decomposition with rank <= 40
    for rank_idx in range(max_rank):
        if time.time() > deadline - 10:
            break
        
        u, v, w, score = find_rank1_component(T_residual)
        
        if score < 1e-6:
            break
        
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
        # Subtract the rank-1 component
        T_residual = T_residual - np.einsum('a,b,c->abc', u, v, w)
        
        # Check if we've converged
        if np.max(np.abs(T_residual)) < 1e-8:
            break
    
    rank = len(U_cols)
    
    if rank > 0:
        U = np.column_stack(U_cols)
        V = np.column_stack(V_cols)
        W = np.column_stack(W_cols)
        
        # Verify the decomposition
        # Reconstruct and check
        reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.max(np.abs(reconstructed - T))
        
        if error < 1e-6:
            # Scale to get integer/half-integer entries if possible
            # For now, return as is
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: trivial decomposition
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
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
