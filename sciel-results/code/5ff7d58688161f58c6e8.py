import numpy as np
from scipy.optimize import minimize
import time
import random

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    # Build the target tensor T[i,j,k] = 1 if we want C[i,k] to include A[i,j]*B[j,k]
    # T has shape (n*m, m*p, p*n) in flattened form
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if the indices are consistent: a% m == b // p, b % p == c // n, a // m == c % n
    
    # Actually, let's think about this more carefully.
    # The matrix multiplication tensor T satisfies:
    # C = A * B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In tensor form: T is a 3-tensor where T[i,j,k] = 1
    # The decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is a vector in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    # and (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Wait, the indexing convention is W[k*n+i, r]
    # So W is indexed by (k, i) where k is the column of B / row of C, and i is the row of A / column of C
    # U is indexed by (i, j) where i is row of A, j is col of A / row of B
    # V is indexed by (j, k) where j is col of A / row of B, k is col of B
    
    # So the term r contributes: U_r[i,j] * V_r[j,k] * W_r[k,i] to T[i,j,k]
    # And we want sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all valid (i,j,k)
    
    # Let's build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now we want to find rank-minimal decomposition
    # Start with a reasonable initial rank and try to reduce
    
    # Known lower bounds and upper bounds:
    # For 4x4x4, the best known upper bound is around 49 (using Strassen-like)
    # For 4x4x7, we need to multiply 4x4 by 4x7
    
    # Let's try a constructive approach:
    # Split B into B1 (4x4) and B2 (4x3)
    # C1 = A * B1 (4x4 result)
    # C2 = A * B2 (4x3 result)
    
    # For C1 = A * B1, use a Strassen-like decomposition for 4x4x4
    # For C2 = A * B2, use schoolbook: 4*4*3 = 48 terms
    
    # For 4x4x4, let's use a known efficient decomposition
    # The best known for 2x2x2 is rank 7 (Strassen)
    # For 4x4x4, we can use block Strassen: split into 2x2 blocks
    # Each 2x2 block multiplication uses 7 terms, but we have 4 block multiplications
    # However, the blocks interact, so it's more complex
    
    # Actually, let's just use a numerical optimization approach
    # Start with a reasonable rank and try to minimize
    
    # Let's try rank 80 first as a starting point
    # We'll use a greedy or iterative approach
    
    def try_decomposition(rank, max_iters=1000, tol=1e-6):
        """Try to find a rank-r decomposition"""
        random.seed(42)
        np.random.seed(42)
        
        # Initialize with random vectors
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        # Scale to get reasonable magnitude
        U *= 2.0
        V *= 2.0
        W *= 2.0
        
        # Define the loss function
        def loss(params):
            U_flat, V_flat, W_flat = params[:n*m*rank], params[n*m*rank:2*n*m*rank], params[2*n*m*rank:]
            U_p = U_flat.reshape(n*m, rank)
            V_p = V_flat.reshape(m*p, rank)
            W_p = W_flat.reshape(p*n, rank)
            
            # Compute the reconstructed tensor
            # T_recon[a,b,c] = sum_r U_p[a,r] * V_p[b,r] * W_p[c,r]
            # This is expensive to compute naively, so let's use a different approach
            
            # Instead, compute the Frobenius norm of the difference
            # T_recon = sum_r np.outer(np.outer(U_p[:,r], V_p[:,r]), W_p[:,r])
            # This is O(rank * n*m * m*p * p*n) which is too expensive
            
            # Alternative: compute the loss as sum over all (a,b,c) of (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
            # We can compute this more efficiently using matrix operations
            
            # For each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # Let's compute the residual directly
            
            # Actually, let's just compute it directly but only for non-zero entries of T
            # T has n*m*p non-zero entries (all 1s)
            
            residual = 0.0
            for r in range(rank):
                u_r = U_p[:, r]
                v_r = V_p[:, r]
                w_r = W_p[:, r]
                
                # For each (i,j,k), the contribution is u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
                # We want this to sum to 1 over r
                
                # Compute the product tensor for this r
                # prod[i,j,k] = u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
                # This is a rank-1 tensor in the (i,j,k) space
                
                # Let's compute the squared error
                # err[i,j,k] = T[i,j,k] - sum_{r'} prod_r'[i,j,k]
                # This is getting complicated. Let's just do it naively for now
                
            # Naive computation
            T_recon = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                u_r = U_p[:, r]
                v_r = V_p[:, r]
                w_r = W_p[:, r]
                
                # Compute outer product
                # This creates a 3D array of size (n*m, m*p, p*n)
                # Which is 16 * 28 * 28 = 12544 elements
                # For rank 80, this is 80 * 12544 = ~1M operations, which is fine
                
                outer = np.einsum('a,b,c->abc', u_r, v_r, w_r)
                T_recon += outer
            
            diff = T - T_recon
            return np.sum(diff * diff)
        
        # This is too slow for optimization. Let's use a different approach.
        # Use a greedy algorithm or a specialized tensor decomposition method
        
        return None
    
    # Let's use a more practical approach:
    # Use known efficient algorithms for matrix multiplication
    
    # For 4x4x4, use the best known decomposition
    # Actually, let's just use schoolbook for simplicity and correctness
    # Schoolbook for 4x4x4: 64 terms
    # Schoolbook for 4x4x3: 48 terms
    # Total: 112 terms
    
    # But we can do better. Let's use Strassen for the 4x4x4 part.
    
    # Strassen for 2x2x2: 7 terms
    # For 4x4x4, we can use a 2-level Strassen:
    # Split 4x4 into 2x2 blocks of 2x2 matrices
    # This gives 7 block-level multiplications, each of which is a 2x2x2 multiplication
    # Each 2x2x2 multiplication can be done in 7 terms
    # Total: 7 * 7 = 49 terms for 4x4x4
    
    # For 4x4x3, use schoolbook: 48 terms
    # Total: 49 + 48 = 97 terms
    
    # Let's implement this
    
    terms = []
    
    def make_u(idx_i, idx_j, val):
        u = [0] * (n * m)
        u[idx_i * m + idx_j] = val
        return u
    
    def make_v(idx_j, idx_k, val):
        v = [0] * (m * p)
        v[idx_j * p + idx_k] = val
        return v
    
    def make_w(idx_k, idx_i, val):
        w = [0] * (p * n)
        w[idx_k * n + idx_i] = val
        return w
    
    # Strassen for 2x2x2
    def strassen_2x2(i0, j0, k0):
        """Add Strassen terms for 2x2 block at row i0, col j0, result col k0"""
        i_vals = [i0, i0+1]
        j_vals = [j0, j0+1]
        k_vals = [k0, k0+1]
        
        # M1 = (A00+A11)*(B00+B11)
        # Contributes to C00 and C11
        u1 = []
        for ii in range(2):
            for jj in range(2):
                u1.append(((i_vals[ii]*m + j_vals[jj]), 1))
        v1 = []
        for jj in range(2):
            for kk in range(2):
                v1.append(((j_vals[jj]*p + k_vals[kk]), 1))
        w1 = []
        for kk in range(2):
            for ii in range(2):
                w1.append(((k_vals[kk]*n + i_vals[ii]), 1))
        
        # We need to add this as multiple rank-1 terms
        # M1 = sum_{ii,jj,kk} A[i0+ii, j0+jj] * B[j0+jj, k0+kk] * (C[k0+kk, i0+ii] + C[k0+kk+1, i0+ii+1])
        # Wait, this isn't right. Let me reconsider.
        
        # Actually, the Strassen algorithm works at the block level.
        # Let me just use schoolbook for simplicity and correctness.
        
        pass
    
    # Let's just use schoolbook for everything to ensure correctness
    # Then we can try to optimize later
    
    # Schoolbook for 4x4x7: 4*4*7 = 112 terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = make_u(i, j, 1)
                v = make_v(j, k, 1)
                w = make_w(k, i, 1)
                terms.append((u, v, w))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
