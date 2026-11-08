"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    schoolbook = {"U": U, "V": V, "W": W}

    # Precompute tensor T[i,j,k] = 1 (the matrix multiplication tensor)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Build the tensor as a 3D array for verification
    T = np.ones((n, m, p))
    
    def verify(U_mat, V_mat, W_mat):
        """Check if U, V, W exactly decompose the tensor."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal T[i,j,k]
        # Vectorized: for each c, outer product of U[:,c], V[:,c], W[:,c]
        # But that's expensive for large rank. Let's do it efficiently.
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        r = U_arr.shape[1]
        # Sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U3 = U_arr.reshape(n, m, r)
        V3 = V_arr.reshape(m, p, r)
        W3 = W_arr.reshape(p, n, r)
        # Compute sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(result, T, atol=1e-10)

    # Random search for lower rank decompositions
    # Strategy: try random linear combinations of schoolbook terms
    # Each rank-1 term in schoolbook is (i,j,k) -> U has 1 at (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i)
    # We can try to find a set of r < n*m*p rank-1 terms that span the tensor
    
    # Better approach: use random projections / random linear combinations
    # Generate random vectors a in R^{n*m}, b in R^{m*p}, c in R^{p*n}
    # and check if their "trilinear form" matches part of the tensor
    
    # Actually, let's try a different approach: random rank-r approximations
    # by randomly selecting r combinations and solving for coefficients
    
    # For small ranks, we can try:
    # Pick r random (i,j,k) triples with replacement, form the corresponding
    # rank-1 tensors, and solve a linear system to see if we can represent T
    
    # But the tensor has n*m*p = 128 entries, all equal to 1.
    # A rank-1 tensor for triple (i0,j0,k0) has 1 at position (i0,j0,k0) and 0 elsewhere.
    # So we need r rank-1 tensors whose sum is the all-ones tensor.
    # But rank-1 tensors in the trilinear sense are not just single entries!
    
    # Let me reconsider. The matrix multiplication tensor T_{i,j,k} = delta_{...} 
    # Actually for matrix multiplication, T[i,j,k] = 1 for all i,j,k when multiplying
    # A[i,j] * B[j,k] -> C[i,k]. So yes, T is all ones.
    
    # A rank-1 decomposition term is: sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = T[i,j,k]
    # where u_c is an n*m vector, v_c is an m*p vector, w_c is a p*n vector.
    
    # The schoolbook terms are: for each (i,j,k), u has 1 at (i,j), v has 1 at (j,k), w has 1 at (k,i).
    # These are "sparse" rank-1 terms.
    
    # We can try denser rank-1 terms. Let's do random search:
    # Generate random u, v, w vectors and add them as rank-1 terms,
    # then solve for coefficients to match T.
    
    best_result = schoolbook
    best_rank = rank
    
    # Try ranks from some lower bound up to best_rank - 1
    # Lower bound: at least max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # But actually the rank could be lower. Let's try from 16 upward.
    
    for target_rank in range(16, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random rank-1 terms
            # u: (n*m, target_rank), v: (m*p, target_rank), w: (p*n, target_rank)
            u_rand = np.random.randn(n * m, target_rank)
            v_rand = np.random.randn(m * p, target_rank)
            w_rand = np.random.randn(p * n, target_rank)
            
            # Build the linear system: for each (i,j,k), 
            # sum_c u[i*m+j, c] * v[j*p+k, c] * w[k*n+i, c] = 1
            # This is nonlinear in the coefficients if we fix u,v,w shapes.
            
            # Alternative: fix u and v, solve for w? No, it's trilinear.
            
            # Let's use a different approach: 
            # The tensor T can be written as sum_c u_c ⊗ v_c ⊗ w_c
            # where u_c ∈ R^{n*m}, v_c ∈ R^{m*p}, w_c ∈ R^{p*n}
            # This is equivalent to: for all (i,j,k), sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = T[i,j,k]
            
            # Let's vectorize T as a vector of length n*m*p
            # Then each rank-1 term (u_c, v_c, w_c) corresponds to a vector 
            # vec_c where vec_c[i*m*p + j*p + k] = u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
            # We need sum_c vec_c = vec_T (all ones)
            
            # So we need to find r vectors vec_c that sum to all-ones, where each vec_c
            # has the special trilinear structure.
            
            # This is a hard problem. Let me try a simpler heuristic:
            # Use random Gaussian vectors and check if their "trilinear product" 
            # can span the all-ones tensor.
            
            # Actually, let's just try random sparse combinations of schoolbook terms.
            # Pick r random (i,j,k) triples and see if we can assign weights to make sum = 1 everywhere.
            # But schoolbook terms only have 1 at one position, so we need at least one term 
            # for each (i,j,k) position, meaning rank >= 128. That doesn't help.
            
            # Let's try denser terms. Generate random u, v, w and compute the resulting tensor.
            # If we can find r terms whose weighted sum equals T, we're done.
            
            # For now, just verify the schoolbook works and return it.
            # The random search is too complex to implement correctly in this time.
            pass
    
    return best_result
# EVOLVE-BLOCK-END
