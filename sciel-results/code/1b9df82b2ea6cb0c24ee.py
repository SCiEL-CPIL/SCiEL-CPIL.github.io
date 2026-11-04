import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # The trivial decomposition has rank n*m*p = 120
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, idx] = 1
                V_trivial[b, idx] = 1
                W_trivial[c, idx] = 1
                idx += 1
    
    best_rank = trivial_rank
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()
    
    # Try to find a lower rank decomposition
    # For matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # and at most n*m*p. Known lower bounds suggest rank >= max(n, m, p) * something
    # But in practice, for small matrices, we can try random searches
    
    # Strategy: Try random low-rank decompositions and check if they work
    # The tensor T[i,j,k] = delta_{i, i'} delta_{j, j'} delta_{k, k'} where
    # we're multiplying A[i,j] * B[j,k] = C[i,k]
    # So T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    # Wait, let me re-read the problem.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed by (k, i) where k is the column of B and row of C, i is row of A and C
    # Actually, let me think about this more carefully.
    
    # For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 (it's the multiplication tensor)
    # We want to decompose T as sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # Where U is indexed by (i,j), V by (j,k), W by (k,i)
    # So U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should give T[i,j,k]
    
    # Let's verify the trivial decomposition works
    # For the trivial case, each (i,j,k) triple gets its own rank-1 term
    # U[i*m+j, r_{ijk}] = 1, V[j*p+k, r_{ijk}] = 1, W[k*n+i, r_{ijk}] = 1
    # Sum over r: only r_{ijk} contributes, giving 1 for T[i,j,k]. Correct.
    
    # Now let's try to find a lower rank decomposition
    # We can use random search with structured initializations
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W correctly decompose the multiplication tensor."""
        rank = U.shape[1]
        # Compute the reconstructed tensor
        # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need this to equal 1 for all valid (i,j,k)
        
        # Efficient check: for each (i,j,k), compute the sum
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :]) * 0  # placeholder
                    # Actually, we need sum_r U[a,r] * V[b,r] * W[c,r]
                    # This is not a simple dot product. Let me compute it properly.
                    vals = U[a, :] * V[b, :] * W[c, :]
                    s = np.sum(vals)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try random low-rank decompositions
    # Start with ranks below the trivial one
    for target_rank in range(40, trivial_rank):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random values, then try to optimize
            # Use a structured initialization based on the trivial decomposition
            # Pick target_rank random (i,j,k) triples and try to merge them
            
            # Alternative: use random Gaussian initialization and check
            U_rand = np.random.randn(n * m, target_rank) * 0.1
            V_rand = np.random.randn(m * p, target_rank) * 0.1
            W_rand = np.random.randn(p * n, target_rank) * 0.1
            
            if check_decomposition(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U = U_rand.copy()
                best_V = V_rand.copy()
                best_W = W_rand.copy()
                break
        
        if best_rank < trivial_rank:
            break
    
    # If we found a better decomposition, return it
    if best_rank < trivial_rank:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    else:
        # Return the trivial decomposition
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
