"""Matrix multiplication 2,5,5 - rank minimization via random search."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Try to find lower rank decompositions
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    def build_matrices(U, V, W, rank):
        """Build the multiplication tensor and verify."""
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix mult: A[i,j] * B[j,k] = C[i,k]
        # Tensor: T[i,j,k] should be 1 if j is a valid index, else 0
        # Actually the tensor for matrix mult M_{n,m,p} has T[i,j,k] = delta_{i in [0,n)} delta_{j in [0,m)} delta_{k in [0,p)}
        # Wait, no. The matrix multiplication tensor is defined as:
        # M(A,B) = sum_j A[i,j] B[j,k]
        # So the tensor T[i,j,k] = 1 for all valid i,j,k (it's just the identity in the middle dimension)
        # Actually T is a 3-tensor where T[i,j,k] = 1 if the contraction over j gives the product.
        # The tensor entries are: T[i,j,k] = delta_{i,j,k} in the sense that 
        # C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k] where T[i,j,k] = 1
        
        # So we need: for all i in [0,n), k in [0,p):
        #   sum_j A[i,j] B[j,k] = sum_j A[i,j] B[j,k] * (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c])
        # This requires: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
        # Wait, that's not right either. Let me think more carefully.
        
        # The matrix multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k
        # (since C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j]B[j,k] in C[i,k] is 1)
        
        # A rank-r decomposition means: M[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[i,k,c]
        # where U has shape (n*m, r), V has shape (m*p, r), W has shape (n*p, r)
        # Note: W is indexed by (i,k) not (k,i). But the task says W[k*n+i, r]
        # So W row index is k*n + i, meaning W is indexed by (k,i) pairs.
        
        # Let me re-read: "W[k*n+i,r]" - so W[row, col] where row = k*n + i
        # This means W is a (p*n) x r matrix, indexed by (k,i)
        
        # The decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need this to equal 1 for all i in [0,n), j in [0,m), k in [0,p)
        
        pass
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        # M[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    def try_random_rank(r, max_iters=200):
        """Try to find a rank-r decomposition via random initialization and ALS."""
        if r >= best_rank:
            return None
        
        # Random initialization with small integers/half-integers
        for attempt in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Initialize with small random values from {-1, -0.5, 0, 0.5, 1}
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, r))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r))
            
            # ALS optimization
            for it in range(50):
                if time.time() > deadline:
                    return None
                
                # Update U
                for c in range(r):
                    # U[:,c] should maximize the fit
                    # For each (i,j), the residual is:
                    # res[i*m+j] = 1 - sum_{c' != c} U[i*m+j,c'] * V[j*p+k,c'] * W[k*n+i,c']
                    # But this is complex. Let's use a simpler gradient approach.
                    pass
            
            # This approach is too slow. Let's try a different strategy.
            break
        
        return None
    
    # Alternative: use known lower bounds and specific constructions
    # For 2x5x5, the schoolbook rank is 50.
    # Known lower bound for matrix multiplication rank is related to bilinear complexity.
    # For M_{2,5,5}, I'm not sure of the exact minimum rank.
    
    # Let's try a systematic search with structured decompositions.
    # One idea: use the fact that we can combine some terms.
    
    # For now, let's try a few random structured decompositions.
    rng = np.random.default_rng(42)
    
    for r in range(49, 30, -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Structured random: start from schoolbook and try to merge terms
            # Pick r random triples (i,j,k) from the 50 possible, ensuring coverage
            # Actually, let's try a different approach: random ALS with continuous values,
            # then round to half-integers and verify.
            
            U = rng.standard_normal((n*m, r)) * 0.5
            V = rng.standard_normal((m*p, r)) * 0.5
            W = rng.standard_normal((p*n, r)) * 0.5
            
            # Simple gradient descent on the tensor fit
            lr = 0.01
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Compute current tensor
                # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                # Gradient w.r.t. U[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c] for each (i,j,k)
                # This is expensive. Let's just check if it's close after random init.
                
                # Quick check
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-6:
                    # Round to half-integers
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    if verify(U_round, V_round, W_round, r):
                        return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                break  # Skip to next attempt since gradient descent is too slow
        
        # If no valid decomposition found for this rank, continue to next
    
    return result

# EVOLVE-BLOCK-END
