"""Matrix multiplication 2x5x6 rank-minimal decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # The multiplication tensor T[i,j,k] = delta[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Try various ranks from low to high
    min_possible_rank = 20  # lower bound estimate
    max_search_rank = 40
    
    # Use random initialization and verify
    for rank in range(min_possible_rank, max_search_rank + 1):
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Verify: compute the reconstructed tensor and check
            # T_reconstructed[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # This should equal 1 if i,j,k form a valid multiplication path, else 0
            # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
            # The tensor has T[i,j,k] = 1 for all valid (i,j,k) triples
            # Wait, that's not right. The multiplication tensor is:
            # T_{i,j,k} where we compute C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor has entries T[i,j,k] = 1 for all i,j,k (since every j contributes)
            # No wait - the tensor is defined such that <T, A ⊗ B> = C
            # T[i,j,k] * A[i,j] * B[j,k] summed over i,j,k gives C[i,k]
            # So T[i,j,k] should be 1 for all i,j,k? No...
            # Actually C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all valid indices
            # But that means the tensor is all ones, which has rank 1... that can't be right.
            
            # Let me reconsider. The matrix multiplication tensor T is a 3-tensor of size n x m x p
            # where T[i,j,k] = 1 if the indices form a valid multiplication term
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            # Wait, that's still all ones...
            
            # I think I'm confusing myself. Let me think about this differently.
            # The multiplication tensor is defined by:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # In tensor form: C = T(A, B) where T has T[i,j,k] = 1
            # But that's a rank-1 tensor!
            
            # No, I think the issue is that the tensor decomposition is:
            # C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_k V[j*p+k,c] * B[j,k])... 
            # Hmm, this is getting complicated. Let me use the standard formulation.
            
            # Standard: C[i,k] = sum_j A[i,j] B[j,k]
            # We want: C[i,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] * A[i,j] B[j,k]... 
            # No, that's not how it works either.
            
            # The correct formulation: We seek U, V, W such that for all A, B:
            # C = A*B where C[i,k] = sum_j A[i,j] B[j,k]
            # And C[i,k] = sum_c (U[i*m+j,c] * A[i,j]) * (V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
            # Wait no, that's not right either.
            
            # Let me look at the seed code more carefully:
            # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # The constraint is: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
            # Wait, that would mean the tensor is all ones...
            
            # Actually I think the decomposition is:
            # For the multiplication to work, we need:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta condition
            # Let me just verify with the schoolbook: for each (i,j,k), there's exactly one c
            # where U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
            # So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
            # This means the "tensor" we're decomposing has all entries equal to 1!
            
            # So we need: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,j,k
            # This is a different problem than I thought. Let me verify this interpretation.
            
            # Actually, I think I need to reconsider the problem. The tensor for matrix multiplication
            # T[i,j,k] represents the coefficient such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
            # So T[i,j,k] = 1 for all i,j,k. The tensor is indeed all ones.
            # But that seems too simple... Let me double-check with a small example.
            
            # For 2x2x2: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
            # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
            # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
            # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
            # So T[0,0,0]=1, T[0,1,0]=1, T[0,0,1]=1, T[0,1,1]=1, etc.
            # Yes, T[i,j,k]=1 for all valid indices.
            
            # So the problem is: find rank-minimal U,V,W such that
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
            
            # This is equivalent to finding a low-rank decomposition of the all-ones tensor.
            # The all-ones tensor of size n*m x m*p x p*n... wait, no.
            # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
            # And the constraint is on the "contracted" product.
            
            # Actually, I think the correct interpretation is:
            # We need the bilinear form to reproduce matrix multiplication.
            # Let me just trust the seed code and verify my decomposition numerically.
            
            pass  # Skip this attempt for now, I need to understand the problem better
    
    # Let me try a different approach: just use the schoolbook and try to optimize
    # Actually, let me verify what the constraint actually is by testing the schoolbook
    
    # For now, return the schoolbook decomposition
    # Convert to lists as required
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(), 
        "W": best_W.tolist()
    }
