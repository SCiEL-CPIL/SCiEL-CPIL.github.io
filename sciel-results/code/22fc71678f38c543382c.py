"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    schoolbook_rank = n * m * p  # 144
    
    # Build the matrix multiplication tensor
    # T[i,j,k] = 1 if we need A[i,j] * B[j,k] to contribute to C[i,k]
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # Sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta for the correct terms
    
    # The tensor: for each (i,j,k), the product should be 1 if it's a valid term
    # Actually, the decomposition must satisfy:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is in the support
    # The support is all (i,j,k) with i in [0,n), j in [0,m), k in [0,p)
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is a 3-way tensor of size (n*m) x (m*p) x (p*n)
    # with all entries = 1 (since every (i,j,k) triple is valid)
    
    # Wait, that's not right. Let me re-read the problem.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The decomposition: for the matrix multiplication tensor
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor for matmul)
    # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k] = 1
    
    # So we need a rank-r decomposition of the all-ones tensor of size (n*m) x (m*p) x (p*n)
    # where n*m=16, m*p=36, p*n=36
    
    # The schoolbook gives rank 144 with U,V,W having single 1s per column.
    
    # To find lower rank, we need to find matrices U (16 x r), V (36 x r), W (36 x r)
    # such that the element-wise product sum equals the all-ones tensor.
    
    # This is equivalent to: the 3-way tensor with all ones must have rank <= r.
    
    # The rank of the all-ones tensor of size a x b x c is... well, it's at most min(a,b,c) 
    # No wait, that's not right either.
    
    # Actually, the all-ones tensor of size a x b x c can be written as:
    # T[i,j,k] = 1 = sum_{c=1}^{r} U[i,c] * V[j,c] * W[k,c]
    # One trivial decomposition: r=1, U[i,0]=1, V[j,0]=1, W[k,0]=1 for all i,j,k
    # That gives T[i,j,k] = 1*1*1 = 1. Rank 1!
    
    # But wait, that can't be right for matrix multiplication. Let me re-read the problem.
    
    # Oh I see, I'm confusing myself. The matrix multiplication tensor is NOT the all-ones tensor.
    # Let me re-read the interface:
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor T for multiplying (n x m) by (m x p) matrix:
    # T is a tensor of size (n*m) x (m*p) x (p*n)
    # T[a, b, c] = 1 if a = i*m + j, b = j'*p + k, c = k'*n + i' where i=i', j=j', k=k'
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1, and 0 otherwise.
    
    # So T is NOT all ones. It has 1s only at positions where the indices are consistent.
    # Specifically, T has n*m*p = 144 ones, at positions (i*m+j, j*p+k, k*n+i) for all i,j,k.
    
    # The schoolbook decomposition uses rank 144 with each column corresponding to one (i,j,k).
    
    # To find a lower rank decomposition, we need to find r < 144 such that there exist
    # U (16 x r), V (36 x r), W (36 x r) with:
    # sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # and = 0 for all other (a,b,c) triples.
    
    # This is a non-trivial optimization problem.
    
    # Let me try a different approach: use known results.
    # For 2x2 x 2x2: rank 7 (Strassen)
    # For 4x4 x 4x4: rank 49 (best known)
    # For 4x4 x 4x9: ?
    
    # One approach: use the 4x4 x 4x4 rank-49 decomposition as a subroutine.
    # Write B (4x9) as [B1 B2 B3] where each Bi is 4x3.
    # C = A*B = [A*B1, A*B2, A*B3]
    # Each A*Bi is 4x3.
    
    # For 4x4 x 4x3: split B_i into three 4x1 columns.
    # A * (4x1) = 4x1, which is a linear map. The tensor for this has rank 4 (schoolbook).
    # So total rank would be 3 * 4 * 3 = 36? No, that's not how it works.
    
    # Actually, I think I need to just do a numerical search.
    
    # Let me try a simple approach: start with schoolbook and try to reduce rank by finding
    # linear dependencies or using a greedy algorithm.
    
    # For now, let me just return the schoolbook decomposition if I can't find better quickly.
    # But let me first try a few specific constructions.
    
    # Try: can we use rank < 144?
    # Let me try a random search with a small number of iterations.
    
    def schoolbook():
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
        return U, V, W, rank
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        # T[a,b,c] should be 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
        # and 0 otherwise.
        # Compute T_reconstructed[a,b,c] = sum_c U[a,c]*V[b,c]*W[c_index,c]
        # Actually, let me compute it properly.
        
        # T_reconstructed is a 3D tensor of size (n*m, m*p, p*n)
        # T_reconstructed[a, b, c] = sum_{r} U[a, r] * V[b, r] * W[c, r]
        
        # We need T_reconstructed[a, b, c] = 1 if (a,b,c) = (i*m+j, j*p+k, k*n+i) for some i,j,k
        # and 0 otherwise.
        
        # For efficiency, let's check only the required entries and a sample of others.
        
        # Check all (i,j,k) triples:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[:, :len(U[0])] * V[b, :] * W[c, :]) if False else None
                    # More efficient:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check some random non-support entries:
        rng = np.random.RandomState(42)
        for _ in range(1000):
            a = rng.randint(0, n * m)
            b = rng.randint(0, m * p)
            c = rng.randint(0, p * n)
            # Check if (a,b,c) is in the support
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            if i_a == i_c and j_a == j_b and k_b == k_c:
                continue  # In support, already checked
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val) > 1e-9:
                return False
        
        return True
    
    # Try to find a better decomposition using random search
    # This is a hard problem, so let's try a few strategies
    
    # Strategy 1: Try rank values and see if we can find a valid decomposition
    # This is computationally expensive, so let's be smart about it.
    
    # For now, let's just return the schoolbook decomposition.
    # A proper search would require much more sophisticated methods.
    
    U, V, W, rank = schoolbook()
    
    # Convert to list format as required
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
