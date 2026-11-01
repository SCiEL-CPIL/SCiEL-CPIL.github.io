"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # First, return the known rank-7 Strassen decomposition as fallback
    rank7_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    rank7_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    rank7_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Search for rank-6 decomposition
    # The tensor T[i,j,k] for 2x2x2 matrix mult is:
    # T[0,0,0]=1, T[0,1,1]=1, T[1,0,0]=1, T[1,1,1]=1 (identity-like)
    # Actually for standard matrix mult: T[i,j,k] = delta_{i,k} * delta_{j,i}... 
    # Let me be more careful. For C = A*B, C[i,j] = sum_k A[i,k]*B[k,j]
    # So the tensor is T[i,k,j] = 1 if indices match appropriately
    # In the convention W[k*n+i, r], we have W indexed by (k,i) where k is from B, i is from C
    # So T[i, j, k] where i=row of C, j=col of C, k=inner index
    # T[i,j,k] = 1 if the k-th element of A's i-th row times B's k-th row j-th col contributes to C[i,j]
    # So T[i,j,k] = 1 always for valid indices? No...
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}
    # because C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
    # So each term A[i,k]*B[k,j] contributes to C[i,j] with coefficient 1
    
    # We need: sum_r U[i,m+r] * V[k,m+r] * W[j*p+r] = 1 for all i,j,k in {0,1}
    # Wait, let me re-read the convention. U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r] summed over r should give T[i,j,k]
    
    # For 2x2x2: i,j,k in {0,1}
    # T[i,j,k] = 1 for all combinations (since each A[i,k]*B[k,j] term has coefficient 1)
    
    # Search over small integer coefficients for rank 6
    # Coefficients in {-2, -1, 0, 1, 2}
    coeff_range = [-2, -1, 0, 1, 2]
    
    # We need to find U (4x6), V (4x6), W (4x6) with entries in coeff_range
    # such that for all i,j,k in {0,1}:
    # sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r] = 1
    
    # This is a 3-variable product sum. Let's try a structured search.
    # Due to symmetry, we can fix some values.
    
    # Actually, let's try a different approach: use the fact that rank-6 would be a 
    # significant breakthrough. Let's do a randomized search with verification.
    
    best_result = None
    
    # Try random integer matrices with small entries
    np.random.seed(42)
    attempts = 0
    
    while time.time() < deadline and attempts < 500000:
        attempts += 1
        
        # Generate random U, V, W with entries in {-1, 0, 1}
        U = np.random.randint(-1, 2, size=(4, 6))
        V = np.random.randint(-1, 2, size=(4, 6))
        W = np.random.randint(-1, 2, size=(4, 6))
        
        # Check if this decomposes the tensor
        valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Compute sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
                    s = 0
                    for r in range(6):
                        s += U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Found a rank-6 decomposition!
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to rank-7 Strassen
    return {"U": rank7_U, "V": rank7_V, "W": rank7_W}
# EVOLVE-BLOCK-END
