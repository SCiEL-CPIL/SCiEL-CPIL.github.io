"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank_school
    
    # Verify a candidate decomposition
    def verify(U, V, W, rank):
        """Check if U, V, W correctly decompose the multiplication tensor."""
        # The multiplication tensor: T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # But actually the tensor is: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor T has entries T[i,j,k] = 1 (it's the identity coupling)
        # The decomposition means: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Wait, that's not right. Let me reconsider.
        
        # The standard tensor decomposition for matrix multiplication:
        # The tensor T is defined such that T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
        # contribute to C[i,k]. Actually T[i,j,k] is just 1 for all valid (i,j,k).
        # No wait - the multiplication tensor is T[i,j,k] where the bilinear form is:
        # <T, A, B> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k]
        # where C = A*B. So T[i,j,k] = 1 for all valid indices.
        
        # A rank-1 tensor is u * v * w where (u*v*w)[i,j,k] = u[i,j]*v[j,k]*w[k,i]
        # Wait, the W index is k*n+i, so w is indexed by (k,i).
        
        # So a rank-1 decomposition term c has:
        # U[:, c] is a vector in R^{n*m} indexed by (i,j)
        # V[:, c] is a vector in R^{m*p} indexed by (j,k)  
        # W[:, c] is a vector in R^{p*n} indexed by (k,i)
        
        # The product is: for each (i,j,k), the contribution is U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
        
        # Let's verify this
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check a sample of entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    # Strategy: Try random sparse decompositions with verification
    # Start from rank slightly below schoolbook and try to find valid ones
    
    # For 2x5x6, the known lower bound is at least n*p = 12 (from Strassen-type arguments)
    # and the upper bound is 60 (schoolbook). Let's try ranks in between.
    
    # Try a structured approach: since n=2 is small, we can try rank = 2*p*m/2 or similar
    # Actually, let's just try various ranks and use random initialization
    
    np.random.seed(42)
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, p*n) = max(10, 30, 12) = 30? No, that's not right.
    # The rank must be at least the maximum of the three matrix dimensions in some sense.
    # Actually, a simple lower bound is: rank >= max(n, m, p) = 6, but that's weak.
    # A better bound: the tensor has a certain structure. For now, let's try ranks from 30 to 60.
    
    for target_rank in range(30, rank_school + 1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 2:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to make entries roughly O(1)
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Quick verification with tolerance
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list, target_rank):
                best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                break
        
        if best_rank < target_rank:
            break
    
    # If we found something better, return it; otherwise return schoolbook
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        U, V, W, _ = make_schoolbook()
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
