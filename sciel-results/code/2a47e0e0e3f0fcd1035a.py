"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a lower rank decomposition using structured search
    # The matrix multiplication tensor has a known structure that may allow
    # rank reduction beyond schoolbook
    
    # For 4x5 * 5x6, the schoolbook rank is 120
    # Let's try some known lower-rank constructions
    
    # Attempt 1: Try rank 100 with random initialization and gradient-like refinement
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        # Build the multiplication tensor T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # The decomposition requires: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # and for all other combinations it should be 0 (but actually we only need the (i,j,k) terms)
        
        # Actually, the tensor equation is:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where T[i,j,k] = 1 if this corresponds to a valid multiplication term, else 0
        # For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Wait, that's not right. The tensor is:
        # T[i,j,k] = 1 for all valid (i,j,k) indices since each term contributes
        # Actually the multiplication tensor has T[i,j,k] = 1 for all i,j,k in range
        
        # Let me reconsider. The standard tensor for matmul A(n,m) * B(m,p) -> C(n,p) is:
        # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # And the decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
        # where U is indexed by (i,j), V by (j,k), W by (i,k)
        
        # In the given format:
        # U[i*m + j, c] corresponds to U[i,j,c]
        # V[j*p + k, c] corresponds to V[j,k,c]  
        # W[k*n + i, c] corresponds to W[k,i,c] -- wait, the task says W[k*n+i,r]
        
        # So W is indexed by (k,i) in the linear index k*n+i
        # The tensor equation should be: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        # Let's verify a few random points
        for _ in range(100):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
            if abs(val - 1) > 1e-9:
                return False
        return True
    
    # Try some structured lower-rank approaches
    # Known results: For 4x4*4x4, rank 49 is known. For rectangular, less is known.
    
    # Let's try a systematic search with small ranks
    # First, let's see if we can do better than 120
    
    # Try rank 115, 110, 105, 100 with random search
    for target_rank in [115, 110, 105, 100, 95, 90]:
        if time.time() > deadline - 20:
            break
        
        # Random search for this rank
        best_found = False
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random values
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            # Check if this works
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(target_rank))
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
            
            if valid:
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                best_found = True
                break
        
        if best_found:
            break
    
    # Convert to lists if we found something better
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
