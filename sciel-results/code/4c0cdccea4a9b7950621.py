"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Check tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: T[i,j,k] = 1 if i+k < n (for the specific structure)
    # Actually for (n,m,p) matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] where output C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = delta(i+k) ... no.
    # T[i,j,k] = 1 if the (i,j) of A and (j,k) of B contribute to C[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's just the identity in the middle)
    # Wait, let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
    # The tensor for matrix multiplication (n x m) * (m x p) -> (n x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = 1 for all i,j,k (each term in the sum)
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_construction(n, m, p):
    """
    Try to find a low-rank decomposition for 2 x m x p case.
    For n=2, we can use the fact that C[0,k] and C[1,k] share structure.
    
    Strategy: For 2 x m x m, try to use rank around 2m - 2.
    Construction idea:
    - For each k, we need to compute C[0,k] = sum_j A[0,j]*B[j,k] and C[1,k] = sum_j A[1,j]*B[j,k]
    - We can try to factor the computation.
    
    Let me try a specific construction for 2x5x5.
    """
    # For 2 x m x m, one known approach:
    # Use rank = 2m - 1 or 2m - 2
    # Let's try to construct explicitly for m=5
    
    m = 5
    # Try rank = 9 (2*5 - 1)
    # Actually, let me try several ranks and see which works
    
    # A systematic approach: set up the equations and try to find a solution
    # For small ranks, we can try random or structured initializations
    
    # Let me try a specific construction:
    # For 2 x m x m, define:
    # U[i*m+j, c] and V[j*m+k, c] and W[k*2+i, c]
    # such that sum_c U[i*m+j,c] V[j*m+k,c] W[k*2+i,c] = 1
    
    # One idea: use the fact that for fixed i, the vectors {V[j*m+k,c]}_k 
    # need to be "orthogonal" in some sense to W[k*2+i',c] for i' != i
    
    # Let me try rank = 10 first with a structured approach
    # Actually, let me just try to find ANY valid decomposition with rank < 50
    # by using a greedy or optimization approach
    
    # For now, let's return schoolbook as fallback
    return _try_rank_schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best = _try_rank_schoolbook(n, m, p)
    
    # Try to find better decompositions
    # For 2 x 5 x 5, let's try to construct a low-rank decomposition
    
    # Approach: For n=2, the problem has special structure.
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    
    # We can think of this as: for each k, we compute two dot products.
    # Total: 2 * 5 * 5 = 50 scalar multiplications in schoolbook.
    
    # Can we do better? For 2 x m x m, the known lower bound is 2m and 
    # upper bound is 2m-1 (I think) for large m.
    
    # Let me try to construct a rank 9 or 10 decomposition.
    
    # Try rank = 10
    rank = 10
    # Initialize with zeros
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a system of n*m*p = 50 equations in the products.
    
    # Let me try a structured assignment:
    # For c in range(rank):
    #   Assign U, V, W entries to satisfy the equations
    
    # Simple structured approach for 2 x 5 x 5:
    # Let's try to make V sparse with one 1 per row (gauge fixing)
    # Then solve for U and W
    
    # Actually, let me try a different approach: use the known result that
    # 2 x m x m can be done in 2m - 1 multiplications for m >= 2.
    
    # For m = 5, try rank = 9.
    
    # Construction for 2 x m x m with rank 2m-1:
    # This is non-trivial. Let me try to find it numerically.
    
    # Let me try random search for small ranks
    np.random.seed(42)
    
    for target_rank in [9, 10, 11, 12, 15, 20, 25]:
        if time.time() > deadline:
            break
            
        # Try random initialization and check
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            U_rand = np.random.randint(0, 3, (n*m, target_rank)) - 1  # -1, 0, 1
            V_rand = np.random.randint(0, 3, (m*p, target_rank)) - 1
            W_rand = np.random.randint(0, 3, (p*n, target_rank)) - 1
            
            # Check if this works
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                best = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                break
    
    # Return the best found
    return best
# EVOLVE-BLOCK-END
