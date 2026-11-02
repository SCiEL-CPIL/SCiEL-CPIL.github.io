import time
import random
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has T[i,k,j] = 1 if we consider A[i,j]*B[j,k] -> C[i,k]
    # U has shape (n*p, rank) = (24, rank) -- indices (i,k) for C
    # V has shape (m*n, rank) = (16, rank) -- indices (j,i) for A? Wait...
    # Let me re-read: U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U: (4*4, rank) = (16, rank) -- for A matrix indices (i,j)
    # V: (4*6, rank) = (24, rank) -- for B matrix indices (j,k)  
    # W: (6*4, rank) = (24, rank) -- for C matrix indices (k*n+i) = (k*4+i)
    
    # The decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right. Let me re-read the contract.
    # "U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]" summed over r should equal delta for the multiplication
    # Actually the tensor entry T[i,j,k] for C[i,k] = sum_j A[i,j]B[j,k]
    # The standard form: T is such that T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # So sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 if the indices are consistent
    # i.e., for each (i,j,k), the product should be 1 (since A[i,j]*B[j,k] contributes to C[i,k])
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 96 equations.
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Start with schoolbook rank 96
    rank_schoolbook = n * m * p
    U_sb = [[0]*rank_schoolbook for _ in range(n*m)]
    V_sb = [[0]*rank_schoolbook for _ in range(m*p)]
    W_sb = [[0]*rank_schoolbook for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Strategy: try ranks from 24 upward, use random initialization and local search
    
    # For efficiency, try a few specific ranks
    candidate_ranks = [24, 25, 26, 27, 28, 30, 32, 35, 40, 48, 56, 64, 72, 80, 88, 95]
    
    random.seed(42)
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values from {-1, 0, 1}
            U = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(p*n)]
            
            # Local search: try to fix violations by flipping signs
            for iteration in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Check current violation
                violated = False
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                            if abs(s - 1.0) > 1e-9:
                                violated = True
                                break
                        if violated:
                            break
                    if violated:
                        break
                
                if not violated:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break
                
                # Pick a random violated triple and try to fix it
                # Try flipping a random element
                i = random.randint(0, n-1)
                j = random.randint(0, m-1)
                k = random.randint(0, p-1)
                
                # Compute current sum
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                
                # Try flipping a random U, V, or W element
                choice = random.randint(0, 2)
                r = random.randint(0, rank-1)
                
                if choice == 0:
                    old_val = U[i*m+j][r]
                    U[i*m+j][r] = -old_val
                elif choice == 1:
                    old_val = V[j*p+k][r]
                    V[j*p+k][r] = -old_val
                else:
                    old_val = W[k*n+i][r]
                    W[k*n+i][r] = -old_val
                
                # Check if this improved things (simplified: just accept)
                # Better: check if the violation decreased
                s_new = 0.0
                for rr in range(rank):
                    s_new += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                
                if abs(s_new - 1.0) >= abs(s - 1.0):
                    # Revert
                    if choice == 0:
                        U[i*m+j][r] = old_val
                    elif choice == 1:
                        V[j*p+k][r] = old_val
                    else:
                        W[k*n+i][r] = old_val
    
    if best_rank < rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
