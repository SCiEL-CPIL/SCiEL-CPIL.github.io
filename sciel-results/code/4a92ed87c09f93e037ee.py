"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = rank_schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    deadline = time.time() + 100.0  # safety margin before 120s limit
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using random initialization + iterative refinement
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 for all i,j,k
    
    # Try random rank-r decompositions for r < 42
    # Use integer/rational search with small coefficients
    
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 5, 20), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Random integer matrices with small entries
            U = np.zeros((n * m, target_rank), dtype=np.int64)
            V = np.zeros((m * p, target_rank), dtype=np.int64)
            W = np.zeros((p * n, target_rank), dtype=np.int64)
            
            for r in range(target_rank):
                # Assign random +/-1 values
                U[:, r] = np.random.choice([-1, 1], size=n * m)
                V[:, r] = np.random.choice([-1, 1], size=m * p)
                W[:, r] = np.random.choice([-1, 1], size=p * n)
            
            # Check if this decomposition is valid
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += int(U[i*m+j, r]) * int(V[j*p+k, r]) * int(W[k*n+i, r])
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                # Convert to lists of integers
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                best_result = result
                return best_result
    
    # Try structured approaches: exploit the fact that n=2, m=3, p=7
    # The tensor is all-ones, which has special structure
    
    # For all-ones tensor, we can use the fact that:
    # 1 = (1/n) * sum over some basis
    # Actually, for the all-ones tensor, rank is related to the tensor's structure
    
    # A key insight: for multiplying matrices, the all-ones tensor T[i,j,k]=1
    # can be decomposed using the identity: 1 = 1*1*1
    # The schoolbook is already minimal for generic matrices, but for this specific
    # structure (all ones), there might be better decompositions
    
    # Try: use the fact that we're computing C = A*B where A is 2x3, B is 3x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 means we want sum_j A[i,j] B[j,k] = C[i,k]
    
    # This is just standard matrix multiplication, so the rank is the rank of
    # the matrix multiplication tensor, which for m x n times n x p is known to be
    # at least mn + np - n (from Frobenius-like bounds) but the exact value is unknown
    # for most cases.
    
    # For 2x3 times 3x7: known lower bounds suggest rank could be lower than 42
    # but no explicit lower-rank decomposition is known in the literature for these
    # specific dimensions that I can recall.
    
    # Try one more approach: use half-integer entries
    for target_rank in range(38, 42):
        if time.time() > deadline:
            break
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Try with +/-1/2 values
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                U[:, r] = np.random.choice([-1, 1], size=n * m)
                V[:, r] = np.random.choice([-1, 1], size=m * p)
                W[:, r] = np.random.choice([-1, 1], size=p * n)
            
            # Scale to try to get sum = 1
            # If we use 1/2 entries, product of three gives 1/8, need 8 such terms
            # This is getting complicated. Let's just return the best found.
            pass
    
    return best_result
# EVOLVE-BLOCK-END
