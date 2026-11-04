"""Find rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# Deadline guard
_DEADLINE = time.time() + 135  # 135s out of 150s limit, safety margin

def _check_deadline():
    return time.time() > _DEADLINE

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x5 matrix by a 5x6 matrix.
    
    Returns dict with U (10 x rank), V (30 x rank), W (12 x rank)
    where W follows trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (30, rank)  
    # W shape: (p*n, rank) = (12, rank)
    
    # The matrix multiplication tensor T has entries T[i, j, k] = 1 if we're computing
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In the decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, let me reconsider the convention.
    #
    # Standard: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T_{i,j,k} where output C[i,k] gets contribution from A[i,j]*B[j,k]
    # Decomposition: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    #
    # Actually, the standard bilinear decomposition of matrix mult:
    # C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_k V[j,k,r]*B[j,k]) ... no
    #
    # Let me use the standard form:
    # C[i,k] = sum_r M_r[i,j] * N_r[j,k] where M_r and N_r are the rank-r factors
    # Actually the tensor decomposition is:
    # T_{i,j,k} = sum_r U_{(i,j),r} * V_{(j,k),r} * W_{(k,i),r}
    # This is a 3-tensor decomposition where each factor is a matrix.
    
    # Schoolbook rank = n*m*p = 60
    # We want to find a lower rank.
    
    # Strategy: Try random integer/rational decompositions of decreasing rank
    # Start from a known upper bound and try to reduce.
    
    # First, let's use the schoolbook as fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using random search + local optimization
    # The tensor T[i,j,k] = delta (it's the identity-like tensor for matmul)
    # Actually for matmul, T[i,j,k] = 1 for all valid (i,j,k) in the sense that
    # C[i,k] = sum_j A[i,j]*B[j,k], so the tensor is the "matmul tensor"
    
    # Let's try random rank-r decompositions with small integer entries
    # and check if they're exact.
    
    # Upper bound: try ranks from a reasonable lower bound up to schoolbook
    # Lower bound for 2x5*5x6: at least max(n*m, m*p, p*n) = max(10,30,12) = 30
    # But actually the lower bound is higher. For matmul, rank >= n*m*p / (some factor)
    # Known: rank of 2x2*2x2 is 7 (Strassen), schoolbook is 8
    # For 2x5*5x6, let's try various ranks
    
    # Try random search with small ranks first
    random.seed(42)
    
    # Try ranks from 30 up to 60, but focus on lower ranks
    # Use a smarter approach: try to find decompositions with entries in {-1, 0, 1} first
    
    for target_rank in range(30, best_rank + 1):
        if _check_deadline():
            break
        if target_rank >= best_rank:
            break
            
        # Try multiple random attempts for this rank
        max_attempts = 50 if target_rank < 45 else 20
        found = False
        
        for attempt in range(max_attempts):
            if _check_deadline():
                found = True  # stop
                break
                
            # Generate random factor matrices with small integer entries
            # U: 10 x target_rank, V: 30 x target_rank, W: 12 x target_rank
            # Entries from {-1, 0, 1}
            
            U = np.zeros((n*m, target_rank), dtype=int)
            V = np.zeros((m*p, target_rank), dtype=int)
            W = np.zeros((p*n, target_rank), dtype=int)
            
            for r in range(target_rank):
                # Random sparse pattern
                for i in range(n*m):
                    if random.random() < 0.5:
                        U[i, r] = random.choice([-1, 1])
                for j in range(m*p):
                    if random.random() < 0.5:
                        V[j, r] = random.choice([-1, 1])
                for k in range(p*n):
                    if random.random() < 0.5:
                        W[k, r] = random.choice([-1, 1])
            
            # Check if this decomposition is exact
            # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # The matmul tensor: T[i,j,k] = 1 for all i,j,k (in the appropriate sense)
            # Actually, the matmul tensor T has T[i,j,k] = 1 if we consider
            # the full tensor where each (i,j,k) triple contributes to C[i,k]
            # via A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid i,j,k.
            
            # Wait, I need to be more careful. The matrix multiplication tensor
            # is defined as: T_{i,j,k} = 1 for all i in [n], j in [m], k in [p]
            # because C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor is all ones!
            
            # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
            
            valid = True
            # Sample some entries first for speed
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Verify all entries
            all_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += int(U[i*m+j, r]) * int(V[j*p+k, r]) * int(W[k*n+i, r])
                        if val != 1:
                            all_valid = False
                            break
                    if not all_valid:
                        break
                if not all_valid:
                    break
            
            if all_valid:
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                found = True
                break
        
        if found:
            continue  # Try next (lower) rank... wait, we're going UP
        # Actually we should try lower ranks first. Let me restructure.
    
    # The above loop goes from 30 up, but we want to find the MINIMUM rank.
    # Let me restructure: try from low to high, and return the first valid one.
    
    # Actually, the loop above already does this - it tries rank 30 first, then 31, etc.
    # And stops at the first valid rank found (lowest).
    
    return best_result
