"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def compute_M(U, V, rank):
        """Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]"""
        M = np.zeros((p*n, rank))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    s = 0.0
                    for j in range(m):
                        s += U[i*m+j, r] * V[j*p+k, r]
                    M[k*n+i, r] = s
        return M
    
    def find_W_for_M(M, rank, max_terms=2, coeff_range=(-4, 4)):
        """
        For each row idx of M, find W[idx, :] such that sum_r M[idx,r]*W[idx,r] = 1
        W entries must be integers or half-integers.
        Returns W if found, None otherwise.
        """
        W = np.zeros((p*n, rank))
        for idx in range(p*n):
            found = False
            # Try single non-zero W entry
            for r in range(rank):
                if abs(M[idx, r]) > 1e-12:
                    w_val = 1.0 / M[idx, r]
                    if abs(w_val - round(w_val)) < 1e-9 or abs(w_val*2 - round(w_val*2)) < 1e-9:
                        W[idx, r] = w_val
                        found = True
                        break
            
            if not found and max_terms >= 2:
                # Try two non-zero W entries
                for r1 in range(rank):
                    if abs(M[idx, r1]) < 1e-12:
                        continue
                    for r2 in range(r1+1, rank):
                        if abs(M[idx, r2]) < 1e-12:
                            continue
                        for a in range(coeff_range[0], coeff_range[1]+1):
                            if a == 0:
                                continue
                            for b in range(coeff_range[0], coeff_range[1]+1):
                                if b == 0:
                                    continue
                                if abs(a*M[idx, r1] + b*M[idx, r2] - 1.0) < 1e-9:
                                    W[idx, r1] = float(a)
                                    W[idx, r2] = float(b)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
            
            if not found:
                return None
        
        return W
    
    def try_structured_decomposition(rank, time_limit=None):
        """
        Try structured decompositions based on known patterns for matrix multiplication.
        Key idea: use the fact that matrix multiplication has a specific algebraic structure.
        """
        if time_limit is not None and time.time() - start_time > time_limit:
            return None
        
        # Strategy 1: Group columns by (i,j) pairs for U and (j,k) pairs for V
        # Each column r corresponds to a "slice" of the computation
        
        # Pattern A: Each column handles one (i,j,k) triple but with sharing
        # This is essentially the schoolbook method, so we need smarter patterns
        
        # Pattern B: Use Kronecker-like structure
        # For 3x4x7, try to exploit the dimensions
        
        num_trials = 50 if rank < 30 else 20
        
        for trial in range(num_trials):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Generate U and V with structured sparsity
            # Each column of U should have non-zeros that correspond to a connected set of (i,j)
            # Each column of V should have non-zeros that correspond to a connected set of (j,k)
            
            for r in range(rank):
                # Choose a random "center" j for this column
                center_j = random.randint(0, m-1)
                
                # For U: pick 1-2 (i, j) pairs centered around center_j
                num_u = random.choice([1, 2])
                if num_u == 1:
                    i = random.randint(0, n-1)
                    U[i*m + center_j, r] = random.choice([-1, 1])
                else:
                    i1 = random.randint(0, n-1)
                    i2 = random.randint(0, n-1)
                    while i2 == i1:
                        i2 = random.randint(0, n-1)
                    U[i1*m + center_j, r] = random.choice([-1, 1])
                    U[i2*m + center_j, r] = random.choice([-1, 1])
                
                # For V: pick 1-2 (j, k) pairs centered around center_j
                num_v = random.choice([1, 2])
                if num_v == 1:
                    k = random.randint(0, p-1)
                    V[center_j*p + k, r] = random.choice([-1, 1])
                else:
                    k1 = random.randint(0, p-1)
                    k2 = random.randint(0, p-1)
                    while k2 == k1:
                        k2 = random.randint(0, p-1)
                    V[center_j*p + k1, r] = random.choice([-1, 1])
                    V[center_j*p + k2, r] = random.choice([-1, 1])
            
            M = compute_M(U, V, rank)
            W = find_W_for_M(M, rank, max_terms=2, coeff_range=(-4, 4))
            
            if W is not None and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_dense_search(rank, time_limit=None):
        """
        Try to find a decomposition by solving a linear system.
        For a fixed U, V, we need W such that M @ W.T = 1 (element-wise on rows).
        This is equivalent to: for each idx, sum_r M[idx,r] * W[idx,r] = 1
        """
        if time_limit is not None and time.time() - start_time > time_limit:
            return None
        
        num_trials = 30 if rank < 25 else 15
        
        for trial in range(num_trials):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            # Generate random U, V with small integer entries
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            for r in range(rank):
                # U column: 2-3 non-zeros with values in {-1, 0, 1, 2}
                num_nz_u = random.choice([2, 3])
                indices_u = random.sample(range(n*m), num_nz_u)
                for idx in indices_u:
                    U[idx, r] = random.choice([-2, -1, 1, 2])
                
                # V column: 2-3 non-zeros with values in {-1, 0, 1, 2}
                num_nz_v = random.choice([2, 3])
                indices_v = random.sample(range(m*p), num_nz_v)
                for idx in indices_v:
                    V[idx, r] = random.choice([-2, -1, 1, 2])
            
            M = compute_M(U, V, rank)
            W = find_W_for_M(M, rank, max_terms=2, coeff_range=(-6, 6))
            
            if W is not None and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_low_rank_specialized(rank, time_limit=None):
        """
        Specialized search for low ranks (15-25) using more targeted patterns.
        """
        if time_limit is not None and time.time() - start_time > time_limit:
            return None
        
        num_trials = 100 if rank <= 20 else 50
        
        for trial in range(num_trials):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Pattern: each column of U has exactly 2 non-zeros (one for each of 2 rows i)
            # Each column of V has exactly 2 non-zeros (one for each of 2 cols k)
            # This creates M with structure that might allow simple W
            
            for r in range(rank):
                # U: pick 2 different i values, same j
                j = random.randint(0, m-1)
                i1 = random.randint(0, n-1)
                i2 = random.randint(0, n-1)
                while i2 == i1:
                    i2 = random.randint(0, n-1)
                U[i1*m + j, r] = random.choice([-1, 1])
                U[i2*m + j, r] = random.choice([-1, 1])
                
                # V: pick 2 different k values, same j
                k1 = random.randint(0, p-1)
                k2 = random.randint(0, p-1)
                while k2 == k1:
                    k2 = random.randint(0, p-1)
                V[j*p + k1, r] = random.choice([-1, 1])
                V[j*p + k2, r] = random.choice([-1, 1])
            
            M = compute_M(U, V, rank)
            W = find_W_for_M(M, rank, max_terms=2, coeff_range=(-3, 3))
            
            if W is not None and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    # Known results: for 3x4x7, the rank is likely between 20-35
    # Start from lower ranks and work up
    
    ranks_to_try = list(range(18, min(n*m*p, 84)))
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Try different search strategies based on rank
        if rank <= 22:
            # Low rank: use specialized search with more trials
            result = try_low_rank_specialized(rank, time_limit=DEADLINE - 5)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
            
            if time.time() - start_time < DEADLINE - 10:
                result = try_structured_decomposition(rank, time_limit=DEADLINE - 5)
                if result is not None:
                    U, V, W = result
                    if rank < best_rank:
                        best_rank = rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    break
        
        elif rank <= 30:
            # Medium rank: use structured decomposition
            result = try_structured_decomposition(rank, time_limit=DEADLINE - 5)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
            
            if time.time() - start_time < DEADLINE - 10:
                result = try_dense_search(rank, time_limit=DEADLINE - 5)
                if result is not None:
                    U, V, W = result
                    if rank < best_rank:
                        best_rank = rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    break
        
        else:
            # Higher rank: use dense search
            result = try_dense_search(rank, time_limit=DEADLINE - 5)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
