"""Matrix multiplication 4,4,5 rank-minimal decomposition search."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit
    
    # Schoolbook decomposition as guaranteed fallback
    schoolbook_rank = n * m * p  # 80
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    def verify(U, V, W, rank):
        """Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
        V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
        W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
        return np.allclose(reconstructed, np.ones((n, m, p)), atol=1e-8)
    
    def try_strassen_block(n1, m1, p1, rank_target, time_budget, rng):
        """Try to find decomposition using Strassen-like block structure."""
        deadline_local = time.time() + time_budget
        # For 4x4 * 4x5, we can think of blocks
        # A is 4x4, B is 4x5, C is 4x5
        # Split: A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where B11,B12 are 2x5/2, B21,B22 are 2x5/2
        # Actually B is 4x5, so split into 2x3 and 2x2 blocks (or 2x5/2)
        
        # Use Strassen for 2x2 * 2x2 base case with rank 7
        # Then extend to handle the 5 columns
        
        # Approach: Use the known 7-multiplication algorithm for 2x2
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # For 4x4 * 4x5, split A into four 2x2 blocks
        # B into 2x3 and 2x2 (or 2x5/2 blocks)
        # This gives a rank of 7 * (number of block pairs)
        
        # Actually, let's use a different approach: 
        # Treat B as 4x5 = [4x3 | 4x2]
        # A*B = [A*[4x3] | A*[4x2]]
        # For 4x4 * 4x3: rank <= 7*2 + something
        # For 4x4 * 4x2: rank <= 7*1 + something
        
        # Simpler: just try random sparse decompositions
        n_rows_u = n * m
        n_rows_v = m * p
        n_rows_w = p * n
        
        for attempt in range(5000):
            if time.time() > deadline_local:
                break
            
            U = np.zeros((n_rows_u, rank_target), dtype=np.float64)
            V = np.zeros((n_rows_v, rank_target), dtype=np.float64)
            W = np.zeros((n_rows_w, rank_target), dtype=np.float64)
            
            # Sparse initialization: each row has ~2 non-zeros
            for c in range(rank_target):
                # Random sparse pattern
                nz_u = rng.randint(1, 3)
                nz_v = rng.randint(1, 3)
                nz_w = rng.randint(1, 3)
                
                idx_u = rng.sample(range(n_rows_u), min(nz_u, n_rows_u))
                idx_v = rng.sample(range(n_rows_v), min(nz_v, n_rows_v))
                idx_w = rng.sample(range(n_rows_w), min(nz_w, n_rows_w))
                
                for idx in idx_u:
                    U[idx, c] = rng.choice([-1.0, -0.5, 0.5, 1.0])
                for idx in idx_v:
                    V[idx, c] = rng.choice([-1.0, -0.5, 0.5, 1.0])
                for idx in idx_w:
                    W[idx, c] = rng.choice([-1.0, -0.5, 0.5, 1.0])
            
            if verify(U.tolist(), V.tolist(), W.tolist(), rank_target):
                return U, V, W
        
        return None
    
    def try_structured_decomposition(rank_target, time_budget, rng):
        """Try structured decompositions based on known algorithms."""
        deadline_local = time.time() + time_budget
        
        # For 4x4 * 4x5, try to use Kronecker product structure
        # 4x4 = 2x2 ⊗ 2x2
        # 4x5 = 2x2 ⊗ 2x2.5 (not clean)
        
        # Alternative: direct approach with patterned matrices
        n_rows_u = n * m
        n_rows_v = m * p
        n_rows_w = p * n
        
        # Try: U[i*m+j, c] depends on (i,j,c) pattern
        # V[j*p+k, c] depends on (j,k,c) pattern  
        # W[k*n+i, c] depends on (k,i,c) pattern
        
        # Use a circulant-like structure
        for variant in range(200):
            if time.time() > deadline_local:
                break
            
            U = np.zeros((n_rows_u, rank_target), dtype=np.float64)
            V = np.zeros((n_rows_v, rank_target), dtype=np.float64)
            W = np.zeros((n_rows_w, rank_target), dtype=np.float64)
            
            # Pattern-based initialization
            for c in range(rank_target):
                phase = 2 * np.pi * c / rank_target
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, c] = 1 if ((i + j + c) % 4 < 2) else 0
                    for k in range(p):
                        V[j*p+k, c] = 1 if ((j + k + c) % 4 < 2) else 0
                    for k2 in range(p):
                        W[k2*n+i, c] = 1 if ((k2 + i + c) % 4 < 2) else 0
            
            if verify(U.tolist(), V.tolist(), W.tolist(), rank_target):
                return U, V, W
        
        return None
    
    # Try various ranks in order of decreasing likelihood
    # Theoretical lower bound for 4x4*4x5 is at least max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # But practical lower bounds are higher
    
    target_ranks = [64, 60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20]
    rng = random.Random(12345)
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 2:
            break
        
        time_per_method = remaining / 3.0  # split time among methods
        
        # Method 1: Structured search
        result = try_structured_decomposition(target_rank, time_per_method, rng)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
        
        # Method 2: Random sparse search
        if time.time() < deadline - 1:
            result = try_strassen_block(n, m, p, target_rank, time_per_method, rng)
            if result is not None:
                U, V, W = result
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best_result
# EVOLVE-BLOCK-END
