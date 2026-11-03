"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    
    Strategy:
    1. The decomposition is separable per (i,k) pair.
    2. For each (i,k), we need sum_c M[i,k,c] * W[k,i,c] = 1 where M[i,k,c] = sum_j U[i,j,c]*V[j,k,c].
    3. We search for structured U, V patterns that yield clean W values.
    4. Try multiple pattern families: random, block-structured, and combinatorial designs.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # 15 seconds margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Helper to check if a value is a clean half-integer
    def is_clean_half_int(val, tol=1e-9):
        val2 = val * 2.0
        nearest_int = round(val2)
        if abs(val2 - nearest_int) < tol:
            return True
        return False
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_decomposition(U, V, target_rank):
        """
        Given U and V, try to find a clean W.
        For each (i,k), solve: sum_c M[i,k,c] * W[k,i,c] = 1
        where M[i,k,c] = sum_j U[i*m+j][c] * V[j*p+k][c]
        
        Strategy: Find sparse W with clean half-integer entries.
        """
        # Compute M[i,k,c] for all i,k,c
        M = {}
        for i in range(n):
            for k in range(p):
                for c in range(target_rank):
                    m_val = 0.0
                    for j in range(m):
                        m_val += U[i * m + j][c] * V[j * p + k][c]
                    M[(i, k, c)] = m_val
        
        W_candidate = [[0.0] * target_rank for _ in range(p * n)]
        
        # For each (i,k), find a set of columns c such that sum of W[k,i,c]*M[i,k,c] = 1
        # with W[k,i,c] being clean half-integers.
        # Try single-column solutions first, then two-column, etc.
        
        for i in range(n):
            for k in range(p):
                found = False
                
                # Try single column: W[k,i,c] = 1/M[i,k,c] must be clean half-int
                for c in range(target_rank):
                    m_val = M[(i, k, c)]
                    if abs(m_val) < 1e-9:
                        continue
                    w_val = 1.0 / m_val
                    if is_clean_half_int(w_val):
                        W_candidate[k * n + i][c] = w_val
                        found = True
                        break
                
                if found:
                    continue
                
                # Try two-column solutions: W1*M1 + W2*M2 = 1
                # where W1, W2 are clean half-integers
                # W1 = (1 - W2*M2) / M1
                # Try small half-integer values for W2: -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2
                for c1 in range(target_rank):
                    m1 = M[(i, k, c1)]
                    if abs(m1) < 1e-9:
                        continue
                    for c2 in range(target_rank):
                        if c2 == c1:
                            continue
                        m2 = M[(i, k, c2)]
                        if abs(m2) < 1e-9:
                            continue
                        for w2_num in range(-4, 5):  # w2 = w2_num / 2
                            w2 = w2_num / 2.0
                            w1 = (1.0 - w2 * m2) / m1
                            if is_clean_half_int(w1):
                                W_candidate[k * n + i][c1] = w1
                                W_candidate[k * n + i][c2] = w2
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                
                if not found:
                    return None  # Failed to find clean W for this (i,k)
        
        return W_candidate
    
    def generate_random_UV(target_rank, density_u=0.35, density_v=0.35):
        """Generate random sparse U and V with entries in {-1, 0, 1}."""
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        
        for c in range(target_rank):
            for i in range(n * m):
                if random.random() < density_u:
                    U[i][c] = random.choice([-1.0, 1.0])
            for j in range(m * p):
                if random.random() < density_v:
                    V[j][c] = random.choice([-1.0, 1.0])
        
        return U, V
    
    def generate_block_UV(target_rank):
        """
        Generate block-structured U and V.
        Divide the rank into blocks, each block handles a subset of (i,k) pairs.
        """
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        
        # Number of (i,k) pairs is n*p = 12
        # Each block should handle 1-2 (i,k) pairs
        num_blocks = min(target_rank, 12)
        block_size = max(1, target_rank // num_blocks)
        
        block_idx = 0
        for i in range(n):
            for k in range(p):
                if block_idx >= num_blocks:
                    break
                start_c = block_idx * block_size
                end_c = min(start_c + block_size, target_rank)
                
                # For this (i,k) pair, assign columns start_c to end_c-1
                # U[i,j,c] should be non-zero for some j
                # V[j,k,c] should be non-zero for some j
                # M[i,k,c] = sum_j U[i,j,c] * V[j,k,c] should be "nice"
                
                for c in range(start_c, end_c):
                    # Pick a random j to make non-zero
                    j = random.randint(0, m - 1)
                    sign_u = random.choice([-1.0, 1.0])
                    sign_v = random.choice([-1.0, 1.0])
                    U[i * m + j][c] = sign_u
                    V[j * p + k][c] = sign_v
                    # M[i,k,c] = sign_u * sign_v = +/-1
                    # So W[k,i,c] = +/-1, which is clean!
                
                block_idx += 1
        
        return U, V
    
    def generate_hadamard_UV(target_rank):
        """
        Generate U and V using Hadamard-like structures for better conditioning.
        """
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        
        # Use a structured pattern based on modular arithmetic
        for c in range(target_rank):
            # For each column, assign a pattern
            pattern_seed = c
            for i in range(n):
                for j in range(m):
                    # Deterministic pseudo-random based on position
                    val = ((i * 7 + j * 13 + pattern_seed * 3) % 5)
                    if val < 2:
                        U[i * m + j][c] = 1.0
                    elif val < 4:
                        U[i * m + j][c] = -1.0
                    else:
                        U[i * m + j][c] = 0.0
            
            for k in range(p):
                for j in range(m):
                    val = ((k * 11 + j * 17 + pattern_seed * 5) % 5)
                    if val < 2:
                        V[j * p + k][c] = 1.0
                    elif val < 4:
                        V[j * p + k][c] = -1.0
                    else:
                        V[j * p + k][c] = 0.0
        
        return U, V
    
    # Try different rank values and pattern types
    pattern_types = ['random', 'block', 'hadamard']
    
    for target_rank in range(24, 36):
        if time.time() > deadline - 5.0:
            break
        
        found = False
        
        # Number of attempts per pattern type
        attempts_per_type = 200 if target_rank <= 26 else 100
        
        for ptype in pattern_types:
            if time.time() > deadline - 5.0:
                break
            
            for attempt in range(attempts_per_type):
                if time.time() > deadline - 5.0:
                    break
                
                if ptype == 'random':
                    U, V = generate_random_UV(target_rank)
                elif ptype == 'block':
                    U, V = generate_block_UV(target_rank)
                elif ptype == 'hadamard':
                    U, V = generate_hadamard_UV(target_rank)
                else:
                    U, V = generate_random_UV(target_rank)
                
                W_candidate = try_decomposition(U, V, target_rank)
                
                if W_candidate is not None:
                    if check_decomposition(U, V, W_candidate, target_rank):
                        best_result = {"U": U, "V": V, "W": W_candidate}
                        best_rank = target_rank
                        found = True
                        break
            
            if found:
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
