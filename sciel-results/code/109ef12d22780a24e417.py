"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 84
    
    # Start with the trivial decomposition as baseline
    best_U = [[0] * trivial_rank for _ in range(n * m)]
    best_V = [[0] * trivial_rank for _ in range(m * p)]
    best_W = [[0] * trivial_rank for _ in range(p * n)]
    best_rank = trivial_rank
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # Strategy: try rank values from trivial_rank - 1 down
    # For each rank, use randomized search
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify if U, V, W form a valid decomposition of M_{n,m,p}"""
        # Use numpy for efficient verification
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: should sum to 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m + j]
                    v_val = V_arr[j*p + k]
                    w_val = W_arr[k*n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1) > 1e-10:
                        return False
        
        # Check invalid entries: should sum to 0
        # Sample some invalid combinations for efficiency
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    
                    is_valid = (i_a == i_c) and (j_a == j_b) and (k_b == k_c)
                    
                    if not is_valid:
                        s = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(s) > 1e-10:
                            return False
        return True
    
    # Try a few randomized searches for lower rank
    # Use small rational numbers (integers or half-integers)
    search_values = [0, 1, -1, Fraction(1,2), Fraction(-1,2)]
    
    for target_rank in range(trivial_rank - 1, max(trivial_rank - 10, 70), -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Randomly initialize with small values
            U = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(search_values) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Quick check: do the valid entries sum to 1?
            valid_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(target_rank):
                            s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                        if s != 1:
                            valid_ok = False
                            break
                    if not valid_ok:
                        break
                if not valid_ok:
                    break
            
            if valid_ok:
                # Full verification
                if verify_decomposition(U, V, W, target_rank, n, m, p):
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = target_rank
                    break
        
        if best_rank < trivial_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
