import numpy as np
import time
from scipy.optimize import linprog

START_TIME = time.time()
TIME_LIMIT = 140  # seconds, safety margin from 150s

DIMENSIONS = [2, 4, 7]

def _build_fixed_solution():
    """Build the known rank-50 solution."""
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        U[a_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        U[c_idx][col] += 1
        U[d_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        W[c11_idx][col] -= 1
        col += 1
        
        U[a_idx][col] += 1
        V[f_idx][col] += 1
        V[h_idx][col] -= 1
        W[c01_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        V[e_idx][col] -= 1
        W[c00_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        U[a_idx][col] += 1
        U[b_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] -= 1
        W[c01_idx][col] += 1
        col += 1
        
        U[c_idx][col] += 1
        U[a_idx][col] -= 1
        V[e_idx][col] += 1
        V[f_idx][col] += 1
        W[c11_idx][col] += 1
        col += 1
        
        U[b_idx][col] += 1
        U[d_idx][col] -= 1
        V[g_idx][col] += 1
        V[h_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx][col] += 1
        V[e_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        U[b_idx][col] += 1
        V[g_idx][col] += 1
        W[c00_idx][col] += 1
        col += 1
        
        U[c_idx][col] += 1
        V[e_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1
        
        U[d_idx][col] += 1
        V[g_idx][col] += 1
        W[c10_idx][col] += 1
        col += 1

    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}

def _verify_solution(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Check all (i, j, k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
                val = 0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # First, get the known solution
    result = _build_fixed_solution()
    best_rank = len(result['U'][0])
    
    # Try to find a lower-rank solution using LP-based approach
    # Strategy: try rank from best_rank-1 down to max(1, best_rank-10)
    # For each rank, set up LP to minimize L1 norm subject to constraints
    
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 30:
        return result
    
    # The constraints are bilinear, so we can't directly use linprog.
    # Instead, we'll try a heuristic: start from the rank-50 solution and
    # try to eliminate columns by checking if the remaining can satisfy constraints.
    
    # Convert to numpy for faster verification
    U_np = np.array(result['U'])
    V_np = np.array(result['V'])
    W_np = np.array(result['W'])
    
    # Try to reduce rank by removing zero-norm columns or redundant columns
    # First, check which columns are non-zero
    col_norms_U = np.linalg.norm(U_np, axis=0)
    col_norms_V = np.linalg.norm(V_np, axis=0)
    col_norms_W = np.linalg.norm(W_np, axis=0)
    
    # All columns should be non-zero in the current solution
    # Try a different approach: use the structure to find if we can merge columns
    
    # Given the complexity of the bilinear constraints, let's try a simple
    # perturbation approach: randomly zero out some columns and see if we can
    # adjust the remaining to still satisfy the constraints.
    
    # For now, return the known solution as it's already quite good
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
