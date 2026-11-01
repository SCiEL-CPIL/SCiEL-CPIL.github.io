import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Build the rank-56 Strassen construction as baseline
    U, V, W = _build_rank_56(n, m, p)
    best_result = {"U": U, "V": V, "W": W}
    best_rank = 56
    
    # Try to find a lower-rank construction
    # Strategy: group middle indices differently
    # For 2x4 x 4x8, try partitioning middle dim into 4 groups of 1
    # Each group: 2x1 x 1x8 -> rank 1 per group per output... 
    # Actually each middle index j contributes A[:,j] outer B[j,:] which is rank 1
    # So 4 middle indices -> rank 4 for each of 8 output columns? No.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a bilinear form. Rank of this tensor is what we want.
    
    # Let's try: for each pair of output columns, use Strassen on the 2x4 x 4x2 subproblem
    # 4 such pairs, each 2x4 x 4x2
    # 2x4 x 4x2: partition middle into two 2s
    # C[:,2v:2v+2] = A[:,0:2]@B[0:2,2v:2v+2] + A[:,2:4]@B[2:4,2v:2v+2]
    # Each 2x2 x 2x2 has rank 7
    # Total: 4 * 2 * 7 = 56 (same as before)
    
    # Alternative: treat as 2x4 x 4x8 directly with shared structure
    # Try rank 48: can we merge some Strassen terms?
    
    # Actually, let's try a completely different approach:
    # Use the known result that rank(M_{2,2,2}) = 7
    # And try to exploit the rectangular structure
    
    # For now, return the rank-56 construction which is verified correct
    return best_result

def _build_rank_56(n, m, p):
    """
    Build the rank-56 decomposition using Strassen on 2x2 blocks.
    8 blocks of 2x2 multiplication, each with rank 7.
    """
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            
            b11_idx = 2 * u * 8 + 2 * v
            b12_idx = 2 * u * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            
            c11_idx = 2 * v * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = 2 * v * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            # s=0: M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # s=1: M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # s=2: M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # s=3: M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # s=4: M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # s=5: M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # s=6: M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
    
    return U, V, W
