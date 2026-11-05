import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

def construct():
    t0 = time.time()
    
    # Schoolbook fallback
    rank_schoolbook = N * M * P
    U = [[0] * rank_schoolbook for _ in range(N * M)]
    V = [[0] * rank_schoolbook for _ in range(M * P)]
    W = [[0] * rank_schoolbook for _ in range(P * N)]
    
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
                
    # Attempt Rank 33 Construction
    # Partition J: [0,1], [2,3], [4]
    # Partition K: [0,1], [2,3], [4]
    # I: [0,1]
    
    # We will build a new U, V, W with rank 33
    # 4 Strassen blocks (rank 7 each) = 28
    # 4 Small blocks (rank 1 each) = 4
    # Total 32? 
    # Let's recount.
    # Blocks:
    # (J0, K0): 2x2x2 -> 7
    # (J0, K1): 2x2x2 -> 7
    # (J1, K0): 2x2x2 -> 7
    # (J1, K1): 2x2x2 -> 7
    # (J0, K2): 2x2x1 -> 1
    # (J1, K2): 2x2x1 -> 1
    # (J2, K0): 2x1x2 -> 1
    # (J2, K1): 2x1x2 -> 1
    # (J2, K2): 2x1x1 -> 1
    # Total: 28 + 1 + 1 + 1 + 1 + 1 = 33.
    
    new_U = [[0] * 33 for _ in range(N * M)]
    new_V = [[0] * 33 for _ in range(M * P)]
    new_W = [[0] * 33 for _ in range(P * N)]
    
    rank_offset = 0
    
    def add_strassen_block(i_start, j_start, k_start):
        nonlocal rank_offset
        # Local indices 0,1
        # A: i in {0,1}, j in {0,1}
        # B: j in {0,1}, k in {0,1}
        # C: k in {0,1}, i in {0,1}
        
        # Strassen factors
        # u: 4 entries (2x2 A)
        # v: 4 entries (2x2 B)
        # w: 4 entries (2x2 C)
        
        strassen_terms = [
            # (u, v, w)
            # u, v are dicts or lists of 4
            # w is list of 4
            # M1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, [1, 0, 0, 1]), # C00=1, C11=1 (Wait, C11 is index 3. C00 is 0. C01 is 1. C10 is 2.)
            # Let's be precise with W indices: k*N+i.
            # C[0,0] -> k=0, i=0 -> idx 0.
            # C[0,1] -> k=1, i=0 -> idx 1.
            # C[1,0] -> k=0, i=1 -> idx 2.
            # C[1,1] -> k=1, i=1 -> idx 3.
            
            # M1: C00+=1, C11+=1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, [1, 0, 0, 1]),
            # M2: C10+=1, C11-=1
            ({(1,0):1, (1,1):1}, {(0,0):1}, [0, 0, 1, -1]),
            # M3: C01+=1, C11+=1
            ({(0,0):1}, {(1,0):1}, [0, 1, 0, 1]),
            # M4: C00+=1, C10+=1
            ({(1,1):1}, {(1,1):1}, [1, 0, 1, 0]),
            # M5: C00-=1, C01+=1
            ({(0,0):1, (0,1):1}, {(1,1):1}, [-1, 1, 0, 0]),
            # M6: C11+=1
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, [0, 0, 0, 1]),
            # M7: C00+=1
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, [1, 0, 0, 0])
        ]
        
        for u_dict, v_dict, w_list in strassen_terms:
            # Add to U
            for (li, lj), val in u_dict.items():
                gi = i_start + li
                gj = j_start + lj
                new_U[gi * M + gj][rank_offset] += val
            
            # Add to V
            for (lj, lk), val in v_dict.items():
                gj = j_start + lj
                gk = k_start + lk
                new_V[gj * P + gk][rank_offset] += val
            
            # Add to W
            for lk in range(2):
                for li in range(2):
                    val = w_list[lk * 2 + li]
                    if val != 0:
                        gk = k_start + lk
                        gi = i_start + li
                        new_W[gk * N + gi][rank_offset] += val
            
            rank_offset += 1
        
        return rank_offset

    def add_rank1_block(i_start, j_start, k_start, j_size, k_size):
        nonlocal rank_offset
        # Block is N x j_size x k_size
        # Rank 1 decomposition:
        # u[i, 0] = 1
        # v[j, 0] = 1
        # w[k, i] = 1
        
        for li in range(N):
            gi = i_start + li
            # U
            for lj in range(j_size):
                gj = j_start + lj
                new_U[gi * M + gj][rank_offset] = 1
            
            # V
            for lj in range(j_size):
                gj = j_start + lj
                for lk in range(k_size):
                    gk = k_start + lk
                    new_V[gj * P + gk][rank_offset] = 1
            
            # W
            for lk in range(k_size):
                gk = k_start + lk
                new_W[gk * N + gi][rank_offset] = 1
        
        rank_offset += 1
        return rank_offset

    # Apply blocks
    # J partitions: 0: [0,1], 1: [2,3], 2: [4]
    # K partitions: 0: [0,1], 1: [2,3], 2: [4]
    
    # Strassen Blocks
    add_strassen_block(0, 0, 0) # J0, K0
    add_strassen_block(0, 0, 2) # J0, K1
    add_strassen_block(0, 2, 0) # J1, K0
    add_strassen_block(0, 2, 2) # J1, K1
    
    # Rank 1 Blocks
    add_rank1_block(0, 0, 4, 2, 1) # J0, K2 (2x2x1)
    add_rank1_block(0, 2, 4, 2, 1) # J1, K2 (2x2x1)
    add_rank1_block(0, 4, 0, 1, 2) # J2, K0 (2x1x2)
    add_rank1_block(0, 4, 2, 1, 2) # J2, K1 (2x1x2)
    add_rank1_block(0, 4, 4, 1, 1) # J2, K2 (2x1x1)
    
    if rank_offset != 33:
        print(f"Warning: Rank offset {rank_offset} != 33")
        # Fall back to schoolbook if construction fails
        return {"U": U, "V": V, "W": W}
        
    # Verify new decomposition
    if _verify_decomp(new_U, new_V, new_W, 33):
        return {"U": new_U, "V": new_V, "W": new_W}
    else:
        # Verification failed, return schoolbook
        return {"U": U, "V": V, "W": W}

def _verify_decomp(U, V, W, rank):
    """Verify that the decomposition is exact for the 2x5x5 matrix multiplication tensor."""
    if len(U) != N * M or len(V) != M * P or len(W) != P * N:
        return False
        
    # Check non-zero entries
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[i * M + j][r]) * Fraction(V[j * P + k][r]) * Fraction(W[k * N + i][r])
                if val != 1:
                    return False
                    
    # Check zero entries (sample for speed, or full if fast enough)
    # Full check: 10 * 25 * 10 = 2500 entries. 33 ranks. 82500 ops. Fast.
    for i in range(N):
        for j in range(M):
            for k in range(P):
                # This is the non-zero part, already checked.
                pass
                
    # Check other entries
    for a in range(N * M):
        for b in range(M * P):
            for c in range(P * N):
                i = a // M
                j = a % M
                j2 = b // P
                k = b % P
                k2 = c // N
                i2 = c % N
                
                if i == i2 and j == j2 and k == k2:
                    continue
                    
                val = Fraction(0)
                for r in range(rank):
                    val += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                if val != 0:
                    return False
                    
    return True
