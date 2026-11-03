"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    total_terms = n * m * p
    
    # Precompute Strassen coefficients for a 2x2x2 block
    # Local indices: i,j,k in {0,1}
    # U index: i*m + j (local: i*2 + j)
    # V index: j*p + k (local: j*2 + k)
    # W index: k*n + i (local: k*2 + i)
    
    # We define the 7 Strassen products and their contributions to U, V, W
    # U_strassen[local_u_idx, t], V_strassen[local_v_idx, t], W_strassen[local_w_idx, t]
    # local_u_idx = i*2 + j
    # local_v_idx = j*2 + k
    # local_w_idx = k*2 + i
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # A00: u=0*2+0=0, A11: u=1*2+1=3
    # B00: v=0*2+0=0, B11: v=1*2+1=3
    # C00: w=0*2+0=0, C11: w=1*2+1=3
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10+=1, C11-=1
    # A10: u=1*2+0=2, A11: u=3
    # B00: v=0
    # C10: w=0*2+1=1, C11: w=3
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[1, 1] = 1; W_s[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01+=1, C11+=1
    # A00: u=0
    # B01: v=0*2+1=1, B11: v=3
    # C01: w=1*2+0=2, C11: w=3
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[2, 2] = 1; W_s[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00+=1, C10+=1
    # A11: u=3
    # B10: v=1*2+0=2, B00: v=0
    # C00: w=0, C10: w=1
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[1, 3] = 1
    
    # M5: (A00+A01)B11 -> C00-=1, C01+=1
    # A00: u=0, A01: u=0*2+1=1
    # B11: v=3
    # C00: w=0, C01: w=2
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C11+=1
    # A10: u=2, A00: u=0
    # B00: v=0, B01: v=1
    # C11: w=3
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00+=1
    # A01: u=1, A11: u=3
    # B10: v=2, B11: v=3
    # C00: w=0
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # Generate all possible 2x2x2 blocks
    # A block is defined by (i0, j0, k0) where i0 in [0, n-2], j0 in [0, m-2], k0 in [0, p-2]
    # It covers terms (i, j, k) for i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                # Create a set of covered terms (i, j, k)
                terms = set()
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            terms.add((i0 + di, j0 + dj, k0 + dk))
                blocks.append((i0, j0, k0, terms))
    
    # Greedy search for max disjoint blocks
    # We want to maximize the number of blocks to minimize rank.
    # Rank reduction per block is 1 (8 schoolbook terms -> 7 Strassen terms).
    # Total rank = (total_terms - 8*num_blocks) + 7*num_blocks = total_terms - num_blocks.
    # So maximizing num_blocks minimizes rank.
    
    best_blocks = []
    best_count = 0
    
    # Randomized restarts to escape local optima
    num_restarts = 500
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        random.shuffle(blocks)
        covered = set()
        current_blocks = []
        
        for block in blocks:
            i0, j0, k0, terms = block
            if not (terms & covered):
                covered |= terms
                current_blocks.append(block)
                
        if len(current_blocks) > best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
            # Early exit if we reach a theoretical limit or good score
            # Max possible blocks?
            # n=3, m=5, p=7.
            # i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3,4,5}
            # Total blocks = 2 * 4 * 6 = 48.
            # But they overlap.
            # If we find a high count, we can stop early?
            # Let's just run all restarts within time.

    # Construct U, V, W
    num_blocks = len(best_blocks)
    rank = total_terms - num_blocks
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    covered = set()
    
    # Add Strassen blocks
    for i0, j0, k0, terms in best_blocks:
        # Mapping local indices to global
        # Local u_idx = li*2 + lj -> Global u_idx = (i0+li)*m + (j0+lj)
        # Local v_idx = lj*2 + lk -> Global v_idx = (j0+lj)*p + (k0+lk)
        # Local w_idx = lk*2 + li -> Global w_idx = (k0+lk)*n + (i0+li)
        
        for t in range(7):
            # U contributions
            for li in range(2):
                for lj in range(2):
                    lu = li * 2 + lj
                    val = U_s[lu, t]
                    if val != 0:
                        gu = (i0 + li) * m + (j0 + lj)
                        U[gu, col] += val
                        
            # V contributions
            for lj in range(2):
                for lk in range(2):
                    lv = lj * 2 + lk
                    val = V_s[lv, t]
                    if val != 0:
                        gv = (j0 + lj) * p + (k0 + lk)
                        V[gv, col] += val
                        
            # W contributions
            for lk in range(2):
                for li in range(2):
                    lw = lk * 2 + li
                    val = W_s[lw, t]
                    if val != 0:
                        gw = (k0 + lk) * n + (i0 + li)
                        W[gw, col] += val
                        
            col += 1
        covered |= terms
        
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
                    
    # Trim if necessary (should be exact)
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col
        
    # Verify
    def verify(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        T = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            T += np.einsum('ij,jk,ki->ijk', U_r[:, :, c], V_r[:, :, c], W_r[:, :, c])
            
        return np.allclose(T, 1.0, atol=1e-9)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        U_sb = np.zeros((n * m, total_terms), dtype=float)
        V_sb = np.zeros((m * p, total_terms), dtype=float)
        W_sb = np.zeros((p * n, total_terms), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
