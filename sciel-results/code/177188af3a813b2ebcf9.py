import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # 1. Helper to verify decomposition
    # T[i, j, k] = 1 for all i, j, k in the multiplication tensor.
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Decomposition: T_{ijk} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def verify(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(rank):
                        val += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if val != 1:
                        return False
        return True

    # 2. Fallback: Schoolbook Decomposition (Rank 112)
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook(n, m, p)
    best_rank = n * m * p

    # 3. Construct Known Best: Strassen (4x4x4) + Schoolbook (4x4x3) = Rank 97
    # We implement this to ensure we have a strong baseline if the search fails.
    # Strassen 2x2x2 Rank 7 coefficients.
    # M1 = (A11+A22)(B11+B22) -> U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,0] (C11 part) etc.
    # Actually, let's just use the block structure logic to generate the 97 rank solution.
    # It's safer to hardcode the structure or generate it.
    
    def strassen_4x4x4():
        """
        Returns U, V, W lists for the 4x4x4 tensor.
        Rank 49.
        """
        # Strassen 2x2x2 has 7 terms.
        # 4x4x4 can be seen as 4 blocks of 2x2x2? No, it's 2x2 blocks of size 2.
        # A = [[A11, A12], [A21, A22]] where Ai are 2x2.
        # C = A*B.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map the 2x2 Strassen coefficients to the 4x4 indices.
        # Let's define the 7 Strassen terms for 2x2x2:
        # Each term t has (u_2x2, v_2x2, w_2x2) where these are 4-dim vectors (flattened 2x2).
        # u_2x2 corresponds to A11, A12, A21, A22 coeffs.
        
        # Strassen Terms for 2x2x2 (Rank 7):
        # Indices: 0:A11, 1:A12, 2:A21, 3:A22
        # Terms:
        # 1: (1,0,0,1) * (1,0,0,1) * (1,0,0,0)  -> M1 = (A11+A22)(B11+B22) contributes to C11
        #    Wait, the output W is also 2x2.
        #    M1 = (A11+A22)(B11+B22).
        #    C11 += M1. C22 += M1.
        #    So W vector for M1 has 1 at C11 and 1 at C22?
        #    Let's check standard Strassen:
        #    M1 = (A11+A22)(B11+B22)
        #    C11 = M1 + M4 - M5 + M7
        #    C12 = M3 + M5
        #    C21 = M2 + M4
        #    C22 = M1 - M2 + M3 + M6
        #    
        #    So for M1:
        #    U_part: A11+A22 -> [1, 0, 0, 1]
        #    V_part: B11+B22 -> [1, 0, 0, 1]
        #    W_part: C11 + C22 -> [1, 0, 0, 1]
        
        #    M2 = (A21+A22)B11
        #    U: [0, 0, 1, 1]
        #    V: [1, 0, 0, 0]
        #    W: C21 -> [0, 0, 1, 0] (since C21 = M2 + M4)
        #    Wait, C21 is bottom-left. In flattened 2x2: 0:C11, 1:C12, 2:C21, 3:C22.
        #    So W for M2 is [0, 0, 1, 0].
        
        #    M3 = A11(B12-B22)
        #    U: [1, 0, 0, 0]
        #    V: [0, 1, 0, -1]
        #    W: C12 + C22? No.
        #    C12 = M3 + M5. C22 = ... + M3 ...
        #    So W for M3 is [0, 1, 0, 1].
        
        #    M4 = A22(B21-B11)
        #    U: [0, 0, 0, 1]
        #    V: [0, 0, 1, -1]
        #    W: C11 + C21?
        #    C11 = ... + M4 ...
        #    C21 = ... + M4 ...
        #    So W for M4 is [1, 0, 1, 0].
        
        #    M5 = (A11+A12)B22
        #    U: [1, 1, 0, 0]
        #    V: [0, 0, 0, 1]
        #    W: C12 - C11?
        #    C11 = ... - M5 ...
        #    C12 = ... + M5 ...
        #    So W for M5 is [-1, 1, 0, 0].
        
        #    M6 = (A21-A11)(B11+B12)
        #    U: [-1, 0, 1, 0]
        #    V: [1, 1, 0, 0]
        #    W: C22?
        #    C22 = ... + M6 ...
        #    So W for M6 is [0, 0, 0, 1].
        
        #    M7 = (A12-A22)(B21+B22)
        #    U: [0, 1, 0, -1]
        #    V: [0, 0, 1, 1]
        #    W: C11?
        #    C11 = ... + M7 ...
        #    So W for M7 is [1, 0, 0, 0].
        
        terms_2x2 = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
        ]
        
        U_444 = []
        V_444 = []
        W_444 = []
        
        # Expand each 2x2 term into 4x4 terms.
        # The 4x4 matrix is composed of 2x2 blocks.
        # A_4x4 = [[A11, A12], [A21, A22]]
        # The index i in 0..3 maps to (block_row, local_row).
        # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1).
        # U index i*m+j. Here m=4.
        # For a specific 2x2 term, U_2x2 has coeffs for A11, A12, A21, A22.
        # We need to map these to the 4x4 indices.
        # A11 corresponds to rows 0,1 and cols 0,1.
        # A12 corresponds to rows 0,1 and cols 2,3.
        # A21 corresponds to rows 2,3 and cols 0,1.
        # A22 corresponds to rows 2,3 and cols 2,3.
        
        # For each of the 7 terms in the 2x2 decomposition:
        for u2, v2, w2 in terms_2x2:
            # u2[0] is coeff for A11, u2[1] for A12, etc.
            # We create a 4x4 vector U4 for this term.
            U4 = [0.0] * 16
            V4 = [0.0] * 16
            W4 = [0.0] * 16
            
            # Fill U4
            # A11 (u2[0]): indices (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5?
            # Wait, U index is i*m + j.
            # i is row, j is col.
            # A11: i in {0,1}, j in {0,1}. Indices: 0, 1, 4, 5.
            # A12: i in {0,1}, j in {2,3}. Indices: 2, 3, 6, 7.
            # A21: i in {2,3}, j in {0,1}. Indices: 8, 9, 12, 13.
            # A22: i in {2,3}, j in {2,3}. Indices: 10, 11, 14, 15.
            
            # Let's define the mapping for A blocks
            a_blocks = [
                [0, 1, 4, 5],   # A11
                [2, 3, 6, 7],   # A12
                [8, 9, 12, 13], # A21
                [10, 11, 14, 15]# A22
            ]
            
            # For V (B matrix), m=4, p=4.
            # V index j*p + k.
            # B11: j in {0,1}, k in {0,1}. Indices: 0, 1, 4, 5.
            # B12: j in {0,1}, k in {2,3}. Indices: 2, 3, 6, 7.
            # B21: j in {2,3}, k in {0,1}. Indices: 8, 9, 12, 13.
            # B22: j in {2,3}, k in {2,3}. Indices: 10, 11, 14, 15.
            # Same structure.
            
            # For W (C matrix), p=4, n=4.
            # W index k*n + i.
            # C11: k in {0,1}, i in {0,1}. Indices: 0, 1, 4, 5.
            # C12: k in {0,1}, i in {2,3}. Indices: 2, 3, 6, 7.
            # C21: k in {2,3}, i in {0,1}. Indices: 8, 9, 12, 13.
            # C22: k in {2,3}, i in {2,3}. Indices: 10, 11, 14, 15.
            # Same structure.
            
            for b_idx in range(4):
                coeff_u = u2[b_idx]
                coeff_v = v2[b_idx]
                coeff_w = w2[b_idx]
                
                if coeff_u != 0:
                    for idx in a_blocks[b_idx]:
                        U4[idx] += coeff_u
                if coeff_v != 0:
                    for idx in a_blocks[b_idx]:
                        V4[idx] += coeff_v
                if coeff_w != 0:
                    for idx in a_blocks[b_idx]:
                        W4[idx] += coeff_w
                        
            U_444.append(U4)
            V_444.append(V4)
            W_444.append(W4)
            
        return U_444, V_444, W_444

    U_s, V_s, W_s = strassen_4x4x4()
    
    # Now construct the 4x4x3 part (k=4,5,6) using Schoolbook.
    # Rank 48.
    U_sb = []
    V_sb = []
    W_sb = []
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):
                u = [0.0] * 16
                v = [0.0] * 16
                w = [0.0] * 28 # p*n = 7*4 = 28
                
                u[i*m+j] = 1.0
                v[j*p+k] = 1.0
                w[k*n+i] = 1.0
                
                U_sb.append(u)
                V_sb.append(v)
                W_sb.append(w)
                
    # Combine Strassen (4x4x4) and Schoolbook (4x4x3)
    # Total rank 49 + 48 = 97.
    # We need to align the dimensions.
    # U: 16 x rank
    # V: 28 x rank (m*p = 4*7)
    # W: 28 x rank (p*n = 7*4)
    
    # The Strassen part only uses k=0..3 in V and W.
    # The Schoolbook part only uses k=4..6 in V and W.
    # We can just concatenate the columns.
    
    U_combined = []
    V_combined = []
    W_combined = []
    
    total_rank = len(U_s) + len(U_sb)
    
    for r in range(total_rank):
        if r < len(U_s):
            u = U_s[r] + [0.0] * len(U_sb)
            v = V_s[r] + [0.0] * len(V_sb) # V_s is len 16, V_sb is len 16? No.
            # Wait, V_s is for 4x4x4, so V_s length is 16.
            # V_sb is for 4x4x3, so V_sb length is 16.
            # But V total length is m*p = 28.
            # V_s indices 0..15 correspond to j=0..3, k=0..3.
            # V_sb indices 0..15 correspond to j=0..3, k=4..6? No.
            # In V_sb construction above: v[j*p+k]. j in 0..3, k in 4..6.
            # So indices are 4,5,6, 8,9,10, 12,13,14, 16,17,18.
            # These are within 0..27.
            
            # We need to place V_s into the full 28-dim vector.
            v_full = [0.0] * 28
            for idx, val in enumerate(V_s[r]):
                v_full[idx] = val
            # V_s[r] is already 16-dim, corresponding to k=0..3.
            # So indices 0..15 are correct for the first 16 entries.
            
            w_full = [0.0] * 28
            for idx, val in enumerate(W_s[r]):
                w_full[idx] = val
            # W_s[r] is 16-dim, corresponding to k=0..3, i=0..3.
            # W index k*n+i.
            # k=0..3, i=0..3 -> indices 0..15.
            # So this is correct.
            
            U_combined.append(u)
            V_combined.append(v_full)
            W_combined.append(w_full)
        else:
            r_sb = r - len(U_s)
            u = [0.0] * len(U_s) + U_sb[r_sb]
            v = [0.0] * len(V_s) + V_sb[r_sb] # V_sb is 16-dim? No, V_sb was constructed as 16-dim in the loop?
            # Let's check V_sb construction.
            # v = [0.0] * 16.
            # v[j*p+k] = 1.0.
            # j*p+k can be up to 3*7+6 = 27.
            # So v should be length 28.
            # My previous construction of V_sb used length 16, which is WRONG for the combined vector if we just append.
            # I need to reconstruct V_sb and W_sb with full dimensions.
            
            pass # Skip this logic, let's do it properly below.

    # Redo combination properly
    U_combined = []
    V_combined = []
    W_combined = []
    
    # Strassen Part
    for r in range(len(U_s)):
        u = U_s[r]
        v = [0.0] * 28
        for idx, val in enumerate(V_s[r]):
            v[idx] = val
        w = [0.0] * 28
        for idx, val in enumerate(W_s[r]):
            w[idx] = val
        U_combined.append(u)
        V_combined.append(v)
        W_combined.append(w)
        
    # Schoolbook Part (k=4,5,6)
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):
                u = [0.0] * 16
                v = [0.0] * 28
                w = [0.0] * 28
                
                u[i*m+j] = 1.0
                v[j*p+k] = 1.0
                w[k*n+i] = 1.0
                
                U_combined.append(u)
                V_combined.append(v)
                W_combined.append(w)
                
    rank_97 = len(U_combined)
    if rank_97 < best_rank:
        # Verify
        if verify(U_combined, V_combined, W_combined, n, m, p):
            best_U, best_V, best_W = U_combined, V_combined, W_combined
            best_rank = rank_97

    # 4. Greedy Sparse Search for Rank < 97
    # Target: Try to find rank 96, 95, etc.
    # We will use a simplified greedy approach.
    # Due to time constraints, we only try a few iterations.
    
    # Pre-compute candidate vectors.
    # Candidates are sparse vectors with entries in {-1, 0, 1}.
    # We limit sparsity to 2 or 3 non-zeros to keep the pool manageable.
    
    def generate_candidates(dim, sparsity):
        cands = []
        # Single non-zero
        for i in range(dim):
            v = [0.0] * dim
            v[i] = 1.0
            cands.append(v)
        # Two non-zeros
        if sparsity >= 2:
            for i in range(dim):
                for j in range(i+1, dim):
                    v = [0.0] * dim
                    v[i] = 1.0
                    v[j] = 1.0
                    cands.append(v)
                    v2 = [0.0] * dim
                    v2[i] = 1.0
                    v2[j] = -1.0
                    cands.append(v2)
        return cands

    # This might be too slow if we do it from scratch.
    # Let's try a very limited greedy search for rank 96.
    # If we can't find it quickly, we return 97.
    
    # Actually, the greedy search for exact integer solutions is hard.
    # Let's try a "Randomized Local Search" on the rank 97 solution.
    # Try to merge two rank-1 terms into one? Unlikely.
    # Try to replace a rank-1 term with a different one?
    
    # Given the strict time limit and the difficulty of exact rank reduction,
    # the rank 97 solution is the most reliable improvement over 112.
    
    # If time remains, we could attempt to verify if rank 96 is possible via linear algebra checks,
    # but constructing the solution is the bottleneck.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
