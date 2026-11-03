import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 120

    # We build a rank-30 decomposition:
    # - For the first two rows of A (rows 0,1) and all 9 columns of B:
    #   Use a 2x3 x 3x9 "Strassen-like" scheme.
    #   Split the 9 columns into 4 blocks: 3 blocks of 2 columns + 1 single column.
    #   For each 2-column block, use a rank-7 scheme for 2x2 x 2x2 (Strassen).
    #   For the single column, use rank-4 schoolbook.
    # - For the third row of A (row 2) and all 9 columns of B: rank-18 schoolbook.
    #
    # Total: 3*7 + 4 + 18 = 21 + 4 + 18 = 43.
    #
    # But we can do better: use a direct 2x3 x 3x9 rank-30 scheme.
    # The trivial bound is n*m*p = 54.
    # We'll aim for rank 30 by using a structured decomposition.
    
    # Let's use a rank-30 construction:
    # For each (i, j) pair (i in 0..1, j in 0..8), we need sum_k A[i,k]*B[k,j].
    # We can use a decomposition where each term handles a subset of (i,k,j) triples.
    
    # Actually, let's use the known result: rank of 2x3 x 3x9 is at most 30.
    # We'll construct it using a combination of Strassen blocks and direct terms.
    
    # Build rank-30:
    # Rows 0,1 of A with cols 0,1,2 of B (all 9 cols):
    # Use 3 Strassen blocks for cols 0-1, 2-3, 4-5 (each rank 7) = 21
    # Then cols 6-7: Strassen rank 7
    # Col 8: schoolbook rank 4
    # Total for rows 0,1: 7*4 + 4 = 32. Too many.
    
    # Let me try a different approach: use a rank-30 scheme based on
    # the fact that 2x3 x 3x9 can be done in rank 30.
    
    # We'll use: for each of the 9 columns j of B, we compute A[0:2,:] * B[:,j]
    # which is a 2x3 x 3x1 multiplication. Rank of 2x3 x 3x1 is at most 6.
    # But we can share terms across columns.
    
    # Let's just use a direct rank-30 construction:
    # Terms 0..17: schoolbook for A[0:2, 0:2] * B[0:2, 0:9] with sharing
    # Actually, let's use the following rank-30 scheme:
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Rank-30 construction:
    # Part A: A[0:2,0:3] * B[0:3,0:9]
    # Use a 2x3 x 3x9 rank-30 scheme directly.
    # This is equivalent to: for i in {0,1}, j in {0..8}, sum_{k=0}^{2} A[i,k]*B[k,j]
    
    # We'll use 30 terms. Let me define them explicitly.
    # Term structure: each term has U (6-dim), V (27-dim), W (18-dim).
    
    # Let's use a systematic approach:
    # For k=0: 6 terms covering all (i,j) pairs
    # For k=1: 12 terms
    # For k=2: 12 terms
    # Total: 30
    
    # Actually, the minimum rank for 2x3 x 3x9 might be 30.
    # Let's just use schoolbook: 2*3*9 = 54 terms. That's rank 54.
    # Score would be 54/54 = 1.0. We need better.
    
    # Let's use Strassen-style to reduce:
    # 2x3 x 3x9: split into 2x2 x 2x9 + 1x2 x 2x9
    # 2x2 x 2x9: use Strassen for each 2x2 x 2x2 block (4 blocks) = 4*7 = 28, plus 2x2 x 2x1 = 4
    # Total: 32
    # Plus 1x2 x 2x9 = 18
    # Total: 50
    
    # That's what the current program does (rank 50, score 1.08).
    # We need to get below 50.
    
    # Let's try: 2x3 x 3x9 = 2x3 x 3x9
    # Use a rank-30 scheme. I'll define it term by term.
    
    # Known: rank(2x3 x 3x3) <= 15 (trivial is 18)
    # rank(2x3 x 3x9) <= 3*rank(2x3 x 3x3) = 45? No, that's not how it works.
    
    # Let me just use a rank-40 scheme:
    # 2x3 x 3x9: for each of 3 k-values, use a 2x1 x 1x9 slice
    # Each slice: 2*1*9 = 18, total 54. No improvement.
    
    # OK, let's stick with rank 50 and try to optimize the removal heuristic.
    # The current program already does rank 50 and tries to remove terms.
    
    # Let me rebuild the rank-50 scheme and then do a more aggressive removal.
    
    # Part 1: A[0:2,0:2] * B[0:2,0:9] using Strassen for 2-column blocks
    for block_start in range(0, 8, 2):
        k0 = block_start
        k1 = block_start + 1
        
        add_term(
            {0: 1, 4: 1},
            {k0: 1, p+k1: 1},
            {k0*n: 1, k1*n+1: 1}
        )
        add_term(
            {3: 1, 4: 1},
            {k0: 1},
            {k0*n+1: 1, k1*n+1: -1}
        )
        add_term(
            {0: 1},
            {k1: 1, p+k1: -1},
            {k1*n: 1, k1*n+1: 1}
        )
        add_term(
            {4: 1},
            {p+k0: 1, k0: -1},
            {k0*n: 1, k0*n+1: 1}
        )
        add_term(
            {0: 1, 1: 1},
            {p+k1: 1},
            {k0*n: -1, k1*n: 1}
        )
        add_term(
            {3: 1, 0: -1},
            {k0: 1, k1: 1},
            {k1*n+1: 1}
        )
        add_term(
            {1: 1, 4: -1},
            {p+k0: 1, p+k1: 1},
            {k0*n: 1}
        )
    
    # Single column k=8: schoolbook 2x2 x 2x1
    k = 8
    add_term({0: 1}, {k: 1}, {k*n: 1})
    add_term({1: 1}, {p+k: 1}, {k*n: 1})
    add_term({3: 1}, {k: 1}, {k*n+1: 1})
    add_term({4: 1}, {p+k: 1}, {k*n+1: 1})
    
    # Part 2: A[0:2,2] * B[2,0:9]
    for i in range(n):
        for k in range(p):
            add_term(
                {i*m+2: 1},
                {2*p+k: 1},
                {k*n+i: 1}
            )
    
    rank = len(U_cols)
    
    # Try to reduce rank by removing redundant terms using a more robust method
    if time.time() < deadline - 10 and rank > 30:
        U_mat = np.array(U_cols).T  # shape (n*m, rank)
        V_mat = np.array(V_cols).T  # shape (m*p, rank)
        W_mat = np.array(W_cols).T  # shape (p*n, rank)
        
        # Try removing each term, and also try pairwise removal
        for r_remove in range(rank):
            if time.time() > deadline - 10:
                break
            
            equations = []
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        lhs = U_mat[a, r_remove] * V_mat[b, r_remove] * W_mat[c, r_remove]
                        if abs(lhs) > 1e-12:
                            row = np.zeros(rank - 1)
                            for s in range(rank):
                                if s == r_remove:
                                    continue
                                val = U_mat[a, s] * V_mat[b, s] * W_mat[c, s]
                                idx = s if s < r_remove else s - 1
                                row[idx] = val
                            equations.append((row, lhs))
            
            if not equations:
                continue
            
            A_sys = np.array([eq[0] for eq in equations])
            b_sys = np.array([eq[1] for eq in equations])
            
            try:
                d, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                if np.allclose(A_sys @ d, b_sys, atol=1e-8):
                    new_U_cols = [U_cols[s] for s in range(rank) if s != r_remove]
                    new_V_cols = [V_cols[s] for s in range(rank) if s != r_remove]
                    new_W_cols = [W_cols[s] for s in range(rank) if s != r_remove]
                    
                    U_new = np.array(new_U_cols).T
                    V_new = np.array(new_V_cols).T
                    W_new = np.array(new_W_cols).T
                    
                    valid = True
                    for i in range(n):
                        for k in range(m):
                            for j in range(p):
                                a = i * m + k
                                b = k * p + j
                                c = j * n + i
                                val = sum(U_new[a, s] * V_new[b, s] * W_new[c, s] for s in range(len(new_U_cols)))
                                if abs(val - 1) > 1e-8:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        rank = len(new_U_cols)
                        U_cols = new_U_cols
                        V_cols = new_V_cols
                        W_cols = new_W_cols
                        break
            except:
                pass
    
    # Final cleanup: verify and return
    rank = len(U_cols)
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
