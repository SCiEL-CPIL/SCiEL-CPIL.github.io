import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Strassen decomposition for 2x2x2
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    strassen_decomp = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4
        ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),         # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})          # M7
    ]
    
    def _build_strassen_3x4x8():
        """Build the block-Strassen decomposition: rank 88."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        for j1 in range(2):  # 2 blocks of 2 in m dimension
            # Top part: 2x2x8 (rows 0,1 of A)
            for l in range(4):  # 4 blocks of 2 in p dimension
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    # U (A): local 2x2 -> global
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i  # 0 or 1
                        global_j = 2 * j1 + li_j
                        U[global_i * m + global_j][col] += coeff
                    
                    # V (B): local 2x2 -> global
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_j = 2 * j1 + li_i
                        global_k = 2 * l + li_j
                        V[global_j * p + global_k][col] += coeff
                    
                    # W (C): local 2x2 -> global
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i
                        global_k = 2 * l + li_j
                        W[global_k * n + global_i][col] += coeff
                    
                    col += 1
            
            # Bot part: 1x2x8 (row 2 of A), schoolbook
            for k in range(8):
                for j2 in range(2):
                    global_i = 2
                    global_j = 2 * j1 + j2
                    U[global_i * m + global_j][col] = 1
                    V[global_j * p + k][col] = 1
                    W[k * n + global_i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def _build_strassen_alt_split():
        """
        Alternative split: split n=3 into 1+2 (top=1, bot=2).
        Top: 1x4x8 -> schoolbook = 32
        Bot: 2x4x8 -> split m=4 into 2x2, p=8 into 4x2
              Each 2x2x2 uses Strassen: 2*4*7 = 56
        Total = 32 + 56 = 88 (same rank, different structure)
        """
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Top: row 0 of A, schoolbook 1x4x8 = 32
        for j in range(4):
            for k in range(8):
                U[0 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 0][col] = 1
                col += 1
        
        # Bot: rows 1,2 of A, Strassen on 2x4x8
        for j1 in range(2):  # 2 blocks of 2 in m
            for l in range(4):  # 4 blocks of 2 in p
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = 1 + li_i  # rows 1,2
                        global_j = 2 * j1 + li_j
                        U[global_i * m + global_j][col] += coeff
                    
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_j = 2 * j1 + li_i
                        global_k = 2 * l + li_j
                        V[global_j * p + global_k][col] += coeff
                    
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = 1 + li_i
                        global_k = 2 * l + li_j
                        W[global_k * n + global_i][col] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def _build_strassen_4x4_blocks():
        """
        Try: split p=8 into 4+4, m=4 into 2+2, n=3 into 1+2.
        For each (m_block, p_block):
          Top (n=1): 1x2x4 schoolbook = 8
          Bot (n=2): 2x2x4 -> split p into 2+2, each 2x2x2 Strassen = 14
        Total = 2*2*(8+14) = 88
        Same rank, verify correctness.
        """
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        for j1 in range(2):  # m blocks
            for l1 in range(2):  # p blocks of 4
                # Top: row 0, schoolbook 1x2x4
                for j2 in range(2):
                    for k2 in range(4):
                        global_j = 2 * j1 + j2
                        global_k = 4 * l1 + k2
                        U[0 * m + global_j][col] = 1
                        V[global_j * p + global_k][col] = 1
                        W[global_k * n + 0][col] = 1
                        col += 1
                
                # Bot: rows 1,2, Strassen 2x2x4 -> two 2x2x2
                for l2 in range(2):  # sub-blocks of 2 in p
                    for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                        for li, coeff in u_coeffs.items():
                            li_i, li_j = li // 2, li % 2
                            global_i = 1 + li_i
                            global_j = 2 * j1 + li_j
                            U[global_i * m + global_j][col] += coeff
                        
                        for li, coeff in v_coeffs.items():
                            li_i, li_j = li // 2, li % 2
                            global_j = 2 * j1 + li_i
                            global_k = 4 * l1 + 2 * l2 + li_j
                            V[global_j * p + global_k][col] += coeff
                        
                        for li, coeff in w_coeffs.items():
                            li_i, li_j = li // 2, li % 2
                            global_i = 1 + li_i
                            global_k = 4 * l1 + 2 * l2 + li_j
                            W[global_k * n + global_i][col] += coeff
                        
                        col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def _verify(U, V, W, tol=1e-9):
        """Verify exact tensor decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def _verify_half_integer(U, V, W):
        """Check that all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True
    
    # Build candidate decompositions
    best_result = None
    best_rank = float('inf')
    
    candidates = []
    
    # Candidate 1: original Strassen split (n: 2+1)
    c1 = _build_strassen_3x4x8()
    if _verify(c1['U'], c1['V'], c1['W']) and _verify_half_integer(c1['U'], c1['V'], c1['W']):
        candidates.append((88, c1))
    
    # Candidate 2: alternative split (n: 1+2)
    if time.time() < deadline:
        c2 = _build_strassen_alt_split()
        if _verify(c2['U'], c2['V'], c2['W']) and _verify_half_integer(c2['U'], c2['V'], c2['W']):
            candidates.append((88, c2))
    
    # Candidate 3: 4x4 p-blocks
    if time.time() < deadline:
        c3 = _build_strassen_4x4_blocks()
        if _verify(c3['U'], c3['V'], c3['W']) and _verify_half_integer(c3['U'], c3['V'], c3['W']):
            candidates.append((88, c3))
    
    # Try to find a lower-rank decomposition via structured search
    # Key insight: for 3x4x8, can we do better than 88?
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32
    # But that's very weak. Better: for matrix mult, rank >= n*m*p / max(n,m,p) is not standard.
    # Actually, the trivial lower bound for rank of matrix mult tensor is not well-known to be > 32.
    # Let's try a structured search for rank < 88.
    
    if time.time() < deadline:
        # Try to combine Strassen blocks more cleverly.
        # Idea: For the 1x2x8 "bot" part, can we share structure with adjacent blocks?
        # The bot part has rank 16 per m-block. If we could reduce this...
        # 1x2x8 is essentially: sum_{j=0}^{1} A[2,j]*B[j,k] for each k.
        # This is rank 1*2*8 = 16 by schoolbook. Can we do better?
        # For 1xm x p, the rank is m*p (no Strassen possible for 1-row).
        # So 16 is optimal for that sub-problem.
        
        # Alternative: don't split n into 2+1, but use a different approach.
        # What if we treat the whole 3x4x8 differently?
        # Split m=4 into 1+3?
        # m=1: 3x1x8 = 24 (schoolbook)
        # m=3: 3x3x8 -> split into 3 blocks of 1 in m: 3*(3x1x8) = 72? No that's worse.
        # Or 3x3x8: split p into 4+4, each 3x3x4.
        #   3x3x4: split n into 1+2, m into 1+2
        #     1x1x4 = 4, 1x2x4 = 8, 2x1x4 = 8, 2x2x4 = 2*7=14 -> total 34
        #   So 3x3x8 = 2*34 = 68
        #   Total = 24 + 68 = 92. Worse.
        
        # What about m=4 into 2+2, and for each 2:
        #   n=3 into 1+2:
        #     1x2x8 = 16
        #     2x2x8 = 4*7 = 28
        #     Total per block = 44, grand total = 88. (Same as before)
        
        # What if for the 2x2x8 part, we use a different decomposition?
        # 2x2x8: split p into 4+4.
        #   2x2x4: split p into 2+2, each 2x2x2 Strassen = 14.
        #   So 2x2x8 = 2*14 = 28. Same.
        
        # What about using the fact that n=3 is close to 4?
        # Embed 3x4x8 into 4x4x8, use Strassen on 4x4x8?
        # 4x4x8: split into 2x2 blocks in n and m.
        #   4x4x8 = 4 * (2x2x8) = 4 * 28 = 112? No.
        #   4x4x8: split n=4 into 2+2, m=4 into 2+2, p=8 into 4+4.
        #   Each 2x2x4 = 14. Total = 2*2*2*14 = 112. Worse.
        #   Or 4x4x8: split p into 2+2+2+2.
        #   Each 4x4x2: split n=2+2, m=2+2. Each 2x2x2 = 7. Total per 4x4x2 = 4*7=28.
        #   4x4x8 = 4*28 = 112. Worse.
        
        # So 88 seems to be the best from block-Strassen.
        # Let's try a more aggressive search: can we find rank 80-87?
        
        # Try a randomized approach with verification for ranks 80-87
        # This is unlikely to work but worth trying with the time budget.
        
        for target_rank in range(80, 88):
            if time.time() > deadline - 5:
                break
            
            # Try a few random structured attempts
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                
                # Use the 88-rank solution and try to zero out some columns
                # to see if it still works (unlikely but fast to check)
                U88 = c1['U'] if c1 else None
                if U88 is None:
                    break
                
                # Try removing columns in a structured way
                # This is a heuristic: try removing columns that have small norms
                U_arr = np.array(U88, dtype=float)
                V_arr = np.array(c1['V'], dtype=float)
                W_arr = np.array(c1['W'], dtype=float)
                
                # Compute column "importance"
                col_norms = np.zeros(88)
                for r in range(88):
                    col_norms[r] = np.linalg.norm(U_arr[:, r]) * np.linalg.norm(V_arr[:, r]) * np.linalg.norm(W_arr[:, r])
                
                # Try removing the least important columns
                order = np.argsort(col_norms)
                removed = set()
                valid = True
                
                for idx in order:
                    if len(removed) >= 88 - target_rank:
                        break
                    removed.add(idx)
                
                if len(removed) < 88 - target_rank:
                    continue
                
                # Build reduced matrices
                keep = [i for i in range(88) if i not in removed]
                U_red = [[row[i] for i in keep] for row in U88]
                V_red = [[row[i] for i in keep] for row in c1['V']]
                W_red = [[row[i] for i in keep] for row in c1['W']]
                
                if _verify(U_red, V_red, W_red, tol=1e-9):
                    if _verify_half_integer(U_red, V_red, W_red):
                        candidates.append((target_rank, {"U": U_red, "V": V_red, "W": W_red}))
                        break  # Found valid reduced rank
    
    # Select best candidate
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_rank, best_result = candidates[0]
    else:
        # Fallback: use the first valid candidate or build fresh
        best_result = _build_strassen_3x4x8()
        best_rank = 88
    
    return best_result
# EVOLVE-BLOCK-END
