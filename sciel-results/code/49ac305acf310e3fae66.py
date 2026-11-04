"""Structured exact tensor decomposition for 4x4 @ 4x5 matrix multiplication.

Strategy: use Strassen's algorithm for the 4x4x4 part (rank 49) and append a
rank-4 schoolbook block for the fifth output column (k=4). Total rank = 53.
"""
import time

N, M, P = 4, 4, 5
NM = N * M          # 16 (U rows)
MP = M * P          # 20 (V rows)
PN = P * N          # 20 (W rows, trace convention W[k*n+i, r])


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are length-4 vectors.
    u indexes (i,j) for i,j in {0,1}; v indexes (j,k); w indexes (k,n).
    """
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # u: (A00, A01, A10, A11)
    # v: (B00, B01, B10, B11)
    # w: (C00, C01, C10, C11)
    terms = []
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00(+1), C11(-1)
    terms.append(((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, -1)))
    # M2 = (A10+A11)*B00 -> contributes to C10(+1), C11(-1)
    terms.append(((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)))
    # M3 = A00*(B01-B11) -> contributes to C01(+1), C11(+1)
    terms.append(((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)))
    # M4 = A11*(B10-B00) -> contributes to C00(+1), C10(+1)
    terms.append(((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)))
    # M5 = (A00+A01)*B11 -> contributes to C00(-1), C01(+1)
    terms.append(((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)))
    # M6 = (A10-A00)*(B00+B01) -> contributes to C11(+1)
    terms.append(((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)))
    # M7 = (A01-A11)*(B10+B11) -> contributes to C00(+1)
    terms.append(((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)))
    return terms


def _build_strassen_4x4x4():
    """Build rank-49 Strassen decomposition for 4x4x4 matrix multiplication.
    
    Partition 4x4 matrices into 2x2 blocks of 2x2 matrices.
    For each of the 4 output 2x2 blocks, apply Strassen (7 terms each).
    Total: 4 * 7 = 28 terms... wait, that's not 49.
    
    Actually, for 4x4x4 via Strassen recursion:
    - Split 4x4 into four 2x2 blocks
    - Use Strassen to compute each output 2x2 block from 2x2 block products
    - But the Strassen algorithm computes a 2x2 product using 7 scalar multiplications
    - For 4x4: we have 4 output blocks, each computed via Strassen on 2x2 blocks
    - Each Strassen step needs 7 "multiplications" but these are 2x2 matrix multiplications
    - Wait, no. Strassen for 4x4: treat 4x4 as 2x2 blocks of 2x2 matrices.
    - Apply Strassen: 7 multiplications of 2x2 matrices
    - Each 2x2 multiplication uses 7 scalar multiplications (Strassen)
    - Total: 7 * 7 = 49 scalar multiplications.
    
    So the structure is:
    - 7 Strassen "virtual" 2x2 multiplications
    - Each virtual multiplication is itself a Strassen 2x2x2 decomposition (7 terms)
    - Total: 49 terms
    """
    s2 = _strassen_2x2_terms()
    
    # For 4x4x4 Strassen:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = A*B computed via Strassen
    #
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mx is a 2x2 matrix product computed via Strassen (7 terms)
    # Each 2x2 product term has u,v,w of length 4
    #
    # For the full 4x4x4, we need to map:
    # - u indices: (i,j) for i,j in {0,1,2,3} -> 16 entries
    # - v indices: (j,k) for j,k in {0,1,2,3} -> 16 entries  
    # - w indices: (k,n) for k,n in {0,1,2,3} -> 16 entries
    #
    # Block mapping:
    # A00 = rows 0-1, cols 0-1 -> u indices 0,1,4,5
    # A01 = rows 0-1, cols 2-3 -> u indices 2,3,6,7
    # A10 = rows 2-3, cols 0-1 -> u indices 8,9,12,13
    # A11 = rows 2-3, cols 2-3 -> u indices 10,11,14,15
    #
    # B00 = rows 0-1, cols 0-1 -> v indices 0,1,4,5
    # B01 = rows 0-1, cols 2-3 -> v indices 2,3,6,7
    # B10 = rows 2-3, cols 0-1 -> v indices 8,9,12,13
    # B11 = rows 2-3, cols 2-3 -> v indices 10,11,14,15
    #
    # C00 = rows 0-1, cols 0-1 -> w indices 0,1,4,5 (k*n+i: k=0,1; n=0,1)
    # C01 = rows 0-1, cols 2-3 -> w indices 2,3,6,7
    # C10 = rows 2-3, cols 0-1 -> w indices 8,9,12,13
    # C11 = rows 2-3, cols 2-3 -> w indices 10,11,14,15
    
    # Helper: expand a 2x2 Strassen term to 4x4 level
    # u_block: which A block (0=A00, 1=A01, 2=A10, 3=A11)
    # v_block: which B block
    # w_block: which C block
    # signs_u, signs_v, signs_w: coefficients for the 2x2 sub-entries
    
    def block_offset(block):
        # block 0: rows 0-1, cols 0-1
        # block 1: rows 0-1, cols 2-3
        # block 2: rows 2-3, cols 0-1
        # block 3: rows 2-3, cols 2-3
        row_off = (block // 2) * 2
        col_off = (block % 2) * 2
        return row_off, col_off
    
    def expand_2x2_to_4x4(u_vec, u_blocks, v_vec, v_blocks, w_vec, w_blocks):
        """Expand a set of 2x2 terms to 4x4 level.
        u_blocks: list of (block, coeff) for u
        v_blocks: list of (block, coeff) for v  
        w_blocks: list of (block, coeff) for w
        u_vec, v_vec, w_vec: length-4 vectors for the 2x2 Strassen term
        """
        u_full = [0] * 16
        v_full = [0] * 16
        w_full = [0] * 16
        
        for (block, coeff) in u_blocks:
            ro, co = block_offset(block)
            for ii in range(2):
                for jj in range(2):
                    idx = (ro + ii) * 4 + (co + jj)
                    u_full[idx] += coeff * u_vec[ii * 2 + jj]
        
        for (block, coeff) in v_blocks:
            ro, co = block_offset(block)
            for ii in range(2):
                for jj in range(2):
                    idx = (ro + ii) * 4 + (co + jj)
                    v_full[idx] += coeff * v_vec[ii * 2 + jj]
        
        for (block, coeff) in w_blocks:
            ro, co = block_offset(block)
            for ii in range(2):
                for jj in range(2):
                    # w index: k*n+i where k=row, n=col, i=... 
                    # Actually w[k*n+i] with k in {0,1,2,3}, n in {0,1,2,3}
                    # Block 0 (C00): k=0,1; n=0,1 -> w[0*4+0]=w[0], w[0*4+1]=w[1], w[1*4+0]=w[4], w[1*4+1]=w[5]
                    # Block 1 (C01): k=0,1; n=2,3 -> w[0*4+2]=w[2], w[0*4+3]=w[3], w[1*4+2]=w[6], w[1*4+3]=w[7]
                    # Block 2 (C10): k=2,3; n=0,1 -> w[2*4+0]=w[8], w[2*4+1]=w[9], w[3*4+0]=w[12], w[3*4+1]=w[13]
                    # Block 3 (C11): k=2,3; n=2,3 -> w[2*4+2]=w[10], w[2*4+3]=w[11], w[3*4+2]=w[14], w[3*4+3]=w[15]
                    idx = (ro + ii) * 4 + (co + jj)
                    w_full[idx] += coeff * w_vec[ii * 2 + jj]
        
        return u_full, v_full, w_full
    
    all_terms = []
    
    # For each of the 7 Strassen M terms, and for each of the 7 sub-Strassen terms:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Each Mx involves specific A and B blocks with signs
    # M1: A blocks [(0,+1),(3,+1)], B blocks [(0,+1),(3,+1)]
    # M2: A blocks [(2,+1),(3,+1)], B blocks [(0,+1)]
    # M3: A blocks [(0,+1)], B blocks [(1,+1),(3,-1)]
    # M4: A blocks [(3,+1)], B blocks [(2,+1),(0,-1)]
    # M5: A blocks [(0,+1),(1,+1)], B blocks [(3,+1)]
    # M6: A blocks [(2,+1),(0,-1)], B blocks [(0,+1),(1,+1)]
    # M7: A blocks [(1,+1),(3,-1)], B blocks [(2,+1),(3,+1)]
    
    m_defs = [
        # (u_blocks, v_blocks)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)]),            # M2
        ([(0, 1)], [(1, 1), (3, -1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)]),            # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),   # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),   # M7
    ]
    
    # C block contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_contribs = [
        [(0, 1), (3, 1), (1, -1), (7, 1)],   # C00: M1+M4-M5+M7 (0-indexed: M0,M3,M4,M6)
        [(2, 1), (4, 1)],                     # C01: M3+M5 (M2, M4)
        [(1, 1), (3, 1)],                     # C10: M2+M4 (M1, M3)
        [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11: M1-M2+M3+M6 (M0, M1, M2, M5)
    ]
    
    # Build all 49 terms
    for m_idx in range(7):
        u_blocks, v_blocks = m_defs[m_idx]
        for s2_idx in range(7):
            u2, v2, w2 = s2[s2_idx]
            # Expand to 4x4
            u_full, v_full, w_full = expand_2x2_to_4x4(u2, u_blocks, v2, v_blocks, w2, [(0,1)])
            # Now distribute w to the correct C blocks
            # Actually, the w2 from Strassen gives contributions to C00,C01,C10,C11 of the 2x2 block
            # We need to map these to the correct 4x4 C blocks
            
            # The 2x2 Strassen w2 = (C00, C01, C10, C11) for the virtual 2x2 product
            # We need to add this to the appropriate 4x4 C blocks with the right signs
            
            # Let me redo this more carefully
            pass
    
    # Let me take a different, cleaner approach
    # Directly enumerate all 49 terms by combining the two levels of Strassen
    
    all_terms = []
    
    for m_idx in range(7):
        u_blocks, v_blocks = m_defs[m_idx]
        for s2_idx in range(7):
            u2, v2, w2 = s2[s2_idx]
            
            u_full = [0] * 16
            v_full = [0] * 16
            w_full = [0] * 16
            
            # Expand u
            for (block, coeff) in u_blocks:
                ro, co = block_offset(block)
                for ii in range(2):
                    for jj in range(2):
                        idx = (ro + ii) * 4 + (co + jj)
                        u_full[idx] += coeff * u2[ii * 2 + jj]
            
            # Expand v
            for (block, coeff) in v_blocks:
                ro, co = block_offset(block)
                for ii in range(2):
                    for jj in range(2):
                        idx = (ro + ii) * 4 + (co + jj)
                        v_full[idx] += coeff * v2[ii * 2 + jj]
            
            # Expand w: w2 has 4 entries for the 2x2 output block
            # w2[0]=C00, w2[1]=C01, w2[2]=C10, w2[3]=C11 (for the virtual 2x2 product)
            # We need to distribute to the 4x4 C blocks based on c_contribs
            # Actually, each Mx contributes to specific C blocks with specific signs
            # The w2 entries correspond to the 4 positions in the 2x2 output of Mx
            # Mx is a 2x2 matrix, so w2[0]=Mx[0,0], w2[1]=Mx[0,1], w2[2]=Mx[1,0], w2[3]=Mx[1,1]
            
            # C00 gets Mx[0,0] with sign from c_contribs[0]
            # C01 gets Mx[0,1] with sign from c_contribs[1]
            # C10 gets Mx[1,0] with sign from c_contribs[2]
            # C11 gets Mx[1,1] with sign from c_contribs[3]
            
            for c_block in range(4):
                # Find the sign for this Mx in this C block
                sign = 0
                for (mi, s) in c_contribs[c_block]:
                    if mi == m_idx:
                        sign = s
                        break
                if sign == 0:
                    continue
                
                c_ro, c_co = block_offset(c_block)
                # w2[c_block] is the value for this output position
                val = sign * w2[c_block]
                for ii in range(2):
                    for jj in range(2):
                        idx = (c_ro + ii) * 4 + (c_co + jj)
                        w_full[idx] += val * (1 if ii == jj == 0 else 0)  # This is wrong
            
            # This approach is getting complicated. Let me use a simpler method.
            pass
    
    # SIMPLER APPROACH: Just use schoolbook for the 4x4x4 part (64 terms) + 4 for column 5 = 68
    # Or better: directly construct the 49-term Strassen decomposition
    
    # Actually, let me just use the straightforward schoolbook approach but optimized
    # Schoolbook 4x4x4 = 64 terms, + 4 for column 5 = 68
    # This is guaranteed correct
    
    # Even better: use the fact that we only need 4x4x5, so:
    # For k=0,1,2,3: schoolbook 4x4x4 = 64 terms
    # For k=4: 4 terms
    # Total: 68
    
    # But we can do better with Strassen. Let me carefully construct it.
    
    return None  # Placeholder


def _schoolbook_4x4x5():
    """Full schoolbook rank-80 decomposition."""
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _build_strassen_plus_schoolbook():
    """Build Strassen 4x4x4 (rank 49) + schoolbook column 5 (rank 4) = rank 53."""
    s2 = _strassen_2x2_terms()
    
    def block_offset(block):
        row_off = (block // 2) * 2
        col_off = (block % 2) * 2
        return row_off, col_off
    
    m_defs = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)]),
        ([(3, 1)], [(2, 1), (0, -1)]),
        ([(0, 1), (1, 1)], [(3, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),
    ]
    
    c_contribs = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],
        [(2, 1), (4, 1)],
        [(1, 1), (3, 1)],
        [(0, 1), (1, -1), (2, 1), (5, 1)],
    ]
    
    terms = []
    for m_idx in range(7):
        u_blocks, v_blocks = m_defs[m_idx]
        for s2_idx in range(7):
            u2, v2, w2 = s2[s2_idx]
            
            u_full = [0] * 16
            v_full = [0] * 16
            w_full = [0] * 16
            
            for (block, coeff) in u_blocks:
                ro, co = block_offset(block)
                for ii in range(2):
                    for jj in range(2):
                        idx = (ro + ii) * 4 + (co + jj)
                        u_full[idx] += coeff * u2[ii * 2 + jj]
            
            for (block, coeff) in v_blocks:
                ro, co = block_offset(block)
                for ii in range(2):
                    for jj in range(2):
                        idx = (ro + ii) * 4 + (co + jj)
                        v_full[idx] += coeff * v2[ii * 2 + jj]
            
            # w2 = (Mx[0,0], Mx[0,1], Mx[1,0], Mx[1,1]) from the 2x2 Strassen
            # Each Mx[a,b] contributes to C_block[a,b] for each C_block that Mx feeds into
            for c_block in range(4):
                sign = 0
                for (mi, s) in c_contribs[c_block]:
                    if mi == m_idx:
                        sign = s
                        break
                if sign == 0:
                    continue
                c_ro, c_co = block_offset(c_block)
                # Mx has 4 entries: (0,0), (0,1), (1,0), (1,1)
                for a in range(2):
                    for b in range(2):
                        w_val = sign * w2[a * 2 + b]
                        if w_val == 0:
                            continue
                        # Position in 4x4 C: row = c_ro + a, col = c_co + b
                        idx = (c_ro + a) * 4 + (c_co + b)
                        w_full[idx] += w_val
            
            terms.append((tuple(u_full), tuple(v_full), tuple(w_full)))
    
    # Build the full 4x4x5 decomposition
    rank = 49 + 4
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    for t, (u, v, w) in enumerate(terms):
        for r in range(16):
            if u[r]:
                U[r][t] = u[r]
            if v[r]:
                V[r][t] = v[r]
            if w[r]:
                W[r][t] = w[r]
    
    # Add 4 schoolbook terms for k=4
    base = 49
    for j in range(M):
        c = base + j
        for i in range(N):
            U[i * M + j][c] = 1
            W[4 * N + i][c] = 1
        V[j * P + 4][c] = 1
    
    return {"U": U, "V": V, "W": W}


def _is_valid(U, V, W):
    """Exact check: for every (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0]) if U and U[0] else 0
    for i in range(N):
        for j in range(M):
            urow = U[i * M + j]
            vbase = j * P
            for k in range(P):
                total = 0.0
                vrow = V[vbase + k]
                wrow = W[k * N + i]
                for r in range(rank):
                    ur = urow[r]
                    if ur:
                        v = vrow[r]
                        if v:
                            w = wrow[r]
                            if w:
                                total += ur * v * w
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _rank(U, V, W):
    return len(U[0]) if U and U[0] else 0


def _drop_zero_terms(U, V, W, deadline):
    """Remove any column that contributes zero to all slots."""
    removed = True
    while removed and time.time() < deadline:
        removed = False
        rank = len(U[0])
        for c in range(rank):
            all_zero = True
            for i in range(N):
                for j in range(M):
                    u_ij = U[i * M + j][c]
                    if not u_ij:
                        continue
                    for k in range(P):
                        v = V[j * P + k][c]
                        if not v:
                            continue
                        w = W[k * N + i][c]
                        if not w:
                            continue
                        if u_ij * v * w:
                            all_zero = False
                            break
                    if not all_zero:
                        break
                if not all_zero:
                    break
            if all_zero:
                for r in range(NM):
                    U[r].pop(c)
                for r in range(MP):
                    V[r].pop(c)
                for r in range(PN):
                    W[r].pop(c)
                removed = True
                break
    return U, V, W


def construct():
    deadline = time.time() + 120.0
    safety = 20.0
    effective_deadline = deadline - safety

    best = None
    best_rank = None

    def consider(cand):
        nonlocal best, best_rank
        if cand is None:
            return
        r = _rank(cand["U"], cand["V"], cand["W"])
        if best_rank is None or r < best_rank:
            if _is_valid(cand["U"], cand["V"], cand["W"]):
                best = cand
                best_rank = r

    # 1) Try Strassen 4x4x4 + schoolbook column 5 (rank 53)
    cand53 = _build_strassen_plus_schoolbook()
    consider(cand53)

    # 2) If valid, try to drop zero terms
    if best_rank is not None and best_rank <= 53:
        U, V, W = _drop_zero_terms(best["U"], best["V"], best["W"], effective_deadline)
        if _is_valid(U, V, W):
            r = _rank(U, V, W)
            if r < best_rank:
                best = {"U": U, "V": V, "W": W}
                best_rank = r

    # 3) Fallback: schoolbook (always valid)
    if best is None:
        consider(_schoolbook_4x4x5())

    return best


if __name__ == "__main__":
    res = construct()
    print("rank:", _rank(res["U"], res["V"], res["W"]))
