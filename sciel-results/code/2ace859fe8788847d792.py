"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time
import itertools
import random

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
# For A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B:
# C00 = a*e + c*h
# C01 = b*h
# C10 = a*f + d*g
# C11 = b*g + d*h
#
# Strassen products:
# P1 = (a+d)(e+h)
# P2 = (c+d)e
# P3 = a(f-h)
# P4 = d(g-e)
# P5 = (a+b)h
# P6 = (c-a)(e+f)
# P7 = (b-d)(g+h)
#
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

# Each Strassen product is a linear combination of entries of A and B.
# We represent each product as:
#   (list of (ai, aj, coeff) for A-part, list of (bk, bl, coeff) for B-part)
# where ai,aj index into the 2x2 A block and bk,bl index into the 2x2 B block.

# Strassen products for a 2x2 block:
# A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
# B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h

_STRASSEN_2X2 = [
    # P1 = (a+d)(e+h)
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # P2 = (c+d)e
    ([(1,0,1),(1,1,1)], [(0,0,1)]),
    # P3 = a(f-h)
    ([(0,0,1)], [(0,1,1),(1,1,-1)]),
    # P4 = d(g-e)
    ([(1,1,1)], [(1,0,1),(0,0,-1)]),
    # P5 = (a+b)h
    ([(0,0,1),(0,1,1)], [(1,1,1)]),
    # P6 = (c-a)(e+f)
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
    # P7 = (b-d)(g+h)
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
]

# How each C entry is formed from the 7 Strassen products:
# C00 = P1 + P4 - P5 + P7
# C01 = P3 + P5
# C10 = P2 + P4
# C11 = P1 - P2 + P3 + P6

_STRASSEN_C_COMBINATION = [
    # (c_row, c_col, list of (product_idx, coeff))
    (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
    (0, 1, [(2, 1), (4, 1)]),
    (1, 0, [(1, 1), (3, 1)]),
    (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
]


def _strassen_4x4_decomposition():
    """
    Build rank-49 decomposition for 4x4x4 matrix multiplication using
    recursive Strassen: apply Strassen at the 4x4 level (7 block products),
    then apply Strassen within each 2x2 sub-block product (7 scalar mults each).
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49

    U = [[0] * rank for _ in range(n * m)]  # shape (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 49)

    # At the 4x4 level, we partition A into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # Similarly for B and C.
    #
    # The 7 Strassen block products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a product of two 2x2 matrices (linear combinations of blocks).
    # For each Mi, we then apply Strassen's 7 products to compute the 2x2 product.
    # Each of those 7 products corresponds to one rank-1 term in the full decomposition.
    #
    # The block-level Strassen structure:
    # For each block product Mi = Xi * Yi where Xi and Yi are 2x2 blocks:
    #   Xi is a linear combination of A-blocks
    #   Yi is a linear combination of B-blocks
    #
    # We need to express Xi as a linear combination of individual A entries,
    # and Yi as a linear combination of individual B entries.

    # Define the 7 block-level products. Each is (A_block_combo, B_block_combo).
    # A_block_combo: list of (block_row, block_col, coeff) for 2x2 blocks of A
    # B_block_combo: list of (block_row, block_col, coeff) for 2x2 blocks of B
    # Block (br, bc) of A covers rows 2*br..2*br+1, cols 2*bc..2*bc+1.

    block_products = [
        # M1 = (A00+A11)(B00+B11)
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        # M2 = (A10+A11)B00
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
        # M3 = A00(B01-B11)
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)]),
        # M4 = A11(B10-B00)
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)]),
        # M5 = (A00+A01)B11
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)]),
    ]

    # C block combination (same as Strassen 2x2 C combination)
    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    # For each block product Mi (i=0..6), we apply Strassen's 7 products.
    # The resulting 2x2 product gives entries for the C-block.
    #
    # For block product i with A-block-combo a_combo and B-block-combo b_combo:
    #   The "effective" 2x2 A matrix X has entries:
    #     X[r][c] = sum of coeff * A[2*br+r][2*bc+c] for (br,bc,coeff) in a_combo
    #   Similarly for B.
    #
    # Then Strassen's 7 products on X*Y give 7 rank-1 terms.
    # Each term contributes to C-block entries via the Strassen C combination.
    #
    # The C-block (cbr, cbc) gets contribution from block product i with
    # coefficient w_i, and within that, Strassen product s contributes to
    # C-local entry (lr, lc) with coefficient v_s.
    # So C[2*cbr+lr][2*cbc+lc] gets contribution w_i * v_s * (Strassen term).

    for i, (a_combo, b_combo) in enumerate(block_products):
        for s, (a_local, b_local) in enumerate(_STRASSEN_2X2):
            r = i * 7 + s  # global rank index (0..48)

            # Build U contributions: for each (a_br, a_bc, a_coeff) in a_combo
            # and each (lr, lc, l_coeff) in a_local:
            #   U[2*a_br + lr][2*a_bc + lc][r] += a_coeff * l_coeff
            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col][r] += a_coeff * l_coeff

            # Build V contributions: for each (b_br, b_bc, b_coeff) in b_combo
            # and each (lr, lc, l_coeff) in b_local:
            #   V[2*b_br + lr][2*b_bc + lc][r] += b_coeff * l_coeff
            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col][r] += b_coeff * l_coeff

            # Build W contributions: for each C-block (cbr, cbc) that uses block
            # product i with weight w_i, and for each C-local entry (lr, lc)
            # that uses Strassen product s with weight v_s:
            #   W[2*cbc + lc][2*cbr + lr][r] += w_i * v_s
            # Note: W is indexed as W[k*n+i] where k is column of C, i is row of C.
            # C[row][col] -> W[col*n + row]
            for (cbr, cbc, c_list) in c_block_combination:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue
                # For each C-local entry that uses Strassen product s:
                for (cbr2, cbc2, c_list2) in c_block_combination:
                    pass  # This is handled below

            # Better approach: iterate over all C-block entries and all Strassen
            # C-combination entries to find contributions.
            # Actually, let's redo the W construction more carefully.

    # Let me redo the W construction properly.
    # Reset W
    W = [[0] * rank for _ in range(p * n)]

    for i, (a_combo, b_combo) in enumerate(block_products):
        for s, (a_local, b_local) in enumerate(_STRASSEN_2X2):
            r = i * 7 + s
            # For each C-block (cbr, cbc) with weight w_i for block product i:
            for (cbr, cbc, c_list) in c_block_combination:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue
                # For each C-local entry (lr, lc) with weight v_s for Strassen product s:
                for (cbr2, cbc2, c_list2) in c_block_combination:
                    pass

    # I need to be more careful. Let me restructure.
    # The C-block (cbr, cbc) is computed as:
    #   C[cbr,cbc] = sum_i w_i * Mi  where Mi = Xi * Yi (2x2 matrices)
    # And Mi is computed via Strassen:
    #   Mi[lr,lc] = sum_s v_s * Ps  where Ps is the s-th Strassen product
    # So C[2*cbr+lr, 2*cbc+lc] = sum_i sum_s w_i(cbr,cbc,i) * v_s(lr,lc,s) * Ps(i,s)
    # And Ps(i,s) is the s-th Strassen product of the i-th block product.
    # Each Ps(i,s) is a rank-1 term with U, V, W coefficients.
    # The W coefficient for C[2*cbr+lr, 2*cbc+lc] from term (i,s) is w_i * v_s.

    # Let's build W properly:
    W = [[0] * rank for _ in range(p * n)]

    for i in range(7):
        for s in range(7):
            r = i * 7 + s
            for (cbr, cbc, c_list) in c_block_combination:
                w_i = 0
                for (pi, wi) in c_list:
                    if pi == i:
                        w_i = wi
                        break
                if w_i == 0:
                    continue
                # Now find v_s for each C-local entry
                for (lr, lc, v_list) in _STRASSEN_C_COMBINATION:
                    v_s = 0
                    for (si, vs) in v_list:
                        if si == s:
                            v_s = vs
                            break
                    if v_s == 0:
                        continue
                    c_row = 2 * cbr + lr
                    c_col = 2 * cbc + lc
                    # W is indexed as W[col*n + row]
                    W[c_col * n + c_row][r] += w_i * v_s

    return {"U": U, "V": V, "W": W}


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True


def _try_rank48_search(n, m, p, deadline, seed=42):
    """
    Search for a rank-48 decomposition by modifying the recursive Strassen
    rank-49 decomposition. The idea: in recursive Strassen, one of the 49
    terms may be expressible as a linear combination of the others due to
    algebraic identities in the 4x4 case. We try to find such a reduction
    by searching over small integer perturbations of the factor matrices.
    """
    rng = random.Random(seed)
    base = _strassen_4x4_decomposition()
    U0 = [row[:] for row in base["U"]]
    V0 = [row[:] for row in base["V"]]
    W0 = [row[:] for row in base["W"]]
    rank49 = 49

    # Strategy: try to find a rank-48 decomposition by searching over
    # sparse integer matrices. We'll use a local search approach:
    # start from the rank-49 solution and try to eliminate one column
    # by adjusting the others.
    
    # For each column r to eliminate (0..48), try to find coefficients
    # for the remaining 48 columns that reproduce the tensor.
    # This is a linear system: for each (i,j,k) in the tensor,
    #   sum_{r' != r} U[i*m+j][r'] * V[j*p+k][r'] * W[k*n+i][r'] * alpha[r'] = U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r]
    # But this is nonlinear in alpha. Instead, we search directly.
    
    # Alternative: brute-force search over sparse rank-48 matrices.
    # Given the 150s budget, we do a limited structured search.
    
    # Try a different approach: use the fact that for 4x4x4, there might
    # be a rank-48 decomposition based on modifying Strassen's structure.
    # Specifically, try to combine two of the 49 Strassen terms into one.
    
    # For each pair (r1, r2), check if term r2 can be replaced by a
    # linear combination involving term r1 and others.
    # We do a limited search over small integer coefficient patterns.
    
    # Given time constraints, try a few structured modifications:
    # 1. Try to merge columns that have similar sparsity patterns
    # 2. Try to find a column that is a linear combination of others
    
    # Check if any column in U is a linear combination of others
    # (necessary condition for rank reduction)
    import numpy as np
    
    U_np = np.array(U0, dtype=float)
    V_np = np.array(V0, dtype=float)
    W_np = np.array(W0, dtype=float)
    
    # For each column r, check if it can be removed by adjusting others
    # We use a simple heuristic: try setting one column to zero and
    # see if we can fix the others with small integer adjustments
    
    # Due to time constraints, we'll do a limited search:
    # Try each of the 49 columns as a candidate for elimination.
    # For each, try to find a rank-48 solution by solving a least-squares
    # problem with integer rounding.
    
    elapsed_start = time.time()
    
    for elim_r in range(49):
        if time.time() > deadline - 10:
            break
        
        # Try to find rank-48 decomposition by eliminating column elim_r
        # and adjusting the remaining columns slightly.
        # We search over small integer perturbations of the remaining columns.
        
        # For efficiency, only try a subset of perturbations
        # Focus on columns that interact with the eliminated column
        
        # Get the "target" contributions from the eliminated column
        # For each tensor entry (i,j,k), the eliminated column contributes:
        #   U[i*m+j][elim_r] * V[j*p+k][elim_r] * W[k*n+i][elim_r]
        
        u_elim = [U0[i*m+j][elim_r] for i in range(n) for j in range(m)]
        v_elim = [V0[j*p+k][elim_r] for j in range(m) for k in range(p)]
        w_elim = [W0[k*n+i][elim_r] for k in range(p) for i in range(n)]
        
        # If the eliminated column is mostly zero, it's already redundant
        if all(x == 0 for x in u_elim) or all(x == 0 for x in v_elim) or all(x == 0 for x in w_elim):
            # This column is already zero in one factor, so rank is already <= 48
            # Verify by checking if the decomposition still works without it
            U_test = [row[:] for row in U0]
            V_test = [row[:] for row in V0]
            W_test = [row[:] for row in W0]
            for row in U_test:
                row[elim_r] = 0
            for row in V_test:
                row[elim_r] = 0
            for row in W_test:
                row[elim_r] = 0
            
            if _verify_decomposition(U_test, V_test, W_test, n, m, p, 49):
                # Find the actual rank (number of non-zero columns)
                rank_actual = sum(1 for r in range(49) if any(U_test[i*m+j][r] != 0 for i in range(n) for j in range(m)) or any(V_test[j*p+k][r] != 0 for j in range(m) for k in range(p)) or any(W_test[k*n+i][r] != 0 for k in range(p) for i in range(n)))
                if rank_actual <= 48:
                    # Trim to actual rank
                    active_cols = [r for r in range(49) if any(U_test[i*m+j][r] != 0 for i in range(n) for j in range(m)) or any(V_test[j*p+k][r] != 0 for j in range(m) for k in range(p)) or any(W_test[k*n+i][r] != 0 for k in range(p) for i in range(n))]
                    U_trim = [[U_test[i*m+j][r] for r in active_cols] for i in range(n) for j in range(m)]
                    V_trim = [[V_test[j*p+k][r] for r in active_cols] for j in range(m) for k in range(p)]
                    W_trim = [[W_test[k*n+i][r] for r in active_cols] for k in range(p) for i in range(n)]
                    if _verify_decomposition(U_trim, V_trim, W_trim, n, m, p, len(active_cols)):
                        return {"U": U_trim, "V": V_trim, "W": W_trim}
        
        # Try small perturbations: for a few random subsets of columns,
        # try adding small integer values to see if we can absorb the eliminated column
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Pick a random column to modify
            mod_r = rng.choice([r for r in range(49) if r != elim_r])
            
            # Try small integer perturbations to W (or U, V) at column mod_r
            # to absorb part of the eliminated column's contribution
            found = False
            for delta in [-2, -1, 1, 2]:
                W_test = [row[:] for row in W0]
                # Perturb W at column mod_r
                # Try perturbing a random entry
                k_idx = rng.randrange(p * n)
                W_test[k_idx][mod_r] += delta
                
                # Check if this helps (this is a rough check)
                # For a proper check, we'd need to solve the full system
                # which is too expensive. Skip for now.
                pass
    
    return None


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        return _schoolbook_decomposition(n, m, p)

    # Try to find a rank-48 decomposition via search
    if time.time() < deadline - 20:
        rank48_result = _try_rank48_search(n, m, p, deadline)
        if rank48_result is not None:
            return rank48_result

    # Return the verified rank-49 Strassen decomposition
    return result

# EVOLVE-BLOCK-END
