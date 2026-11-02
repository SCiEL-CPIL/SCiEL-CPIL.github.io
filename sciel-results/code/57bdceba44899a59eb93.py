"""Recursive Strassen decomposition of 4x4x4 matrix multiplication (rank 49)."""
import time
import random

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# Strassen's 7 products for 2x2 matrices.
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
_STRASSEN_C_COMBINATION = [
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

    c_block_combination = [
        (0, 0, [(0, 1), (3, 1), (4, -1), (6, 1)]),
        (0, 1, [(2, 1), (4, 1)]),
        (1, 0, [(1, 1), (3, 1)]),
        (1, 1, [(0, 1), (1, -1), (2, 1), (5, 1)]),
    ]

    for i, (a_combo, b_combo) in enumerate(block_products):
        for s, (a_local, b_local) in enumerate(_STRASSEN_2X2):
            r = i * 7 + s

            for (a_br, a_bc, a_coeff) in a_combo:
                for (lr, lc, l_coeff) in a_local:
                    row = 2 * a_br + lr
                    col = 2 * a_bc + lc
                    U[row * m + col][r] += a_coeff * l_coeff

            for (b_br, b_bc, b_coeff) in b_combo:
                for (lr, lc, l_coeff) in b_local:
                    row = 2 * b_br + lr
                    col = 2 * b_bc + lc
                    V[row * p + col][r] += b_coeff * l_coeff

    # Build W properly
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


def _try_rank_reduction(U, V, W, n, m, p, rank, deadline):
    """
    Try to reduce the rank by checking if any column can be eliminated
    and the remaining columns adjusted to still satisfy the tensor equation.
    
    For each candidate column r to eliminate, we check if the contribution
    of column r can be expressed as a linear combination of other columns'
    contributions. This requires solving a system where for each tensor
    entry (i,j,k), the residual from removing column r must be zero.
    
    A more practical approach: try setting one column to zero and see if
    the decomposition still works. If not, try small integer perturbations.
    """
    import numpy as np
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each column, check if it's already redundant (all zeros in one factor)
    for r in range(rank):
        if time.time() > deadline - 5:
            break
        
        u_col = U_np[:, r]
        v_col = V_np[:, r]
        w_col = W_np[:, r]
        
        if np.all(u_col == 0) or np.all(v_col == 0) or np.all(w_col == 0):
            # Column is already zero in one factor - try removing it
            U_test = U_np.copy()
            V_test = V_np.copy()
            W_test = W_np.copy()
            U_test[:, r] = 0
            V_test[:, r] = 0
            W_test[:, r] = 0
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for rr in range(rank):
                            total += U_test[i*m+j, rr] * V_test[j*p+k, rr] * W_test[k*n+i, rr]
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Find active columns
                active = []
                for rr in range(rank):
                    if np.any(U_test[:, rr] != 0) or np.any(V_test[:, rr] != 0) or np.any(W_test[:, rr] != 0):
                        active.append(rr)
                
                if len(active) < rank:
                    U_trim = [[int(U_test[i*m+j, rr]) for rr in active] for i in range(n) for j in range(m)]
                    V_trim = [[int(V_test[j*p+k, rr]) for rr in active] for j in range(m) for k in range(p)]
                    W_trim = [[int(W_test[k*n+i, rr]) for rr in active] for k in range(p) for i in range(n)]
                    if _verify_decomposition(U_trim, V_trim, W_trim, n, m, p, len(active)):
                        return {"U": U_trim, "V": V_trim, "W": W_trim}
    
    # Try a more aggressive search: for each pair of columns, try to see if
    # one can be absorbed by modifying the other with a small integer shift
    rng = random.Random(12345)
    
    for elim_r in range(rank):
        if time.time() > deadline - 5:
            break
        
        # Get the target tensor contribution from column elim_r
        # For each (i,j,k), target[i,j,k] = U[i*m+j][elim_r] * V[j*p+k][elim_r] * W[k*n+i][elim_r]
        
        # Try to find a single column mod_r and a scalar alpha such that
        # adding alpha * (U[:,mod_r] ⊗ V[:,mod_r] ⊗ W[:,mod_r]) cancels the elim_r contribution
        # This is generally not possible with a single column, but let's try small shifts
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            mod_r = rng.choice([r for r in range(rank) if r != elim_r])
            
            # Try shifting W at a random entry
            k_idx = rng.randrange(p * n)
            for delta in [-2, -1, 1, 2]:
                W_test = [row[:] for row in W]
                W_test[k_idx][mod_r] += delta
                W_test[k_idx][elim_r] = 0
                # Also zero out the elim column in U and V
                # Actually, we need to zero out the entire elim column
                U_test = [row[:] for row in U]
                V_test = [row[:] for row in V]
                for row in U_test:
                    row[elim_r] = 0
                for row in V_test:
                    row[elim_r] = 0
                for row in W_test:
                    row[elim_r] = 0
                
                # Quick check: verify a subset of tensor entries
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0
                            for rr in range(rank):
                                total += U_test[i*m+j][rr] * V_test[j*p+k][rr] * W_test[k*n+i][rr]
                            if total != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    active = [rr for rr in range(rank) if any(U_test[i*m+j][rr] != 0 for i in range(n) for j in range(m)) or any(V_test[j*p+k][rr] != 0 for j in range(m) for k in range(p)) or any(W_test[k*n+i][rr] != 0 for k in range(p) for i in range(n))]
                    if len(active) < rank:
                        U_trim = [[U_test[i*m+j][rr] for rr in active] for i in range(n) for j in range(m)]
                        V_trim = [[V_test[j*p+k][rr] for rr in active] for j in range(m) for k in range(p)]
                        W_trim = [[W_test[k*n+i][rr] for rr in active] for k in range(p) for i in range(n)]
                        if _verify_decomposition(U_trim, V_trim, W_trim, n, m, p, len(active)):
                            return {"U": U_trim, "V": V_trim, "W": W_trim}
    
    return None


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Build the recursive Strassen rank-49 decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p, rank):
        # Fallback to schoolbook
        return _schoolbook_decomposition(n, m, p)

    # Try to find a lower-rank decomposition
    if time.time() < deadline - 20:
        reduced = _try_rank_reduction(U, V, W, n, m, p, rank, deadline)
        if reduced is not None:
            return reduced

    # Return the verified rank-49 Strassen decomposition
    return result

# EVOLVE-BLOCK-END
