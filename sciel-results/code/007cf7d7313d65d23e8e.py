"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_full(U, V, W, n, m, p, rank):
    """Full verification: correct entries = 1, all other tensor entries = 0."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    nm, mp, pn = n*m, m*p, p*n
    # Check correct entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    # Check zero entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i_a, j_a = divmod(a, m)
                j_b, k_b = divmod(b, p)
                k_c, i_c = divmod(c, n)
                if i_a == i_c and j_a == j_b and k_b == k_c:
                    continue
                s = np.dot(U_np[a], V_np[b] * W_np[c])
                if abs(s) > 1e-9:
                    return False
    return True

def _verify_correct_only(U, V, W, n, m, p, rank):
    """Only check that the 40 correct entries sum to 1."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(U, V, W, u_off, v_off, w_off, col):
    """
    Add a Strassen 2x2x2 block.
    U rows: u_off..u_off+3 (i=0,1; j=0,1) -> indices u_off + i*2 + j
    V rows: v_off..v_off+3 (j=0,1; k=0,1) -> indices v_off + j*2 + k
    W rows: w_off..w_off+3 (k=0,1; i=0,1) -> indices w_off + k*2 + i
    """
    # M1 = (A00+A11)*(B00+B11)
    U[u_off + 0][col] = 1; U[u_off + 3][col] = 1
    V[v_off + 0][col] = 1; V[v_off + 3][col] = 1
    W[w_off + 0][col] = 1; W[w_off + 3][col] = 1
    col += 1
    # M2 = (A10+A11)*B00
    U[u_off + 2][col] = 1; U[u_off + 3][col] = 1
    V[v_off + 0][col] = 1
    W[w_off + 2][col] = 1; W[w_off + 3][col] = -1
    col += 1
    # M3 = A00*(B01-B11)
    U[u_off + 0][col] = 1
    V[v_off + 1][col] = 1; V[v_off + 3][col] = -1
    W[w_off + 0][col] = 1; W[w_off + 3][col] = 1
    col += 1
    # M4 = A11*(B10-B00)
    U[u_off + 3][col] = 1
    V[v_off + 2][col] = 1; V[v_off + 0][col] = -1
    W[w_off + 0][col] = 1; W[w_off + 2][col] = 1
    col += 1
    # M5 = (A00+A01)*B11
    U[u_off + 0][col] = 1; U[u_off + 1][col] = 1
    V[v_off + 3][col] = 1
    W[w_off + 0][col] = -1; W[w_off + 1][col] = 1
    col += 1
    # M6 = (A10-A00)*(B00+B01)
    U[u_off + 2][col] = 1; U[u_off + 0][col] = -1
    V[v_off + 0][col] = 1; V[v_off + 1][col] = 1
    W[w_off + 3][col] = 1
    col += 1
    # M7 = (A01-A11)*(B10+B11)
    U[u_off + 1][col] = 1; U[u_off + 3][col] = -1
    V[v_off + 2][col] = 1; V[v_off + 3][col] = 1
    W[w_off + 0][col] = 1
    col += 1
    return col

def _construct_hybrid_rank36():
    """Construct the rank-36 hybrid decomposition."""
    n, m, p = 2, 4, 5
    total_rank = 36
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    # Block 1: A row 0 (U rows 0-3) * B (V rows 0-19)
    # Cols 0-1: Strassen 2x2x2 for B cols 0-1
    col = _strassen_2x2x2(U, V, W, 0, 0, 0, col)
    # Col 2: schoolbook for B col 2
    U[0][col] = 1; V[2][col] = 1; W[4][col] = 1; col += 1
    U[1][col] = 1; V[3][col] = 1; W[4][col] = 1; col += 1
    U[2][col] = 1; V[2][col] = 1; W[5][col] = 1; col += 1
    U[3][col] = 1; V[3][col] = 1; W[5][col] = 1; col += 1
    # Cols 3-4: Strassen 2x2x2 for B cols 3-4
    col = _strassen_2x2x2(U, V, W, 0, 6, 6, col)
    
    # Block 2: A row 1 (U rows 4-7) * B (V rows 10-19)
    # Cols 0-1: Strassen 2x2x2 for B cols 0-1
    col = _strassen_2x2x2(U, V, W, 4, 10, 0, col)
    # Col 2: schoolbook for B col 2
    U[4][col] = 1; V[12][col] = 1; W[4][col] = 1; col += 1
    U[5][col] = 1; V[13][col] = 1; W[4][col] = 1; col += 1
    U[6][col] = 1; V[12][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[13][col] = 1; W[5][col] = 1; col += 1
    # Cols 3-4: Strassen 2x2x2 for B cols 3-4
    col = _strassen_2x2x2(U, V, W, 4, 16, 6, col)
    
    return {"U": U, "V": V, "W": W}

def _construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # First try the known rank-36 construction
    result = _construct_hybrid_rank36()
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    if _verify_full(U, V, W, n, m, p, rank):
        # Check if we have time to attempt rank reduction
        if time.time() < deadline - 30:
            # Try to find a rank-35 or lower decomposition
            reduced = _try_reduce_rank(U, V, W, n, m, p, start_time, deadline)
            if reduced is not None:
                return reduced
        return result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U_orig, V_orig, W_orig, n, m, p, start_time, deadline):
    """
    Attempt to reduce the rank by trying to merge or eliminate terms.
    This uses a structured search over small perturbations and linear combinations.
    """
    nm, mp, pn = n*m, m*p, p*n
    orig_rank = len(U_orig[0])
    
    U_np = np.array(U_orig, dtype=np.float64)
    V_np = np.array(V_orig, dtype=np.float64)
    W_np = np.array(W_orig, dtype=np.float64)
    
    # Try to find a rank (orig_rank - 1) decomposition by attempting to
    # express one column as a linear combination of others with integer/half-integer coefficients
    
    # For each pair of columns, try to see if one can be eliminated
    # by adding a multiple of another
    for target_col in range(orig_rank):
        if time.time() > deadline - 5:
            break
        for other_col in range(orig_rank):
            if other_col == target_col:
                continue
            # Try small integer/half-integer coefficients
            for coeff in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
                # Create a modified decomposition
                U_mod = U_np.copy()
                V_mod = V_np.copy()
                W_mod = W_np.copy()
                
                # Subtract coeff * other_col from target_col
                U_mod[:, target_col] -= coeff * U_mod[:, other_col]
                V_mod[:, target_col] -= coeff * V_mod[:, other_col]
                W_mod[:, target_col] -= coeff * W_mod[:, target_col]
                
                # Check if target_col is now all zeros
                if np.allclose(U_mod[:, target_col], 0, atol=1e-9) and \
                   np.allclose(V_mod[:, target_col], 0, atol=1e-9) and \
                   np.allclose(W_mod[:, target_col], 0, atol=1e-9):
                    # We can eliminate this column
                    new_rank = orig_rank - 1
                    U_new = np.delete(U_mod, target_col, axis=1)
                    V_new = np.delete(V_mod, target_col, axis=1)
                    W_new = np.delete(W_mod, target_col, axis=1)
                    
                    # Verify the reduced decomposition
                    if _verify_full_np(U_new, V_new, W_new, n, m, p):
                        return _to_list_format(U_new, V_new, W_new)
    
    # Try more complex reductions: replace two columns with a combination
    # This is more expensive, so only do it if we have time
    if time.time() < deadline - 10:
        for c1 in range(orig_rank):
            for c2 in range(c1+1, orig_rank):
                if time.time() > deadline - 5:
                    break
                # Try to see if c1 and c2 can be replaced by a single column
                # This requires finding a rank-1 tensor that approximates both
                # For now, skip this as it's very complex
                pass
    
    return None

def _verify_full_np(U_np, V_np, W_np, n, m, p):
    """Verify using numpy arrays directly."""
    nm, mp, pn = n*m, m*p, p*n
    rank = U_np.shape[1]
    # Check correct entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1) > 1e-9:
                    return False
    # Check zero entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                i_a, j_a = divmod(a, m)
                j_b, k_b = divmod(b, p)
                k_c, i_c = divmod(c, n)
                if i_a == i_c and j_a == j_b and k_b == k_c:
                    continue
                s = np.dot(U_np[a], V_np[b] * W_np[c])
                if abs(s) > 1e-9:
                    return False
    return True

def _to_list_format(U_np, V_np, W_np):
    """Convert numpy arrays to list format with integer/half-integer values."""
    rank = U_np.shape[1]
    U_list = [[int(round(U_np[i, r] * 2)) / 2 for r in range(rank)] for i in range(U_np.shape[0])]
    V_list = [[int(round(V_np[i, r] * 2)) / 2 for r in range(rank)] for i in range(V_np.shape[0])]
    W_list = [[int(round(W_np[i, r] * 2)) / 2 for r in range(rank)] for i in range(W_np.shape[0])]
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    return _construct()
# EVOLVE-BLOCK-END
