"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)

    # Try the structured 4-block rank-11 scheme (total rank 44)
    best = None
    best_rank = n * m * p

    if time.time() < deadline - 20:
        try:
            U1, V1, W1, r1 = _four_block_rank11(n, m, p)
            if _verify(n, m, p, U1, V1, W1):
                best = (U1, V1, W1)
                best_rank = r1
        except Exception:
            pass

    # Try further refinement: attempt to remove redundant columns
    if best is not None and time.time() < deadline - 15:
        try:
            U2, V2, W2, r2 = _refine_by_column_removal(n, m, p, best[0], best[1], best[2], best_rank)
            if r2 < best_rank and _verify(n, m, p, U2, V2, W2):
                best = (U2, V2, W2)
                best_rank = r2
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _four_block_rank11(n, m, p):
    """
    Split p=8 into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition.
    
    A is 2x3: rows i=0,1; cols j=0,1,2
    B is 3x2: rows j=0,1,2; cols k=0,1
    C is 2x2: C[i,k] = sum_j A[i,j] * B[j,k]
    
    Decomposition:
    Let A = [A11 A12 A13; A21 A22 A23]
    Let B = [B00 B01; B10 B11; B20 B21]
    
    C[0,0] = A11*B00 + A12*B10 + A13*B20
    C[0,1] = A11*B01 + A12*B11 + A13*B21
    C[1,0] = A21*B00 + A22*B10 + A23*B20
    C[1,1] = A21*B01 + A22*B11 + A23*B21
    
    We use 11 rank-1 terms. The construction:
    - 7 terms from Strassen-like decomposition of the 2x2x2x2 core (j=0,1; k=0,1)
    - 4 terms for the j=2 cross terms
    
    Strassen for 2x2 * 2x2 (indices j in {0,1}, k in {0,1}):
    M1 = (A11+A21)*(B10+B20)  -> contributes to C[1,1]
    M2 = (A21)*(B20)          -> contributes to C[1,1]
    M3 = (A11)*(B10)          -> contributes to C[0,0]
    M4 = (A22)*(B21)          -> contributes to C[1,1]
    M5 = (A11+A12)*(B21)      -> contributes to C[0,1]
    M6 = (A21-A11)*(B10+B11)  -> contributes to C[1,0]
    M7 = (A12-A22)*(B20+B21)  -> contributes to C[0,1]
    
    Wait, let me use the standard Strassen formulation more carefully.
    
    For matrices X (2x2) and Y (2x2):
    M1 = (X00+X11)(Y00+Y11)
    M2 = (X10+X11)Y00
    M3 = X00(Y10-Y11)
    M4 = X11(Y01-Y00)
    M5 = (X00+X01)Y11
    M6 = (X10-X00)(Y00+Y01)
    M7 = (X01-X11)(Y10+Y11)
    
    Z00 = M1 + M4 - M5 + M7
    Z01 = M3 + M5
    Z10 = M2 + M4
    Z11 = M1 - M2 + M3 + M6
    
    For our case, the "2x2 core" uses j in {0,1} and k in {0,1}:
    X = [[A11, A12], [A21, A22]] (rows i, cols j)
    Y = [[B00, B01], [B10, B11]] (rows j, cols k)
    
    So:
    M1 = (A11+A21)*(B00+B11)
    M2 = (A21+A22)*B00
    M3 = A11*(B10-B11)
    M4 = A22*(B01-B00)
    M5 = (A11+A12)*B11
    M6 = (A21-A11)*(B00+B01)
    M7 = (A12-A22)*(B10+B11)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    
    Plus the j=2 terms:
    C[0,0] += A13*B20
    C[0,1] += A13*B21
    C[1,0] += A23*B20
    C[1,1] += A23*B21
    
    These 4 extra terms are rank-1 each.
    
    Total: 7 + 4 = 11 terms per block.
    """
    n, m, p = 2, 3, 8
    block_size = 2
    num_blocks = p // block_size  # 4
    
    rank_per_block = 11
    total_rank = rank_per_block * num_blocks  # 44
    
    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44
    
    for b in range(num_blocks):
        k_off = b * block_size  # offset in k: 0, 2, 4, 6
        col_off = b * rank_per_block  # offset in rank: 0, 11, 22, 33
        
        # Strassen terms (indices 0-6 in the block)
        # M1: (A11+A21)*(B00+B11)
        # U: A11 at (i=0,j=0) -> row 0; A21 at (i=1,j=0) -> row 3
        # V: B00 at (j=0,k=0) -> row 0*8+k_off; B11 at (j=1,k=1) -> row 1*8+k_off+1
        # W: contributes to C[0,0], C[1,1]
        U[0, col_off + 0] += 1   # A11
        U[3, col_off + 0] += 1   # A21
        V[0 * p + k_off + 0, col_off + 0] += 1   # B00
        V[1 * p + k_off + 1, col_off + 0] += 1   # B11
        W[(k_off + 0) * n + 0, col_off + 0] += 1  # C[0,0]
        W[(k_off + 1) * n + 1, col_off + 0] += 1  # C[1,1]
        
        # M2: (A21+A22)*B00
        U[3, col_off + 1] += 1   # A21
        U[4, col_off + 1] += 1   # A22
        V[0 * p + k_off + 0, col_off + 1] += 1   # B00
        W[(k_off + 0) * n + 1, col_off + 1] += 1  # C[1,0]
        W[(k_off + 1) * n + 1, col_off + 1] -= 1  # C[1,1]
        
        # M3: A11*(B10-B11)
        U[0, col_off + 2] += 1   # A11
        V[1 * p + k_off + 0, col_off + 2] += 1   # B10
        V[1 * p + k_off + 1, col_off + 2] -= 1   # B11
        W[(k_off + 1) * n + 0, col_off + 2] += 1  # C[0,1]
        W[(k_off + 1) * n + 1, col_off + 2] += 1  # C[1,1]
        
        # M4: A22*(B01-B00)
        U[4, col_off + 3] += 1   # A22
        V[0 * p + k_off + 1, col_off + 3] += 1   # B01
        V[0 * p + k_off + 0, col_off + 3] -= 1   # B00
        W[(k_off + 0) * n + 0, col_off + 3] += 1  # C[0,0]
        W[(k_off + 0) * n + 1, col_off + 3] += 1  # C[1,0]
        
        # M5: (A11+A12)*B11
        U[0, col_off + 4] += 1   # A11
        U[1, col_off + 4] += 1   # A12
        V[1 * p + k_off + 1, col_off + 4] += 1   # B11
        W[(k_off + 0) * n + 0, col_off + 4] -= 1  # C[0,0]
        W[(k_off + 1) * n + 0, col_off + 4] += 1  # C[0,1]
        
        # M6: (A21-A11)*(B00+B01)
        U[3, col_off + 5] += 1   # A21
        U[0, col_off + 5] -= 1   # A11
        V[0 * p + k_off + 0, col_off + 5] += 1   # B00
        V[0 * p + k_off + 1, col_off + 5] += 1   # B01
        W[(k_off + 1) * n + 1, col_off + 5] += 1  # C[1,1]
        
        # M7: (A12-A22)*(B10+B11)
        U[1, col_off + 6] += 1   # A12
        U[4, col_off + 6] -= 1   # A22
        V[1 * p + k_off + 0, col_off + 6] += 1   # B10
        V[1 * p + k_off + 1, col_off + 6] += 1   # B11
        W[(k_off + 0) * n + 0, col_off + 6] += 1  # C[0,0]
        W[(k_off + 1) * n + 0, col_off + 6] += 1  # C[0,1]
        
        # 4 extra rank-1 terms for j=2 cross terms
        # Term 7: A13*B20 -> C[0,0]
        U[2, col_off + 7] = 1   # A13 (i=0, j=2)
        V[2 * p + k_off + 0, col_off + 7] = 1   # B20
        W[(k_off + 0) * n + 0, col_off + 7] = 1  # C[0,0]
        
        # Term 8: A13*B21 -> C[0,1]
        U[2, col_off + 8] = 1   # A13
        V[2 * p + k_off + 1, col_off + 8] = 1   # B21
        W[(k_off + 1) * n + 0, col_off + 8] = 1  # C[0,1]
        
        # Term 9: A23*B20 -> C[1,0]
        U[5, col_off + 9] = 1   # A23 (i=1, j=2)
        V[2 * p + k_off + 0, col_off + 9] = 1   # B20
        W[(k_off + 0) * n + 1, col_off + 9] = 1  # C[1,0]
        
        # Term 10: A23*B21 -> C[1,1]
        U[5, col_off + 10] = 1   # A23
        V[2 * p + k_off + 1, col_off + 10] = 1   # B21
        W[(k_off + 1) * n + 1, col_off + 10] = 1  # C[1,1]
    
    return U, V, W, total_rank


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact using the bilinear form:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k])
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for c in range(U.shape[1]):
                u_val = 0.0
                for j in range(m):
                    u_val += U[i * m + j, c] * A[i, j]
                v_val = 0.0
                for l in range(m):
                    v_val += V[l * p + k, c] * B[l, k]
                C_comp[i, k] += W[k * n + i, c] * u_val * v_val
    
    if not np.allclose(C_comp, C_true, atol=1e-9):
        return False
    return True


def _refine_by_column_removal(n, m, p, U, V, W, current_rank):
    """
    Try to reduce rank by removing columns that are linearly dependent
    in the tensor sense. For each column, check if removing it still
    yields a valid decomposition.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    
    # Try removing each column one at a time
    for c_remove in range(rank - 1, -1, -1):
        mask = [c for c in range(rank) if c != c_remove]
        U_new = U[:, mask]
        V_new = V[:, mask]
        W_new = W[:, mask]
        
        if _verify(n, m, p, U_new, V_new, W_new):
            U, V, W = U_new, V_new, W_new
            rank -= 1
    
    return U, V, W, rank


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            # Round to nearest half-integer
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
