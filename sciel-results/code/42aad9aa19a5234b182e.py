"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _decomp_2x2x3():
    """
    Return (U, V, W) for 2x2x3 matrix multiplication.
    A is 2x2, B is 2x3, C is 2x3.
    U shape: (4, rank), V shape: (6, rank), W shape: (6, rank)
    """
    U1, V1, W1 = _strassen_2x2x2()
    
    rank = 11
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((6, rank), dtype=int)
    W = np.zeros((6, rank), dtype=int)
    
    # Strassen part for first 2 columns (terms 0-6)
    for c in range(7):
        U[0][c] = U1[0][c]
        U[1][c] = U1[1][c]
        U[2][c] = U1[2][c]
        U[3][c] = U1[3][c]
        
        # V1 rows: j*2+k -> V rows: j*3+k
        # j=0,k=0 -> V[0], j=0,k=1 -> V[1], j=1,k=0 -> V[3], j=1,k=1 -> V[4]
        V[0][c] = V1[0][c]
        V[1][c] = V1[1][c]
        V[3][c] = V1[2][c]
        V[4][c] = V1[3][c]
        V[2][c] = 0
        V[5][c] = 0
        
        # W1 rows: k*2+i -> W rows: k*2+i
        W[0][c] = W1[0][c]
        W[1][c] = W1[1][c]
        W[2][c] = W1[2][c]
        W[3][c] = W1[3][c]
        W[4][c] = 0
        W[5][c] = 0
    
    # Trivial part for column 2 (terms 7-10)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    U[0][7] = 1
    V[2][7] = 1   # j=0, k=2
    W[4][7] = 1   # k=2, i=0
    
    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    U[1][8] = 1
    V[5][8] = 1   # j=1, k=2
    W[4][8] = 1   # k=2, i=0
    
    # Term 9: A[1,0]*B[0,2] -> C[1,2]
    U[2][9] = 1
    V[2][9] = 1   # j=0, k=2
    W[5][9] = 1   # k=2, i=1
    
    # Term 10: A[1,1]*B[1,2] -> C[1,2]
    U[3][10] = 1
    V[5][10] = 1  # j=1, k=2
    W[5][10] = 1  # k=2, i=1
    
    return U, V, W

def _kron_decompose():
    """
    Build the full 4x4x6 decomposition via Kronecker product of 2x2x2 and 2x2x3.
    """
    U1, V1, W1 = _strassen_2x2x2()
    U2, V2, W2 = _decomp_2x2x3()
    
    rank = 7 * 11
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1][c1]
                            u2 = U2[i2*2+j2][c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row][c] = u1 * u2
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1][c1]
                            v2 = V2[j2*3+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row][c] = v1 * v2
            
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1][c1]
                            w2 = W2[k2*2+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row][c] = w1 * w2
    
    return U, V, W

def _verify(U_mat, V_mat, W_mat, n, m, p, seed=42):
    """Verify decomposition with a random test."""
    rng = np.random.RandomState(seed)
    A_test = rng.randn(n, m)
    B_test = rng.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U_mat.shape[1]):
        alpha_r = np.sum(A_test * U_mat[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V_mat[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
    
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _verify_multi(U_mat, V_mat, W_mat, n, m, p, seeds=None):
    """Verify with multiple random tests."""
    if seeds is None:
        seeds = [42, 123, 999, 7, 555]
    for s in seeds:
        if not _verify(U_mat, V_mat, W_mat, n, m, p, seed=s):
            return False
    return True

def _greedy_reduce(U, V, W, n, m, p, deadline):
    """
    Aggressive greedy column removal with multiple passes and orderings.
    """
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    num_cols = U.shape[1]
    keep = np.ones(num_cols, dtype=bool)
    
    # Precompute test matrices
    rng = np.random.RandomState(0)
    A_tests = [rng.randn(n, m) for _ in range(5)]
    B_tests = [rng.randn(m, p) for _ in range(5)]
    C_tests = [A @ B for A, B in zip(A_tests, B_tests)]
    
    def _quick_verify(Uc, Vc, Wc):
        """Fast verification using precomputed tests."""
        nc = Uc.shape[1]
        for idx in range(5):
            A_t = A_tests[idx]
            B_t = B_tests[idx]
            C_exp = C_tests[idx]
            C_comp = np.zeros((n, p))
            for r in range(nc):
                a_r = np.sum(A_t * Uc[:, r].reshape(n, m))
                b_r = np.sum(B_t * Vc[:, r].reshape(m, p))
                C_comp += a_r * b_r * Wc[:, r].reshape(p, n).T
            if not np.allclose(C_comp, C_exp, atol=1e-10):
                return False
        return True
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = num_cols
    
    # Multiple passes with different orderings
    orderings = [
        list(range(num_cols)),
        list(range(num_cols - 1, -1, -1)),
    ]
    for seed in [1, 2, 3, 4]:
        ord_list = list(range(num_cols))
        np.random.RandomState(seed).shuffle(ord_list)
        orderings.append(ord_list)
    
    for pass_idx, order in enumerate(orderings):
        if time.time() > deadline - 5:
            break
        
        current_rank = best_U.shape[1]
        if current_rank != best_rank:
            # Rebuild order to match current columns
            order = [i for i in order if i < current_rank]
        
        mask = np.ones(current_rank, dtype=bool)
        
        for r in order:
            if time.time() > deadline - 3:
                break
            if not mask[r]:
                continue
            
            mask[r] = False
            U_trial = best_U[:, mask]
            V_trial = best_V[:, mask]
            W_trial = best_W[:, mask]
            
            if _quick_verify(U_trial, V_trial, W_trial):
                best_U = U_trial
                best_V = V_trial
                best_W = W_trial
                best_rank = mask.sum()
            else:
                mask[r] = True
    
    # Final multi-seed verification
    if not _verify_multi(best_U, best_V, best_W, n, m, p, seeds=[42, 123, 999, 7, 555, 31415, 271828]):
        # Fallback: return original
        return U, V, W
    
    return best_U, best_V, best_W

def _linear_reduction(U, V, W, n, m, p, deadline):
    """
    Try to find linear dependencies among columns to reduce rank.
    For each set of columns, check if one is a linear combination of others.
    """
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    num_cols = U.shape[1]
    
    # Check for columns that are zero
    zero_cols = []
    for r in range(num_cols):
        if np.all(U[:, r] == 0) or np.all(V[:, r] == 0) or np.all(W[:, r] == 0):
            zero_cols.append(r)
    
    if zero_cols:
        mask = np.ones(num_cols, dtype=bool)
        for r in zero_cols:
            mask[r] = False
        U = U[:, mask]
        V = V[:, mask]
        W = W[:, mask]
        num_cols = U.shape[1]
    
    # Try to find linear dependencies by checking if removing a column
    # and solving for its contribution is possible
    # This is expensive, so we do a limited search
    
    if time.time() > deadline - 10:
        return U, V, W
    
    # For each column, check if it can be expressed as a linear combination
    # of a subset of other columns. We use a simple approach:
    # Check pairs of columns for linear dependence first.
    
    mask = np.ones(num_cols, dtype=bool)
    
    # Check for duplicate or proportional columns
    for i in range(num_cols):
        if time.time() > deadline - 5:
            break
        if not mask[i]:
            continue
        for j in range(i + 1, num_cols):
            if not mask[j]:
                continue
            
            # Check if column i is proportional to column j
            # (across U, V, W simultaneously)
            Ui = U[:, i]
            Uj = U[:, j]
            if np.linalg.norm(Ui) < 1e-15 or np.linalg.norm(Uj) < 1e-15:
                continue
            
            # Try to find scalar s such that Ui = s * Uj
            s = np.dot(Ui, Uj) / np.dot(Uj, Uj)
            if np.allclose(Ui, s * Uj, atol=1e-10):
                # Check V and W
                Vi = V[:, i]
                Vj = V[:, j]
                s2 = np.dot(Vi, Vj) / np.dot(Vj, Vj)
                if np.allclose(Vi, s2 * Vj, atol=1e-10) and np.isclose(s, s2, atol=1e-10):
                    Wi = W[:, i]
                    Wj = W[:, j]
                    s3 = np.dot(Wi, Wj) / np.dot(Wj, Wj)
                    if np.allclose(Wi, s3 * Wj, atol=1e-10) and np.isclose(s, s3, atol=1e-10):
                        # Column i is redundant given column j
                        mask[i] = False
                        break
    
    if mask.sum() < num_cols:
        U = U[:, mask]
        V = V[:, mask]
        W = W[:, mask]
    
    return U, V, W

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy:
    1. Build Kronecker product decomposition: (2x2x2) ⊗ (2x2x3)
       - 2x2x2 uses Strassen (rank 7)
       - 2x2x3 uses hybrid (rank 11)
       - Total rank = 77
    2. Apply greedy column removal
    3. Apply linear dependency detection
    4. Return best result found
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Build initial Kronecker decomposition
    U, V, W = _kron_decompose()
    
    # Verify initial decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # First pass: linear dependency reduction (fast)
    if time.time() < deadline - 20:
        U, V, W = _linear_reduction(U, V, W, n, m, p, deadline)
    
    # Second pass: greedy column removal
    if time.time() < deadline - 10:
        U, V, W = _greedy_reduce(U, V, W, n, m, p, deadline)
    
    # Another round of linear reduction after greedy
    if time.time() < deadline - 5:
        U, V, W = _linear_reduction(U, V, W, n, m, p, deadline)
    
    # Final verification
    if not _verify_multi(U, V, W, n, m, p, seeds=[42, 123, 999, 7, 555, 31415, 271828, 161803, 141421]):
        # Rebuild from scratch and verify
        U, V, W = _kron_decompose()
        if not _verify(U, V, W, n, m, p):
            # Absolute fallback: schoolbook
            rank_school = n * m * p
            U = np.zeros((n * m, rank_school), dtype=int)
            V = np.zeros((m * p, rank_school), dtype=int)
            W = np.zeros((p * n, rank_school), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to int if possible
    if np.allclose(U, np.round(U), atol=1e-10):
        U = np.round(U).astype(int)
    if np.allclose(V, np.round(V), atol=1e-10):
        V = np.round(V).astype(int)
    if np.allclose(W, np.round(W), atol=1e-10):
        W = np.round(W).astype(int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
