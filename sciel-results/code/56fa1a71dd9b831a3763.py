import numpy as np
import time

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s kill

def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]      # 4 x 32
    V = [[0] * rank for _ in range(m * p)]      # 18 x 32
    W = [[0] * rank for _ in range(p * n)]      # 18 x 32

    # Strassen terms: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (U_index, coeff)
    # v_coeffs: list of (V_index_offset, coeff)
    # w_coeffs: list of (W_index_offset, coeff)
    strassen_terms = [
        # P0 = (A00 + A11) * (B00 + B11)
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1), (1, -1), (2, 1) ]),
        # P1 = (A10 + A11) * B00
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        # P2 = A00 * (B01 - B11)
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        # P3 = A11 * (B10 - B00)
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, -1) ]),
        # P4 = (A00 + A01) * B11
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (3, 1) ]),
        # P5 = (A10 - A00) * (B00 + B01)
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (0, -1), (2, 1) ]),
        # P6 = (A01 - A11) * (B10 + B11)
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (1, 1) ]),
    ]

    # Fill in 4 Strassen blocks for columns 0-7
    for block in range(4):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]

            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff

            for v_idx_off, coeff in v_coeffs:
                v_idx = 2 * block + v_idx_off
                V[v_idx][col] += coeff

            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off
                W[w_idx][col] += coeff

    # Schoolbook terms for column 8 (2x2x1 block)
    # V indices: V[j*p + k], so for k=8: V[8] (j=0), V[17] (j=1)
    # W indices: W[k*n + i], so for k=8: W[16] (i=0), W[17] (i=1)
    schoolbook_terms = [
        ([ (0, 1) ], [ (8, 1) ],  [ (16, 1) ]),   # A00 * B0 -> C00
        ([ (1, 1) ], [ (17, 1) ], [ (16, 1) ]),   # A01 * B1 -> C00
        ([ (2, 1) ], [ (8, 1) ],  [ (17, 1) ]),   # A10 * B0 -> C10
        ([ (3, 1) ], [ (17, 1) ], [ (17, 1) ]),   # A11 * B1 -> C10
    ]

    col_start = 4 * 7  # column 28
    for t in range(4):
        col = col_start + t
        u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]

        for u_idx, coeff in u_coeffs:
            U[u_idx][col] += coeff

        for v_idx, coeff in v_coeffs:
            V[v_idx][col] += coeff

        for w_idx, coeff in w_coeffs:
            W[w_idx][col] += coeff

    return U, V, W

def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the multiplication tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)

def _try_rank_31_via_dependency(n, m, p):
    """
    Try to find a rank-31 decomposition by checking if the 32 terms
    in the rank-32 solution have a linear dependency.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
        return None

    U32, V32, W32 = _build_rank_32(n, m, p)
    Ua = np.array(U32, dtype=float)
    Va = np.array(V32, dtype=float)
    Wa = np.array(W32, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Check if removing any single term still gives a valid decomposition
    for r in range(32):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        mask = [i for i in range(32) if i != r]
        U = Ua[:, mask]
        V = Va[:, mask]
        W = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T, T_recon, atol=1e-10):
            return (
                [[U[i][j] for j in range(31)] for i in range(n * m)],
                [[V[i][j] for j in range(31)] for i in range(m * p)],
                [[W[i][j] for j in range(31)] for i in range(p * n)],
            )
    return None

def _try_rank_31_randomized(n, m, p, max_iter=5000):
    """
    Structural approach: start from rank-32 solution, perturb one term,
    and use Newton's method to find a rank-31 solution.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 20:
        return None

    U32, V32, W32 = _build_rank_32(n, m, p)
    Ua = np.array(U32, dtype=float)
    Va = np.array(V32, dtype=float)
    Wa = np.array(W32, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try each term as the one to "absorb" into others
    for r_target in range(32):
        if time.time() - _START_TIME > _TIMEOUT - 10:
            return None
        
        # Initialize: copy rank-32, then try to make term r_target zero
        # by adjusting other terms
        U = Ua.copy()
        V = Va.copy()
        W = Wa.copy()
        
        # Zero out term r_target
        U[:, r_target] = 0
        V[:, r_target] = 0
        W[:, r_target] = 0
        
        # Now we have 31 active terms but they don't sum to T
        # We need to adjust the 31 terms to compensate
        # Use iterative refinement: for each iteration, compute error and distribute
        # This is a simplified approach
        
        # Actually, let's try a different structural approach:
        # Group the 9 columns differently. Use 3 Strassen blocks (covering 6 cols)
        # and 3 schoolbook blocks (covering 3 cols) = 21 + 12 = 33 terms
        # Then check for dependencies among these 33 terms.
        
        if r_target == 0:  # Only do this expensive check once
            result = _try_rank_33_then_reduce(n, m, p)
            if result is not None:
                return result
    
    return None

def _try_rank_33_then_reduce(n, m, p):
    """
    Build a rank-33 decomposition (3 Strassen + 3 schoolbook blocks),
    then check if any single term can be removed.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 15:
        return None

    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_terms = [
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1), (1, -1), (2, 1) ]),
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, -1) ]),
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (1, 1), (3, 1) ]),
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (0, -1), (2, 1) ]),
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1), (1, 1) ]),
    ]

    # 3 Strassen blocks for columns 0-5
    for block in range(3):
        col_start = block * 7
        for t in range(7):
            col = col_start + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]

            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff

            for v_idx_off, coeff in v_coeffs:
                v_idx = 2 * block + v_idx_off
                V[v_idx][col] += coeff

            for w_idx_off, coeff in w_coeffs:
                w_idx = 2 * block + w_idx_off
                W[w_idx][col] += coeff

    # 3 schoolbook blocks for columns 6, 7, 8
    col_start = 3 * 7  # column 21
    for block in range(3):
        k = 6 + block
        schoolbook_terms = [
            ([ (0, 1) ], [ (k, 1) ],  [ (k * n, 1) ]),
            ([ (1, 1) ], [ (k + n, 1) ], [ (k * n, 1) ]),
            ([ (2, 1) ], [ (k, 1) ],  [ (k * n + 1, 1) ]),
            ([ (3, 1) ], [ (k + n, 1) ], [ (k * n + 1, 1) ]),
        ]
        for t in range(4):
            col = col_start + block * 4 + t
            u_coeffs, v_coeffs, w_coeffs = schoolbook_terms[t]

            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff

            for v_idx, coeff in v_coeffs:
                V[v_idx][col] += coeff

            for w_idx, coeff in w_coeffs:
                W[w_idx][col] += coeff

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Validate rank-33
    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    if not np.allclose(T, T_recon, atol=1e-10):
        return None

    # Try removing each term
    for r in range(33):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        mask = [i for i in range(33) if i != r]
        U = Ua[:, mask]
        V = Va[:, mask]
        W = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T, T_recon, atol=1e-10):
            return (
                [[U[i][j] for j in range(32)] for i in range(n * m)],
                [[V[i][j] for j in range(32)] for i in range(m * p)],
                [[W[i][j] for j in range(32)] for i in range(p * n)],
            )
    return None

def _try_rank_31_structured(n, m, p):
    """
    Different structural approach: exploit the 2x2 x 2x9 structure.
    View as: C = A @ B where A is 2x2, B is 2x9.
    Each column of C is A @ B[:,k].
    
    Try a mixed approach: use Strassen-like tricks that work across
    multiple output columns simultaneously.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 20:
        return None

    # For 2x2 x 2x9, we can try to find a rank-31 decomposition
    # by setting up the tensor equations and using a structured ansatz.
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # For n=2, m=2, p=9:
    # T[0, 0, 0] = 1  (i=0,j=0,k=0)
    # T[0, 1, 2] = 1  (i=0,j=0,k=1)
    # T[0, 2, 4] = 1  (i=0,j=0,k=2)
    # T[0, 3, 6] = 1  (i=0,j=0,k=3)
    # T[0, 4, 8] = 1  (i=0,j=0,k=4)
    # T[0, 5, 10] = 1 (i=0,j=0,k=5)
    # T[0, 6, 12] = 1 (i=0,j=0,k=6)
    # T[0, 7, 14] = 1 (i=0,j=0,k=7)
    # T[0, 8, 16] = 1 (i=0,j=0,k=8)
    # T[1, 0, 1] = 1  (i=0,j=1,k=0)
    # T[1, 1, 3] = 1  (i=0,j=1,k=1)
    # T[1, 2, 5] = 1  (i=0,j=1,k=2)
    # T[1, 3, 7] = 1  (i=0,j=1,k=3)
    # T[1, 4, 9] = 1  (i=0,j=1,k=4)
    # T[1, 5, 11] = 1 (i=0,j=1,k=5)
    # T[1, 6, 13] = 1 (i=0,j=1,k=6)
    # T[1, 7, 15] = 1 (i=0,j=1,k=7)
    # T[1, 8, 17] = 1 (i=0,j=1,k=8)
    # T[2, 0, 0] = 1  (i=1,j=0,k=0)
    # T[2, 1, 2] = 1  (i=1,j=0,k=1)
    # T[2, 2, 4] = 1  (i=1,j=0,k=2)
    # T[2, 3, 6] = 1  (i=1,j=0,k=3)
    # T[2, 4, 8] = 1  (i=1,j=0,k=4)
    # T[2, 5, 10] = 1 (i=1,j=0,k=5)
    # T[2, 6, 12] = 1 (i=1,j=0,k=6)
    # T[2, 7, 14] = 1 (i=1,j=0,k=7)
    # T[2, 8, 16] = 1 (i=1,j=0,k=8)
    # T[3, 0, 1] = 1  (i=1,j=1,k=0)
    # T[3, 1, 3] = 1  (i=1,j=1,k=1)
    # T[3, 2, 5] = 1  (i=1,j=1,k=2)
    # T[3, 3, 7] = 1  (i=1,j=1,k=3)
    # T[3, 4, 9] = 1  (i=1,j=1,k=4)
    # T[3, 5, 11] = 1 (i=1,j=1,k=5)
    # T[3, 6, 13] = 1 (i=1,j=1,k=6)
    # T[3, 7, 15] = 1 (i=1,j=1,k=7)
    # T[3, 8, 17] = 1 (i=1,j=1,k=8)
    
    # Total: 36 non-zero entries.
    
    # For rank r, we need r terms (u_r, v_r, w_r) such that
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = T[i*m+j, j*p+k, k*n+i]
    
    # This is a system of 36 equations in 3*r*(2+2+9) = 3*r*13 variables
    # For r=31: 3*31*13 = 1209 variables, 36 equations - very underdetermined
    # But the equations are nonlinear (trilinear).
    
    # Let's try a different approach: use the known rank-32 solution
    # and try to find a linear combination of terms that equals zero.
    # If sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0, then we can eliminate one term.
    
    # Actually, let's just try all possible single-term removals from rank-32
    # and also try a few random perturbations.
    
    U32, V32, W32 = _build_rank_32(n, m, p)
    Ua = np.array(U32, dtype=float)
    Va = np.array(V32, dtype=float)
    Wa = np.array(W32, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try removing each term
    for r in range(32):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        mask = [i for i in range(32) if i != r]
        U = Ua[:, mask]
        V = Va[:, mask]
        W = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T, T_recon, atol=1e-10):
            return (
                [[U[i][j] for j in range(31)] for i in range(n * m)],
                [[V[i][j] for j in range(31)] for i in range(m * p)],
                [[W[i][j] for j in range(31)] for i in range(p * n)],
            )
    
    return None

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - _START_TIME

    # Try rank 31 via multiple structural approaches
    if elapsed < _TIMEOUT - 15:
        # Approach 1: Check if any term in rank-32 is redundant
        result = _try_rank_31_via_dependency(n, m, p)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    if elapsed < _TIMEOUT - 15:
        # Approach 2: Try rank-33 then reduce
        result = _try_rank_33_then_reduce(n, m, p)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    if elapsed < _TIMEOUT - 10:
        # Approach 3: Structured search
        result = _try_rank_31_structured(n, m, p)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # Fall back to rank-32 Strassen-based decomposition
    U, V, W = _build_rank_32(n, m, p)
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Final fallback: schoolbook rank 36
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
