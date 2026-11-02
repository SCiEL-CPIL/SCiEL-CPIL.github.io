"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Check all 12 (i,j,k) triples exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    """
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11) -> contributes to C[0,0] and C[1,1]
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> contributes to C[0,1] and C[1,1]
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> contributes to C[1,0] and C[1,1]
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> contributes to C[0,0] and C[0,1]
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> contributes to C[0,0] and C[1,0]
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> contributes to C[1,1]
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> contributes to C[0,0]
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
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


def _try_rank10_column_compression(deadline):
    """
    Rank-10 search exploiting column structure: C[:,2] only depends on B[:,2].
    This creates a natural compression where the third column equations can be
    satisfied with fewer independent terms.
    
    Key insight: For k=2, the equations are:
      sum_r U[i*2+j, r] * V[j*3+2, r] * W[2*2+i, r] = 1 for all i,j
    
    This means V[:,2] and W[4:6,:] are the only B/C factors for column 2.
    We can search for a rank-10 decomposition where the first 7 terms handle
    the 2x2 block (k=0,1) and 3 additional terms handle k=2 with shared structure.
    """
    if time.time() > deadline - 15:
        return None
    
    rank = 10
    
    # Standard Strassen for the 2x2x2 subproblem (terms 0-6)
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 1, 0, 0, 0, 1, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ], dtype=float)  # 4 x 7
    
    V_strassen = np.array([
        [1, 0, 0, 0, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, -1, 1, 0, 1], # B11
    ], dtype=float)  # 4 x 7 (only first 4 rows for k=0,1)
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0]
        [0, 1, 1, 0, 0, 0, 0],   # C[0,1]
        [0, 0, 0, 0, 0, 1, 0],   # C[1,0]
        [1, -1, 1, 0, 1, 0, 0],  # C[1,1]
    ], dtype=float)  # 4 x 7 (only first 4 rows for k=0,1)
    
    def _objective(U_mat, V_mat, W_mat):
        """Compute sum of squared residuals for the 12 equations."""
        res = np.empty(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = np.dot(U_mat[i*2+j], V_mat[j*3+k] * W_mat[k*2+i])
                    res[idx] = s - 1.0
                    idx += 1
        return np.sum(res * res)
    
    def _solve_all_given_UW(U_mat, W_mat):
        """Solve for V given U and W."""
        V_mat = np.zeros((6, rank))
        for j in range(2):
            for k in range(3):
                A = np.zeros((2, rank))
                b = np.ones(2)
                for i in range(2):
                    for r in range(rank):
                        A[i, r] = U_mat[i*2+j, r] * W_mat[k*2+i, r]
                try:
                    V_mat[j*3+k] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
        return V_mat
    
    def _solve_all_given_UV(U_mat, V_mat):
        """Solve for W given U and V."""
        W_mat = np.zeros((6, rank))
        for k in range(3):
            for i in range(2):
                A = np.zeros((2, rank))
                b = np.ones(2)
                for j in range(2):
                    for r in range(rank):
                        A[j, r] = U_mat[i*2+j, r] * V_mat[j*3+k, r]
                try:
                    W_mat[k*2+i] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
        return W_mat
    
    def _solve_U_given_VW(V_mat, W_mat):
        """Solve for U given V and W."""
        U_mat = np.zeros((4, rank))
        for i in range(2):
            for j in range(2):
                A = np.zeros((3, rank))
                b = np.ones(3)
                for k in range(3):
                    for r in range(rank):
                        A[k, r] = V_mat[j*3+k, r] * W_mat[k*2+i, r]
                try:
                    U_mat[i*2+j] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
        return U_mat
    
    rng = np.random.default_rng(777)
    best_cost = float('inf')
    best_UW = None
    
    # Strategy: Initialize with Strassen for k=0,1 and random for k=2
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random initialization for the 3 extra columns (indices 7,8,9)
        U_extra = rng.normal(0, 1.0, (4, 3))
        V_extra = rng.normal(0, 1.0, (6, 3))
        W_extra = rng.normal(0, 1.0, (6, 3))
        
        # Build full matrices
        U_mat = np.hstack([U_strassen, U_extra])
        
        # V: rows 0-3 for k=0,1 from Strassen, rows 4-5 for k=2 random
        V_mat = np.zeros((6, rank))
        V_mat[:4, :7] = V_strassen
        V_mat[4:6, :7] = 0  # Strassen doesn't contribute to k=2
        V_mat[:, 7:] = V_extra
        
        # W: rows 0-3 for k=0,1 from Strassen, rows 4-5 for k=2 random
        W_mat = np.zeros((6, rank))
        W_mat[:4, :7] = W_strassen
        W_mat[4:6, :7] = 0  # Strassen doesn't contribute to k=2
        W_mat[:, 7:] = W_extra
        
        # Alternating minimization with more iterations
        for iteration in range(80):
            if time.time() > deadline - 10:
                break
            
            # Update V
            V_mat = _solve_all_given_UW(U_mat, W_mat)
            # Update W
            W_mat = _solve_all_given_UV(U_mat, V_mat)
            # Update U
            U_mat = _solve_U_given_VW(V_mat, W_mat)
            
            cost = _objective(U_mat, V_mat, W_mat)
            
            if cost < best_cost:
                best_cost = cost
                best_UW = (U_mat.copy(), V_mat.copy(), W_mat.copy())
            
            if cost < 1e-14:
                break
            if iteration > 50 and cost > 1e-8:
                break  # Early termination if stuck
    
    if best_UW is None or best_cost > 1e-10:
        return None
    
    U_mat, V_mat, W_mat = best_UW
    
    # Try rounding to half-integers and integers
    for round_func in [lambda x: round(x * 2) / 2, lambda x: round(x)]:
        U = [[round_func(u) for u in row] for row in U_mat.tolist()]
        V = [[round_func(v) for v in row] for row in V_mat.tolist()]
        W = [[round_func(w) for w in row] for row in W_mat.tolist()]
        
        if _verify(U, V, W, 2, 2, 3):
            return {"U": U, "V": V, "W": W}
    
    return None


def _try_rank10_projection(deadline):
    """
    Rank-10 search by dropping one schoolbook term from rank-11 and re-optimizing
    via alternating minimization.
    """
    if time.time() > deadline - 15:
        return None
    
    rank = 10
    
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)
    V11 = np.array(base["V"], dtype=float)
    W11 = np.array(base["W"], dtype=float)
    
    def _objective(U_mat, V_mat, W_mat):
        res = np.empty(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = np.dot(U_mat[i*2+j], V_mat[j*3+k] * W_mat[k*2+i])
                    res[idx] = s - 1.0
                    idx += 1
        return np.sum(res * res)
    
    def _solve_V_given_UW(U_mat, W_mat):
        V_mat = np.zeros((6, rank))
        for j in range(2):
            for k in range(3):
                A = np.zeros((2, rank))
                b = np.ones(2)
                for i in range(2):
                    for r in range(rank):
                        A[i, r] = U_mat[i*2+j, r] * W_mat[k*2+i, r]
                try:
                    V_mat[j*3+k] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
        return V_mat
    
    def _solve_W_given_UV(U_mat, V_mat):
        W_mat = np.zeros((6, rank))
        for k in range(3):
            for i in range(2):
                A = np.zeros((2, rank))
                b = np.ones(2)
                for j in range(2):
                    for r in range(rank):
                        A[j, r] = U_mat[i*2+j, r] * V_mat[j*3+k, r]
                try:
                    W_mat[k*2+i] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
        return W_mat
    
    def _solve_U_given_VW(V_mat, W_mat):
        U_mat = np.zeros((4, rank))
        for i in range(2):
            for j in range(2):
                A = np.zeros((3, rank))
                b = np.ones(3)
                for k in range(3):
                    for r in range(rank):
                        A[k, r] = V_mat[j*3+k, r] * W_mat[k*2+i, r]
                try:
                    U_mat[i*2+j] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
        return U_mat
    
    rng = np.random.default_rng(54321)
    best_cost = float('inf')
    best_UW = None
    
    for drop_idx in [7, 8, 9, 10]:
        if time.time() > deadline - 10:
            break
        
        keep = [r for r in range(11) if r != drop_idx]
        U0 = U11[:, keep].copy()
        V0 = V11[:, keep].copy()
        W0 = W11[:, keep].copy()
        
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            U_mat = U0 + rng.normal(0, 0.1, (4, 10))
            V_mat = V0 + rng.normal(0, 0.1, (6, 10))
            W_mat = W0 + rng.normal(0, 0.1, (6, 10))
            
            for iteration in range(40):
                if time.time() > deadline - 10:
                    break
                
                V_mat = _solve_V_given_UW(U_mat, W_mat)
                W_mat = _solve_W_given_UV(U_mat, V_mat)
                U_mat = _solve_U_given_VW(V_mat, W_mat)
                
                cost = _objective(U_mat, V_mat, W_mat)
                if cost < best_cost:
                    best_cost = cost
                    best_UW = (U_mat.copy(), V_mat.copy(), W_mat.copy())
                
                if cost < 1e-14:
                    break
    
    if best_UW is None or best_cost > 1e-10:
        return None
    
    U_mat, V_mat, W_mat = best_UW
    
    U = [[round(u * 2) / 2 for u in row] for row in U_mat.tolist()]
    V = [[round(v * 2) / 2 for v in row] for row in V_mat.tolist()]
    W = [[round(w * 2) / 2 for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    U = [[round(u) for u in row] for row in U_mat.tolist()]
    V = [[round(v) for v in row] for row in V_mat.tolist()]
    W = [[round(w) for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    return None


def _try_rank10_structured(deadline):
    """
    Structured rank-10 search: fix U to Strassen pattern + 3 free columns,
    alternate solving for V and W via least squares.
    """
    if time.time() > deadline - 15:
        return None
    
    rank = 10
    
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=float)
    
    V_strassen = np.array([
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)
    
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)
    
    def _objective(U_mat, V_mat, W_mat):
        res = np.empty(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = np.dot(U_mat[i*2+j], V_mat[j*3+k] * W_mat[k*2+i])
                    res[idx] = s - 1.0
                    idx += 1
        return np.sum(res * res)
    
    def _solve_V_given_UW(U_mat, W_mat):
        V_mat = np.zeros((6, rank))
        for j in range(2):
            for k in range(3):
                A = np.zeros((2, rank))
                b = np.ones(2)
                for i in range(2):
                    for r in range(rank):
                        A[i, r] = U_mat[i*2+j, r] * W_mat[k*2+i, r]
                try:
                    V_mat[j*3+k] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
        return V_mat
    
    def _solve_W_given_UV(U_mat, V_mat):
        W_mat = np.zeros((6, rank))
        for k in range(3):
            for i in range(2):
                A = np.zeros((2, rank))
                b = np.ones(2)
                for j in range(2):
                    for r in range(rank):
                        A[j, r] = U_mat[i*2+j, r] * V_mat[j*3+k, r]
                try:
                    W_mat[k*2+i] = np.linalg.lstsq(A, b, rcond=None)[0]
                except Exception:
                    pass
        return W_mat
    
    def _solve_U_extra_given_VW(V_mat, W_mat):
        """Solve only for U_extra (last 3 cols), keeping U_strassen fixed."""
        U_extra = np.zeros((4, 3))
        for i in range(2):
            for j in range(2):
                # sum_r U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] = 1
                # Split into known (first 7) and unknown (last 3)
                rhs = np.ones(3)
                for k in range(3):
                    known_sum = 0.0
                    for r in range(7):
                        known_sum += U_strassen[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                    rhs[k] = 1.0 - known_sum
                
                A = np.zeros((3, 3))
                for k in range(3):
                    for col in range(3):
                        r = 7 + col
                        A[k, col] = V_mat[j*3+k, r] * W_mat[k*2+i, r]
                
                try:
                    U_extra[i*2+j] = np.linalg.solve(A, rhs)
                except Exception:
                    try:
                        U_extra[i*2+j] = np.linalg.lstsq(A, rhs, rcond=None)[0]
                    except Exception:
                        pass
        return U_extra
    
    rng = np.random.default_rng(99)
    best_cost = float('inf')
    best_UW = None
    
    for trial in range(15):
        if time.time() > deadline - 10:
            break
        
        V_extra0 = rng.normal(0, 0.5, (6, 3))
        W_extra0 = rng.normal(0, 0.5, (6, 3))
        
        V_mat = np.hstack([V_strassen, V_extra0])
        W_mat = np.hstack([W_strassen, W_extra0])
        U_extra = rng.normal(0, 0.5, (4, 3))
        U_mat = np.hstack([U_strassen, U_extra])
        
        for iteration in range(40):
            if time.time() > deadline - 10:
                break
            
            V_mat = _solve_V_given_UW(U_mat, W_mat)
            W_mat = _solve_W_given_UV(U_mat, V_mat)
            U_extra = _solve_U_extra_given_VW(V_mat, W_mat)
            U_mat = np.hstack([U_strassen, U_extra])
            
            cost = _objective(U_mat, V_mat, W_mat)
            if cost < best_cost:
                best_cost = cost
                best_UW = (U_mat.copy(), V_mat.copy(), W_mat.copy())
            
            if cost < 1e-14:
                break
    
    if best_UW is None or best_cost > 1e-10:
        return None
    
    U_mat, V_mat, W_mat = best_UW
    
    U = [[round(u * 2) / 2 for u in row] for row in U_mat.tolist()]
    V = [[round(v * 2) / 2 for v in row] for row in V_mat.tolist()]
    W = [[round(w * 2) / 2 for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    U = [[round(u) for u in row] for row in U_mat.tolist()]
    V = [[round(v) for v in row] for row in V_mat.tolist()]
    W = [[round(w) for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    
    return None


def construct():
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    # First, get the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        best = _schoolbook()
        best_rank = 12
    
    # Try column compression approach for rank 10
    if best_rank > 10 and time.time() < deadline - 60:
        result10_c = _try_rank10_column_compression(deadline)
        if result10_c is not None:
            best = result10_c
            best_rank = 10
    
    # Try linear projection approach for rank 10
    if best_rank > 10 and time.time() < deadline - 40:
        result10_p = _try_rank10_projection(deadline)
        if result10_p is not None:
            best = result10_p
            best_rank = 10
    
    # Try structured rank-10 search
    if best_rank > 10 and time.time() < deadline - 20:
        result10_s = _try_rank10_structured(deadline)
        if result10_s is not None:
            best = result10_s
            best_rank = 10
    
    return best
