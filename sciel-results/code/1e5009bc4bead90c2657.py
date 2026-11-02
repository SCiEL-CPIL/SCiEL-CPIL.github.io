"""Tensor decomposition search for 2x2 by 2x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    rng = np.random.default_rng(42)
    for _ in range(50):
        A = rng.integers(-10, 11, size=(n, m)).astype(float)
        B = rng.integers(-10, 11, size=(m, p)).astype(float)
        C = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_vec = np.array(U)[:, r]
            v_vec = np.array(V)[:, r]
            w_vec = np.array(W)[:, r]
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            # C[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                    C_decomp[i, k] += s
        if np.max(np.abs(C - C_decomp)) > tol:
            return False
    return True


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    This is the known best construction from inspiration 2.
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

    # Term 0: (A00+A11)(B00+B11) -> M1
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> M2
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> M3
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> M4
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> M5
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> M6
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> M7
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


def _try_rank10_search(deadline):
    """
    Attempt to find a rank-10 decomposition via numerical search.
    
    Strategy: Use structured initialization from the rank-11 solution (merge two
    schoolbook terms into one Strassen-like term), then polish with least_squares.
    Also try random initializations with multiple restarts.
    """
    if time.time() > deadline - 30:
        return None
    
    from scipy.optimize import least_squares
    
    rng = np.random.default_rng(123)
    
    def residuals(params):
        U_flat = params[:40]
        V_flat = params[40:100]
        W_flat = params[100:160]
        
        U_mat = U_flat.reshape(4, 10)
        V_mat = V_flat.reshape(6, 10)
        W_mat = W_flat.reshape(6, 10)
        
        res = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = 0.0
                    for r in range(10):
                        s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                    res.append(s - 1.0)
        return res
    
    def try_polish(x0, max_nfev=2000):
        """Polish a candidate solution using least_squares, then verify."""
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=max_nfev)
            if result.cost < 1e-14:
                U_flat = result.x[:40]
                V_flat = result.x[40:100]
                W_flat = result.x[100:160]
                
                U = U_flat.reshape(4, 10).tolist()
                V = V_flat.reshape(6, 10).tolist()
                W = W_flat.reshape(6, 10).tolist()
                
                # Round to nearest half-integer
                U = [[round(u * 2) / 2 for u in row] for row in U]
                V = [[round(v * 2) / 2 for v in row] for row in V]
                W = [[round(w * 2) / 2 for w in row] for row in W]
                
                if _verify(U, V, W, 2, 2, 3):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            pass
        return None
    
    # Strategy 1: Structured initialization - merge two schoolbook terms
    # Take the rank-11 solution and try to merge terms 7-10 (schoolbook for col 2)
    # into a single term combined with Strassen terms.
    
    # Get the rank-11 solution as starting point
    r11 = _rank11_decomposition()
    U11 = np.array(r11["U"])  # 4x11
    V11 = np.array(r11["V"])  # 6x11
    W11 = np.array(r11["W"])  # 6x11
    
    # Try different ways to merge: pick a pair of terms to combine
    # and see if the remaining 9 can be adjusted to absorb the merged contribution.
    
    # Approach: Initialize 10 terms where 9 are copied from rank-11 (first 9 terms)
    # and the 10th is a combination of terms 9 and 10 (0-indexed: terms 9,10)
    for merge_pair in [(7, 8), (7, 9), (7, 10), (8, 9), (8, 10), (9, 10),
                       (5, 7), (5, 8), (6, 9), (6, 10), (4, 7), (4, 8)]:
        if time.time() > deadline - 15:
            break
        
        a, b = merge_pair
        # Create 10-term initialization
        U_init = np.zeros((4, 10))
        V_init = np.zeros((6, 10))
        W_init = np.zeros((6, 10))
        
        # Copy first 9 terms, skipping the merged pair
        idx = 0
        for r in range(11):
            if r == a or r == b:
                continue
            if idx < 9:
                U_init[:, idx] = U11[:, r]
                V_init[:, idx] = V11[:, r]
                W_init[:, idx] = W11[:, r]
                idx += 1
        
        # 10th term: linear combination of terms a and b
        # Try several weight combinations
        for wa in [0.5, 1.0, 1.5, 2.0, -0.5, -1.0]:
            if time.time() > deadline - 10:
                break
            wb = 1.0 / wa if wa != 0 else 1.0
            U_init[:, 9] = wa * U11[:, a] + wb * U11[:, b]
            V_init[:, 9] = wa * V11[:, a] + wb * V11[:, b]
            W_init[:, 9] = wa * W11[:, a] + wb * W11[:, b]
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            result = try_polish(x0)
            if result is not None:
                return result
    
    # Strategy 2: Random initializations with more trials
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        
        x0 = rng.normal(0, 1, 160)
        result = try_polish(x0, max_nfev=1500)
        if result is not None:
            return result
    
    # Strategy 3: Small perturbations of the rank-11 solution (drop one term, adjust)
    for drop_term in range(11):
        if time.time() > deadline - 10:
            break
        
        # Initialize 10 terms from rank-11, dropping one term
        U_init = np.zeros((4, 10))
        V_init = np.zeros((6, 10))
        W_init = np.zeros((6, 10))
        
        idx = 0
        for r in range(11):
            if r == drop_term:
                continue
            U_init[:, idx] = U11[:, r]
            V_init[:, idx] = V11[:, r]
            W_init[:, idx] = W11[:, r]
            idx += 1
        
        # Add small random perturbation to help escape local minima
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        x0 += rng.normal(0, 0.1, 160)
        
        result = try_polish(x0, max_nfev=2000)
        if result is not None:
            return result
    
    return None


def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # First, try the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        # Fallback to schoolbook
        best = _schoolbook()
        best_rank = 12
    
    # Try to find a rank-10 solution
    if best_rank > 10 and time.time() < deadline - 30:
        result10 = _try_rank10_search(deadline)
        if result10 is not None:
            best = result10
            best_rank = 10
    
    return best
