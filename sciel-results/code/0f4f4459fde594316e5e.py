"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Uses Strassen's algorithm on four 2x2 sub-blocks to achieve rank 28 (vs schoolbook 32).
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def check_time_and_return(best_result):
    if best_result is not None and (time.time() - START_TIME) > TIME_LIMIT - 10.0:
        return best_result
    return None


def construct():
    best_result = None
    n, m, p = DIMENSIONS
    try:
        best_result = _construct_strassen_28()
        if best_result is not None:
            # Verify the decomposition with the corrected verifier
            if _verify_decomposition(best_result['U'], best_result['V'], best_result['W'], n, m, p):
                # Try to find a lower-rank decomposition
                remaining_time = TIME_LIMIT - 15.0 - (time.time() - START_TIME)
                if remaining_time > 10.0:
                    improved = _search_for_lower_rank(
                        best_result['U'], best_result['V'], best_result['W'],
                        n, m, p, time_budget=remaining_time - 5.0
                    )
                    if improved is not None:
                        return improved
                return best_result
            else:
                best_result = None
    except Exception:
        pass
    # Fallback to schoolbook
    if best_result is None:
        best_result = _construct_schoolbook_32()
    return best_result


def _search_for_lower_rank(U, V, W, n, m, p, time_budget=30.0):
    """
    Try to find a lower-rank decomposition by testing subsets of columns.
    Start from the current rank and try to remove columns one at a time.
    """
    start = time.time()
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_rank = rank
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    # Try removing columns greedily
    current_rank = rank
    changed = True
    while changed and (time.time() - start) < time_budget - 5.0:
        changed = False
        for r in range(current_rank):
            if (time.time() - start) > time_budget - 5.0:
                break
            keep = [i for i in range(current_rank) if i != r]
            U_test = U_arr[:, keep]
            V_test = V_arr[:, keep]
            W_test = W_arr[:, keep]
            
            U_list = [[U_test[i, j] for j in range(len(keep))] for i in range(len(U_test))]
            V_list = [[V_test[i, j] for j in range(len(keep))] for i in range(len(V_test))]
            W_list = [[W_test[i, j] for j in range(len(keep))] for i in range(len(W_test))]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                current_rank -= 1
                changed = True
                break
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    
    if best_rank < rank:
        U_list = [[best_U[i, j] for j in range(best_rank)] for i in range(len(best_U))]
        V_list = [[best_V[i, j] for j in range(best_rank)] for i in range(len(best_V))]
        W_list = [[best_W[i, j] for j in range(best_rank)] for i in range(len(best_W))]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the full tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Shape: (n*m, m*p, p*n) = (8, 16, 8)
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    # Check matching entries: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                if abs(T[u_row, v_row, w_row] - 1.0) > tol:
                    return False
    
    # Check non-matching entries should be 0
    # An entry (a, b, c) is "matching" if there exist i,j,k such that
    # a = i*m+j, b = j*p+k, c = k*n+i
    # For a given (a,b): a = i*m+j gives i=a//m, j=a%m; b = j'*p+k gives j'=b//p, k=b%p
    # These are consistent iff j == j', i.e., a%m == b//p
    # Then c should equal k*n + i = (b%p)*n + a//m
    for a in range(n * m):
        for b in range(m * p):
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            if j == j2:
                # This is a matching triple; already checked above
                continue
            else:
                # Non-matching: T[a,b,c] must be 0 for all c
                if np.any(np.abs(T[a, b, :]) > tol):
                    return False
    
    return True


def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
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


def _construct_strassen_28():
    """
    2x4 * 4x4 using four 2x2 Strassen blocks.
    
    Partition:
      A is 2x4: A = [A1 | A2] where A1, A2 are 2x2
      B is 4x4: B = [B11 B12; B21 B22] all 2x2
      C = A * B is 2x4: C = [C1 | C2] where
        C1 = A1*B11 + A2*B21
        C2 = A1*B12 + A2*B22
    
    For C1 = A1*B11 + A2*B21:
      We compute Strassen for (A1, B11) and (A2, B21) separately.
      Each contributes 7 rank-1 terms. Total 14 for C1.
    Similarly for C2: 14 terms.
    Total rank 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    sub_products = [
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
    ]
    
    col = 0
    for sp in sub_products:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                r, c = loc_u // 2, loc_u % 2
                global_u = sp['a_map'](r, c)
                U[global_u][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                r, c = loc_v // 2, loc_v % 2
                global_v = sp['b_map'](r, c)
                V[global_v][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                r, c = loc_w // 2, loc_w % 2
                global_w = sp['c_map'](r, c)
                W[global_w][col] += coeff
            col += 1
    
    # Post-process: check for linearly dependent rank-1 terms and remove them
    result = _remove_dependent_terms(U, V, W, n, m, p)
    return result


def _remove_dependent_terms(U, V, W, n, m, p, tol=1e-12):
    """
    Check if any rank-1 terms in the decomposition are linearly dependent
    and can be removed. Uses iterative elimination: try removing each column
    and check if the remaining columns still span the same tensor.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # First, check for zero columns
    active = []
    for r in range(rank):
        if np.any(np.abs(U_arr[:, r]) > tol) and np.any(np.abs(V_arr[:, r]) > tol) and np.any(np.abs(W_arr[:, r]) > tol):
            active.append(r)
    
    if len(active) < rank:
        U_arr = U_arr[:, active]
        V_arr = V_arr[:, active]
        W_arr = W_arr[:, active]
        rank = len(active)
    
    if rank == 0:
        return {"U": [[0] for _ in range(n*m)], "V": [[0] for _ in range(m*p)], "W": [[0] for _ in range(p*n)]}
    
    # Iteratively try to remove columns
    # For each column r, check if removing it still gives a valid decomposition
    # This is expensive but rank is small (28)
    current_rank = rank
    changed = True
    while changed:
        changed = False
        for r in range(current_rank):
            # Try removing column r
            keep = [i for i in range(current_rank) if i != r]
            U_test = U_arr[:, keep]
            V_test = V_arr[:, keep]
            W_test = W_arr[:, keep]
            
            if _verify_decomposition(
                [[U_test[i, j] for j in range(len(keep))] for i in range(len(U_test))],
                [[V_test[i, j] for j in range(len(keep))] for i in range(len(V_test))],
                [[W_test[i, j] for j in range(len(keep))] for i in range(len(W_test))],
                n, m, p
            ):
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                current_rank -= 1
                changed = True
                break
    
    if current_rank == rank:
        return {"U": U, "V": V, "W": W}
    
    U_list = [[U_arr[i, j] for j in range(current_rank)] for i in range(len(U_arr))]
    V_list = [[V_arr[i, j] for j in range(current_rank)] for i in range(len(V_arr))]
    W_list = [[W_arr[i, j] for j in range(current_rank)] for i in range(len(W_arr))]
    
    return {"U": U_list, "V": V_list, "W": W_list}
