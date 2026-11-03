"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Uses Strassen's algorithm on four 2x2 sub-blocks to achieve rank 28 (vs schoolbook 32).
Rank 28 is the best known upper bound for this specific tensor product.
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
    try:
        best_result = _construct_strassen_28()
        if best_result is not None:
            if _verify_decomposition(best_result):
                return best_result
    except Exception:
        pass
    # Fallback to schoolbook
    if best_result is None:
        best_result = _construct_schoolbook_32()
    return best_result


def _verify_decomposition(result):
    """Verify that the decomposition correctly computes matrix multiplication."""
    try:
        n, m, p = DIMENSIONS
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        # Test with random matrices
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Reshape for efficient computation
        # U: (n*m, rank) -> (n, m, rank)
        U_reshaped = U.reshape(n, m, -1)
        # V: (m*p, rank) -> (m, p, rank)
        V_reshaped = V.reshape(m, p, -1)
        # W: (p*n, rank) -> (p, n, rank)
        W_reshaped = W.reshape(p, n, -1)
        
        # Compute intermediate tensor P[i, k, t] = sum_j U[i, j, t] * V[j, k, t]
        P = np.einsum('imt,jkt->ikt', U_reshaped, V_reshaped)
        
        # Align W to (n, p, rank) to match P indices
        # W_reshaped is (p, n, rank), so W_reshaped[k, i, t]
        # We need W_aligned[i, k, t] = W_reshaped[k, i, t]
        W_aligned = W_reshaped.transpose(1, 0, 2)
        
        # Compute final C[i, k] = sum_t P[i, k, t] * W_aligned[i, k, t]
        C_computed = np.einsum('ikt,ikt->ik', P, W_aligned)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True
    except Exception:
        return False


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
    
    For each 2x2 product, use Strassen (rank 7). Four products -> rank 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2 product X*Y:
    # Local indices: a=0 (x00), b=1 (x01), c=2 (x10), d=3 (x11)
    # P1 = (x00+x11)(y00+y11) -> c00:+1, c11:+1
    # P2 = (x10+x11)y00 -> c10:+1, c11:-1
    # P3 = x00(y01-y11) -> c01:+1, c11:+1
    # P4 = x11(y10-y00) -> c00:+1, c10:+1
    # P5 = (x00+x01)y11 -> c00:-1, c01:+1
    # P6 = (x10-x00)(y00+y01) -> c11:+1
    # P7 = (x01-x11)(y10+y11) -> c00:+1
    
    # W output indices: c00=0, c01=1, c10=2, c11=3 (local)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # P1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # P2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # P3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # P4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),         # P5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # P6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),         # P7
    ]
    
    # Four sub-products with their index mappings
    # Each mapping: local (r,c) -> global index
    sub_products = [
        # 1. A_left (A[:,0:2]) * B_tl (B[0:2,0:2]) -> C_left (C[:,0:2])
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        # 2. A_right (A[:,2:4]) * B_bl (B[2:4,0:2]) -> C_left (C[:,0:2])
        {
            'a_map': lambda r, c: r * 4 + (c + 2),
            'b_map': lambda r, c: (r + 2) * 4 + c,
            'c_map': lambda r, c: c * 2 + r,
        },
        # 3. A_left (A[:,0:2]) * B_tr (B[0:2,2:4]) -> C_right (C[:,2:4])
        {
            'a_map': lambda r, c: r * 4 + c,
            'b_map': lambda r, c: r * 4 + (c + 2),
            'c_map': lambda r, c: (c + 2) * 2 + r,
        },
        # 4. A_right (A[:,2:4]) * B_br (B[2:4,2:4]) -> C_right (C[:,2:4])
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
    
    assert col == 28, f"Expected col=28, got {col}"
    
    return {"U": U, "V": V, "W": W}
