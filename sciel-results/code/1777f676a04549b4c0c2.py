import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def _verify_decomposition(U, V, W, n=6, m=6, p=6):
    """
    Verifies the tensor decomposition for matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: C[i, k] = sum_j A[i, j] B[j, k]
    Tensor: T[i, j, k] where i is flat index of A, j is flat index of B, k is flat index of C.
    k corresponds to (i_row, k_col) such that C[i_row, k_col] is the entry.
    W index convention: W[k*n + i, r] where k is col of C, i is row of C?
    Wait, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    Let's check the current code's verification:
    C_flat[k] = sum_r W[k, r] x_r y_r
    C_computed = C_flat.reshape(6, 6).T
    
    Let's trace the indices carefully.
    A is (n, m). A_flat[i] where i = i_row * m + i_col.
    B is (m, p). B_flat[j] where j = j_row * p + j_col.
    C is (n, p). C[i_row, k_col].
    The tensor T relates A[i, j], B[j, k] to C[i, k].
    The index in the tensor for C is usually mapped to a single index.
    The task specifies W has shape (p*n, rank).
    And the convention is W[k*n + i, r].
    Here i is likely the row index of C (0..n-1) and k is the column index of C (0..p-1).
    So the flat index in W is k * n + i.
    
    In the verification in the current program:
    C_flat = W @ (x * y)
    C_computed = C_flat.reshape(6, 6).T
    
    If C_flat has index `idx` corresponding to `k*n + i`, then:
    C_flat[idx] = C[k, i] (if we use C[row, col] = C[i, k] standard)
    Wait, standard C is n x p. C[i, k].
    If W index is k*n + i, then C_flat[k*n+i] = C[i, k].
    So C_flat.reshape(n, p, order='F') would give C[i, k] at [i, k]?
    No, reshape(6, 6).T means:
    If C_flat is 1D of length 36.
    Let M = C_flat.reshape(6, 6). Then M[a, b] = C_flat[a*6 + b].
    C_computed = M.T. So C_computed[i, k] = M[k, i] = C_flat[k*6 + i].
    This matches W index k*n + i (with n=6).
    So C_computed[i, k] corresponds to W index k*6 + i.
    This is consistent.
    """
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randint(-2, 3, size=(n, m)).astype(np.float64)
        B = np.random.randint(-2, 3, size=(m, p)).astype(np.float64)
        
        C_expected = A @ B
        
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        # x_r = sum_i U[i, r] A_flat[i]
        x = np.dot(U_arr.T, A_flat)
        # y_r = sum_j V[j, r] B_flat[j]
        y = np.dot(V_arr.T, B_flat)
        
        # C_flat[k_idx] = sum_r W[k_idx, r] x_r y_r
        C_flat = np.dot(W_arr, x * y)
        
        C_computed = C_flat.reshape(n, p).T
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 6, 6, 6
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Rank 189 Decomposition
    # Structure: 3x3 grid of 2x2 blocks.
    # We can view 6x6 mat mult as 3x3 mat mult of 2x2 blocks.
    # Let A = [[A00, A01, A02], [A10, A11, A12], [A20, A21, A22]] where Aij are 2x2.
    # C = A @ B.
    # C_ii = sum_j A_ij B_ji.
    # The multiplication of 2x2 blocks can be done with rank 7 (Strassen).
    # The outer 3x3 multiplication can be done with rank 27 (Schoolbook) or 23 (Bini).
    # Total rank = rank_outer * rank_inner.
    
    # Let's implement the Kronecker product construction directly.
    # Outer terms: list of dicts with 'U', 'V', 'W' for 3x3.
    # Inner terms: list of dicts with 'U', 'V', 'W' for 2x2.
    
    def _get_strassen_2x2():
        return [
            {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)]},
            {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
            {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (1,0,1)]},
            {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)]},
            {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
        ]

    def _get_schoolbook_3x3():
        terms = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    terms.append({
                        'U': [(i, j, 1)],
                        'V': [(j, k, 1)],
                        'W': [(i, k, 1)]
                    })
        return terms

    def _build_kronecker(terms_outer, terms_inner):
        rank = len(terms_outer) * len(terms_inner)
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        for u_idx, term_o in enumerate(terms_outer):
            for t_idx, term_i in enumerate(terms_inner):
                col = u_idx * len(terms_inner) + t_idx
                
                # U: A is n x m.
                # A_flat index: i_row * m + i_col.
                # i_row = r_o * 2 + r_i, i_col = c_o * 2 + c_i.
                # U_flat_idx = (r_o * 2 + r_i) * 6 + (c_o * 2 + c_i)
                for r_o, c_o, coeff_o in term_o['U']:
                    for r_i, c_i, coeff_i in term_i['U']:
                        r = r_o * 2 + r_i
                        c = c_o * 2 + c_i
                        U[r * m + c, col] += coeff_o * coeff_i
                
                # V: B is m x p.
                # B_flat index: j_row * p + j_col.
                # j_row = r_o * 2 + r_i, j_col = c_o * 2 + c_i.
                # V_flat_idx = (r_o * 2 + r_i) * 6 + (c_o * 2 + c_i)
                for r_o, c_o, coeff_o in term_o['V']:
                    for r_i, c_i, coeff_i in term_i['V']:
                        r = r_o * 2 + r_i
                        c = c_o * 2 + c_i
                        V[r * p + c, col] += coeff_o * coeff_i
                        
                # W: C is n x p.
                # W convention: W[k*n + i, r] where k is col of C, i is row of C.
                # C[i, k]. i = r_o * 2 + r_i, k = c_o * 2 + c_i.
                # W_flat_idx = k * n + i = (c_o * 2 + c_i) * 6 + (r_o * 2 + r_i)
                for r_o, c_o, coeff_o in term_o['W']:
                    for r_i, c_i, coeff_i in term_i['W']:
                        i = r_o * 2 + r_i
                        k = c_o * 2 + c_i
                        W[k * n + i, col] += coeff_o * coeff_i
                        
        return U.tolist(), V.tolist(), W.tolist()

    # Try Rank 189 (27 * 7)
    try:
        if time.time() < deadline:
            terms_27 = _get_schoolbook_3x3()
            terms_7 = _get_strassen_2x2()
            U, V, W = _build_kronecker(terms_27, terms_7)
            if _verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback: Direct Schoolbook 6x6 (Rank 216)
    # T_{i,j,k} = 1 if i_col == j_row and k_row == i_row and k_col == j_col
    try:
        if time.time() < deadline:
            rank = 216
            U = np.zeros((36, rank), dtype=np.float64)
            V = np.zeros((36, rank), dtype=np.float64)
            W = np.zeros((36, rank), dtype=np.float64)
            
            idx = 0
            for i_row in range(6):
                for i_col in range(6):
                    for k_col in range(6):
                        j = i_col # i_col == j_row
                        # k_row must be i_row
                        # W index: k_col * 6 + i_row
                        # U index: i_row * 6 + i_col
                        # V index: j * 6 + k_col = i_col * 6 + k_col
                        
                        U[i_row * 6 + i_col, idx] = 1.0
                        V[i_col * 6 + k_col, idx] = 1.0
                        W[k_col * 6 + i_row, idx] = 1.0
                        idx += 1
                        
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify_decomposition(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass

    # Should never reach here
    return {"U": [], "V": [], "W": []}
