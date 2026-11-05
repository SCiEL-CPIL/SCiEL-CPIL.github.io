import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _start_time)

def _add_strassen(U_list, V_list, W_list, i_start, j_start, k_start, n=4, m=5, p=5):
    """
    Adds a Strassen decomposition for a 2x2x2 block.
    i_start, j_start, k_start are the starting indices for the 2x2, 2x2, 2x2 submatrices.
    """
    # Strassen's 7 multiplications
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices
    # A is 2x2, B is 2x2, C is 2x2
    # A indices: (i_start, j_start), (i_start+1, j_start), (i_start, j_start+1), (i_start+1, j_start+1)
    # B indices: (j_start, k_start), (j_start, k_start+1), (j_start+1, k_start), (j_start+1, k_start+1)
    # C indices: (i_start, k_start), (i_start, k_start+1), (i_start+1, k_start), (i_start+1, k_start+1)
    
    # We need to add 7 rank-1 terms.
    # Each term is defined by vectors u, v, w of length 4 (for the 2x2 blocks)
    # But we are filling the global U, V, W matrices.
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        # u_vals is dict {(i,j): val}
        # v_vals is dict {(j,k): val}
        # w_vals is dict {(k,i): val}
        
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for (i, j), val in u_vals.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_vals.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_vals.items():
            w_vec[k * n + i] = val
            
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    # Define the 7 terms
    # M1
    u1 = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
    v1 = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
    w1 = {(k_start, i_start): 1, (k_start+1, i_start+1): 1, (k_start, i_start+1): -1, (k_start+1, i_start): 1}
    # Wait, C00 = M1 + ...
    # The W matrix entries correspond to the coefficients in the sum for C_ik.
    # C_ik = sum_c U_ic V_ck W_ki
    # So W_ki is the coefficient for the product U_ic V_ck.
    
    # Let's re-verify the Strassen coefficients for W.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Term 1 (M1):
    # U: A00, A11
    # V: B00, B11
    # W: C00 (+1), C11 (+1)
    add_term(
        {(i_start, j_start): 1, (i_start+1, j_start+1): 1},
        {(j_start, k_start): 1, (j_start+1, k_start+1): 1},
        {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
    )
    
    # Term 2 (M2):
    # U: A10, A11
    # V: B00
    # W: C10 (+1), C11 (-1)
    add_term(
        {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1},
        {(j_start, k_start): 1},
        {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
    )
    
    # Term 3 (M3):
    # U: A00
    # V: B01, B11
    # W: C01 (+1), C11 (+1)
    add_term(
        {(i_start, j_start): 1},
        {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1},
        {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
    )
    
    # Term 4 (M4):
    # U: A11
    # V: B10, B00
    # W: C00 (+1), C10 (+1)
    add_term(
        {(i_start+1, j_start+1): 1},
        {(j_start+1, k_start): 1, (j_start, k_start): -1},
        {(k_start, i_start): 1, (k_start, i_start+1): 1}
    )
    
    # Term 5 (M5):
    # U: A00, A01
    # V: B11
    # W: C00 (-1), C01 (+1)
    add_term(
        {(i_start, j_start): 1, (i_start, j_start+1): 1},
        {(j_start+1, k_start+1): 1},
        {(k_start+1, i_start): -1, (k_start+1, i_start+1): 1}
    )
    
    # Term 6 (M6):
    # U: A10, A00
    # V: B00, B01
    # W: C11 (+1)
    add_term(
        {(i_start+1, j_start): 1, (i_start, j_start): -1},
        {(j_start, k_start): 1, (j_start, k_start+1): 1},
        {(k_start+1, i_start+1): 1}
    )
    
    # Term 7 (M7):
    # U: A01, A11
    # V: B10, B11
    # W: C00 (+1)
    add_term(
        {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1},
        {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1},
        {(k_start, i_start): 1}
    )

def _add_schoolbook_2x2x1(U_list, V_list, W_list, i_start, j_start, k_start, n=4, m=5, p=5):
    """
    Adds schoolbook decomposition for A(2x2) * B(2x1) -> C(2x1).
    Rank 4.
    """
    def add_term(u_vals, v_vals, w_vals):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for (i, j), val in u_vals.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_vals.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_vals.items():
            w_vec[k * n + i] = val
            
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    # C0 = A00 B00 + A01 B10
    # C1 = A10 B00 + A11 B10
    
    # Term 1: A00 B00 -> C0
    add_term(
        {(i_start, j_start): 1},
        {(j_start, k_start): 1},
        {(k_start, i_start): 1}
    )
    # Term 2: A01 B10 -> C0
    add_term(
        {(i_start, j_start+1): 1},
        {(j_start+1, k_start): 1},
        {(k_start, i_start): 1}
    )
    # Term 3: A10 B00 -> C1
    add_term(
        {(i_start+1, j_start): 1},
        {(j_start, k_start): 1},
        {(k_start, i_start+1): 1}
    )
    # Term 4: A11 B10 -> C1
    add_term(
        {(i_start+1, j_start+1): 1},
        {(j_start+1, k_start): 1},
        {(k_start, i_start+1): 1}
    )

def _add_schoolbook_2x1x5(U_list, V_list, W_list, i_start, j_start, k_start, n=4, m=5, p=5):
    """
    Adds schoolbook decomposition for A(2x1) * B(1x5) -> C(2x5).
    Rank 10.
    """
    def add_term(u_vals, v_vals, w_vals):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for (i, j), val in u_vals.items():
            u_vec[i * m + j] = val
        for (j, k), val in v_vals.items():
            v_vec[j * p + k] = val
        for (k, i), val in w_vals.items():
            w_vec[k * n + i] = val
            
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)

    # C_ik = A_i B_k
    for i in range(2):
        for k in range(5):
            add_term(
                {(i_start + i, j_start): 1},
                {(j_start, k_start + k): 1},
                {(k_start + k, i_start + i): 1}
            )

def construct():
    n, m, p = DIMENSIONS
    
    U_list = []
    V_list = []
    W_list = []
    
    # Block 1: A[0:2, 0:4] * B[0:4, 0:5]
    # Split into 2x2x2 and 2x2x1 blocks
    # A[0:2, 0:2] * B[0:2, 0:2] (Strassen)
    _add_strassen(U_list, V_list, W_list, 0, 0, 0)
    # A[0:2, 0:2] * B[0:2, 2:4] (Strassen)
    _add_strassen(U_list, V_list, W_list, 0, 0, 2)
    # A[0:2, 0:2] * B[0:2, 4:5] (Schoolbook 2x2x1)
    _add_schoolbook_2x2x1(U_list, V_list, W_list, 0, 0, 4)
    
    # A[0:2, 2:4] * B[2:4, 0:2] (Strassen)
    _add_strassen(U_list, V_list, W_list, 0, 2, 0)
    # A[0:2, 2:4] * B[2:4, 2:4] (Strassen)
    _add_strassen(U_list, V_list, W_list, 0, 2, 2)
    # A[0:2, 2:4] * B[2:4, 4:5] (Schoolbook 2x2x1)
    _add_schoolbook_2x2x1(U_list, V_list, W_list, 0, 2, 4)
    
    # Block 2: A[2:4, 0:4] * B[0:4, 0:5]
    # A[2:4, 0:2] * B[0:2, 0:2] (Strassen)
    _add_strassen(U_list, V_list, W_list, 2, 0, 0)
    # A[2:4, 0:2] * B[0:2, 2:4] (Strassen)
    _add_strassen(U_list, V_list, W_list, 2, 0, 2)
    # A[2:4, 0:2] * B[0:2, 4:5] (Schoolbook 2x2x1)
    _add_schoolbook_2x2x1(U_list, V_list, W_list, 2, 0, 4)
    
    # A[2:4, 2:4] * B[2:4, 0:2] (Strassen)
    _add_strassen(U_list, V_list, W_list, 2, 2, 0)
    # A[2:4, 2:4] * B[2:4, 2:4] (Strassen)
    _add_strassen(U_list, V_list, W_list, 2, 2, 2)
    # A[2:4, 2:4] * B[2:4, 4:5] (Schoolbook 2x2x1)
    _add_schoolbook_2x2x1(U_list, V_list, W_list, 2, 2, 4)
    
    # Block 3: A[0:4, 4:5] * B[4:5, 0:5]
    # A[0:2, 4] * B[4, 0:5] (Schoolbook 2x1x5)
    _add_schoolbook_2x1x5(U_list, V_list, W_list, 0, 4, 0)
    # A[2:4, 4] * B[4, 0:5] (Schoolbook 2x1x5)
    _add_schoolbook_2x1x5(U_list, V_list, W_list, 2, 4, 0)
    
    # Convert to numpy arrays
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
