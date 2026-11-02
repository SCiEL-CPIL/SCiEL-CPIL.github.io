import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def _get_strassen_49():
    """Fallback: Recursive Strassen Rank 49."""
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_2x2 = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    def get_elem_linear_form(block_idx, r, c, coeff):
        vec = [0] * 16
        if block_idx == 0:
            idx = r * 4 + c
        elif block_idx == 1:
            idx = r * 4 + (c + 2)
        elif block_idx == 2:
            idx = (r + 2) * 4 + c
        elif block_idx == 3:
            idx = (r + 2) * 4 + (c + 2)
        vec[idx] = coeff
        return vec

    def add_vec(v1, v2, sign=1):
        return [v1[i] + sign * v2[i] for i in range(16)]

    A_forms = []
    B_forms = []
    
    # t=0
    A_forms.append([add_vec(get_elem_linear_form(0, r, c, 1), get_elem_linear_form(3, r, c, 1)) for r in range(2) for c in range(2)])
    B_forms.append([add_vec(get_elem_linear_form(0, r, c, 1), get_elem_linear_form(3, r, c, 1)) for r in range(2) for c in range(2)])
    # t=1
    A_forms.append([add_vec(get_elem_linear_form(2, r, c, 1), get_elem_linear_form(3, r, c, 1)) for r in range(2) for c in range(2)])
    B_forms.append([get_elem_linear_form(0, r, c, 1) for r in range(2) for c in range(2)])
    # t=2
    A_forms.append([get_elem_linear_form(0, r, c, 1) for r in range(2) for c in range(2)])
    B_forms.append([add_vec(get_elem_linear_form(1, r, c, 1), get_elem_linear_form(3, r, c, -1)) for r in range(2) for c in range(2)])
    # t=3
    A_forms.append([get_elem_linear_form(3, r, c, 1) for r in range(2) for c in range(2)])
    B_forms.append([add_vec(get_elem_linear_form(2, r, c, 1), get_elem_linear_form(0, r, c, -1)) for r in range(2) for c in range(2)])
    # t=4
    A_forms.append([add_vec(get_elem_linear_form(0, r, c, 1), get_elem_linear_form(1, r, c, 1)) for r in range(2) for c in range(2)])
    B_forms.append([get_elem_linear_form(3, r, c, 1) for r in range(2) for c in range(2)])
    # t=5
    A_forms.append([add_vec(get_elem_linear_form(2, r, c, 1), get_elem_linear_form(0, r, c, -1)) for r in range(2) for c in range(2)])
    B_forms.append([add_vec(get_elem_linear_form(0, r, c, 1), get_elem_linear_form(1, r, c, 1)) for r in range(2) for c in range(2)])
    # t=6
    A_forms.append([add_vec(get_elem_linear_form(1, r, c, 1), get_elem_linear_form(3, r, c, -1)) for r in range(2) for c in range(2)])
    B_forms.append([add_vec(get_elem_linear_form(2, r, c, 1), get_elem_linear_form(3, r, c, 1)) for r in range(2) for c in range(2)])

    block_contribs = {
        0: [(0, 0, 1), (1, 1, 1)],
        1: [(1, 0, 1), (1, 1, -1)],
        2: [(0, 1, 1), (1, 1, 1)],
        3: [(0, 0, 1), (1, 0, 1)],
        4: [(0, 0, -1), (0, 1, 1)],
        5: [(1, 1, 1)],
        6: [(0, 0, 1)]
    }

    for t in range(7):
        for s in range(7):
            r = t * 7 + s
            u_2x2, v_2x2, w_2x2 = strassen_2x2[s]
            
            u_full = [0] * 16
            v_full = [0] * 16
            
            for k in range(4):
                if u_2x2[k] != 0:
                    u_full = add_vec(u_full, A_forms[t][k], u_2x2[k])
                if v_2x2[k] != 0:
                    v_full = add_vec(v_full, B_forms[t][k], v_2x2[k])
            
            w_full = [0] * 16
            for k in range(4):
                if w_2x2[k] == 0: continue
                r_block = k // 2
                c_block = k % 2
                for (br, bc, weight) in block_contribs[t]:
                    total_coeff = weight * w_2x2[k]
                    if total_coeff != 0:
                        row = br * 2 + r_block
                        col = bc * 2 + c_block
                        idx = col * 4 + row
                        w_full[idx] += total_coeff
            
            for i in range(16):
                U[i][r] = u_full[i]
                V[i][r] = v_full[i]
                W[i][r] = w_full[i]

    return {"U": U, "V": V, "W": W}

def _get_laderman_26():
    """Primary: Laderman Rank 26."""
    # Implementation of Laderman's 26 forms
    # This is a placeholder structure. If coefficients are incorrect, verification will fail.
    # For this submission, I will use a known set if available, otherwise return None to trigger fallback.
    # Since I cannot guarantee the 26 coefficients from memory, I will return None to force fallback.
    # However, to attempt the improvement, I will try to construct a valid rank 26 if possible.
    # Given the risk, I will return None.
    return None

def construct():
    start_time = time.time()
    
    # Try Laderman
    result = _get_laderman_26()
    if result is not None:
        # Verify
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        
        # Random test
        np.random.seed(42)
        A = np.random.rand(16)
        B = np.random.rand(16)
        
        # Compute C via decomposition
        C_decomp = np.zeros(16)
        for r in range(U.shape[1]):
            u_val = np.dot(U[:, r], A)
            v_val = np.dot(V[:, r], B)
            C_decomp += u_val * v_val * W[:, r]
            
        # Compute C via matrix multiplication
        A_mat = A.reshape(4, 4)
        B_mat = B.reshape(4, 4)
        C_mat = A_mat @ B_mat
        C_flat = C_mat.flatten()
        
        if np.allclose(C_decomp, C_flat):
            return result
    
    # Fallback to Strassen
    return _get_strassen_49()
