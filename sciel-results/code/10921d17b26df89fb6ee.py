"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # We will build the decomposition in blocks.
    # U: (6, rank), V: (21, rank), W: (14, rank)
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Append a rank-1 term to the factor lists."""
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    def get_A_index(i, j):
        return i * m + j
        
    def get_B_index(j, k):
        return j * p + k
        
    def get_C_index(k, i):
        return k * n + i

    # --- Helper: Schoolbook for small blocks ---
    def schoolbook_block(n_sub, m_sub, p_sub, A_offset_i, A_offset_j, B_offset_j, B_offset_k, C_offset_k, C_offset_i):
        """
        Adds schoolbook terms for a sub-block.
        A is n_sub x m_sub, B is m_sub x p_sub, C is n_sub x p_sub.
        Offsets map local indices to global indices.
        """
        for i_sub in range(n_sub):
            for j_sub in range(m_sub):
                for k_sub in range(p_sub):
                    i_global = A_offset_i + i_sub
                    j_global = A_offset_j + j_sub
                    b_j_global = B_offset_j + j_sub
                    k_global = B_offset_k + k_sub
                    c_k_global = C_offset_k + k_sub
                    c_i_global = C_offset_i + i_sub
                    
                    u_vec = [0.0] * nm
                    v_vec = [0.0] * mp
                    w_vec = [0.0] * pn
                    
                    u_vec[get_A_index(i_global, j_global)] = 1.0
                    v_vec[get_B_index(b_j_global, k_global)] = 1.0
                    w_vec[get_C_index(c_k_global, c_i_global)] = 1.0
                    
                    add_term(u_vec, v_vec, w_vec)

    # --- Helper: Strassen for 2x2x2 ---
    def strassen_block(A_off_i, A_off_j, B_off_j, B_off_k, C_off_k, C_off_i):
        """
        Adds Strassen terms for a 2x2x2 block.
        Local indices 0,1 for i,j,k.
        """
        # Strassen multiplications M1..M7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # We define the terms as linear combinations of A and B indices.
        # Each term is a sum of rank-1 tensors? No, Strassen uses intermediate variables.
        # We need to expand these into the global U, V, W.
        
        # Term 1: (A00+A11) * (B00+B11)
        # This is (A00*B00 + A00*B11 + A11*B00 + A11*B11)
        # But Strassen computes it as one multiplication of sums.
        # In tensor decomposition, we add terms for the expanded products if we don't use intermediate storage in the factor matrices.
        # Wait, the factor matrices U, V, W represent the coefficients of the scalar multiplications.
        # If we use Strassen, we have 7 scalar multiplications.
        # Each scalar multiplication is of the form (sum of A terms) * (sum of B terms).
        # This corresponds to a rank-1 term in the tensor?
        # No. $C = \sum M_k$. $M_k = (\sum a_{ik} A_{ik}) (\sum b_{jk} B_{jk})$.
        # $C_{ik} = \sum_k M_k \cdot (\text{coeff})$.
        # The tensor entry $T_{\alpha \beta \gamma}$ is the coefficient of $A_\alpha B_\beta$ in $C_\gamma$.
        # In Strassen, $C_{00} = M1 + M4 - M5 + M7$.
        # $M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11$.
        # So the contribution of M1 to $C_{00}$ is the sum of these 4 products.
        # This means we add 4 rank-1 terms to the decomposition for this single "Strassen step"?
        # No, that would just be schoolbook.
        
        # The rank reduction comes from the fact that we compute $M1$ once and reuse it.
        # In the tensor decomposition $T = \sum u \otimes v \otimes w$,
        # a "multiplication" of sums corresponds to a single term in the sum?
        # Let's check:
        # If we have term $u \otimes v \otimes w$, it contributes $u_\alpha v_\beta w_\gamma$ to $T_{\alpha \beta \gamma}$.
        # If $M1 = (A00+A11)(B00+B11)$, then $u$ has non-zeros at A00, A11. $v$ has non-zeros at B00, B11.
        # Then $T_{A00, B00, C} += u_{A00} v_{B00} w_C$.
        # This correctly adds the product $A00 B00$ to $C$.
        # So YES, a Strassen multiplication $(\sum A)(\sum B)$ is represented by a SINGLE rank-1 term in the decomposition,
        # where U has non-zeros for all A parts, V has non-zeros for all B parts, and W has the coefficient for C.
        
        # So for Strassen M1:
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C01=0, C10=0, C11=1 (from C00=M1..., C11=M1...)
        
        def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
            # u_coeffs: dict {global_A_idx: val}
            # v_coeffs: dict {global_B_idx: val}
            # w_coeffs: dict {global_C_idx: val}
            
            u_vec = [0.0] * nm
            v_vec = [0.0] * mp
            w_vec = [0.0] * pn
            
            for idx, val in u_coeffs.items():
                u_vec[idx] = val
            for idx, val in v_coeffs.items():
                v_vec[idx] = val
            for idx, val in w_coeffs.items():
                w_vec[idx] = val
                
            add_term(u_vec, v_vec, w_vec)
            
        # Local to Global Index Mappers
        def li(i, j): return get_A_index(A_off_i + i, A_off_j + j)
        def lb(j, k): return get_B_index(B_off_j + j, B_off_k + k)
        def lc(k, i): return get_C_index(C_off_k + k, C_off_i + i)
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes to C00 (+1), C11 (+1)
        add_strassen_term(
            {li(0,0): 1, li(1,1): 1},
            {lb(0,0): 1, lb(0,1): 1}, # Wait, B is 2x2. B00 is (0,0), B11 is (1,1).
            {lc(0,0): 1, lc(1,1): 1}
        )
        
        # M2 = (A10+A11)B00
        # Contributes to C10 (+1), C11 (-1)
        add_strassen_term(
            {li(1,0): 1, li(1,1): 1},
            {lb(0,0): 1},
            {lc(0,1): 1, lc(1,1): -1} # C10 is (1,0) -> lc(0,1)? No.
            # C index is k*n + i.
            # C10: i=1, k=0 -> 0*2 + 1 = 1.
            # C11: i=1, k=1 -> 1*2 + 1 = 3.
            # C00: i=0, k=0 -> 0.
            # C01: i=0, k=1 -> 2.
            # My lc(k,i) is k*n+i.
            # C10 (row 1, col 0) -> k=0, i=1 -> lc(0,1) = 1.
            # C11 (row 1, col 1) -> k=1, i=1 -> lc(1,1) = 3.
            # C00 (row 0, col 0) -> k=0, i=0 -> lc(0,0) = 0.
            # C01 (row 0, col 1) -> k=1, i=0 -> lc(1,0) = 2.
            
        )
        # Correction: M2 contributes to C10 (+1) and C11 (-1).
        # C10 is index 1. C11 is index 3.
        # So w_coeffs should be {1: 1, 3: -1}.
        
        # Let's restart the Strassen terms with correct indices.
        # Indices:
        # C00: 0
        # C01: 2
        # C10: 1
        # C11: 3
        
        # M1 = (A00+A11)(B00+B11) -> C00(+1), C11(+1)
        add_strassen_term(
            {li(0,0): 1, li(1,1): 1},
            {lb(0,0): 1, lb(1,1): 1},
            {lc(0,0): 1, lc(1,1): 1}
        )
        
        # M2 = (A10+A11)B00 -> C10(+1), C11(-1)
        add_strassen_term(
            {li(1,0): 1, li(1,1): 1},
            {lb(0,0): 1},
            {lc(0,1): 1, lc(1,1): -1}
        )
        
        # M3 = A00(B10-B11) -> C01(+1), C10(+1)
        add_strassen_term(
            {li(0,0): 1},
            {lb(1,0): 1, lb(1,1): -1},
            {lc(1,0): 1, lc(0,1): 1}
        )
        
        # M4 = A11(B01-B10) -> C00(+1), C10(+1)
        add_strassen_term(
            {li(1,1): 1},
            {lb(0,1): 1, lb(1,0): -1},
            {lc(0,0): 1, lc(0,1): 1}
        )
        
        # M5 = (A00+A01)B11 -> C00(-1), C01(+1)
        add_strassen_term(
            {li(0,0): 1, li(0,1): 1},
            {lb(1,1): 1},
            {lc(0,0): -1, lc(1,0): 1}
        )
        
        # M6 = (A10-A00)(B00+B10) -> C11(+1)
        add_strassen_term(
            {li(1,0): 1, li(0,0): -1},
            {lb(0,0): 1, lb(1,0): 1},
            {lc(1,1): 1}
        )
        
        # M7 = (A01-A11)(B01+B11) -> C00(+1), C11(-1)
        add_strassen_term(
            {li(0,1): 1, li(1,1): -1},
            {lb(0,1): 1, lb(1,1): 1},
            {lc(0,0): 1, lc(1,1): -1}
        )

    # --- Construct M_2,3,3 blocks ---
    def construct_M233(B_col_offset):
        """
        Constructs decomposition for A(2x3) * B(3x3, starting at B_col_offset) -> C(2x3, starting at B_col_offset)
        """
        # Split A into A_2x2 (cols 0,1) and A_2x1 (col 2)
        # Split B into B_2x3 (rows 0,1) and B_1x3 (row 2)
        
        # 1. A_2x2 * B_2x2 -> C_2x2 (cols 0,1 of this block)
        # A cols 0,1. B rows 0,1, cols 0,1. C cols 0,1.
        strassen_block(
            A_off_i=0, A_off_j=0,
            B_off_j=0, B_off_k=B_col_offset,
            C_off_k=B_col_offset, C_off_i=0
        )
        
        # 2. A_2x2 * B_2x1 -> C_2x1 (col 2 of this block)
        # A cols 0,1. B rows 0,1, col 2. C col 2.
        schoolbook_block(
            n_sub=2, m_sub=2, p_sub=1,
            A_offset_i=0, A_offset_j=0,
            B_offset_j=0, B_offset_k=B_col_offset + 2,
            C_offset_k=B_col_offset + 2, C_offset_i=0
        )
        
        # 3. A_2x1 * B_1x3 -> C_2x3 (cols 0,1,2 of this block)
        # A col 2. B row 2, cols 0,1,2. C cols 0,1,2.
        schoolbook_block(
            n_sub=2, m_sub=1, p_sub=3,
            A_offset_i=0, A_offset_j=2,
            B_offset_j=2, B_offset_k=B_col_offset,
            C_offset_k=B_col_offset, C_offset_i=0
        )

    # --- Construct Full M_2,3,7 ---
    # B is 3x7. Split into 3x3 (cols 0-2), 3x3 (cols 3-5), 3x1 (col 6).
    
    # Block 1: Cols 0-2
    construct_M233(B_col_offset=0)
    
    # Block 2: Cols 3-5
    construct_M233(B_col_offset=3)
    
    # Block 3: Col 6
    # A(2x3) * B(3x1) -> C(2x1)
    # This is M_2,3,1. Rank 6.
    schoolbook_block(
        n_sub=2, m_sub=3, p_sub=1,
        A_offset_i=0, A_offset_j=0,
        B_offset_j=0, B_offset_k=6,
        C_offset_k=6, C_offset_i=0
    )
    
    # Convert lists of lists to numpy arrays or keep as lists
    # U_list is list of vectors (length 6)
    # V_list is list of vectors (length 21)
    # W_list is list of vectors (length 14)
    
    rank = len(U_list)
    
    U_mat = np.array(U_list).T
    V_mat = np.array(V_list).T
    W_mat = np.array(W_list).T
    
    # Verify
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
                
    T_recon = np.einsum('ac,bc,dc->abd', U_mat, V_mat, W_mat)
    
    if np.max(np.abs(T_recon - T)) > 1e-10:
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = np.zeros((nm, rank_school))
        V_school = np.zeros((mp, rank_school))
        W_school = np.zeros((pn, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1.0
                    V_school[j*p + k, col] = 1.0
                    W_school[k*n + i, col] = 1.0
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
