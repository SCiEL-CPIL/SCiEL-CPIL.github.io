import time
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 79
    rank = 79
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_indices, v_indices, w_indices):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_indices.items():
            U[idx][col] += val
        for idx, val in v_indices.items():
            V[idx][col] += val
        for idx, val in w_indices.items():
            W[idx][col] += val
        col += 1

    # Helper to get global indices for A (4x5)
    # A[i, j] -> i * m + j
    def a_idx(i, j):
        return i * m + j

    # Helper to get global indices for B (5x6)
    # B[j, k] -> j * p + k
    def b_idx(j, k):
        return j * p + k

    # Helper to get global indices for C (4x6)
    # C[i, k] -> W[k * n + i]
    def w_idx(i, k):
        return k * n + i

    # --- Part 1: A_L (4x4) * B_LL (4x4) -> C_LL (4x4) ---
    # A_L is A[0:4, 0:4]
    # B_LL is B[0:4, 0:4]
    # C_LL is C[0:4, 0:4]
    
    # Strassen 4x4x4 using 7x 2x2x2
    # Block indices in 2x2 grid:
    # A: (0,0), (0,1), (1,0), (1,1) -> sizes 2x2
    # B: (0,0), (0,1), (1,0), (1,1)
    # C: (0,0), (0,1), (1,0), (1,1)
    
    def strassen_2x2(A_base, B_base, C_base):
        # A_base, B_base, C_base are (i, j) offsets in the 4x4 grid?
        # No, this function handles a specific 2x2x2 multiplication
        # It assumes the 2x2 blocks are at specific global positions
        pass

    # We need to map the Strassen terms for 4x4 to the global U, V, W.
    # The 4x4 multiplication is decomposed into 7 sub-multiplications of 2x2 blocks.
    # Each sub-multiplication is a 2x2x2 tensor.
    # We will define the 2x2x2 Strassen decomposition and apply it 7 times.
    
    # Strassen 2x2x2 coefficients
    # M1: (A11+A22)(B11+B22) -> C11, C22
    # M2: (A21+A22)B11 -> C21, C22
    # M3: A11(B12-B22) -> C12, C22
    # M4: A22(B21-B11) -> C11, C21
    # M5: (A11+A12)B22 -> C11, C12
    # M6: (A21-A11)(B11+B12) -> C22
    # M7: (A12-A22)(B21+B22) -> C11
    
    # We will create a generic function to add a 2x2x2 Strassen decomposition
    # given the global row/col offsets for A, B, C blocks.
    
    def add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        # a_r, a_c: top-left row/col of A block (2x2)
        # b_r, b_c: top-left row/col of B block (2x2)
        # c_r, c_c: top-left row/col of C block (2x2)
        
        # A indices:
        # A11: (a_r, a_c), A12: (a_r, a_c+1)
        # A21: (a_r+1, a_c), A22: (a_r+1, a_c+1)
        
        # B indices:
        # B11: (b_r, b_c), B12: (b_r, b_c+1)
        # B21: (b_r+1, b_c), B22: (b_r+1, b_c+1)
        
        # C indices:
        # C11: (c_r, c_c), C12: (c_r, c_c+1)
        # C21: (c_r+1, c_c), C22: (c_r+1, c_c+1)
        
        # Term 1: (A11+A22)(B11+B22)
        add_term(
            {a_idx(a_r, a_c): 1, a_idx(a_r+1, a_c+1): 1},
            {b_idx(b_r, b_c): 1, b_idx(b_r+1, b_c+1): 1},
            {w_idx(c_r, c_c): 1, w_idx(c_r+1, c_c+1): 1}
        )
        
        # Term 2: (A21+A22)B11
        add_term(
            {a_idx(a_r+1, a_c): 1, a_idx(a_r+1, a_c+1): 1},
            {b_idx(b_r, b_c): 1},
            {w_idx(c_r+1, c_c): 1, w_idx(c_r+1, c_c+1): 1}
        )
        
        # Term 3: A11(B12-B22)
        add_term(
            {a_idx(a_r, a_c): 1},
            {b_idx(b_r, b_c+1): 1, b_idx(b_r+1, b_c+1): -1},
            {w_idx(c_r, c_c+1): 1, w_idx(c_r+1, c_c+1): 1}
        )
        
        # Term 4: A22(B21-B11)
        add_term(
            {a_idx(a_r+1, a_c+1): 1},
            {b_idx(b_r+1, b_c): 1, b_idx(b_r, b_c): -1},
            {w_idx(c_r, c_c): 1, w_idx(c_r+1, c_c): 1}
        )
        
        # Term 5: (A11+A12)B22
        add_term(
            {a_idx(a_r, a_c): 1, a_idx(a_r, a_c+1): 1},
            {b_idx(b_r+1, b_c+1): 1},
            {w_idx(c_r, c_c): 1, w_idx(c_r, c_c+1): 1}
        )
        
        # Term 6: (A21-A11)(B11+B12)
        add_term(
            {a_idx(a_r+1, a_c): 1, a_idx(a_r, a_c): -1},
            {b_idx(b_r, b_c): 1, b_idx(b_r, b_c+1): 1},
            {w_idx(c_r+1, c_c+1): 1}
        )
        
        # Term 7: (A12-A22)(B21+B22)
        add_term(
            {a_idx(a_r, a_c+1): 1, a_idx(a_r+1, a_c+1): -1},
            {b_idx(b_r+1, b_c): 1, b_idx(b_r+1, b_c+1): 1},
            {w_idx(c_r, c_c): 1}
        )

    # Apply to 4x4x4 part (Top-Left 4x4 of A, Top-Left 4x4 of B, Top-Left 4x4 of C)
    # A blocks: (0,0), (0,2), (2,0), (2,2)
    # B blocks: (0,0), (0,2), (2,0), (2,2)
    # C blocks: (0,0), (0,2), (2,0), (2,2)
    
    # Strassen Formulas for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We call add_strassen_2x2 for each Mi, with appropriate offsets.
    
    # M1: A(0,0)+A(2,2) * B(0,0)+B(2,2) -> C(0,0), C(2,2)
    # Note: add_strassen_2x2 handles the internal 2x2 structure.
    # We need to pass the base offsets for the 2x2 blocks involved.
    
    # M1 involves A11 (0,0), A22 (2,2).
    # But add_strassen_2x2 expects a single 2x2 block for A?
    # No, my add_strassen_2x2 function adds a *single* 2x2x2 tensor term.
    # Strassen for 4x4 requires 7 *such* tensors.
    # So we call add_strassen_2x2 7 times, but we need to map the coefficients.
    # Wait, the coefficients in Strassen 4x4 are sums of blocks.
    # My add_strassen_2x2 function adds a term like $A_{sub} B_{sub} C_{sub}$ where $A_{sub}$ is a 2x2 matrix.
    # But in Strassen 4x4, $M_1 = (A_{11}+A_{22})(B_{11}+B_{22})$.
    # This is NOT a single 2x2x2 tensor. It is a product of *sums*.
    # To decompose $(A_{11}+A_{22})(B_{11}+B_{22})$ into rank-1 tensors, we must expand it?
    # No, $(A_{11}+A_{22})$ is a 4x4 matrix? No, $A_{11}$ is 2x2.
    # $A_{11}+A_{22}$ is a 2x2 matrix (sum of two 2x2 blocks).
    # So $M_1$ is a 2x2 matrix multiplication: $(A_{11}+A_{22}) (B_{11}+B_{22})$.
    # This 2x2 multiplication can be decomposed using Strassen 2x2 (Rank 7).
    # So yes, we call add_strassen_2x2 once for M1, but we need to pass the *combined* coefficients?
    # My add_strassen_2x2 function hardcodes the Strassen coefficients for the 2x2 blocks.
    # It assumes the input A is a 2x2 matrix.
    # Here, the "input A" for the M1 multiplication is $A_{11}+A_{22}$.
    # So we need to modify the function to accept coefficient vectors for the 4 sub-blocks of the 2x2 A, B, C.
    
    # Let's redefine add_strassen_2x2 to take coefficients.
    
    def add_strassen_2x2_coeff(a_coeffs, b_coeffs, c_coeffs):
        # a_coeffs: dict { (ri, ci): val } for the 2x2 A block
        # b_coeffs: dict { (ri, ci): val } for the 2x2 B block
        # c_coeffs: dict { (ri, ci): val } for the 2x2 C block
        # This adds 7 terms.
        
        # Strassen terms for 2x2:
        # M1: (A11+A22)(B11+B22)
        # We need to form the linear combinations.
        
        # This is getting complex. Let's stick to the 79 rank construction but implement it carefully.
        
        pass

    # Given the complexity and time limit, I will return the schoolbook decomposition if I cannot implement the 79-rank one quickly and correctly.
    # But 79 is a big jump.
    
    # Let's try to implement the 79-rank version.
    
    # Re-implementing add_strassen_2x2 to handle linear combinations.
    
    def add_strassen_2x2_lin(a_comb, b_comb, c_comb):
        # a_comb: list of (coeff, r, c) for A 2x2
        # b_comb: list of (coeff, r, c) for B 2x2
        # c_comb: list of (coeff, r, c) for C 2x2
        # We add 7 terms corresponding to Strassen 2x2.
        
        # A11, A12, A21, A22 are linear combinations of global A elements.
        # We build the U, V, W contributions for each of the 7 Strassen terms.
        
        # Strassen Term 1: (A11+A22)(B11+B22) -> C11, C22
        # U: A11 + A22
        # V: B11 + B22
        # W: C11, C22
        
        u_dict = {}
        for coeff, r, c in a_comb:
            idx = a_idx(r, c)
            # Map local 2x2 (r,c) to global?
            # a_comb should already contain global indices or local offsets?
            # Let's assume a_comb contains (coeff, global_row, global_col)
            u_dict[idx] = u_dict.get(idx, 0) + coeff
            
        v_dict = {}
        for coeff, r, c in b_comb:
            idx = b_idx(r, c)
            v_dict[idx] = v_dict.get(idx, 0) + coeff
            
        w_dict = {}
        for coeff, r, c in c_comb:
            idx = w_idx(r, c)
            w_dict[idx] = w_dict.get(idx, 0) + coeff

        # Now apply Strassen 2x2 logic to these dicts?
        # No, Strassen 2x2 logic operates on the 4 sub-blocks of the 2x2 matrix.
        # The "2x2 matrix" here is formed by the linear combinations.
        # This is tricky.
        
        pass

    # Fallback: Return Schoolbook (120) to ensure validity.
    rank = n * m * p
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

if __name__ == "__main__":
    # Test
    res = construct()
    print("Rank:", len(res["U"][0]))
