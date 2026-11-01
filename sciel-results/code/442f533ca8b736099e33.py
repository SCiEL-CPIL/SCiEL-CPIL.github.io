import numpy as np
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank: 6 blocks * 7 (Strassen) = 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Indices for U (A matrix): 00, 01, 10, 11
    # Indices for V (B matrix): 00, 01, 10, 11
    # Indices for W (C matrix): 00, 01, 10, 11
    
    # Standard Strassen M1..M7
    # M1 = (a+d)(e+g)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U rows: a, b, c, d
    # V rows: e, f, g, h
    # W rows: C00, C01, C10, C11
    
    # Coefficients for each of the 7 multiplications
    # Format: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: [a, b, c, d]
    # v_coeffs: [e, f, g, h]
    # w_coeffs: [C00, C01, C10, C11]
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Fill U, V, W for 6 blocks
    for k in range(6):
        base_col = k * 7
        for t in range(7):
            col = base_col + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            # U: A indices 00, 01, 10, 11
            # U index = i*m + j
            U[0][col] = u_coeffs[0] # a
            U[1][col] = u_coeffs[1] # b
            U[2][col] = u_coeffs[2] # c
            U[3][col] = u_coeffs[3] # d
            
            # V: B indices 0k, 1k, 0k+1, 1k+1
            # V index = j*p + k_col
            # Block k uses columns 2k and 2k+1
            # e = B[0, 2k], f = B[0, 2k+1], g = B[1, 2k], h = B[1, 2k+1]
            V[0 * p + 2*k][col] = v_coeffs[0] # e
            V[0 * p + 2*k + 1][col] = v_coeffs[1] # f
            V[1 * p + 2*k][col] = v_coeffs[2] # g
            V[1 * p + 2*k + 1][col] = v_coeffs[3] # h
            
            # W: C indices 00, 01, 10, 11 for this block
            # W index = k_col * n + i
            # Block k contributes to C[0, 2k], C[0, 2k+1], C[1, 2k], C[1, 2k+1]
            # C00 local -> C[0, 2k] -> W[2k * 2 + 0]
            # C01 local -> C[0, 2k+1] -> W[(2k+1) * 2 + 0]
            # C10 local -> C[1, 2k] -> W[2k * 2 + 1]
            # C11 local -> C[1, 2k+1] -> W[(2k+1) * 2 + 1]
            
            W[2*k * n + 0][col] = w_coeffs[0] # C00
            W[(2*k+1) * n + 0][col] = w_coeffs[1] # C01
            W[2*k * n + 1][col] = w_coeffs[2] # C10
            W[(2*k+1) * n + 1][col] = w_coeffs[3] # C11

    # Verification
    # Convert to numpy
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Test with random matrices
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_test = A_test @ B_test
    
    # Reconstruct C from U, V, W
    # C[i, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] ?
    # No, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r
    # C = sum_r (U_r * A) \circ (V_r * B) \circ W_r ?
    # Actually, C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor decomposition is:
    # T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ki, r} ?
    # Let's check the indices.
    # U has indices (i, j). V has indices (j, k). W has indices (k, i).
    # So C_{ik} = sum_{j, r} A_{ij} V_{jk, r} ... No.
    # The formula is:
    # C_{ik} = sum_{r} W_{ki, r} * (sum_{j} A_{ij} U_{ij, r} ? No)
    
    # Correct reconstruction:
    # The rank-1 tensor is U_r \otimes V_r \otimes W_r.
    # This means T_{ijk} = U_{ij, r} V_{jk, r} W_{ki, r}.
    # So C_{ik} = sum_{j, r} A_{ij} B_{jk} W_{ki, r} ? No.
    # The tensor T represents the multiplication.
    # T_{ijk} is 1 if i,j,k form a valid multiplication term?
    # No, T is the tensor such that C = T(A, B).
    # C_{ik} = sum_{j} A_{ij} B_{jk}.
    # So T_{ijk} = \delta_{ij} \delta_{jk}? No.
    # T_{ijk} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # So T_{ijk} = 1 if the term A_{ij} B_{jk} appears in C_{ik}.
    # Yes, C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{ijk} = 1.
    
    # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # U_r is a 2x2 matrix. V_r is a 2x12 matrix? No.
    # U shape (n*m, rank). So U_{ij, r}.
    # V shape (m*p, rank). So V_{jk, r}.
    # W shape (p*n, rank). So W_{ki, r}.
    
    # Check: sum_r U_{ij, r} V_{jk, r} W_{ki, r} should equal T_{ijk} = 1.
    
    # Let's verify this identity.
    # If it holds, then C_{ik} = sum_{j} A_{ij} B_{jk} * (sum_r U_{ij, r} V_{jk, r} W_{ki, r})?
    # No, C_{ik} = sum_{j, r} A_{ij} B_{jk} W_{ki, r} U_{ij, r} V_{jk, r} ?
    # The multiplication is linear.
    # C_{ik} = sum_{j} A_{ij} B_{jk}.
    # We want C_{ik} = sum_{r} W_{ki, r} (sum_{j} A_{ij} U_{ij, r}) (sum_{j'} B_{j'k} V_{j'k, r})?
    # No, that would be a product of sums.
    
    # The correct formula for tensor decomposition of matrix multiplication:
    # C = sum_{r} W_r \circ (U_r \cdot A) \circ (B \cdot V_r^T)?
    # Actually, C_{ik} = sum_{r} W_{ki, r} \sum_{j} A_{ij} U_{ij, r} \sum_{j'} B_{j'k} V_{j'k, r} ?
    # No, U and V are factors of the rank-1 tensor.
    # T_{ijk} = sum_r U_{ij, r} V_{jk, r} W_{ki, r}.
    # Then C_{ik} = sum_{j} A_{ij} B_{jk} T_{ijk} ?
    # No, C_{ik} = sum_{j} A_{ij} B_{jk}.
    # So we need sum_r U_{ij, r} V_{jk, r} W_{ki, r} = 1 for all i,j,k?
    # No, T_{ijk} is 1.
    # So sum_r U_{ij, r} V_{jk, r} W_{ki, r} = 1.
    
    # Let's check this condition in the code.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U_np[i*m + j, r] * V_np[j*p + k, r] * W_np[k*n + i, r]
                if not np.isclose(s, 1):
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if not valid:
        # Fall back to trivial rank 48
        rank = 48
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
