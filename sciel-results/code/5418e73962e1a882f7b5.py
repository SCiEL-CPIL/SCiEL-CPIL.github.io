# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2 blocks
    def add_strassen_block(rA, cA, rB, cB, cC, col_offset):
        # rA, rB, cB, cC are block indices
        # A block is at rows 2*rA..2*rA+1, cols 2*cA..2*cA+1
        # B block is at rows 2*rB..2*rB+1, cols 2*cB..2*cB+1
        # C block is at rows 2*rA..2*rA+1, cols 2*cC..2*cC+1
        
        # Local indices i, j, k in {0, 1}
        # Global A index: (2*rA + i) * 4 + (2*cA + j)
        # Global B index: (2*rB + j) * 6 + (2*cB + k)
        # Global C index: (2*cC + k) * 4 + (2*rA + i)
        
        # Strassen components
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # We iterate through the 7 multiplications
        # For each multiplication s, we have coefficients for A, B, C
        # A coeffs: vector of 4 values (for X00, X01, X10, X11)
        # B coeffs: vector of 4 values (for Y00, Y01, Y10, Y11)
        # C coeffs: vector of 4 values (for Z00, Z01, Z10, Z11)
        
        # Note: The tensor decomposition is sum_r U_r V_r W_r
        # Here, for a specific Strassen step 's', the contribution to C is 
        # (sum_a a_a A_a) * (sum_b b_b B_b) * c_c
        # This is not a single rank-1 tensor in the global sense unless we expand the products.
        # Wait. The Strassen decomposition expresses the *product* as a sum of 7 scalar products of linear forms.
        # Specifically, Z_{ik} = sum_s (sum_j A_{ij} u_{s,ij}) (sum_k B_{jk} v_{s,jk}) w_{s,ik} ?
        # No, Strassen says:
        # Z_{ik} = sum_{s=1}^7 ( sum_{j} A_{ij} alpha_{s,ij} ) * ( sum_{l} B_{jl} beta_{s,jl} ) * gamma_{s,ik} ??
        # Actually, the standard tensor decomposition for 2x2x2 is:
        # T_{i,j,k} = sum_{s=1}^7 U_{i,j,s} V_{j,k,s} W_{k,i,s}
        # Let's use the explicit rank-1 tensors derived from Strassen.
        
        # The 7 rank-1 tensors for 2x2x2 are:
        # 1. (A00+A11) * (B00+B11) -> contributes to C00, C11
        #    This term is (A00+B11 term) * (B00+B11 term).
        #    It contributes to C00 with +1, C11 with +1.
        #    Wait, C00 = M1 + ... so M1 contributes +1 to C00.
        #    C11 = M1 - ... so M1 contributes +1 to C11.
        #    So this rank-1 tensor has W components: W[C00]=1, W[C11]=1.
        
        # Let's list the 7 tensors (U, V, W) for local 2x2x2:
        # Indices: A(i,j), B(j,k), C(i,k)
        
        # Tensor 1:
        # U: A00=1, A11=1. (Others 0)
        # V: B00=1, B11=1. (Others 0)
        # W: C00=1, C11=1. (Others 0)
        # Check: C00 += A00*B00 + A00*B11 + A11*B00 + A11*B11.
        # C11 += A00*B00 + A00*B11 + A11*B00 + A11*B11.
        # In Strassen: M1 = (A00+A11)(B00+B11).
        # C00 = M1 + M4 - M5 + M7.
        # C11 = M1 - M2 + M3 + M6.
        
        # Tensor 2:
        # M2 = (A10+A11)B00
        # U: A10=1, A11=1.
        # V: B00=1.
        # W: C10=1 (from C10=M2+M4), C11=-1 (from C11=M1-M2+...).
        
        # Tensor 3:
        # M3 = A00(B01-B11)
        # U: A00=1.
        # V: B01=1, B11=-1.
        # W: C01=1 (from C01=M3+M5), C11=1 (from C11=...+M3+...).
        
        # Tensor 4:
        # M4 = A11(B10-B00)
        # U: A11=1.
        # V: B10=1, B00=-1.
        # W: C00=1 (from C00=...+M4+...), C10=1 (from C10=M2+M4).
        
        # Tensor 5:
        # M5 = (A00+A01)B11
        # U: A00=1, A01=1.
        # V: B11=1.
        # W: C00=-1 (from C00=...-M5+...), C01=1 (from C01=M3+M5).
        
        # Tensor 6:
        # M6 = (A10-A00)(B00+B01)
        # U: A10=1, A00=-1.
        # V: B00=1, B01=1.
        # W: C11=1 (from C11=...+M6).
        
        # Tensor 7:
        # M7 = (A01-A11)(B10+B11)
        # U: A01=1, A11=-1.
        # V: B10=1, B11=1.
        # W: C00=1 (from C00=...+M7).
        
        # Let's verify C00:
        # T1: +1
        # T4: +1
        # T5: -1
        # T7: +1
        # Sum: 1+1-1+1 = 2? No.
        # Strassen: C00 = M1 + M4 - M5 + M7.
        # My W coefficients:
        # T1 (M1): C00=1
        # T4 (M4): C00=1
        # T5 (M5): C00=-1
        # T7 (M7): C00=1
        # Sum = 2. This is wrong. The coefficient should be 1 for the final sum?
        # No, the decomposition is $C = \sum T_s$.
        # If $T_s$ contributes $M_s$ to $C_{ik}$, then the coefficient in $W$ is 1 for the term $M_s$ in the formula for $C_{ik}$.
        # But $M_s$ is a scalar product.
        # So yes, if $C_{00} = 1 \cdot M_1 + 1 \cdot M_4 - 1 \cdot M_5 + 1 \cdot M_7$, then the W coefficient for $C_{00}$ in tensor 1 is 1, in tensor 4 is 1, in tensor 5 is -1, in tensor 7 is 1.
        # The sum of these contributions reconstructs $C_{00}$.
        
        # Let's define the 7 tensors explicitly.
        # U, V, W are dicts or arrays of size 4.
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        tensors = [
            # M1
            {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
            # M2
            {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 0, 1, -1]},
            # M3
            {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 1, 0, 1]},
            # M4
            {'U': [0, 0, 0, 1], 'V': [0, 0, 1, -1], 'W': [1, 0, 1, 0]},
            # M5
            {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 1, 0, 0]},
            # M6
            {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
            # M7
            {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
        ]
        
        for t in tensors:
            # Map local indices to global
            for li in range(2):
                for lj in range(2):
                    u_val = t['U'][li*2 + lj]
                    if u_val != 0:
                        gi = 2*rA + li
                        gj = 2*cA + lj
                        U[gi*m + gj, col_offset] += u_val
                        
            for lj in range(2):
                for lk in range(2):
                    v_val = t['V'][lj*2 + lk]
                    if v_val != 0:
                        gj = 2*rB + lj
                        gk = 2*cB + lk
                        V[gj*p + gk, col_offset] += v_val
                        
            for li in range(2):
                for lk in range(2):
                    w_val = t['W'][li*2 + lk]
                    if w_val != 0:
                        gi = 2*rA + li
                        gk = 2*cC + lk
                        W[gk*n + gi, col_offset] += w_val
            
            col_offset += 1
            
        return col_offset

    # Iterate over blocks
    # rA in 0..1, cB in 0..2
    for rA in range(2):
        for cB in range(3):
            # Term 1: A[rA, 0] * B[0, cB]
            col = add_strassen_block(rA, 0, 0, cB, cB, col)
            # Term 2: A[rA, 1] * B[1, cB]
            col = add_strassen_block(rA, 1, 1, cB, cB, col)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
