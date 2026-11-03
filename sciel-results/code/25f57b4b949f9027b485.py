"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 12
    V_size = m * p  # 28
    W_size = p * n  # 21
    
    # Accumulate rank-1 terms as lists of (U_vec, V_vec, W_vec)
    terms = []
    
    def add_term(U_vec, V_vec, W_vec):
        """Add a rank-1 term. Vectors are dicts mapping index->value."""
        # Convert to sparse representation
        terms.append((U_vec, V_vec, W_vec))
    
    # Helper to create a vector with specific non-zero entries
    def make_vec(size, entries):
        """entries is dict {index: value}"""
        return entries
    
    # Block index mappings
    # A blocks: (i_range, j_range) -> U indices
    # A11: i in {0,1}, j in {0,1}
    # A12: i in {0,1}, j in {2,3}
    # A21: i in {2}, j in {0,1}
    # A22: i in {2}, j in {2,3}
    
    # B blocks: (j_range, k_range) -> V indices
    # B11: j in {0,1}, k in {0,1}
    # B12: j in {0,1}, k in {2,3}
    # B13: j in {0,1}, k in {4,5}
    # B14: j in {0,1}, k in {6}
    # B21: j in {2,3}, k in {0,1}
    # B22: j in {2,3}, k in {2,3}
    # B23: j in {2,3}, k in {4,5}
    # B24: j in {2,3}, k in {6}
    
    # C blocks: (i_range, k_range) -> W indices (W[k*n+i])
    # C11: i in {0,1}, k in {0,1}
    # C12: i in {0,1}, k in {2,3}
    # C13: i in {0,1}, k in {4,5}
    # C14: i in {0,1}, k in {6}
    # C21: i in {2}, k in {0,1}
    # C22: i in {2}, k in {2,3}
    # C23: i in {2}, k in {4,5}
    # C24: i in {2}, k in {6}
    
    # For a 2x2x2 Strassen multiplication of A_block (2x2) * B_block (2x2) -> C_block (2x2)
    # A_block indices: a00, a01, a10, a11 (row, col within block)
    # B_block indices: b00, b01, b10, b11
    # C_block indices: c00, c01, c10, c11
    #
    # Strassen 7 terms:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    def strassen_222(a_indices, b_indices, c_indices):
        """
        a_indices: [a00, a01, a10, a11] - U indices for the 2x2 A block
        b_indices: [b00, b01, b10, b11] - V indices for the 2x2 B block
        c_indices: [c00, c01, c10, c11] - W indices for the 2x2 C block
        """
        a00, a01, a10, a11 = a_indices
        b00, b01, b10, b11 = b_indices
        c00, c01, c10, c11 = c_indices
        
        # M1 = (a00+a11)(b00+b11)
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1})
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c11: 1})
        # Wait, this is wrong. Each Mi is ONE rank-1 term that contributes to multiple C entries.
        # I need to think about this differently.
        
        # Actually, each Mi is a product of a linear combination of A entries and a linear combination of B entries.
        # This single product contributes to one or more C entries.
        # In the tensor decomposition, each rank-1 term is (U_vec, V_vec, W_vec) where
        # the tensor element T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
        # 
        # For a Strassen term Mi = (sum_a alpha_a * A[a]) * (sum_b beta_b * B[b]),
        # this contributes to C entries. If Mi contributes to C[c] with coefficient gamma_c,
        # then we need: for each c, U_vec has the alpha coefficients on A indices,
        # V_vec has the beta coefficients on B indices, and W_vec has gamma_c on C index c.
        # But wait, one rank-1 term can only have one W entry per C position... 
        # Actually no, W_vec can have multiple non-zero entries.
        # 
        # So for M1 = (a00+a11)(b00+b11), which contributes to c00 with +1 and c11 with +1:
        # U_vec = {a00: 1, a11: 1}
        # V_vec = {b00: 1, b11: 1}
        # W_vec = {c00: 1, c11: 1}
        # This is ONE rank-1 term!
        
        # Let me redo this properly.
        pass
    
    # Clear terms and restart
    terms.clear()
    
    def strassen_222(a_indices, b_indices, c_indices):
        """
        a_indices: [a00, a01, a10, a11] - U indices for the 2x2 A block (row-major within block)
        b_indices: [b00, b01, b10, b11] - V indices for the 2x2 B block
        c_indices: [c00, c01, c10, c11] - W indices for the 2x2 C block
        """
        a00, a01, a10, a11 = a_indices
        b00, b01, b10, b11 = b_indices
        c00, c01, c10, c11 = c_indices
        
        # M1 = (a00+a11)(b00+b11) -> contributes to c00 (+1), c11 (+1)
        add_term({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1})
        
        # M2 = (a10+a11)(b00) -> contributes to c10 (+1), c11 (-1)
        add_term({a10: 1, a11: 1}, {b00: 1}, {c10: 1, c11: -1})
        
        # M3 = (a00)(b01-b11) -> contributes to c01 (+1), c11 (+1)
        add_term({a00: 1}, {b01: 1, b11: -1}, {c01: 1, c11: 1})
        
        # M4 = (a11)(b10-b00) -> contributes to c00 (+1), c10 (+1)
        add_term({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1})
        
        # M5 = (a00+a01)(b11) -> contributes to c00 (-1), c01 (+1)
        add_term({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1})
        
        # M6 = (a10-a00)(b00+b01) -> contributes to c11 (+1)
        add_term({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1})
        
        # M7 = (a01-a11)(b10+b11) -> contributes to c00 (+1)
        add_term({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1})
    
    # Schoolbook for 2x2x1: A (2x2) * B (2x1) -> C (2x1)
    # a00*b00 -> c00, a01*b01 -> c00
    # a10*b00 -> c10, a11*b01 -> c10
    def schoolbook_221(a_indices, b_indices, c_indices):
        """
        a_indices: [a00, a01, a10, a11] - U indices for 2x2 A block
        b_indices: [b00, b01] - V indices for 2x1 B block
        c_indices: [c00, c10] - W indices for 2x1 C block
        """
        a00, a01, a10, a11 = a_indices
        b00, b01 = b_indices
        c00, c10 = c_indices
        
        add_term({a00: 1}, {b00: 1}, {c00: 1})
        add_term({a01: 1}, {b01: 1}, {c00: 1})
        add_term({a10: 1}, {b00: 1}, {c10: 1})
        add_term({a11: 1}, {b01: 1}, {c10: 1})
    
    # Schoolbook for 1x2x2: A (1x2) * B (2x2) -> C (1x2)
    # a0*b00 -> c0, a1*b10 -> c0
    # a0*b01 -> c1, a1*b11 -> c1
    def schoolbook_122(a_indices, b_indices, c_indices):
        """
        a_indices: [a0, a1] - U indices for 1x2 A block
        b_indices: [b00, b01, b10, b11] - V indices for 2x2 B block
        c_indices: [c0, c1] - W indices for 1x2 C block
        """
        a0, a1 = a_indices
        b00, b01, b10, b11 = b_indices
        c0, c1 = c_indices
        
        add_term({a0: 1}, {b00: 1}, {c0: 1})
        add_term({a1: 1}, {b10: 1}, {c0: 1})
        add_term({a0: 1}, {b01: 1}, {c1: 1})
        add_term({a1: 1}, {b11: 1}, {c1: 1})
    
    # Schoolbook for 1x2x1: A (1x2) * B (2x1) -> C (1x1)
    # a0*b0 + a1*b1 -> c
    def schoolbook_121(a_indices, b_indices, c_index):
        """
        a_indices: [a0, a1] - U indices for 1x2 A block
        b_indices: [b0, b1] - V indices for 2x1 B block
        c_index: W index for 1x1 C block
        """
        a0, a1 = a_indices
        b0, b1 = b_indices
        
        add_term({a0: 1}, {b0: 1}, {c_index: 1})
        add_term({a1: 1}, {b1: 1}, {c_index: 1})
    
    # Now define the block index mappings
    # A11: i in {0,1}, j in {0,1} -> U indices: i*4+j
    # a00=(0,0)=0, a01=(0,1)=1, a10=(1,0)=4, a11=(1,1)=5
    A11 = [0, 1, 4, 5]
    # A12: i in {0,1}, j in {2,3} -> U indices: i*4+j
    # a00=(0,2)=2, a01=(0,3)=3, a10=(1,2)=6, a11=(1,3)=7
    A12 = [2, 3, 6, 7]
    # A21: i in {2}, j in {0,1} -> U indices: 2*4+j
    # a0=(2,0)=8, a1=(2,1)=9
    A21 = [8, 9]
    # A22: i in {2}, j in {2,3} -> U indices: 2*4+j
    # a0=(2,2)=10, a1=(2,3)=11
    A22 = [10, 11]
    
    # B blocks: V index = j*7+k
    # B11: j in {0,1}, k in {0,1}
    # b00=(0,0)=0, b01=(0,1)=1, b10=(1,0)=7, b11=(1,1)=8
    B11 = [0, 1, 7, 8]
    # B12: j in {0,1}, k in {2,3}
    # b00=(0,2)=2, b01=(0,3)=3, b10=(1,2)=9, b11=(1,3)=10
    B12 = [2, 3, 9, 10]
    # B13: j in {0,1}, k in {4,5}
    # b00=(0,4)=4, b01=(0,5)=5, b10=(1,4)=11, b11=(1,5)=12
    B13 = [4, 5, 11, 12]
    # B14: j in {0,1}, k in {6}
    # b0=(0,6)=6, b1=(1,6)=13
    B14 = [6, 13]
    # B21: j in {2,3}, k in {0,1}
    # b00=(2,0)=14, b01=(2,1)=15, b10=(3,0)=21, b11=(3,1)=22
    B21 = [14, 15, 21, 22]
    # B22: j in {2,3}, k in {2,3}
    # b00=(2,2)=16, b01=(2,3)=17, b10=(3,2)=23, b11=(3,3)=24
    B22 = [16, 17, 23, 24]
    # B23: j in {2,3}, k in {4,5}
    # b00=(2,4)=18, b01=(2,5)=19, b10=(3,4)=25, b11=(3,5)=26
    B23 = [18, 19, 25, 26]
    # B24: j in {2,3}, k in {6}
    # b0=(2,6)=20, b1=(3,6)=27
    B24 = [20, 27]
    
    # C blocks: W index = k*3+i
    # C11: i in {0,1}, k in {0,1}
    # c00=(k=0,i=0)=0, c01=(k=1,i=0)=3, c10=(k=0,i=1)=1, c11=(k=1,i=1)=4
    # Wait, I need to be careful. C11 is 2x2 with rows i=0,1 and cols k=0,1
    # c00 = C[0,0] -> W[0*3+0] = W[0]
    # c01 = C[0,1] -> W[1*3+0] = W[3]
    # c10 = C[1,0] -> W[0*3+1] = W[1]
    # c11 = C[1,1] -> W[1*3+1] = W[4]
    C11 = [0, 3, 1, 4]  # [c00, c01, c10, c11]
    
    # C12: i in {0,1}, k in {2,3}
    # c00 = C[0,2] -> W[2*3+0] = W[6]
    # c01 = C[0,3] -> W[3*3+0] = W[9]
    # c10 = C[1,2] -> W[2*3+1] = W[7]
    # c11 = C[1,3] -> W[3*3+1] = W[10]
    C12 = [6, 9, 7, 10]
    
    # C13: i in {0,1}, k in {4,5}
    # c00 = C[0,4] -> W[4*3+0] = W[12]
    # c01 = C[0,5] -> W[5*3+0] = W[15]
    # c10 = C[1,4] -> W[4*3+1] = W[13]
    # c11 = C[1,5] -> W[5*3+1] = W[16]
    C13 = [12, 15, 13, 16]
    
    # C14: i in {0,1}, k in {6}
    # c0 = C[0,6] -> W[6*3+0] = W[18]
    # c1 = C[1,6] -> W[6*3+1] = W[19]
    C14 = [18, 19]
    
    # C21: i in {2}, k in {0,1}
    # c0 = C[2,0] -> W[0*3+2] = W[2]
    # c1 = C[2,1] -> W[1*3+2] = W[5]
    C21 = [2, 5]
    
    # C22: i in {2}, k in {2,3}
    # c0 = C[2,2] -> W[2*3+2] = W[8]
    # c1 = C[2,3] -> W[3*3+2] = W[11]
    C22 = [8, 11]
    
    # C23: i in {2}, k in {4,5}
    # c0 = C[2,4] -> W[4*3+2] = W[14]
    # c1 = C[2,5] -> W[5*3+2] = W[17]
    C23 = [14, 17]
    
    # C24: i in {2}, k in {6}
    # c = C[2,6] -> W[6*3+2] = W[20]
    C24 = 20
    
    # Now add all the block multiplications
    # C11 = A11*B11 + A12*B21 (both 2x2x2, use Strassen)
    strassen_222(A11, B11, C11)
    strassen_222(A12, B21, C11)
    
    # C12 = A11*B12 + A12*B22
    strassen_222(A11, B12, C12)
    strassen_222(A12, B22, C12)
    
    # C13 = A11*B13 + A12*B23
    strassen_222(A11, B13, C13)
    strassen_222(A12, B23, C13)
    
    # C14 = A11*B14 + A12*B24 (2x2x1, use schoolbook)
    schoolbook_221(A11, B14, C14)
    schoolbook_221(A12, B24, C14)
    
    # C21 = A21*B11 + A22*B21 (1x2x2, use schoolbook)
    schoolbook_122(A21, B11, C21)
    schoolbook_122(A22, B21, C21)
    
    # C22 = A21*B12 + A22*B22
    schoolbook_122(A21, B12, C22)
    schoolbook_122(A22, B22, C22)
    
    # C23 = A21*B13 + A22*B23
    schoolbook_122(A21, B13, C23)
    schoolbook_122(A22, B23, C23)
    
    # C24 = A21*B14 + A22*B24 (1x2x1, use schoolbook)
    schoolbook_121(A21, B14, C24)
    schoolbook_121(A22, B24, C24)
    
    # Now convert terms to U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(U_size)]
    V = [[0] * rank for _ in range(V_size)]
    W = [[0] * rank for _ in range(W_size)]
    
    for c, (U_dict, V_dict, W_dict) in enumerate(terms):
        for idx, val in U_dict.items():
            U[idx][c] = val
        for idx, val in V_dict.items():
            V[idx][c] = val
        for idx, val in W_dict.items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
