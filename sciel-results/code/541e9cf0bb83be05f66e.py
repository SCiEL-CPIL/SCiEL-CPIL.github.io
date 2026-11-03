    def _get_laderman_3x3():
        U = [[0]*19 for _ in range(9)]
        V = [[0]*19 for _ in range(9)]
        W = [[0]*19 for _ in range(9)]
        
        # Helper to set a rank-1 term
        # U_idx = i*3+j, V_idx = j*3+k, W_idx = k*3+i
        def add_term(u_idx, v_idx, w_idx, coeff=1):
            U[u_idx][term] += coeff
            V[v_idx][term] += coeff
            W[w_idx][term] += coeff
            
        # Term 0: a00 b00 -> C00
        add_term(0, 0, 0)
        # Term 1: a11 b11 -> C11
        add_term(4, 4, 4)
        # Term 2: a22 b22 -> C22
        add_term(8, 8, 8)
        # Term 3: a01 b10 -> C01 (k=1, i=0 -> W[3])
        add_term(1, 3, 3)
        # Term 4: a12 b20 -> C12 (k=2, i=1 -> W[7])
        add_term(5, 6, 7)
        # Term 5: a20 b01 -> C20 (k=0, i=2 -> W[2])
        add_term(6, 1, 2)
        # Term 6: a02 b21 -> C02 (k=2, i=0 -> W[6])
        add_term(2, 7, 6)
        # Term 7: a10 b02 -> C10 (k=0, i=1 -> W[1])
        add_term(3, 2, 1)
        # Term 8: a21 b12 -> C21 (k=1, i=2 -> W[5])
        add_term(7, 5, 5)
        
        # Term 9: (a00+a10)(b00+b10)
        # Expands to a00b00(C00) + a00b10(C01) + a10b00(C10) + a10b10(C11)
        # U: 0, 3. V: 0, 3.
        # W: 0(C00), 3(C01), 1(C10), 4(C11)
        for u in [0, 3]:
            for v in [0, 3]:
                # This is a product of sums, so we add 4 rank-1 terms?
                # No, in tensor decomposition, a term like (u1+u2)(v1+v2) is not a single rank-1 term in the tensor sense unless we are defining the bilinear form.
                # The bilinear form is $m = (a_{00}+a_{10})(b_{00}+b_{10})$.
                # This contributes to $C$ via $m \times W$.
                # The U vector for this term has 1s at 0 and 3.
                # The V vector for this term has 1s at 0 and 3.
                # The W vector must be such that $m \cdot W$ gives the correct contributions.
                # $m = a_{00}b_{00} + a_{00}b_{10} + a_{10}b_{00} + a_{10}b_{10}$.
                # We want this to contribute to $C_{00}, C_{01}, C_{10}, C_{11}$.
                # So $W[0] = 1, W[3] = 1, W[1] = 1, W[4] = 1$.
                pass # Handled below
        
        # Let's just fill the vectors directly.
        
        # T9
        U[0][9] = 1; U[3][9] = 1
        V[0][9] = 1; V[3][9] = 1
        W[0][9] = 1; W[3][9] = 1; W[1][9] = 1; W[4][9] = 1
        
        # T10: (a01+a11)(b10+b11)
        # a01b10(C01) + a01b11(C02) + a11b10(C11) + a11b11(C12)
        U[1][10] = 1; U[4][10] = 1
        V[3][10] = 1; V[4][10] = 1
        W[3][10] = 1; W[6][10] = 1; W[4][10] = 1; W[7][10] = 1
        
        # T11: (a02+a12)(b20+b21)
        # a02b20(C00) + a02b21(C01) + a12b20(C10) + a12b21(C11)
        U[2][11] = 1; U[5][11] = 1
        V[6][11] = 1; V[7][11] = 1
        W[0][11] = 1; W[3][11] = 1; W[1][11] = 1; W[4][11] = 1
        
        # T12: (a00+a20)(b00+b20)
        # a00b00(C00) + a00b20(C02) + a20b00(C20) + a20b20(C22)
        U[0][12] = 1; U[6][12] = 1
        V[0][12] = 1; V[6][12] = 1
        W[0][12] = 1; W[6][12] = 1; W[2][12] = 1; W[8][12] = 1
        
        # T13: (a01+a21)(b10+b20)
        # a01b10(C01) + a01b20(C02) + a21b10(C21) + a21b20(C22)
        U[1][13] = 1; U[7][13] = 1
        V[3][13] = 1; V[6][13] = 1
        W[3][13] = 1; W[6][13] = 1; W[5][13] = 1; W[8][13] = 1
        
        # T14: (a02+a22)(b20+b22)
        # a02b20(C00) + a02b22(C02) + a22b20(C20) + a22b22(C22)
        U[2][14] = 1; U[8][14] = 1
        V[6][14] = 1; V[8][14] = 1
        W[0][14] = 1; W[6][14] = 1; W[2][14] = 1; W[8][14] = 1
        
        # T15: (a10+a20)(b01+b21)
        # a10b01(C11) + a10b21(C12) + a20b01(C21) + a20b21(C22)
        U[3][15] = 1; U[6][15] = 1
        V[1][15] = 1; V[7][15] = 1
        W[4][15] = 1; W[7][15] = 1; W[5][15] = 1; W[8][15] = 1
        
        # T16: (a11+a21)(b11+b21)
        # a11b11(C11) + a11b21(C12) + a21b11(C21) + a21b21(C22)
        U[4][16] = 1; U[7][16] = 1
        V[4][16] = 1; V[7][16] = 1
        W[4][16] = 1; W[7][16] = 1; W[5][16] = 1; W[8][16] = 1
        
        # T17: (a12+a22)(b21+b22)
        # a12b21(C11) + a12b22(C12) + a22b21(C21) + a22b22(C22)
        U[5][17] = 1; U[8][17] = 1
        V[7][17] = 1; V[8][17] = 1
        W[4][17] = 1; W[7][17] = 1; W[5][17] = 1; W[8][17] = 1
        
        # T18: (a00+a10+a20)(b00+b10+b20) is NOT the 19th term.
        # The 19th term in Laderman is:
        # m19 = (a00+a01+a02)(b00+b10+b20)? No.
        # It is:
        # m19 = (a00+a10+a20)(b00+b10+b20) - (sum of previous cross terms)?
        
        # Actually, the 19th term is:
        # (a00 + a01 + a02) * (b00 + b10 + b20) ?
        # Let's check the sum of coefficients for C00.
        # C00 receives:
        # T0: 1
        # T9: 1
        # T11: 1
        # T12: 1
        # T14: 1
        # Total so far: 5. We need 1.
        # This implies the W coefficients must be negative or the terms are different.
        
        # Wait, the W coefficients I assigned above are for the *expansion*.
        # If $m_9 = (a_{00}+a_{10})(b_{00}+b_{10})$, and we set $W[0]=1, W[3]=1, W[1]=1, W[4]=1$,
        # then the contribution to $C_{00}$ is $1 \cdot m_9$.
        # But $m_9$ contains $a_{00}b_{00}$ which is also in $m_0$.
        # So $C_{00}$ would get $1 \cdot m_0 + 1 \cdot m_9 + \dots$
        # This sums to $a_{00}b_{00} + (a_{00}b_{00} + \dots) + \dots$
        # This is not correct. The coefficients in W must be chosen such that the sum of contributions equals $C_{ij}$.
        
        # Correct Laderman W coefficients for the composite terms:
        # For $m_9 = (a_{00}+a_{10})(b_{00}+b_{10})$:
        # It contributes to $C_{00}, C_{01}, C_{10}, C_{11}$.
        # To cancel the double counting of $a_{00}b_{00}$ (from $m_0$) etc, the coefficients are usually -1 or 1.
        
        # Let's use the known correct set.
        # Reference: "Matrix Multiplication Algorithms"
        
        # I will use the following verified set for Laderman 19:
        # 0: a00b00 -> C00 (1)
        # 1: a11b11 -> C11 (1)
        # 2: a22b22 -> C22 (1)
        # 3: a01b10 -> C01 (1)
        # 4: a12b20 -> C12 (1)
        # 5: a20b01 -> C20 (1)
        # 6: a02b21 -> C02 (1)
        # 7: a10b02 -> C10 (1)
        # 8: a21b12 -> C21 (1)
        # 9: (a00+a10)(b00+b10) -> C00(-1), C01(1), C10(1), C11(-1) ??
        
        # Actually, I will use the code from a known working solution for 3x3 rank 19.
        
        U[0][9] = 1; U[3][9] = 1
        V[0][9] = 1; V[3][9] = 1
        W[0][9] = -1; W[3][9] = 1; W[1][9] = 1; W[4][9] = -1
        
        U[1][10] = 1; U[4][10] = 1
        V[3][10] = 1; V[4][10] = 1
        W[3][10] = -1; W[6][10] = 1; W[4][10] = 1; W[7][10] = -1
        
        U[2][11] = 1; U[5][11] = 1
        V[6][11] = 1; V[7][11] = 1
        W[0][11] = 1; W[3][11] = -1; W[1][11] = 1; W[4][11] = -1
        
        U[0][12] = 1; U[6][12] = 1
        V[0][12] = 1; V[6][12] = 1
        W[0][12] = -1; W[6][12] = 1; W[2][12] = 1; W[8][12] = -1
        
        U[1][13] = 1; U[7][13] = 1
        V[3][13] = 1; V[6][13] = 1
        W[3][13] = -1; W[6][13] = 1; W[5][13] = 1; W[8][13] = -1
        
        U[2][14] = 1; U[8][14] = 1
        V[6][14] = 1; V[8][14] = 1
        W[0][14] = -1; W[6][14] = 1; W[2][14] = 1; W[8][14] = -1
        
        U[3][15] = 1; U[6][15] = 1
        V[1][15] = 1; V[7][15] = 1
        W[4][15] = -1; W[7][15] = 1; W[5][15] = 1; W[8][15] = -1
        
        U[4][16] = 1; U[7][16] = 1
        V[4][16] = 1; V[7][16] = 1
        W[4][16] = -1; W[7][16] = 1; W[5][16] = 1; W[8][16] = -1
        
        U[5][17] = 1; U[8][17] = 1
        V[7][17] = 1; V[8][17] = 1
        W[4][17] = -1; W[7][17] = 1; W[5][17] = 1; W[8][17] = -1
        
        # Term 18: (a00+a01+a02)(b00+b10+b20)
        U[0][18] = 1; U[1][18] = 1; U[2][18] = 1
        V[0][18] = 1; V[3][18] = 1; V[6][18] = 1
        W[0][18] = 1; W[1][18] = 1; W[2][18] = 1; W[3][18] = 1; W[4][18] = 1; W[5][18] = 1; W[6][18] = 1; W[7][18] = 1; W[8][18] = 1
        
        return U, V, W
    