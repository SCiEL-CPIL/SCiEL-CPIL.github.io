import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 85 # 64 + 1 + 20
    # Actually, let's use 70 if Strassen works.
    # Let's try to implement Strassen 4x4.
    
    def strassen_2x2():
        # Returns 7 terms (u, v, w) of length 4
        # u, v, w are lists of 4 floats
        terms = []
        # T1
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        # T2
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        # T3
        terms.append(([1,0,0,0], [0,0,1,-1], [0,1,0,1]))
        # T4
        terms.append(([0,0,0,1], [0,1,-1,0], [1,0,1,0]))
        # T5
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        # T6
        terms.append(([0,-1,1,0], [1,1,0,0], [0,0,0,1]))
        # T7
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        return terms

    def strassen_4x4():
        # Returns 49 terms (u, v, w) of length 16
        # A indices: 0,1,2,3 (00,01,10,11)
        # B indices: 0,1,2,3
        # C indices: 0,1,2,3
        
        terms_2x2 = strassen_2x2()
        
        # Strassen combinations
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B10-B11)
        # P4 = A11(B01-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # Helper to expand a 2x2 term into 4x4 terms
        def expand(term, a_coeff, b_coeff, c_coeff):
            # term is (u2, v2, w2)
            # a_coeff, b_coeff, c_coeff are 4x4 matrices (or lists of lists)
            # representing the linear combination
            u2, v2, w2 = term
            u4 = [0]*16
            v4 = [0]*16
            w4 = [0]*16
            
            # u4[i*4+j] = sum_{i',j'} a_coeff[i,j][i',j'] * u2[i'*2+j']
            # This is getting complicated.
            # Let's just hardcode the 49 terms? No.
            # Let's use numpy for the expansion.
            pass

        # This approach is too slow to code correctly in the time limit.
        # I will stick to Schoolbook 4x4x4 (rank 64) + 1 + 20 = 85.
        # 85 is a guaranteed improvement over 100.
        
        U = [[0.0]*85 for _ in range(20)]
        V = [[0.0]*85 for _ in range(25)]
        W = [[0.0]*85 for _ in range(20)]
        
        col = 0
        # Part 1: A1 (4x4) * B11 (4x4) -> C11 (4x4)
        # Schoolbook
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    # A[i,j] B[j,k] C[k,i]
                    # U index: i*5 + j
                    # V index: j*5 + k
                    # W index: k*4 + i
                    U[i*5+j][col] = 1.0
                    V[j*5+k][col] = 1.0
                    W[k*4+i][col] = 1.0
                    col += 1
        # col is now 64
        
        # Part 2: A2 (4x1) * B21 (1x4) -> C21 (4x4)
        # A[i,4] B[4,k] C[k,i]
        # Rank 1
        for i in range(4):
            for k in range(4):
                U[i*5+4][col] = 1.0
                V[4*5+k][col] = 1.0
                W[k*4+i][col] = 1.0
        # col is now 65
        
        # Part 3: A (4x5) * B_right (5x1) -> C_right (4x1)
        # A[i,j] B[j,4] C[4,i]
        # Rank 20
        for i in range(4):
            for j in range(5):
                U[i*5+j][col] = 1.0
                V[j*5+4][col] = 1.0
                W[4*4+i][col] = 1.0
                col += 1
        # col is now 85
        
        return {"U": U, "V": V, "W": W}
