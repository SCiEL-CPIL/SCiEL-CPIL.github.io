"""Program for matrix multiplication 3x3x4 tensor decomposition."""
DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 9 x 32
    V = [[0] * rank for _ in range(m * p)]  # 12 x 32
    W = [[0] * rank for _ in range(p * n)]  # 12 x 32
    
    col = 0
    
    def add_signed(u_dict, v_dict, w_dict):
        nonlocal col
        for i, val in u_dict.items():
            U[i][col] += val
        for i, val in v_dict.items():
            V[i][col] += val
        for i, val in w_dict.items():
            W[i][col] += val
        col += 1
    
    # Index conventions:
    # A[i,j] -> i*3+j: A00=0,A01=1,A02=2,A10=3,A11=4,A12=5,A20=6,A21=7,A22=8
    # B[j,k] -> j*4+k: B00=0,B01=1,B02=2,B03=3,B10=4,B11=5,B12=6,B13=7,B20=8,B21=9,B22=10,B23=11
    # C[i,j] -> j*3+i: C00=0,C01=1,C02=2,C03=3,C10=4,C11=5,C12=6,C13=7,C20=8,C21=9,C22=10,C23=11
    
    # Strassen 2x2x2: A[0:2,0:2]*B[0:2,0:2]->C[0:2,0:2]
    # C00=0,C01=1,C10=4,C11=5
    
    # P1=(A00+A11)(B00+B11): contributes +1 to C00, +1 to C11
    add_signed({0:1, 4:1}, {0:1, 5:1}, {0:1, 5:1})
    
    # P2=(A10+A11)B00: contributes +1 to C10, -1 to C11
    add_signed({3:1, 4:1}, {0:1}, {4:1, 5:-1})
    
    # P3=A00(B10-B00): contributes -1 to C00, +1 to C10
    add_signed({0:1}, {4:1, 0:-1}, {0:-1, 4:1})
    
    # P4=A11(B01-B10): contributes +1 to C01, +1 to C11
    add_signed({4:1}, {1:1, 4:-1}, {1:1, 5:1})
    
    # P5=(A00+A01)B11: contributes +1 to C01, -1 to C11
    add_signed({0:1, 1:1}, {5:1}, {1:1, 5:-1})
    
    # P6=(A10-A00)(B00+B01): contributes +1 to C00, -1 to C10
    add_signed({3:1, 0:-1}, {0:1, 1:1}, {0:1, 4:-1})
    
    # P7=(A01-A11)(B10+B11): contributes +1 to C01, -1 to C11
    add_signed({1:1, 4:-1}, {4:1, 5:1}, {1:1, 5:-1})
    
    # A[0:2,2]*B[2,0:2]->C[0:2,0:2]: 4 terms
    add_signed({2:1}, {8:1}, {0:1})
    add_signed({2:1}, {9:1}, {1:1})
    add_signed({5:1}, {8:1}, {4:1})
    add_signed({5:1}, {9:1}, {5:1})
    
    # A[0:2,0:2]*B[0:2,2:4]->C[0:2,2:4]: 4 terms
    add_signed({0:1}, {2:1}, {2:1})
    add_signed({0:1}, {3:1}, {3:1})
    add_signed({1:1}, {2:1}, {6:1})
    add_signed({1:1}, {3:1}, {7:1})
    
    # A[0:2,2]*B[2,2:4]->C[0:2,2:4]: 4 terms
    add_signed({2:1}, {10:1}, {2:1})
    add_signed({2:1}, {11:1}, {3:1})
    add_signed({5:1}, {10:1}, {6:1})
    add_signed({5:1}, {11:1}, {7:1})
    
    # A[2,0:2]*B[0:2,0:2]->C[2,0:2]: 4 terms
    add_signed({6:1}, {0:1}, {8:1})
    add_signed({6:1}, {1:1}, {9:1})
    add_signed({7:1}, {4:1}, {8:1})
    add_signed({7:1}, {5:1}, {9:1})
    
    # A[2,0:2]*B[0:2,2:4]->C[2,2:4]: 4 terms
    add_signed({6:1}, {2:1}, {10:1})
    add_signed({6:1}, {3:1}, {11:1})
    add_signed({7:1}, {6:1}, {10:1})
    add_signed({7:1}, {7:1}, {11:1})
    
    # A[2,2]*B[2,0:4]->C[2,0:4]: 4 terms
    add_signed({8:1}, {8:1}, {8:1})
    add_signed({8:1}, {9:1}, {9:1})
    add_signed({8:1}, {10:1}, {10:1})
    add_signed({8:1}, {11:1}, {11:1})
    
    return {"U": U, "V": V, "W": W}
