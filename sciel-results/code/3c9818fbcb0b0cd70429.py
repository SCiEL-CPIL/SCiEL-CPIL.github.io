"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 26 decomposition: 7 (Strassen) + 19 (Border)
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a tensor
    def add_tensor(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Indices:
    # U: i*3 + j
    # V: j*3 + k
    # W: k*3 + i
    
    # Strassen Tensors (Core 0,1)
    # P1: (A00+A11)(B00+B11)
    # U: 00, 11. V: 00, 11. W: 00 (1), 11 (1)
    add_tensor(0, 1, 0, 1, 0, 1, 0)
    add_tensor(4, 1, 4, 1, 4, 1, 0)
    
    # P2: (A10+A11)B00
    # U: 10, 11. V: 00. W: 10 (1), 11 (-1)
    add_tensor(3, 1, 0, 1, 1, 1, 1)
    add_tensor(4, 1, 0, 1, 4, -1, 1)
    
    # P3: A00(B01-B11)
    # U: 00. V: 01, 11. W: 01 (1), 11 (1)
    add_tensor(0, 1, 1, 1, 1, 1, 2)
    add_tensor(0, 1, 4, -1, 4, 1, 2)
    
    # P4: A11(B10-B00)
    # U: 11. V: 10, 00. W: 00 (1), 10 (1)
    add_tensor(4, 1, 3, 1, 0, 1, 3)
    add_tensor(4, 1, 0, -1, 1, 1, 3)
    
    # P5: (A00+A01)B11
    # U: 00, 01. V: 11. W: 00 (-1), 01 (1)
    add_tensor(0, 1, 4, 1, 0, -1, 4)
    add_tensor(1, 1, 4, 1, 1, 1, 4)
    
    # P6: (A10-A00)(B00+B01)
    # U: 10, 00. V: 00, 01. W: 11 (1)
    add_tensor(3, 1, 0, 1, 4, 1, 5)
    add_tensor(0, -1, 0, 1, 4, 1, 5)
    add_tensor(3, 1, 1, 1, 4, 1, 5)
    add_tensor(0, -1, 1, 1, 4, 1, 5)
    
    # P7: (A01-A11)(B10+B11)
    # U: 01, 11. V: 10, 11. W: 00 (1)
    add_tensor(1, 1, 3, 1, 0, 1, 6)
    add_tensor(4, -1, 3, 1, 0, 1, 6)
    add_tensor(1, 1, 4, 1, 0, 1, 6)
    add_tensor(4, -1, 4, 1, 0, 1, 6)
    
    # Border Tensors
    # 1-4: j=2, i,k in 0,1
    # A_i2 B_2k -> C_ik
    add_tensor(2, 1, 6, 1, 0, 1, 7)   # A02 B20 -> C00
    add_tensor(2, 1, 7, 1, 1, 1, 8)   # A02 B21 -> C01
    add_tensor(5, 1, 6, 1, 3, 1, 9)   # A12 B20 -> C10
    add_tensor(5, 1, 7, 1, 4, 1, 10)  # A12 B21 -> C11
    
    # 5-8: k=2, i in 0,1, j in 0,1
    # A_ij B_j2 -> C_i2
    add_tensor(0, 1, 2, 1, 2, 1, 11)  # A00 B02 -> C02
    add_tensor(1, 1, 5, 1, 2, 1, 12)  # A01 B12 -> C02
    add_tensor(3, 1, 2, 1, 5, 1, 13)  # A10 B02 -> C12
    add_tensor(4, 1, 5, 1, 5, 1, 14)  # A11 B12 -> C12
    
    # 9-10: k=2, i in 0,1, j=2
    # A_i2 B_22 -> C_i2
    add_tensor(2, 1, 8, 1, 2, 1, 15)  # A02 B22 -> C02
    add_tensor(5, 1, 8, 1, 5, 1, 16)  # A12 B22 -> C12
    
    # 11-14: i=2, k in 0,1, j in 0,1
    # A_2j B_jk -> C_2k
    add_tensor(6, 1, 0, 1, 6, 1, 17)  # A20 B00 -> C20
    add_tensor(7, 1, 3, 1, 6, 1, 18)  # A21 B10 -> C20
    add_tensor(6, 1, 1, 1, 7, 1, 19)  # A20 B01 -> C21
    add_tensor(7, 1, 4, 1, 7, 1, 20)  # A21 B11 -> C21
    
    # 15-16: i=2, k in 0,1, j=2
    # A_22 B_2k -> C_2k
    add_tensor(8, 1, 6, 1, 6, 1, 21)  # A22 B20 -> C20
    add_tensor(8, 1, 7, 1, 7, 1, 22)  # A22 B21 -> C21
    
    # 17-18: i=2, k=2, j in 0,1
    # A_2j B_j2 -> C_22
    add_tensor(6, 1, 2, 1, 8, 1, 23)  # A20 B02 -> C22
    add_tensor(7, 1, 5, 1, 8, 1, 24)  # A21 B12 -> C22
    
    # 19: i=2, k=2, j=2
    # A_22 B_22 -> C_22
    add_tensor(8, 1, 8, 1, 8, 1, 25)  # A22 B22 -> C22

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
