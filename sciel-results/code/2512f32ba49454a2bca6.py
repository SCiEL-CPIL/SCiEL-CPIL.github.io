import numpy as np
import time

def construct():
    n, m, p = 3, 4, 4
    
    # Rank 7 decomposition for 2x2x2 (Strassen)
    # Indices: A(i,j), B(j,k), C(i,k) with i,j,k in {0,1}
    # We return U, V, W for this sub-problem
    def strassen_2x2x2():
        # U: 4x7, V: 4x7, W: 4x7
        # Local indices:
        # A: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
        # B: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
        # C: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11=0, A22=2. B11=0, B22=2.
        U[0][0] = 1; U[2][0] = 1
        V[0][0] = 1; V[2][0] = 1
        # C11 += M1, C22 += M1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A21+A22)B11
        # A21=2, A22=2. B11=0.
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        # C21 += M2
        W[2][1] = 1
        
        # M3 = A11(B12-B22)
        # A11=0. B12=1, B22=2.
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        # C12 += M3
        W[1][2] = 1
        
        # M4 = A22(B21-B11)
        # A22=2. B21=2, B11=0.
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        # C11 += M4, C21 += M4
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A11+A12)B22
        # A11=0, A12=1. B22=2.
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        # C11 -= M5, C12 += M5
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        # A21=2, A11=0. B11=0, B12=1.
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        # C22 += M6
        W[3][5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        # A12=1, A22=2. B21=2, B22=3.
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        # C11 += M7, C22 -= M7?
        # Formula: C22 = M1 - M2 + M3 + M6
        # Wait, standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Let's re-verify M7 contribution.
        # M7 is used in C11?
        # Some versions use M7 in C11 and C22?
        # Let's check the formula again.
        # Standard:
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
        
        # So M7 is in C11.
        W[0][6] = 1
        
        return U, V, W

    # Rank 4 decomposition for 1x2x2 (Schoolbook)
    def schoolbook_1x2x2():
        U = [[0]*4 for _ in range(2)] # 1x2 -> 2 entries
        V = [[0]*4 for _ in range(4)] # 2x2 -> 4 entries
        W = [[0]*4 for _ in range(2)] # 1x2 -> 2 entries
        
        # C0 = A0 B00 + A1 B10
        # C1 = A0 B01 + A1 B11
        
        # Term 1: A0 B00 -> C0
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # Term 2: A1 B10 -> C0
        U[1][1] = 1; V[2][1] = 1; W[0][1] = 1
        # Term 3: A0 B01 -> C1
        U[0][2] = 1; V[1][2] = 1; W[1][2] = 1
        # Term 4: A1 B11 -> C1
        U[1][3] = 1; V[3][3] = 1; W[1][3] = 1
        
        return U, V, W

    # Construct 3x4x4
    # Split into 2x4x4 and 1x4x4
    
    total_U = []
    total_V = []
    total_W = []
    
    # Helper to add a sub-decomposition with index mapping
    def add_decomp(U_sub, V_sub, W_sub, i_map, j_map, k_map, rank_sub):
        # i_map: list of global i indices for each local i index
        # j_map: list of global j indices for each local j index
        # k_map: list of global k indices for each local k index
        # U_sub: (n_sub * m_sub) x rank_sub
        # V_sub: (m_sub * p_sub) x rank_sub
        # W_sub: (p_sub * n_sub) x rank_sub
        
        # We need to expand U_sub to global size (n*m) x rank_sub
        # But we are just appending to the list of columns.
        # So we create new columns for U, V, W.
        
        for r in range(rank_sub):
            u_col = [0] * (3*4)
            v_col = [0] * (4*4)
            w_col = [0] * (4*3)
            
            # Map U
            for local_i in range(len(i_map)):
                for local_j in range(len(j_map)):
                    local_idx = local_i * len(j_map) + local_j
                    global_i = i_map[local_i]
                    global_j = j_map[local_j]
                    global_idx = global_i * 4 + global_j
                    u_col[global_idx] = U_sub[local_idx][r]
            
            # Map V
            for local_j in range(len(j_map)):
                for local_k in range(len(k_map)):
                    local_idx = local_j * len(k_map) + local_k
                    global_j = j_map[local_j]
                    global_k = k_map[local_k]
                    global_idx = global_j * 4 + global_k
                    v_col[global_idx] = V_sub[local_idx][r]
            
            # Map W
            # W convention: W[k*n+i, r]
            # local k, local i
            for local_k in range(len(k_map)):
                for local_i in range(len(i_map)):
                    local_idx = local_k * len(i_map) + local_i
                    global_k = k_map[local_k]
                    global_i = i_map[local_i]
                    global_idx = global_k * 3 + global_i
                    w_col[global_idx] = W_sub[local_idx][r]
            
            total_U.append(u_col)
            total_V.append(v_col)
            total_W.append(w_col)

    # 1. 2x4x4 part (Top 2 rows of A)
    # A is 2x4. i in {0,1}. j in {0,1,2,3}.
    # B is 4x4. j in {0,1,2,3}. k in {0,1,2,3}.
    # C is 2x4. i in {0,1}. k in {0,1,2,3}.
    
    # Split j into {0,1} and {2,3}
    # Split k into {0,1} and {2,3}
    
    # Block 1: i{0,1}, j{0,1}, k{0,1} -> 2x2x2
    U1, V1, W1 = strassen_2x2x2()
    i_map = [0, 1]
    j_map = [0, 1]
    k_map = [0, 1]
    add_decomp(U1, V1, W1, i_map, j_map, k_map, 7)
    
    # Block 2: i{0,1}, j{0,1}, k{2,3} -> 2x2x2
    i_map = [0, 1]
    j_map = [0, 1]
    k_map = [2, 3]
    add_decomp(U1, V1, W1, i_map, j_map, k_map, 7)
    
    # Block 3: i{0,1}, j{2,3}, k{0,1} -> 2x2x2
    i_map = [0, 1]
    j_map = [2, 3]
    k_map = [0, 1]
    add_decomp(U1, V1, W1, i_map, j_map, k_map, 7)
    
    # Block 4: i{0,1}, j{2,3}, k{2,3} -> 2x2x2
    i_map = [0, 1]
    j_map = [2, 3]
    k_map = [2, 3]
    add_decomp(U1, V1, W1, i_map, j_map, k_map, 7)
    
    # 2. 1x4x4 part (Bottom 1 row of A)
    # A is 1x4. i=0. j in {0,1,2,3}.
    # B is 4x4. j in {0,1,2,3}. k in {0,1,2,3}.
    # C is 1x4. i=0. k in {0,1,2,3}.
    
    # Split j into {0,1} and {2,3}
    # Split k into {0,1} and {2,3}
    
    U_s, V_s, W_s = schoolbook_1x2x2()
    
    # Block 5: i{0}, j{0,1}, k{0,1} -> 1x2x2
    i_map = [0]
    j_map = [0, 1]
    k_map = [0, 1]
    add_decomp(U_s, V_s, W_s, i_map, j_map, k_map, 4)
    
    # Block 6: i{0}, j{0,1}, k{2,3} -> 1x2x2
    i_map = [0]
    j_map = [0, 1]
    k_map = [2, 3]
    add_decomp(U_s, V_s, W_s, i_map, j_map, k_map, 4)
    
    # Block 7: i{0}, j{2,3}, k{0,1} -> 1x2x2
    i_map = [0]
    j_map = [2, 3]
    k_map = [0, 1]
    add_decomp(U_s, V_s, W_s, i_map, j_map, k_map, 4)
    
    # Block 8: i{0}, j{2,3}, k{2,3} -> 1x2x2
    i_map = [0]
    j_map = [2, 3]
    k_map = [2, 3]
    add_decomp(U_s, V_s, W_s, i_map, j_map, k_map, 4)
    
    return {"U": total_U, "V": total_V, "W": total_W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
