import time
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create zero matrices
    def zeros(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # Helper to append columns
    def append_cols(mat, cols_to_append):
        for i in range(len(mat)):
            mat[i].extend(cols_to_append[i])
            
    # Strassen 2x2x2 decomposition
    # Returns U (4x7), V (4x7), W (4x7)
    def strassen_2x2x2():
        U = zeros(4, 7)
        V = zeros(4, 7)
        W = zeros(4, 7)
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1 # C10 = M2+M4, C11 = M1-M2+...
        
        # M3 = A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1 # C01 = M3+M5, C11 = ...+M3
        
        # M4 = A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1 # C00 = ...+M4, C10 = ...+M4
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1 # C00 = ...-M5, C01 = ...+M5
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C11 = ...+M6
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1 # C00 = ...+M7
        
        return U, V, W

    # Schoolbook r x c x k decomposition
    def schoolbook(r, c, k):
        rank = r * c * k
        U = zeros(r * c, rank)
        V = zeros(c * k, rank)
        W = zeros(k * r, rank)
        
        col = 0
        for i in range(r):
            for j in range(c):
                for l in range(k):
                    U[i * c + j][col] = 1
                    V[j * k + l][col] = 1
                    W[l * r + i][col] = 1
                    col += 1
        return U, V, W

    # Global matrices
    U = zeros(n * m, 0)
    V = zeros(m * p, 0)
    W = zeros(p * n, 0)

    # Define blocks
    # i ranges: [0,1] and [2]
    # j ranges: [0,1], [2,3], [4]
    # k ranges: [0,1], [2,3], [4,5]
    
    i_blocks = [(0, 2), (2, 3)] # start, end
    j_blocks = [(0, 2), (2, 4), (4, 5)]
    k_blocks = [(0, 2), (2, 4), (4, 6)]
    
    for i_start, i_end in i_blocks:
        r = i_end - i_start
        for j_start, j_end in j_blocks:
            c = j_end - j_start
            for k_start, k_end in k_blocks:
                k = k_end - k_start
                
                # Determine if we use Strassen (2x2x2) or Schoolbook
                if r == 2 and c == 2 and k == 2:
                    sub_U, sub_V, sub_W = strassen_2x2x2()
                else:
                    sub_U, sub_V, sub_W = schoolbook(r, c, k)
                
                # Append to global U, V, W
                # U rows: i in [i_start, i_end), j in [j_start, j_end)
                # V rows: j in [j_start, j_end), k in [k_start, k_end)
                # W rows: k in [k_start, k_end), i in [i_start, i_end)
                
                # We need to map the local indices to global indices
                # Local U index: li * c + lj -> Global: (i_start + li) * m + (j_start + lj)
                # Local V index: lj * k + lk -> Global: (j_start + lj) * p + (k_start + lk)
                # Local W index: lk * r + li -> Global: (k_start + lk) * n + (i_start + li)
                
                # Since we are appending columns, we just need to place the values
                # into the correct rows of the global matrices (which are currently 0)
                
                for col_idx in range(sub_U[0].__len__() if sub_U else 0):
                    # U
                    for li in range(r):
                        for lj in range(c):
                            global_u_idx = (i_start + li) * m + (j_start + lj)
                            U[global_u_idx].append(sub_U[li * c + lj][col_idx])
                            
                    # V
                    for lj in range(c):
                        for lk in range(k):
                            global_v_idx = (j_start + lj) * p + (k_start + lk)
                            V[global_v_idx].append(sub_V[lj * k + lk][col_idx])
                            
                    # W
                    for lk in range(k):
                        for li in range(r):
                            global_w_idx = (k_start + lk) * n + (i_start + li)
                            W[global_w_idx].append(sub_W[lk * r + li][col_idx])

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    rank = len(res["U"][0]) if res["U"] else 0
    print(f"Rank: {rank}")
