import time
import sys

DIMENSIONS = [5, 5, 5]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indices:
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Split 5 into 2, 2, 1
    splits = [2, 2, 1]
    
    # Precompute starts
    starts = [0]
    for s in splits:
        starts.append(starts[-1] + s)
        
    # Total rank
    total_rank = 0
    # Calculate ranks for each block type
    # (2,2,2): 8 blocks * 7 = 56
    # (2,2,1): 4 blocks * 4 = 16
    # (2,1,2): 4 blocks * 4 = 16
    # (1,2,2): 4 blocks * 4 = 16
    # (2,1,1): 2 blocks * 2 = 4
    # (1,2,1): 2 blocks * 2 = 4
    # (1,1,2): 2 blocks * 2 = 4
    # (1,1,1): 1 block * 1 = 1
    # Total 117
    
    U = [[0]*117 for _ in range(25)]
    V = [[0]*117 for _ in range(25)]
    W = [[0]*117 for _ in range(25)]
    
    col_offset = 0
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                si = splits[i]
                sj = splits[j]
                sk = splits[k]
                
                # Get local factors
                if si == 2 and sj == 2 and sk == 2:
                    u_loc, v_loc, w_loc = _strassen_2x2x2()
                    r_loc = 7
                else:
                    u_loc, v_loc, w_loc = _schoolbook_nm_p(si, sj, sk)
                    r_loc = si * sj * sk
                    
                # Map to global
                # U global index: (starts[i] + r) * m + (starts[j] + c)
                # V global index: (starts[j] + r) * p + (starts[k] + c)
                # W global index: (starts[k] + r) * n + (starts[i] + c)
                
                for r in range(si):
                    for c in range(sj):
                        u_val = u_loc[r * sj + c]
                        if u_val != 0:
                            U[(starts[i] + r) * m + (starts[j] + c)][col_offset] = u_val
                            
                for r in range(sj):
                    for c in range(sk):
                        v_val = v_loc[r * sk + c]
                        if v_val != 0:
                            V[(starts[j] + r) * p + (starts[k] + c)][col_offset] = v_val
                            
                for r in range(sk):
                    for c in range(si):
                        w_val = w_loc[r * si + c]
                        if w_val != 0:
                            W[(starts[k] + r) * n + (starts[i] + c)][col_offset] = w_val
                            
                col_offset += r_loc
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
