import time
import sys

def construct():
    """
    Returns a rank-44 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(u_map, v_map, w_map):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        U[u_map[0]][col] += 1
        U[u_map[3]][col] += 1
        V[v_map[0]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M2: (A10+A11)B00
        U[u_map[2]][col] += 1
        U[u_map[3]][col] += 1
        V[v_map[0]][col] += 1
        W[w_map[2]][col] += 1
        W[w_map[3]][col] += -1
        col += 1
        
        # M3: A00(B01-B11)
        U[u_map[0]][col] += 1
        V[v_map[1]][col] += 1
        V[v_map[3]][col] += -1
        W[w_map[1]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M4: A11(B10-B00)
        U[u_map[3]][col] += 1
        V[v_map[2]][col] += 1
        V[v_map[0]][col] += -1
        W[w_map[0]][col] += 1
        W[w_map[2]][col] += 1
        col += 1
        
        # M5: (A00+A01)B11
        U[u_map[0]][col] += 1
        U[u_map[1]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += -1
        W[w_map[1]][col] += 1
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[u_map[2]][col] += 1
        U[u_map[0]][col] += -1
        V[v_map[0]][col] += 1
        V[v_map[1]][col] += 1
        W[w_map[3]][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[u_map[1]][col] += 1
        U[u_map[3]][col] += -1
        V[v_map[2]][col] += 1
        V[v_map[3]][col] += 1
        W[w_map[0]][col] += 1
        col += 1

    # Process left half (k = 0..3) and right half (k = 4..7)
    for half in range(2):
        u_map = {0: 0, 1: 1, 2: 3, 3: 4}
        
        b_start = half * 4
        v_map_b1 = {
            0: 0 * p + (b_start + 0),
            1: 0 * p + (b_start + 1),
            2: 1 * p + (b_start + 0),
            3: 1 * p + (b_start + 1)
        }
        w_map_c1 = {
            0: (b_start + 0) * n + 0,
            1: (b_start + 1) * n + 0,
            2: (b_start + 0) * n + 1,
            3: (b_start + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b1, w_map_c1)
        
        b_start2 = half * 4 + 2
        v_map_b2 = {
            0: 0 * p + (b_start2 + 0),
            1: 0 * p + (b_start2 + 1),
            2: 1 * p + (b_start2 + 0),
            3: 1 * p + (b_start2 + 1)
        }
        w_map_c2 = {
            0: (b_start2 + 0) * n + 0,
            1: (b_start2 + 1) * n + 0,
            2: (b_start2 + 0) * n + 1,
            3: (b_start2 + 1) * n + 1
        }
        add_strassen_block(u_map, v_map_b2, w_map_c2)
        
        for i in range(2):
            for k in range(4):
                u_idx = i * m + 2
                v_idx = 2 * p + (half * 4 + k)
                w_idx = (half * 4 + k) * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
