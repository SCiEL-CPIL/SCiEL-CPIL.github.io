import time
import sys

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (8, rank)
    # V: (m*p, rank) -> (20, rank)
    # W: (p*n, rank) -> (10, rank)
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen(u_idx_map, v_idx_map, w_idx_map, offset):
        nonlocal col
        # u_idx_map: {local_i_j: global_u_idx}
        # local indices: 0:00, 1:01, 2:10, 3:11 ? No, flat index i*m+j.
        # In our context:
        # A part: i in {0,1}, j in {0,1} for block 1. Flat: i*2+j? No, m=4.
        # Flat index in U is i*m+j.
        
        # Let's define local indices for the 2x2 subblock of A
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 2
        # i=1, j=1 -> 3
        # But U row is i*m+j.
        # For block 1 (j=0,1): rows are 0, 1, 4, 5.
        # For block 2 (j=2,3): rows are 2, 3, 6, 7.
        
        # V part: j in {0,1}, k in {0,1} for subblock.
        # V row is j*p+k.
        # For block 1 (j=0,1):
        # k=0,1: rows 0,1 (j=0), 5,6 (j=1).
        
        # W part: k in {0,1}, i in {0,1}.
        # W row is k*n+i.
        # k=0: 0,1. k=1: 2,3.
        
        # We pass maps to translate local (0,1) to global.
        
        # Strassen steps
        # m1 = (a00+a11)(b00+b11)
        U[u_idx_map[0]][col] += 1
        U[u_idx_map[3]][col] += 1
        V[v_idx_map[0]][col] += 1
        V[v_idx_map[3]][col] += 1
        W[w_idx_map[0]][col] += 1
        W[w_idx_map[3]][col] += 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[u_idx_map[2]][col] += 1
        U[u_idx_map[3]][col] += 1
        V[v_idx_map[0]][col] += 1
        W[w_idx_map[2]][col] += 1
        col += 1
        
        # m3 = a00(b01-b11)
        U[u_idx_map[0]][col] += 1
        V[v_idx_map[1]][col] += 1
        V[v_idx_map[3]][col] += -1
        W[w_idx_map[1]][col] += 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[u_idx_map[3]][col] += 1
        V[v_idx_map[2]][col] += 1
        V[v_idx_map[0]][col] += -1
        W[w_idx_map[0]][col] += -1
        W[w_idx_map[1]][col] += 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[u_idx_map[0]][col] += 1
        U[u_idx_map[1]][col] += 1
        V[v_idx_map[3]][col] += 1
        W[w_idx_map[0]][col] += 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[u_idx_map[2]][col] += 1
        U[u_idx_map[0]][col] += -1
        V[v_idx_map[0]][col] += 1
        V[v_idx_map[1]][col] += 1
        W[w_idx_map[3]][col] += 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[u_idx_map[1]][col] += 1
        U[u_idx_map[3]][col] += -1
        V[v_idx_map[2]][col] += 1
        V[v_idx_map[3]][col] += 1
        W[w_idx_map[2]][col] += 1
        W[w_idx_map[0]][col] += -1
        col += 1
        
    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1(u_idx_map, v_idx_map, w_idx_map, offset):
        nonlocal col
        # v_idx_map: {0: b00, 1: b10}
        # w_idx_map: {0: c00, 1: c10}
        
        # p1 = a00*b00
        U[u_idx_map[0]][col] += 1
        V[v_idx_map[0]][col] += 1
        W[w_idx_map[0]][col] += 1
        col += 1
        
        # p2 = a01*b10
        U[u_idx_map[1]][col] += 1
        V[v_idx_map[1]][col] += 1
        W[w_idx_map[0]][col] += 1
        col += 1
        
        # p3 = a10*b00
        U[u_idx_map[2]][col] += 1
        V[v_idx_map[0]][col] += 1
        W[w_idx_map[1]][col] += 1
        col += 1
        
        # p4 = a11*b10
        U[u_idx_map[3]][col] += 1
        V[v_idx_map[1]][col] += 1
        W[w_idx_map[1]][col] += 1
        col += 1

    # --- Block 1: A1 B1 ---
    # A1: i in {0,1}, j in {0,1}. U rows: 0,1,4,5.
    u_map_1 = {0: 0, 1: 1, 2: 4, 3: 5}
    
    # B1 subblock 1: j in {0,1}, k in {0,1}. V rows: 0,1,5,6.
    v_map_1a = {0: 0, 1: 1, 2: 5, 3: 6}
    # W for k in {0,1}, i in {0,1}. W rows: 0,1,2,3.
    w_map_1a = {0: 0, 1: 1, 2: 2, 3: 3}
    add_strassen(u_map_1, v_map_1a, w_map_1a, 0)
    
    # B1 subblock 2: j in {0,1}, k in {2,3}. V rows: 2,3,7,8.
    v_map_1b = {0: 2, 1: 3, 2: 7, 3: 8}
    # W for k in {2,3}, i in {0,1}. W rows: 4,5,7,8.
    w_map_1b = {0: 4, 1: 5, 2: 7, 3: 8}
    add_strassen(u_map_1, v_map_1b, w_map_1b, 0)
    
    # B1 subblock 3: j in {0,1}, k in {4}. V rows: 4,9.
    v_map_1c = {0: 4, 1: 9}
    # W for k in {4}, i in {0,1}. W rows: 6,9.
    w_map_1c = {0: 6, 1: 9}
    add_schoolbook_2x2x1(u_map_1, v_map_1c, w_map_1c, 0)
    
    # --- Block 2: A2 B2 ---
    # A2: i in {0,1}, j in {2,3}. U rows: 2,3,6,7.
    u_map_2 = {0: 2, 1: 3, 2: 6, 3: 7}
    
    # B2 subblock 1: j in {2,3}, k in {0,1}. V rows: 10,11,15,16.
    v_map_2a = {0: 10, 1: 11, 2: 15, 3: 16}
    # W for k in {0,1}, i in {0,1}. W rows: 0,1,2,3.
    w_map_2a = {0: 0, 1: 1, 2: 2, 3: 3}
    add_strassen(u_map_2, v_map_2a, w_map_2a, 0)
    
    # B2 subblock 2: j in {2,3}, k in {2,3}. V rows: 12,13,17,18.
    v_map_2b = {0: 12, 1: 13, 2: 17, 3: 18}
    # W for k in {2,3}, i in {0,1}. W rows: 4,5,7,8.
    w_map_2b = {0: 4, 1: 5, 2: 7, 3: 8}
    add_strassen(u_map_2, v_map_2b, w_map_2b, 0)
    
    # B2 subblock 3: j in {2,3}, k in {4}. V rows: 14,19.
    v_map_2c = {0: 14, 1: 19}
    # W for k in {4}, i in {0,1}. W rows: 6,9.
    w_map_2c = {0: 6, 1: 9}
    add_schoolbook_2x2x1(u_map_2, v_map_2c, w_map_2c, 0)
    
    return {"U": U, "V": V, "W": W}
