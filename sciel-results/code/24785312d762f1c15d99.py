import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=13
    # Rank 46: 6 * 7 (Strassen) + 1 * 4 (Schoolbook)
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Strassen coefficients for 2x2
    # P1: (a00+a11)(b00+b11) -> C00:1, C11:1
    # P2: (a10+a11)b00 -> C10:1, C11:-1
    # P3: a00(b01-b11) -> C01:1, C11:1
    # P4: a11(b10-b00) -> C00:1, C10:1
    # P5: (a00+a01)b11 -> C00:-1, C01:1
    # P6: (a10-a00)(b00+b01) -> C11:1
    # P7: (a01-a11)(b10+b11) -> C00:1
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [a00, a01, a10, a11]
        # V: [b00, b01, b10, b11]
        # W: [C00, C01, C10, C11] (local indices)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # P1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # P2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # P3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # P4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # P5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),  # P6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])   # P7
    ]
    
    # 6 blocks of 2x2
    for block in range(6):
        c0 = block * 2
        c1 = block * 2 + 1
        
        for u_c, v_c, w_c in strassen_terms:
            # Update U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col_idx] += u_c[i * m + j]
            
            # Update V
            # Local V indices: 00, 01, 10, 11
            # Global V indices:
            # 00 -> 0*p + c0
            # 01 -> 0*p + c1
            # 10 -> 1*p + c0
            # 11 -> 1*p + c1
            v_map = [
                0 * p + c0,
                0 * p + c1,
                1 * p + c0,
                1 * p + c1
            ]
            for local_idx, global_idx in enumerate(v_map):
                V[global_idx][col_idx] += v_c[local_idx]
            
            # Update W
            # Local W indices: C00, C01, C10, C11
            # Global W indices: k*n + i
            # C00 -> k=c0, i=0 -> c0*n + 0
            # C01 -> k=c1, i=0 -> c1*n + 0
            # C10 -> k=c0, i=1 -> c0*n + 1
            # C11 -> k=c1, i=1 -> c1*n + 1
            w_map = [
                c0 * n + 0,
                c1 * n + 0,
                c0 * n + 1,
                c1 * n + 1
            ]
            for local_idx, global_idx in enumerate(w_map):
                W[global_idx][col_idx] += w_c[local_idx]
            
            col_idx += 1
    
    # 1 block of 2x1 (column 12)
    c0 = 12
    
    # Schoolbook terms for 2x2 * 2x1
    # 1. a00 * b0 -> C0
    # 2. a01 * b1 -> C0
    # 3. a10 * b0 -> C1
    # 4. a11 * b1 -> C1
    
    schoolbook_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [a00, a01, a10, a11]
        # V: [b0, b1] (local)
        # W: [C0, C1] (local)
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1])
    ]
    
    for u_c, v_c, w_c in schoolbook_terms:
        # Update U
        for i in range(n):
            for j in range(m):
                U[i * m + j][col_idx] += u_c[i * m + j]
        
        # Update V
        # Local V: 0 -> b0, 1 -> b1
        # Global V:
        # 0 -> 0*p + c0
        # 1 -> 1*p + c0
        v_map = [
            0 * p + c0,
            1 * p + c0
        ]
        for local_idx, global_idx in enumerate(v_map):
            V[global_idx][col_idx] += v_c[local_idx]
        
        # Update W
        # Local W: 0 -> C0, 1 -> C1
        # Global W: k=c0, i=0 -> c0*n + 0
        #            k=c0, i=1 -> c0*n + 1
        w_map = [
            c0 * n + 0,
            c0 * n + 1
        ]
        for local_idx, global_idx in enumerate(w_map):
            W[global_idx][col_idx] += w_c[local_idx]
        
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}
