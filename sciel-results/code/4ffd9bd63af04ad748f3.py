import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen_2x2_block(i_off, j_off, k_off):
        """Adds 7 Strassen terms for 2x2 * 2x2 block."""
        # M1 = (A11+A22)*(B11+B22) -> C11 += M1, C22 += M1
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M2 = (A21+A22)*B11 -> C21 += M2, C22 -= M2
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        w = [0] * (p * n)
        w[k_off * n + (i_off+1)] = 1
        w[(k_off+1) * n + (i_off+1)] = -1
        add_term(u, v, w)
        
        # M3 = A11*(B12-B22) -> C12 += M3, C22 += M3
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        v = [0] * (m * p)
        v[j_off * p + (k_off+1)] = 1
        v[(j_off+1) * p + (k_off+1)] = -1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M4 = A22*(B21-B11) -> C11 += M4, C21 += M4
        u = [0] * (n * m)
        u[(i_off+1) * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[j_off * p + k_off] = -1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        w[k_off * n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M5 = (A11+A12)*B22 -> C12 += M5, C11 -= M5
        u = [0] * (n * m)
        u[i_off * m + j_off] = 1
        u[i_off * m + (j_off+1)] = 1
        v = [0] * (m * p)
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + i_off] = 1
        w[k_off * n + i_off] = -1
        add_term(u, v, w)
        
        # M6 = (A21-A11)*(B11+B12) -> C22 += M6
        u = [0] * (n * m)
        u[(i_off+1) * m + j_off] = 1
        u[i_off * m + j_off] = -1
        v = [0] * (m * p)
        v[j_off * p + k_off] = 1
        v[j_off * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[(k_off+1) * n + (i_off+1)] = 1
        add_term(u, v, w)
        
        # M7 = (A12-A22)*(B21+B22) -> C11 += M7
        u = [0] * (n * m)
        u[i_off * m + (j_off+1)] = 1
        u[(i_off+1) * m + (j_off+1)] = -1
        v = [0] * (m * p)
        v[(j_off+1) * p + k_off] = 1
        v[(j_off+1) * p + (k_off+1)] = 1
        w = [0] * (p * n)
        w[k_off * n + i_off] = 1
        add_term(u, v, w)

    def strassen_4x4_head():
        """Generates 49 terms for 4x4 * 4x4 using recursive Strassen."""
        outer_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        inner_terms = [
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
            ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
        ]
        
        def get_block_indices(block_idx, local_idx, dim_type):
            br = block_idx // 2
            bc = block_idx % 2
            lr = local_idx // 2
            lc = local_idx % 2
            row = br * 2 + lr
            col = bc * 2 + lc
            if dim_type == 'A':
                return row * m + col
            elif dim_type == 'B':
                return row * p + col
            elif dim_type == 'C':
                return col * n + row
        
        for outer_a, outer_b, outer_c in outer_terms:
            for inner_x, inner_y, inner_z in inner_terms:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                for x_local, x_coeff in inner_x:
                    for a_block, a_coeff in outer_a:
                        idx = get_block_indices(a_block, x_local, 'A')
                        u[idx] += a_coeff * x_coeff
                        
                for y_local, y_coeff in inner_y:
                    for b_block, b_coeff in outer_b:
                        idx = get_block_indices(b_block, y_local, 'B')
                        v[idx] += b_coeff * y_coeff
                        
                for z_local, z_coeff in inner_z:
                    for c_block, c_coeff in outer_c:
                        idx = get_block_indices(c_block, z_local, 'C')
                        w[idx] += c_coeff * z_coeff
                        
                add_term(u, v, w)

    def tail_2x4x3(i_off):
        """
        Generates terms for 2x4 * 4x3 block.
        i_off: row offset (0 or 2)
        Columns of C: 4, 5, 6
        """
        # j in {0,1}, k in {4,5}: Strassen 2x2x2
        strassen_2x2_block(i_off, 0, 4)
        
        # j in {0,1}, k in {6}: schoolbook 2x2 * 2x1
        for i in range(2):
            for j in range(2):
                u = [0] * (n * m)
                u[(i_off + i) * m + j] = 1
                v = [0] * (m * p)
                v[j * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + (i_off + i)] = 1
                add_term(u, v, w)
        
        # j in {2,3}, k in {4,5}: Strassen 2x2x2
        strassen_2x2_block(i_off, 2, 4)
        
        # j in {2,3}, k in {6}: schoolbook 2x2 * 2x1
        for i in range(2):
            for j in range(2):
                u = [0] * (n * m)
                u[(i_off + i) * m + (2 + j)] = 1
                v = [0] * (m * p)
                v[(2 + j) * p + 6] = 1
                w = [0] * (p * n)
                w[6 * n + (i_off + i)] = 1
                add_term(u, v, w)
    
    strassen_4x4_head()
    tail_2x4x3(0)
    tail_2x4x3(2)
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    # Filter out zero terms
    filtered_terms = []
    for u, v, w in terms:
        if any(x != 0 for x in u) and any(x != 0 for x in v) and any(x != 0 for x in w):
            filtered_terms.append((u, v, w))
    
    rank = len(filtered_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(filtered_terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
