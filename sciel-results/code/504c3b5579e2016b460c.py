import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    def strassen_2x2_block(i_off, j_off, k_off, a_scale=1, b_scale=1, c_scale=1):
        # 7 terms for 2x2x2 Strassen
        # M1 = (A11+A22)(B11+B22)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i_off*m+j_off] = a_scale; u[(i_off+1)*m+(j_off+1)] = a_scale
        v[j_off*p+k_off] = b_scale; v[(j_off+1)*p+(k_off+1)] = b_scale
        w[k_off*n+i_off] = c_scale; w[(k_off+1)*n+(i_off+1)] = c_scale
        add_term(u,v,w)
        # M2 = A22(B21-B11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+1)*m+j_off] = a_scale; u[(i_off+1)*m+(j_off+1)] = a_scale
        v[j_off*p+k_off] = b_scale
        w[k_off*n+(i_off+1)] = c_scale; w[(k_off+1)*n+(i_off+1)] = -c_scale
        add_term(u,v,w)
        # M3 = (A11-A22)(B22+B21)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i_off*m+j_off] = a_scale
        v[j_off*p+(k_off+1)] = b_scale; v[(j_off+1)*p+(k_off+1)] = -b_scale
        w[(k_off+1)*n+i_off] = c_scale; w[(k_off+1)*n+(i_off+1)] = c_scale
        add_term(u,v,w)
        # M4 = A11(B11-B22)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+1)*m+(j_off+1)] = a_scale
        v[(j_off+1)*p+k_off] = b_scale; v[j_off*p+k_off] = -b_scale
        w[k_off*n+i_off] = c_scale; w[k_off*n+(i_off+1)] = c_scale
        add_term(u,v,w)
        # M5 = (A21+A11)B11
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i_off*m+j_off] = a_scale; u[i_off*m+(j_off+1)] = a_scale
        v[(j_off+1)*p+(k_off+1)] = b_scale
        w[(k_off+1)*n+i_off] = c_scale; w[k_off*n+i_off] = -c_scale
        add_term(u,v,w)
        # M6 = (A12-A22)B22
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(i_off+1)*m+j_off] = a_scale; u[i_off*m+j_off] = -a_scale
        v[j_off*p+k_off] = b_scale; v[j_off*p+(k_off+1)] = b_scale
        w[(k_off+1)*n+(i_off+1)] = c_scale
        add_term(u,v,w)
        # M7 = (A21-A11)B12
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[i_off*m+(j_off+1)] = a_scale; u[(i_off+1)*m+(j_off+1)] = -a_scale
        v[(j_off+1)*p+k_off] = b_scale; v[(j_off+1)*p+(k_off+1)] = b_scale
        w[k_off*n+i_off] = c_scale
        add_term(u,v,w)
    
    def strassen_4x4x4():
        # 4x4x4 via Kronecker product of two 2x2x2 Strassen decompositions
        # Outer blocks: 2x2 block structure
        # Inner: 2x2 element structure within each block
        # Total: 7 * 7 = 49 terms
        
        # Define the 7 Strassen patterns for 2x2
        # Each pattern: list of (position, coefficient) for A, B, C
        # Position: (row, col) in 2x2
        strassen_patterns = [
            # M1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
            # M3
            ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
            # M4
            ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            # M5
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(1,0,1),(0,0,-1)]),
            # M6
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            # M7
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
        ]
        
        for outer_a, outer_b, outer_c in strassen_patterns:
            for inner_a, inner_b, inner_c in strassen_patterns:
                u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
                for (or_, oc, coeff_a) in outer_a:
                    for (ir, ic, coeff_inner_a) in inner_a:
                        row = or_*2 + ir
                        col = oc*2 + ic
                        u[row*m + col] += coeff_a * coeff_inner_a
                for (or_, oc, coeff_b) in outer_b:
                    for (ir, ic, coeff_inner_b) in inner_b:
                        row = or_*2 + ir
                        col = oc*2 + ic
                        v[row*p + col] += coeff_b * coeff_inner_b
                for (or_, oc, coeff_c) in outer_c:
                    for (ir, ic, coeff_inner_c) in inner_c:
                        row = or_*2 + ir
                        col = oc*2 + ic
                        w[col*n + row] += coeff_c * coeff_inner_c
                add_term(u,v,w)
    
    def strassen_4x4x2():
        # 4x4x2: split into two 4x4x1 blocks
        # Use Strassen for the 4x4 part with zero padding for the 2nd column
        # Actually, let's use 2x2x2 blocks
        # Split A into 2x2 blocks, B into 2x2 blocks (only first column used), C into 2x2 blocks
        for i_off in range(0, 4, 2):
            for j_off in range(0, 4, 2):
                k_off = 0
                strassen_2x2_block(i_off, j_off, k_off)
    
    def schoolbook_4x4x1():
        for i in range(4):
            for j in range(4):
                u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
                u[i*m+j] = 1
                v[j*p+6] = 1
                w[6*n+i] = 1
                add_term(u,v,w)
    
    strassen_4x4x4()
    strassen_4x4x2()
    schoolbook_4x4x1()
    
    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}
    
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c, (u,v,w) in enumerate(terms):
        for r in range(n*m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m*p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p*n):
            if w[r] != 0:
                W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
