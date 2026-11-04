"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 multiplication decomposition
    # Returns list of 7 tuples (u_vec, v_vec, w_vec)
    # u_vec: coefficients for x00, x01, x10, x11
    # v_vec: coefficients for y00, y01, y10, y11
    # w_vec: coefficients for z00, z01, z10, z11 (from the result combination)
    # Note: The w_vec here is for the SINGLE multiplication M_k.
    # The final Z is a linear combination of M_k.
    # We need to map M_k to Z components.
    
    # Strassen M's:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y01-y11)
    # M4 = x11(y10-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x01-x11)(y10+y11)
    
    # Z components:
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # We define the contribution of each M_k to U, V, W.
    # For a given block mapping, we scale these.
    
    def get_strassen_terms():
        # Each term is (u, v, w)
        # u is 4-vector for X, v is 4-vector for Y, w is 4-vector for Z
        # We will sum these into the global U, V, W based on the block mapping.
        
        terms = []
        
        # M1
        u = [1, 0, 0, 1] # x00, x01, x10, x11
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1] # z00, z01, z10, z11 contributions
        # Wait, w is the coefficient of M1 in Z.
        # z00 gets +1, z11 gets +1.
        # So w = [1, 0, 0, 1] is correct for M1.
        terms.append((u, v, w))
        
        # M2
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, -1] # z10 gets +1, z11 gets -1
        terms.append((u, v, w))
        
        # M3
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 1] # z01 gets +1, z11 gets +1
        terms.append((u, v, w))
        
        # M4
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 1, 0] # z00 gets +1, z10 gets +1
        terms.append((u, v, w))
        
        # M5
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 1, 0, 0] # z00 gets -1, z01 gets +1
        terms.append((u, v, w))
        
        # M6
        u = [0, 0, 1, -1] # x10 - x00
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1] # z11 gets +1
        terms.append((u, v, w))
        
        # M7
        u = [0, 1, 0, -1] # x01 - x11
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0] # z00 gets +1
        terms.append((u, v, w))
        
        return terms

    strassen_terms = get_strassen_terms()
    
    col = 0
    for j in range(3):
        # Block j
        # C_j = A00 * B0j + A01 * B1j
        
        # Mapping for A00:
        # x00 -> A[0,0] (idx 0)
        # x01 -> A[0,1] (idx 1)
        # x10 -> A[1,0] (idx 4)
        # x11 -> A[1,1] (idx 5)
        map_A00 = [0, 1, 4, 5]
        
        # Mapping for A01:
        # x00 -> A[0,2] (idx 2)
        # x01 -> A[0,3] (idx 3)
        # x10 -> A[1,2] (idx 6)
        # x11 -> A[1,3] (idx 7)
        map_A01 = [2, 3, 6, 7]
        
        # Mapping for B0j:
        # B is 4x6.
        # B0j is rows 0,1 and cols 2j, 2j+1.
        # y00 -> B[0, 2j] (idx 0*6 + 2j)
        # y01 -> B[0, 2j+1] (idx 0*6 + 2j+1)
        # y10 -> B[1, 2j] (idx 1*6 + 2j)
        # y11 -> B[1, 2j+1] (idx 1*6 + 2j+1)
        map_B0j = [0*6 + 2*j, 0*6 + 2*j + 1, 1*6 + 2*j, 1*6 + 2*j + 1]
        
        # Mapping for B1j:
        # B1j is rows 2,3 and cols 2j, 2j+1.
        # y00 -> B[2, 2j] (idx 2*6 + 2j)
        # y01 -> B[2, 2j+1] (idx 2*6 + 2*j + 1)
        # y10 -> B[3, 2j] (idx 3*6 + 2j)
        # y11 -> B[3, 2j+1] (idx 3*6 + 2*j + 1)
        map_B1j = [2*6 + 2*j, 2*6 + 2*j + 1, 3*6 + 2*j, 3*6 + 2*j + 1]
        
        # Mapping for Cj:
        # C is 2x6.
        # Cj is rows 0,1 and cols 2j, 2j+1.
        # W index: l*n + i.
        # z00 -> C[0, 2j] -> W[2j*2 + 0]
        # z01 -> C[0, 2j+1] -> W[(2j+1)*2 + 0]
        # z10 -> C[1, 2j] -> W[2j*2 + 1]
        # z11 -> C[1, 2j+1] -> W[(2j+1)*2 + 1]
        map_Cj = [2*j*2 + 0, (2*j+1)*2 + 0, 2*j*2 + 1, (2*j+1)*2 + 1]
        
        # Process A00 * B0j
        for u, v, w in strassen_terms:
            for k in range(4):
                if u[k] != 0:
                    U[map_A00[k]][col] += u[k]
                if v[k] != 0:
                    V[map_B0j[k]][col] += v[k]
                if w[k] != 0:
                    W[map_Cj[k]][col] += w[k]
            col += 1
            
        # Process A01 * B1j
        for u, v, w in strassen_terms:
            for k in range(4):
                if u[k] != 0:
                    U[map_A01[k]][col] += u[k]
                if v[k] != 0:
                    V[map_B1j[k]][col] += v[k]
                if w[k] != 0:
                    W[map_Cj[k]][col] += w[k]
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
