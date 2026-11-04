"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 46.
    # Blocks:
    # 1. n[0:3], m[0:3], p[0:3] : Hybrid 3x3x3 (Rank 21)
    # 2. n[0:3], m[0:3], p[3:6] : Hybrid 3x3x3 (Rank 21)
    # 3. n[0:3], m[3:5], p[0:3] : 3x2x3 (Rank 2)
    # 4. n[0:3], m[3:5], p[3:6] : 3x2x3 (Rank 2)
    
    total_rank = 46
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 3x2x3 block with rank 2
    def add_3x2x3(n_start, m_start, p_start):
        nonlocal col
        # n_start, m_start, p_start are 0-indexed starts
        # n_size=3, m_size=2, p_size=3
        # U is 3x2, V is 2x2, W is 3x2
        # U[i, r], V[j, r], W[k, r]
        # r=0: U=1, V[0]=1, W=1
        # r=1: U=1, V[1]=1, W=1
        for i in range(3):
            for r in range(2):
                U[(n_start + i) * m + (m_start + 0)][col + r] = 1
                # Wait, U index is i*m + j?
                # U shape is (n*m, rank). Index is i*m + j.
                # In this block, j is the local middle index.
                # The global middle index is m_start + j_local.
                # So U index is (n_start + i) * m + (m_start + j_local).
                # But j_local is 0 or 1.
                # So for r=0, j_local=0.
                # U[(n_start + i) * m + (m_start + 0)][col + r] = 1
                # V index is j*p + k.
                # V[(m_start + j_local) * p + (p_start + k_local)]
                # For r=0, j_local=0.
                # V[(m_start + 0) * p + (p_start + k_local)][col + r]
                # W index is k*n + i.
                # W[(p_start + k_local) * n + (n_start + i)][col + r]
                
        # Let's just loop explicitly
        for i in range(3):
            for k in range(3):
                # r=0
                U[(n_start + i) * m + (m_start + 0)][col] = 1
                V[(m_start + 0) * p + (p_start + k)][col] = 1
                W[(p_start + k) * n + (n_start + i)][col] = 1
                # r=1
                U[(n_start + i) * m + (m_start + 1)][col + 1] = 1
                V[(m_start + 1) * p + (p_start + k)][col + 1] = 1
                W[(p_start + k) * n + (n_start + i)][col + 1] = 1
        col += 2

    # Helper to add a Hybrid 3x3x3 block with rank 21
    def add_3x3x3_hybrid(n_start, m_start, p_start):
        nonlocal col
        # Sub-block 1: 2x2x2 (Rank 7) at n[0:2], m[0:2], p[0:2]
        # Sub-block 2: 1x3x3 (Rank 9) at n[2:3], m[0:3], p[0:3]
        # Sub-block 3: 2x1x2 (Rank 1) at n[0:2], m[2:3], p[0:2]
        # Sub-block 4: 2x2x1 (Rank 4) at n[0:2], m[0:2], p[2:3]
        
        # 1. 2x2x2 (Strassen Rank 7)
        # Indices: i in 0..1, j in 0..1, k in 0..1
        # Strassen decomposition for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to map these to U, V, W
        # U[i, r], V[j, r], W[k, r]
        
        # r=0: M1
        U[(n_start + 0) * m + (m_start + 0)][col] = 1
        U[(n_start + 1) * m + (m_start + 1)][col] = 1
        V[(m_start + 0) * p + (p_start + 0)][col] = 1
        V[(m_start + 1) * p + (p_start + 1)][col] = 1
        W[(p_start + 0) * n + (n_start + 0)][col] = 1
        W[(p_start + 1) * n + (n_start + 1)][col] = 1
        col += 1
        
        # r=1: M2
        U[(n_start + 1) * m + (m_start + 0)][col] = 1
        U[(n_start + 1) * m + (m_start + 1)][col] = 1
        V[(m_start + 0) * p + (p_start + 0)][col] = 1
        W[(p_start + 0) * n + (n_start + 1)][col] = 1
        col += 1
        
        # r=2: M3
        U[(n_start + 0) * m + (m_start + 0)][col] = 1
        V[(m_start + 1) * p + (p_start + 0)][col] = 1
        V[(m_start + 1) * p + (p_start + 1)][col] = -1
        W[(p_start + 0) * n + (n_start + 0)][col] = 1
        W[(p_start + 1) * n + (n_start + 0)][col] = -1
        col += 1
        
        # r=3: M4
        U[(n_start + 1) * m + (m_start + 1)][col] = 1
        V[(m_start + 0) * p + (p_start + 1)][col] = 1
        V[(m_start + 0) * p + (p_start + 0)][col] = -1
        W[(p_start + 1) * n + (n_start + 1)][col] = 1
        W[(p_start + 0) * n + (n_start + 1)][col] = -1
        col += 1
        
        # r=4: M5
        U[(n_start + 0) * m + (m_start + 0)][col] = 1
        U[(n_start + 0) * m + (m_start + 1)][col] = 1
        V[(m_start + 1) * p + (p_start + 1)][col] = 1
        W[(p_start + 1) * n + (n_start + 0)][col] = 1
        col += 1
        
        # r=5: M6
        U[(n_start + 1) * m + (m_start + 0)][col] = 1
        U[(n_start + 0) * m + (m_start + 0)][col] = -1
        V[(m_start + 0) * p + (p_start + 0)][col] = 1
        V[(m_start + 0) * p + (p_start + 1)][col] = 1
        W[(p_start + 0) * n + (n_start + 1)][col] = 1
        W[(p_start + 1) * n + (n_start + 1)][col] = 1
        col += 1
        
        # r=6: M7
        U[(n_start + 0) * m + (m_start + 1)][col] = 1
        U[(n_start + 1) * m + (m_start + 1)][col] = -1
        V[(m_start + 1) * p + (p_start + 0)][col] = 1
        V[(m_start + 1) * p + (p_start + 1)][col] = 1
        W[(p_start + 0) * n + (n_start + 0)][col] = 1
        W[(p_start + 1) * n + (n_start + 0)][col] = 1
        col += 1
        
        # 2. 1x3x3 (Rank 9) at n[2:3], m[0:3], p[0:3]
        # U[2, r]=1, V[j, r]=1, W[k, r]=1 for each j,k
        for j in range(3):
            for k in range(3):
                U[(n_start + 2) * m + (m_start + j)][col] = 1
                V[(m_start + j) * p + (p_start + k)][col] = 1
                W[(p_start + k) * n + (n_start + 2)][col] = 1
                col += 1
                
        # 3. 2x1x2 (Rank 1) at n[0:2], m[2:3], p[0:2]
        # U[i, r]=1, V[2, r]=1, W[k, r]=1
        for i in range(2):
            for k in range(2):
                U[(n_start + i) * m + (m_start + 2)][col] = 1
                V[(m_start + 2) * p + (p_start + k)][col] = 1
                W[(p_start + k) * n + (n_start + i)][col] = 1
        col += 1
        
        # 4. 2x2x1 (Rank 4) at n[0:2], m[0:2], p[2:3]
        # C[i, 2] = A[i, 0]B[0, 2] + A[i, 1]B[1, 2]
        # 4 distinct terms
        for i in range(2):
            for j in range(2):
                U[(n_start + i) * m + (m_start + j)][col] = 1
                V[(m_start + j) * p + (p_start + 2)][col] = 1
                W[(p_start + 2) * n + (n_start + i)][col] = 1
                col += 1

    # Add Block 1
    add_3x3x3_hybrid(0, 0, 0)
    # Add Block 2
    add_3x3x3_hybrid(0, 0, 3)
    # Add Block 3
    add_3x2x3(0, 3, 0)
    # Add Block 4
    add_3x2x3(0, 3, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
