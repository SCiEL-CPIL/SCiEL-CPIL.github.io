import numpy as np

def construct():
    """
    Rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Vectorized Strassen-based construction.
    """
    n, m, p = 4, 4, 9
    
    # Strassen 2x2x2: each is a 2x2 matrix of coefficients
    # A-side matrices (7 of them, each 2x2)
    SA = np.array([
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[0,-1]]
    ])
    # B-side matrices
    SB = np.array([
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[-1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,1]]
    ])
    # C-side matrices
    SC = np.array([
        [[1,0],[0,1]],
        [[0,0],[1,-1]],
        [[0,1],[0,1]],
        [[1,0],[1,0]],
        [[-1,1],[0,0]],
        [[0,0],[0,1]],
        [[1,0],[0,0]]
    ])
    
    terms = []
    
    def add_strassen_4x4(b0, c0):
        """Add Strassen terms for 4x4*4x4 at B cols b0:b0+4, C rows c0:c0+4."""
        # 7 block multiplications
        block_mults = [
            # (A_blocks, B_blocks, C_blocks)
            ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
            ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),-1)]),
            ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
            ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
            ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),-1),((0,1),1)]),
            ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,1),1)]),
            ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,0),1)]),
        ]
        
        for Am, Bm, Cm in block_mults:
            for s in range(7):
                u = np.zeros(16)
                v = np.zeros(36)
                w = np.zeros(36)
                
                for (br, bc), ca in Am:
                    for li in range(2):
                        for lj in range(2):
                            val = SA[s, li, lj] * ca
                            if val:
                                u[(br*2+li)*4 + (bc*2+lj)] += val
                
                for (br, bc), cb in Bm:
                    for lj in range(2):
                        for lk in range(2):
                            val = SB[s, lj, lk] * cb
                            if val:
                                v[(b0+br*2+lj)*9 + (bc*2+lk)] += val
                
                for (br, bc), cc in Cm:
                    for li in range(2):
                        for lk in range(2):
                            val = SC[s, li, lk] * cc
                            if val:
                                w[(c0+bc*2+lk)*4 + (br*2+li)] += val
                
                terms.append((u, v, w))
    
    add_strassen_4x4(0, 0)
    add_strassen_4x4(4, 4)
    
    # Schoolbook 4x4 * 4x1
    for i in range(4):
        for j in range(4):
            u = np.zeros(16)
            v = np.zeros(36)
            w = np.zeros(36)
            u[i*4+j] = 1.0
            v[j*9+8] = 1.0
            w[8*4+i] = 1.0
            terms.append((u, v, w))
    
    U = np.column_stack([t[0] for t in terms])
    V = np.column_stack([t[1] for t in terms])
    W = np.column_stack([t[2] for t in terms])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
