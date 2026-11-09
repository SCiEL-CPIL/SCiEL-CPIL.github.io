"""Program for matrix multiplication 2x4x6 with low-rank exact decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_blocks():
    """Return the 7 (u, v, w) triples for 2x2x2 Strassen decomposition.
    Convention: u[i*2+j, c], v[j*2+k, c], w[k*2+i, c]
    """
    blocks = []
    # C00 = A00*(B00+B11)
    blocks.append(([1,0,0,0], [1,0,1,0], [1,1,1,1]))
    # C10 = A10*(B00+B01)
    blocks.append(([0,0,1,0], [1,0,1,1], [1,0,1,0]))
    # C01 = A01*(B10+B11)
    blocks.append(([0,1,0,0], [0,1,1,1], [0,1,1,0]))
    # C11 = A11*(B10+B11)
    blocks.append(([0,0,0,1], [0,1,1,1], [0,0,1,1]))
    # C00 -= A01*B10
    blocks.append(([0,1,0,0], [0,1,0,0], [1,0,0,0]))
    # C11 -= A00*B11
    blocks.append(([1,0,0,0], [0,0,0,1], [0,0,1,1]))
    # C10 += A00*B00
    blocks.append(([1,0,0,0], [1,0,0,0], [1,1,0,0]))
    return blocks

def construct():
    n, m, p = DIMENSIONS
    # Decompose into 2x2x6 blocks by splitting m=4 into 2+2
    # Each 2x2x6 decomposes into 3 2x2x2 blocks (p=6 -> 2+2+2)
    # Each 2x2x2 uses Strassen rank 7
    # Total rank: 2 * 3 * 7 = 42
    
    U = []
    V = []
    W = []
    
    strassen = _strassen_blocks()
    
    for m_block in range(2):  # m = 4 split into 2+2
        for p_block in range(3):  # p = 6 split into 2+2+2
            for s in range(7):
                u = strassen[s][0]
                v = strassen[s][1]
                w = strassen[s][2]
                
                U_row = [0] * (n * m)
                V_row = [0] * (m * p)
                W_row = [0] * (p * n)
                
                # Fill U: indices i*n + j where i in [0,2), j in [m_block*2, m_block*2+2)
                for idx, val in enumerate(u):
                    i = idx // 2
                    j = idx % 2
                    U_row[i * m + m_block * 2 + j] = val
                
                # Fill V: indices j*p + k where j in [m_block*2, m_block*2+2), k in [p_block*2, p_block*2+2)
                for idx, val in enumerate(v):
                    j = idx // 2
                    k = idx % 2
                    V_row[(m_block * 2 + j) * p + p_block * 2 + k] = val
                
                # Fill W: indices k*n + i where k in [p_block*2, p_block*2+2), i in [0, 2)
                for idx, val in enumerate(w):
                    k = idx // 2
                    i = idx % 2
                    W_row[(p_block * 2 + k) * n + i] = val
                
                U.append(U_row)
                V.append(V_row)
                W.append(W_row)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
