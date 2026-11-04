"""Block-Strassen decomposition for 3x5x6 matrix multiplication."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block partitions:
    # n-blocks: (0,2), (2,1)  -> rows [0,1], [2]
    # m-blocks: (0,2), (2,2), (4,1) -> cols [0,1], [2,3], [4]
    # p-blocks: (0,2), (2,2), (4,2) -> cols [0,1], [2,3], [4,5]
    n_blocks = [(0, 2), (2, 1)]
    m_blocks = [(0, 2), (2, 2), (4, 1)]
    p_blocks = [(0, 2), (2, 2), (4, 2)]

    # Count total rank needed
    rank = 0
    for (ni, ns) in n_blocks:
        for (mj, ms) in m_blocks:
            for (pk, ps) in p_blocks:
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 8  # Strassen
                else:
                    rank += ns * ms * ps  # schoolbook

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for (ni, ns) in n_blocks:
        for (mj, ms) in m_blocks:
            for (pk, ps) in p_blocks:
                if ns == 2 and ms == 2 and ps == 2:
                    # Strassen 2x2x2
                    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
                    # Global: i_g = ni+i, j_g = mj+j, k_g = pk+k
                    # Strassen products M1..M8
                    # M1 = (A00+A11)*(B00+B11): U[0,0]=1,U[1,1]=1; V[0,0]=1,V[1,1]=1; W: C00+=1,C11+=1
                    # M2 = (A10+A11)*B00: U[1,0]=1,U[1,1]=1; V[0,0]=1; W: C10+=1,C11-=-1
                    # M3 = A00*(B10-B00): U[0,0]=1; V[1,0]=1,V[0,0]=-1; W: C01+=1,C11+=1
                    # M4 = A11*(B01-B11): U[1,1]=1; V[0,1]=1,V[1,1]=-1; W: C00+=1,C10+=1
                    # M5 = (A00+A01)*B11: U[0,0]=1,U[0,1]=1; V[1,1]=1; W: C00-=-1,C01+=1
                    # M6 = (A10-A00)*(B00+B10): U[1,0]=1,U[0,0]=-1; V[0,0]=1,V[1,0]=1; W: C11+=1
                    # M7 = (A01-A11)*(B11+B01): U[0,1]=1,U[1,1]=-1; V[1,1]=1,V[0,1]=1; W: C00+=1,C11+=1
                    # M8 = (A00+A11)*(B00-B10): U[0,0]=1,U[1,1]=1; V[0,0]=1,V[1,0]=-1; W: C00+=1,C10-=-1

                    strassen_defs = [
                        # (U_coeffs, V_coeffs, W_coeffs)
                        # U_coeffs: list of (local_i, local_j, val)
                        # V_coeffs: list of (local_j, local_k, val)
                        # W_coeffs: list of (local_k, local_i, val)
                        ([((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))], [((0,0,1),(1,1,1))]),
                        ([((1,0,1),(1,1,1))], [((0,0,1))], [((1,0,1),(1,1,-1))]),
                        ([((0,0,1))], [((1,0,1),(0,0,-1))], [((0,1,1),(1,1,1))]),
                        ([((1,1,1))], [((0,1,1),(1,1,-1))], [((0,0,1),(1,0,1))]),
                        ([((0,0,1),(0,1,1))], [((1,1,1))], [((0,0,-1),(0,1,1))]),
                        ([((1,0,1),(0,0,-1))], [((0,0,1),(1,0,1))], [((1,1,1))]),
                        ([((0,1,1),(1,1,-1))], [((1,1,1),(0,1,1))], [((0,0,1),(1,1,1))]),
                        ([((0,0,1),(1,1,1))], [((0,0,1),(1,0,-1))], [((0,0,1),(1,0,-1))]),
                    ]

                    for u_defs, v_defs, w_defs in strassen_defs:
                        for (li, lj, val) in u_defs[0]:
                            U[ni + li][mj + lj] = U[ni + li][mj + lj] + val if False else 0
                            # Actually U is (n*m, rank), so index is [i*m + j, col]
                            U[(ni + li) * m + (mj + lj)][col] += val
                        for (lj, lk, val) in v_defs[0]:
                            V[(mj + lj) * p + (pk + lk)][col] += val
                        for (lk, li, val) in w_defs[0]:
                            W[(pk + lk) * n + (ni + li)][col] += val
                        col += 1
                else:
                    # Schoolbook for this block
                    for li in range(ns):
                        for lj in range(ms):
                            for lk in range(ps):
                                U[(ni + li) * m + (mj + lj)][col] = 1
                                V[(mj + lj) * p + (pk + lk)][col] = 1
                                W[(pk + lk) * n + (ni + li)][col] = 1
                                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
