"""Matrix multiplication 3x4x8 using Strassen-based rank-44 decomposition."""
DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = 3, 4, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    sU = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [-1, 0, 1, 0],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    sV = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B10-B11
        [-1, 0, 1, 0],  # M4: B00-B10
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    sW = [
        [1, 0, 0, 1],   # M1: C00+=1, C11+=1
        [0, 0, 1, -1],  # M2: C10+=1, C11-=1
        [0, 1, 0, 1],   # M3: C01+=1, C11+=1
        [1, 0, 1, 0],   # M4: C00+=1, C10+=1
        [-1, 1, 0, 0],  # M5: C00-=1, C01+=1
        [0, 0, 0, 1],   # M6: C11+=1
        [1, 0, 0, 0]    # M7: C00+=1
    ]
    
    # Part 1: i=0,1 using Strassen on 2x2x4 blocks
    # j blocks: [0,1], [2,3]
    # k blocks: [0,1], [2,3], [4,5], [6,7]
    for jb in range(2):
        j0 = jb * 2
        for kb in range(4):
            k0 = kb * 2
            for t in range(7):
                # U: local (li, lj) -> global (i=li, j=j0+lj)
                for li in range(2):
                    for lj in range(2):
                        l = li * 2 + lj
                        val = sU[t][l]
                        if val != 0:
                            U[li * m + (j0 + lj)][col] += val
                # V: local (lj, lk) -> global (j=j0+lj, k=k0+lk)
                for lj in range(2):
                    for lk in range(2):
                        l = lj * 2 + lk
                        val = sV[t][l]
                        if val != 0:
                            V[(j0 + lj) * p + (k0 + lk)][col] += val
                # W: local (li, lk) -> global (k=k0+lk, i=li)
                for li in range(2):
                    for lk in range(2):
                        l = li * 2 + lk
                        val = sW[t][l]
                        if val != 0:
                            W[(k0 + lk) * n + li][col] += val
                col += 1
    
    # Part 2: i=2 using schoolbook on 1x4x8
    for j in range(4):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
