"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 6, 6, 6
    rank = 189  # 7 * 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the 7 Strassen products
    # Each product: (factor1_terms, factor2_terms, output_terms)
    # factor1_terms: list of (i, j, coeff) for 3x3 indices
    # factor2_terms: list of (j, k, coeff) for 3x3 indices
    # output_terms: list of (i_out, k_out, coeff) for the 6x6 output
    
    strassen_products = []
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # M1 = (A11+A22)(B11+B22)
                f1 = [((i, j), 1), ((i+3, j+3), 1)]
                f2 = [((j, k), 1), ((j+3, k+3), 1)]
                out = [((i, k), 1), ((i+3, k+3), 1)]
                strassen_products.append((f1, f2, out))
                
                # M2 = (A21+A22)(B11)
                f1 = [((i+3, j), 1), ((i+3, j+3), 1)]
                f2 = [((j, k), 1)]
                out = [((i+3, k), 1), ((i+3, k+3), -1)]
                strassen_products.append((f1, f2, out))
                
                # M3 = (A11)(B12-B22)
                f1 = [((i, j), 1)]
                f2 = [((j, k+3), 1), ((j+3, k+3), -1)]
                out = [((i, k+3), 1), ((i+3, k+3), 1)]
                strassen_products.append((f1, f2, out))
                
                # M4 = (A22)(B21-B11)
                f1 = [((i+3, j+3), 1)]
                f2 = [((j+3, k), 1), ((j, k), -1)]
                out = [((i, k), 1), ((i+3, k), 1)]
                strassen_products.append((f1, f2, out))
                
                # M5 = (A11+A12)(B22)
                f1 = [((i, j), 1), ((i, j+3), 1)]
                f2 = [((j+3, k+3), 1)]
                out = [((i, k), -1), ((i, k+3), 1)]
                strassen_products.append((f1, f2, out))
                
                # M6 = (A21-A11)(B11+B12)
                f1 = [((i+3, j), 1), ((i, j), -1)]
                f2 = [((j, k), 1), ((j, k+3), 1)]
                out = [((i+3, k+3), 1)]
                strassen_products.append((f1, f2, out))
                
                # M7 = (A12-A22)(B21+B22)
                f1 = [((i, j+3), 1), ((i+3, j+3), -1)]
                f2 = [((j+3, k), 1), ((j+3, k+3), 1)]
                out = [((i, k), 1)]
                strassen_products.append((f1, f2, out))
    
    for f1, f2, out in strassen_products:
        for (i1, j1), c1 in f1:
            U[i1 * m + j1][col] += c1
        for (j2, k2), c2 in f2:
            V[j2 * p + k2][col] += c2
        for (i3, k3), c3 in out:
            W[k3 * n + i3][col] += c3
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
