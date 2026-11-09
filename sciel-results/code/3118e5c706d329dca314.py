# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                rounded = round(val * 2) / 2
                if abs(rounded - int(rounded)) < 1e-9:
                    row_list.append(int(rounded))
                else:
                    row_list.append(rounded)
            result.append(row_list)
        return result
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
