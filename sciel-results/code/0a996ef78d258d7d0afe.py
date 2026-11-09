<<<<<<< SEARCH
    # Convert to final format
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, float) and val == int(val):
                    new_row.append(int(val))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U = clean_matrix(U)
    V = clean_matrix(V)
    W = clean_matrix(W)
    
    return {"U": U, "V": V, "W": W}
=======
    # Convert to final format
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    def clean_matrix(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                if isinstance(val, float) and val == int(val):
                    new_row.append(int(val))
                elif isinstance(val, (int, np.integer)):
                    new_row.append(int(val))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U = clean_matrix(U)
    V = clean_matrix(V)
    W = clean_matrix(W)
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

