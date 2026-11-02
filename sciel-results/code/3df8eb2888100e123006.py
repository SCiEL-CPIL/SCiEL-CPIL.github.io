    def get_laderman_3x3():
        # Returns U, V, W for 3x3 * 3x3
        # U: 9x20, V: 9x20, W: 9x20
        # Indices: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
        
        U = [[0.0]*20 for _ in range(9)]
        V = [[0.0]*20 for _ in range(9)]
        W = [[0.0]*20 for _ in range(9)]
        
        # Helper to add to a matrix
        def add(U, V, W, i, j, k, c, val=1.0):
            U[i*3+j][c] += val
            V[j*3+k][c] += val
            W[k*3+i][c] += val
            # Note: For Laderman, the coefficients in U and V are not always 1.
            # This simple add function assumes coeff 1.
            # I need to handle coefficients.
