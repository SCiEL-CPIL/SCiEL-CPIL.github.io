<<<<<<< SEARCH
    # 2x3x1 rank-6
    def get_2x3x1_rank6():
        U = np.zeros((6, 6))
        V = np.zeros((3, 6))
        W = np.zeros((2, 6))
        col = 0
        for i in range(2):
            for j in range(3):
                U[i*3+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W
=======
    # 2x3x1 rank-6
    def get_2x3x1_rank6():
        U = np.zeros((6, 6))
        V = np.zeros((3, 6))
        W = np.zeros((2, 6))
        col = 0
        for i in range(2):
            for j in range(3):
                U[i*3+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W

    # Rank-5 algorithm for 1x3x4 polynomial evaluation
    # This is used for the 4-column part of B and C
    def get_1x3x4_rank5():
        rank = 5
        U = np.zeros((3, rank))  # A is 1x3, so only 3 rows, but we'll use 3
        V = np.zeros((12, rank))  # B is 3x4
        W = np.zeros((4, rank))  # C is 1x4
        
        # U rows: A[0,0]=0, A[0,1]=1, A[0,2]=2
        # V rows: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[0,3]=3, B[1,0]=4, B[1,1]=5, B[1,2]=6, B[1,3]=7, B[2,0]=8, B[2,1]=9, B[2,2]=10, B[2,3]=11
        # W rows: C[0,0]=0, C[0,1]=1, C[0,2]=2, C[0,3]=3
        
        # This is a polynomial evaluation problem.
        # C[k] = sum_{j=0}^{2} A[j] * B[j, k]
        # We want to compute this with rank 5.
        
        # For now, let's use schoolbook (rank 12) and see if we can improve.
        U[:] = 0
        V[:] = 0
        W[:] = 0
        
        col = 0
        for j in range(3):
            for k in range(4):
                U[j, col] = 1
                V[j*4+k, col] = 1
                W[k, col] = 1
                col += 1
        
        return U, V, W
>>>>>>> REPLACE

