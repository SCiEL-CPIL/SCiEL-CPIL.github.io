        U = np.zeros((2*3, 11))
        V = np.zeros((3*2, 11))
        W = np.zeros((2*2, 11))
        
        # Part 1: j=0,1 (first 2 of 3 columns)
        U[:, :7] = U_s  # U_s has shape (2*2, 7) = (4, 7), but we need (6, 11)
