        # Fill Laderman terms (columns 0-22)
        for r, (u9, v9, w9) in enumerate(laderman_23):
            U[0:9, r] = u9
            # V mapping: v9 index is j*3+k. V index is j*7+k.
            for j in range(3):
                for k in range(3):
                    V[j*7+k, r] = v9[j*3+k]
            # W mapping: w9 index is k*3+i. W index is k*3+i. Same.
            W[0:9, r] = w9
