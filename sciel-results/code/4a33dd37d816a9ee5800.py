    def get_4x4x4_decomp():
        U1, V1, W1 = strassen_2x2()
        U = np.zeros((16, 49), dtype=int)
        V = np.zeros((16, 49), dtype=int)
        W = np.zeros((16, 49), dtype=int)
        
        for alpha in range(7):
            for beta in range(7):
                r = alpha * 7 + beta
                # Kronecker product of tensors
                # U[i, j, r] = U1[i1, j1, alpha] * U1[i2, j2, beta]
                # where i = i1*2 + i2, j = j1*2 + j2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                u_val = U1[i1 * 2 + j1, alpha] * U1[i2 * 2 + j2, beta]
                                U[i * 4 + j, r] = u_val
                                
                                for k1 in range(2):
                                    for k2 in range(2):
                                        k = k1 * 2 + k2
                                        # V[j, k, r] = V1[j1, k1, alpha] * V1[j2, k2, beta]
                                        v_val = V1[j1 * 2 + k1, alpha] * V1[j2 * 2 + k2, beta]
                                        V[j * 4 + k, r] = v_val
                                        
                                        # W[k, i, r] = W1[k1, i1, alpha] * W1[k2, i2, beta]
                                        w_val = W1[k1 * 2 + i1, alpha] * W1[k2 * 2 + i2, beta]
                                        W[k * 4 + i, r] = w_val
        return U, V, W
    