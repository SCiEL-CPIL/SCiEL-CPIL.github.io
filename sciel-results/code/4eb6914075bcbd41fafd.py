        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 4)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u_vec = U_arr[i*4+k, :]
                    v_vec = V_arr[k*4+j, :]
                    w_vec = W_arr[j*4+i, :]
                    C_computed[i,j] += np.dot(np.dot(u_vec, v_vec), w_vec)
