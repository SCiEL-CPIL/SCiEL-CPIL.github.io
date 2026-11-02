<<<<<<< SEARCH
    # Verification pass: check that the decomposition is correct
    if time.time() - t_start < deadline - 5:
        np.random.seed(42)
        for trial in range(3):
            A_test = np.random.randint(-3, 4, (4, 4)).astype(float)
            B_test = np.random.randint(-3, 4, (4, 4)).astype(float)
            C_expected = A_test @ B_test
            
            C_decomp = np.zeros((4, 4), dtype=float)
            for r in range(rank):
                u_col = U[:, r]
                v_col = V[:, r]
                w_col = W[:, r]
                for i in range(4):
                    for k in range(4):
                        for j in range(4):
                            C_decomp[i, j] += u_col[i*4+k] * v_col[k*4+j] * w_col[j*4+i]
            
            if not np.allclose(C_decomp, C_expected, atol=1e-10):
                pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
