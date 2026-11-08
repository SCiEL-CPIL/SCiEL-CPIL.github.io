def _verify_strassen_4x4():
    # Generate random 4x4 matrices
    np.random.seed(42)
    A = np.random.randn(4, 4)
    B = np.random.randn(4, 4)
    C = A @ B
    
    # Strassen
    A11, A12 = A[0:2, 0:2], A[0:2, 2:4]
    A21, A22 = A[2:4, 0:2], A[2:4, 2:4]
    B11, B12 = B[0:2, 0:2], B[0:2, 2:4]
    B21, B22 = B[2:4, 0:2], B[2:4, 2:4]
    
    M1 = (A11 + A22) @ (B11 + B22)
    M2 = (A21 + A22) @ B11
    M3 = A11 @ (B12 - B22)
    M4 = A22 @ (B21 - B11)
    M5 = (A11 + A12) @ B22
    M6 = (A21 - A11) @ (B11 + B12)
    M7 = (A12 - A22) @ (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    C_strassen = np.block([[C11, C12], [C21, C22]])
    
    assert np.allclose(C, C_strassen), "Strassen verification failed"
    print("Strassen 4x4 x 4x4 verified")
