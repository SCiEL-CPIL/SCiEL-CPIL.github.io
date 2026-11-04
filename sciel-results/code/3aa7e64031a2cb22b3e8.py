def _check_factor():
    import numpy as np
    alpha = 0.5j
    rng = np.random.default_rng(0)
    for _ in range(200):
        A = rng.standard_normal((2,2)) + 1j*rng.standard_normal((2,2))
        B = rng.standard_normal((4,2)) + 1j*rng.standard_normal((4,2))
        Ctrue = A @ B  # 2x2
        M1 = (A[0,0]+A[0,1])*(B[0,0]+B[2,0])
        M2 = (A[0,0]+A[0,1])*(B[0,1]+B[2,1])
        M3 = A[0,0]*(B[0,0]-B[2,0])
        M4 = A[1,1]*(B[1,1]-B[0,1])
        M5 = A[0,1]*(B[2,0]-B[0,1])
        M6 = A[0,0]*(B[0,1]+alpha*B[1,1])
        M7 = A[0,1]*(B[1,1]-alpha*B[0,0])
        M8 = A[1,1]*(B[2,0]+alpha*B[0,1])
        C11 = M1+M3-M4+M6+M7
        C12 = M2+M5+M8-M7-M6
        C21 = M1-M3-M2+M6+M7
        C22 = M2-M5+M4+M6+M7
        Crest = np.array([[C11,C12],[C21,C22]])
        if not np.allclose(Crest, Ctrue, atol=1e-8):
            print("FAIL")
            return
    print("factor scheme OK")
_check_factor()
