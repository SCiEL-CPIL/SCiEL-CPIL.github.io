import numpy as np

def construct():
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # ... fill U2, V2, W2 as before ...

    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = 7*r1 + r2
                            val = U2[2*i1+j1, r1] * U2[2*i2+j2, r2]
                            U4[i*m + j, r] = val # m=4? No, U is n*m. n=4, m=4.
                            # U index is i*m + j.
                            # V index is j*p + k.
                            # W index is k*n + i.
                            
    # This is O(49*16) which is fine.
