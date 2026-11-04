import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    n, m, p = DIMENSIONS
    
    # Base cases and recursive construction
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_222():
        # Rank 7 decomposition of M_{2,2,2}
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # U indices (i,j), V indices (j,k), W indices (k,i)
        # n=2, m=2, p=2
        # i in 0,1; j in 0,1; k in 0,1
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1: (A11+A22)(B11+B22)
        # A11 (0,0), A22 (1,1) -> U[0], U[3]
        # B11 (0,0), B22 (1,1) -> V[0], V[3]
        # C11 (0,0), C22 (1,1) -> W[0*2+0]=0, W[1*2+1]=3
        # Wait, W index is k*n + i.
        # C_ik. i=row, k=col?
        # C = A B. C[i,k].
        # P1 contributes to C_00 and C_11.
        # C_00: i=0, k=0. W index 0*2+0=0.
        # C_11: i=1, k=1. W index 1*2+1=3.
        # But Strassen P1 is (A11+A22)(B11+B22).
        # This product is a 2x2 matrix? No, A11 is scalar.
        # A11+A22 is scalar. B11+B22 is scalar.
        # So P1 is scalar.
        # It contributes to C_00 and C_11.
        # C_00 = P1 + P4 - P5 + P7
        # C_11 = P1 - P2 + P3 + P6
        # So for P1, W[0]=1, W[3]=1.
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # P2: (A21+A22)B11
        # A21 (1,0), A22 (1,1) -> U[2], U[3]
        # B11 (0,0) -> V[0]
        # Contributes to C_21? No.
        # C_21 is not in 2x2.
        # Strassen: C_01?
        # C_01 = P3 + P5
        # C_10 = P2 + P4
        # C_11 = P1 - P2 + P3 + P6
        # So P2 contributes to C_10 and C_11.
        # C_10: i=1, k=0. W index 0*2+1=1.
        # C_11: i=1, k=1. W index 1*2+1=3.
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1 # C11 has -P2
        
        # P3: A11(B12-B22)
        # A11 (0,0) -> U[0]
        # B12 (0,1), B22 (1,1) -> V[1], V[3]
        # Contributes to C_01 and C_11.
        # C_01: i=0, k=1. W index 1*2+0=2.
        # C_11: i=1, k=1. W index 3.
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # P4: A22(B21-B11)
        # A22 (1,1) -> U[3]
        # B21 (1,0), B11 (0,0) -> V[2], V[0]
        # Contributes to C_10 and C_00.
        # C_10: i=1, k=0. W index 1.
        # C_00: i=0, k=0. W index 0.
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1; W[0, 3] = 1
        
        # P5: (A11+A12)B22
        # A11 (0,0), A12 (0,1) -> U[0], U[1]
        # B22 (1,1) -> V[3]
        # Contributes to C_01 and C_00.
        # C_01: i=0, k=1. W index 2.
        # C_00: i=0, k=0. W index 0.
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[2, 4] = 1; W[0, 4] = -1
        
        # P6: (A21-A11)(B11+B12)
        # A21 (1,0), A11 (0,0) -> U[2], U[0]
        # B11 (0,0), B12 (0,1) -> V[0], V[1]
        # Contributes to C_11 and C_10?
        # C_11 = ... + P6.
        # C_10? No.
        # Wait, C_10 = P2 + P4.
        # Where does P6 go?
        # C_11 = P1 - P2 + P3 + P6.
        # Does P6 contribute elsewhere?
        # Let's check C_01 = P3 + P5.
        # C_00 = P1 + P4 - P5 + P7.
        # C_10 = P2 + P4.
        # It seems P6 only in C_11?
        # No, Strassen formulas:
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        # So P6 is only in C11.
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # P7: (A12-A22)(B21+B22)
        # A12 (0,1), A22 (1,1) -> U[1], U[3]
        # B21 (1,0), B22 (1,1) -> V[2], V[3]
        # Contributes to C_00 and C_11?
        # C_00 = ... + P7.
        # C_11? No.
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1

        return U, V, W

    def combine_n(U1, V1, W1, U2, V2, W2, n1, n2, m, p):
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        # U: vstack
        U = np.vstack([U1, U2])
        
        # V: hstack
        V = np.hstack([V1, V2])
        
        # W: scatter
        W = np.zeros((p * (n1 + n2), r1 + r2))
        for k in range(p):
            for i1 in range(n1):
                W[k * (n1 + n2) + i1, :r1] = W1[k * n1 + i1, :]
            for i2 in range(n2):
                W[k * (n1 + n2) + n1 + i2, r1:] = W2[k * n2 + i2, :]
        
        return U, V, W

    def combine_m(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        # U: 
        # Rows i*m + j.
        # Sub1: j in [0, m1). Rows i*m1 + j.
        # Global: j in [0, m1). Row i*(m1+m2) + j.
        # Sub2: j in [0, m2). Rows i*m2 + j.
        # Global: j in [m1, m1+m2). Row i*(m1+m2) + (m1+j).
        U = np.zeros((n * (m1 + m2), r1 + r2))
        for i in range(n):
            for j1 in range(m1):
                U[i * (m1 + m2) + j1, :r1] = U1[i * m1 + j1, :]
            for j2 in range(m2):
                U[i * (m1 + m2) + m1 + j2, r1:] = U2[i * m2 + j2, :]
        
        # V:
        # Rows j*p + k.
        # Sub1: j in [0, m1).
        # Sub2: j in [0, m2).
        V = np.zeros(((m1 + m2) * p, r1 + r2))
        for j1 in range(m1):
            for k in range(p):
                V[j1 * p + k, :r1] = V1[j1 * p + k, :]
        for j2 in range(m2):
            for k in range(p):
                V[(m1 + j2) * p + k, r1:] = V2[j2 * p + k, :]
        
        # W: hstack
        W = np.hstack([W1, W2])
        
        return U, V, W

    def combine_p(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        
        # U: hstack
        U = np.hstack([U1, U2])
        
        # V:
        # Rows j*p + k.
        # Sub1: k in [0, p1).
        # Sub2: k in [0, p2).
        V = np.zeros((m * (p1 + p2), r1 + r2))
        for j in range(m):
            for k1 in range(p1):
                V[j * (p1 + p2) + k1, :r1] = V1[j * p1 + k1, :]
            for k2 in range(p2):
                V[j * (p1 + p2) + p1 + k2, r1:] = V2[j * p2 + k2, :]
        
        # W:
        # Rows k*n + i.
        # Sub1: k in [0, p1).
        # Sub2: k in [0, p2).
        W = np.zeros(((p1 + p2) * n, r1 + r2))
        for k1 in range(p1):
            for i in range(n):
                W[k1 * n + i, :r1] = W1[k1 * n + i, :]
        for k2 in range(p2):
            for i in range(n):
                W[(p1 + k2) * n + i, r1:] = W2[k2 * n + i, :]
        
        return U, V, W

    # Construct M_{3,1,3}
    U313, V313, W313 = schoolbook(3, 1, 3)
    
    # Construct M_{3,2,3} = M_{3,1,3} + M_{3,1,3}
    U323, V323, W323 = combine_m(U313, V313, W313, U313, V313, W313, 3, 1, 1, 3)
    
    # Construct M_{2,2,2}
    U222, V222, W222 = strassen_222()
    
    # Construct M_{1,2,2}
    U122, V122, W122 = schoolbook(1, 2, 2)
    
    # Construct M_{3,2,2} = M_{1,2,2} + M_{2,2,2}
    U322, V322, W322 = combine_n(U122, V122, W122, U222, V222, W222, 1, 2, 2, 2)
    
    # Construct M_{3,2,4} = M_{3,2,2} + M_{3,2,2}
    U324, V324, W324 = combine_p(U322, V322, W322, U322, V322, W322, 3, 2, 2, 2)
    
    # Construct M_{3,2,7} = M_{3,2,3} + M_{3,2,4}
    U327, V327, W327 = combine_p(U323, V323, W323, U324, V324, W324, 3, 2, 3, 4)
    
    # Construct M_{3,4,7} = M_{3,2,7} + M_{3,2,7}
    U347, V347, W347 = combine_m(U327, V327, W327, U327, V327, W327, 3, 2, 2, 7)
    
    # Verify
    rank = U347.shape[1]
    # Check validity
    # T_ijk = sum_c U_ijk V_ijk W_ijk
    # We expect T_ijk = 1 for all i,j,k
    
    # Convert to lists for return
    return {
        "U": U347.tolist(),
        "V": V347.tolist(),
        "W": W347.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
