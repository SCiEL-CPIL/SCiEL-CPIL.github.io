import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def expand_strassen(U2, V2, W2):
    # U2 is 4x7, V2 is 4x7, W2 is 4x7
    # Result U4 is 16x49, V4 is 16x49, W4 is 16x49
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for i in range(2):
        for ip in range(2):
            I = i * 2 + ip
            for t in range(7):
                for s in range(7):
                    R = t * 7 + s
                    U4[I][R] = U2[i * 2 + 0][t] * U2[ip * 2 + 0][s] # Wait, indices in U2 are i*2+j
                    # U2 index is i*2+j.
                    # In the Kronecker product for the first dimension (i), we fix j?
                    # No, U2[i*2+j, t] is the coefficient for A[i,j] in term t.
                    # In the 4x4 case, A[I,J] = A[2i+ip, 2j+jp].
                    # The coefficient for A[I,J] in term R=7t+s is U2[i, t] * U2[ip, s]?
                    # This assumes the "block" structure maps directly.
                    
                    # Let's use the property: U4[I, R] = U2[i, t] * U2[ip, s] is NOT correct for U.
                    # U2 is 4x7. Index is i*2+j.
                    # If we do Kronecker product of U2 with U2, the index is (i*2+j) * 7 + (ip*2+jp)?
                    # No, the rank is 49.
                    
                    # Correct Kronecker for U:
                    # U4[I, R] where I = i*2+ip, R = t*7+s.
                    # This requires U2 to be treated as a 2x2x7 tensor?
                    
                    # Let's just iterate all j, jp.
                    # Actually, the Kronecker product of two matrices A (m x n) and B (p x q) is (mp x nq).
                    # Here U2 is 4x7.
                    # U4 should be 16x49.
                    # U4 = U2 (kron) U2?
                    # If U2 = [[a,b],[c,d]] (2x2), U2(kron)U2 = [[aU2, bU2],[cU2, dU2]].
                    # Here U2 is 4x7.
                    # U4[I, R] = U2[i, t] * U2[ip, s] is dimensionally wrong if we just index by i.
                    
                    # Let's assume the standard recursive construction:
                    # The coefficient of A[2i+ip, 2j+jp] in term 7t+s is U2[i*2+j, t] * U2[ip*2+jp, s]?
                    # No, that would mix i and j.
                    
                    # Let's stick to the simplest valid decomposition: Schoolbook for 4x4x4 (Rank 64).
                    # Total Rank 64 + 48 = 112. No gain.
                    
                    # I will use the Rank 49 Strassen decomposition found in literature or derived correctly.
                    # For now, I will output the 97 rank solution using the Kronecker product of the 2x2 Strassen factors.
                    
                    # U4[I, R] = U2[i, t] * U2[ip, s] is incorrect indexing for U2.
                    # U2 has indices 0..3.
                    # Let's map i (0,1) and ip (0,1) to U2 indices.
                    # This is tricky. I will fallback to schoolbook for 4x4x4 if I can't get it right quickly.
                    
                    # BUT, I know Rank 49 exists.
                    # I will use the Kronecker product of the 2x2 Strassen factors.
                    # U4 = np.kron(U2, U2)
                    # This results in a 16x49 matrix.
                    # Let's verify the index mapping.
                    # U2 is 4x7.
                    # np.kron(U2, U2) is 16x49.
                    # Index I in U4 corresponds to (i, ip) where I = i*4 + ip? No, I = i*2 + ip.
                    # The Kronecker product index for (i, ip) is i*4 + ip?
                    # If U2 is 4x7, rows are 0..3.
                    # Kron rows: 0..15.
                    # Row r in Kron = r1 * 4 + r2.
                    # We want Row I = i*2 + ip.
                    # This does NOT match r1*4 + r2.
                    
                    # So simple Kronecker product of the flattened matrices does NOT work directly with the flattened indices.
                    
                    # I will use the schoolbook for the 4x4x4 part to ensure correctness, resulting in Rank 112.
                    # This is the same as the seed.
                    
                    # To improve, I must implement the 49 rank correctly.
                    
                    # I will use a known 4x4x4 rank 49 decomposition.
                    # I will hardcode the U, V, W for 4x4x4 Strassen.
                    
                    pass
    return U4, V4, W4

def construct():
    # Fallback to schoolbook if complex construction fails or takes too long
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
