import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 4
    rank = 34
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Strassen Left (A11 * B1L) - indices 0-4 in U, 0-5 in V, 0-4 in W
    # P1: (A11+A22)*(B11+B22)
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[5, col] = 1
    W[0, col] = 1; W[4, col] = 1
    col += 1
    
    # P2: (A21+A22)*B11
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1
    W[1, col] = 1; W[4, col] = -1
    col += 1
    
    # P3: A11*(B21-B11)
    U[0, col] = 1
    V[1, col] = 1; V[5, col] = -1
    W[3, col] = 1; W[4, col] = 1
    col += 1
    
    # P4: A22*(B12-B22)
    U[4, col] = 1
    V[4, col] = 1; V[0, col] = -1
    W[0, col] = 1; W[1, col] = 1
    col += 1
    
    # P5: (A11+A12)*B22
    U[0, col] = 1; U[1, col] = 1
    V[5, col] = 1
    W[0, col] = -1; W[3, col] = 1
    col += 1
    
    # P6: (A21-A11)*(B11+B12)
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[4, col] = 1
    col += 1
    
    # P7: (A12-A22)*(B21+B22)
    U[1, col] = 1; U[4, col] = -1
    V[4, col] = 1; V[5, col] = 1
    W[0, col] = 1
    col += 1
    
    # A12 * B2L - direct multiplications
    U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1
    U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1
    U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1
    U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1
    
    # A21 * B1L - direct multiplications
    U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
    U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
    U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
    U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
    
    # A22 * B2L - direct multiplications
    U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
    U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1
    
    # Strassen Right (A11 * B1R) - indices 2-7 in V, 6-11 in W
    # P1: (A11+A22)*(B11+B22)
    U[0, col] = 1; U[4, col] = 1
    V[2, col] = 1; V[7, col] = 1
    W[6, col] = 1; W[10, col] = 1
    col += 1
    
    # P2: (A21+A22)*B11
    U[3, col] = 1; U[4, col] = 1
    V[2, col] = 1
    W[7, col] = 1; W[10, col] = -1
    col += 1
    
    # P3: A11*(B21-B11)
    U[0, col] = 1
    V[3, col] = 1; V[7, col] = -1
    W[9, col] = 1; W[10, col] = 1
    col += 1
    
    # P4: A22*(B12-B22)
    U[4, col] = 1
    V[6, col] = 1; V[2, col] = -1
    W[6, col] = 1; W[7, col] = 1
    col += 1
    
    # P5: (A11+A12)*B22
    U[0, col] = 1; U[1, col] = 1
    V[7, col] = 1
    W[6, col] = -1; W[9, col] = 1
    col += 1
    
    # P6: (A21-A11)*(B11+B12)
    U[3, col] = 1; U[0, col] = -1
    V[2, col] = 1; V[3, col] = 1
    W[10, col] = 1
    col += 1
    
    # P7: (A12-A22)*(B21+B22)
    U[1, col] = 1; U[4, col] = -1
    V[6, col] = 1; V[7, col] = 1
    W[6, col] = 1
    col += 1
    
    # A12 * B2R - direct multiplications
    U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1
    U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1
    U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1
    U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1
    
    # A21 * B1R - direct multiplications
    U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
    U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1
    U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1
    U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1
    
    # A22 * B2R - direct multiplications
    U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1
    U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1
    
    # Verify correctness (fast check on a few random inputs)
    def check_correct():
        rng = np.random.RandomState(42)
        for _ in range(3):
            A = rng.randint(-2, 3, size=(3, 3)).astype(float)
            B = rng.randint(-2, 3, size=(3, 4)).astype(float)
            C_expected = A @ B
            C_recon = np.zeros((3, 4))
            for r in range(rank):
                C_recon += np.outer(
                    np.dot(U[:, r].astype(float), A.flatten()),
                    np.dot(V[:, r].astype(float), B.flatten())
                ) * 0  # This approach is wrong, let me use the tensor approach
            # Use proper tensor evaluation
            C_recon = np.zeros((3, 4))
            for r in range(rank):
                u = U[:, r].astype(float)
                v = V[:, r].astype(float)
                w = W[:, r].astype(float)
                # u has 9 entries (A elements), v has 12 entries (B elements), w has 12 entries (C elements)
                # Contribution: sum_{a,b} u[a]*v[b]*w[c] where c is determined by a,b
                # Actually, the tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # And C[i,k] = sum_j T[i*3+j, j*4+k, k*3+i]
                # So C_recon[i,k] = sum_j sum_r U[i*3+j,r]*V[j*4+k,r]*W[k*3+i,r]
                for i in range(3):
                    for k in range(4):
                        for j in range(3):
                            C_recon[i, k] += u[i*3+j] * v[j*4+k] * w[k*3+i]
            if not np.allclose(C_recon, C_expected, atol=1e-9):
                return False
        return True
    
    if not check_correct():
        # Fallback to trivial rank-36
        rank = 36
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((12, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    U[i*3+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
