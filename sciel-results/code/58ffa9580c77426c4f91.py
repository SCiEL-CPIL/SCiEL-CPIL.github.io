import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Recursive Strassen-like decomposition generator
    # Returns U, V, W as numpy arrays for the given dimensions
    def generate_strassen(n, m, p):
        # Base case: 2x2x2
        if n == 2 and m == 2 and p == 2:
            rank = 7
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Strassen coefficients for 2x2x2
            # M1 = (A00 + A11)(B00 + B11)
            # M2 = (A10 + A11)B00
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            
            # Output C:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
            
            # r=0: M1
            U[0*2+0, 0] = 1; U[1*2+1, 0] = 1
            V[0*2+0, 0] = 1; V[1*2+1, 0] = 1
            W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
            
            # r=1: M2
            U[1*2+0, 1] = 1; U[1*2+1, 1] = 1
            V[0*2+0, 1] = 1
            W[0*2+1, 1] = 1; W[1*2+1, 1] = -1
            
            # r=2: M3
            U[0*2+0, 2] = 1
            V[0*2+1, 2] = 1; V[1*2+1, 2] = -1
            W[1*2+0, 2] = 1; W[1*2+1, 2] = 1
            
            # r=3: M4
            U[1*2+1, 3] = 1
            V[1*2+0, 3] = 1; V[0*2+0, 3] = -1
            W[0*2+0, 3] = 1; W[0*2+1, 3] = 1
            
            # r=4: M5
            U[0*2+0, 4] = 1; U[0*2+1, 4] = 1
            V[1*2+1, 4] = 1
            W[0*2+0, 4] = -1; W[1*2+0, 4] = 1
            
            # r=5: M6
            U[1*2+0, 5] = 1; U[0*2+0, 5] = -1
            V[0*2+0, 5] = 1; V[0*2+1, 5] = 1
            W[1*2+1, 5] = 1
            
            # r=6: M7
            U[0*2+1, 6] = 1; U[1*2+1, 6] = -1
            V[1*2+0, 6] = 1; V[1*2+1, 6] = 1
            W[0*2+0, 6] = 1
            
            return U, V, W

        # Recursive case
        # Split the dimension that is > 2
        if m > 2:
            m1 = m // 2
            m2 = m - m1
            # Split A (n x m) into A1 (n x m1), A2 (n x m2)
            # Split B (m x p) into B1 (m1 x p), B2 (m2 x p)
            # C = A1 B1 + A2 B2
            
            U1, V1, W1 = generate_strassen(n, m1, p)
            U2, V2, W2 = generate_strassen(n, m2, p)
            
            rank = U1.shape[1] + U2.shape[1]
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            
            # Place U1, U2
            # U1 corresponds to A1 (cols 0..m1-1 of A)
            # U2 corresponds to A2 (cols m1..m-1 of A)
            for i in range(n):
                for j in range(m1):
                    U[i*m + j, 0:r1] = U1[i*m1 + j, :]
                for j in range(m2):
                    U[i*m + m1 + j, r1:r1+r2] = U2[i*m2 + j, :]
                    
            # Place V1, V2
            # V1 corresponds to B1 (rows 0..m1-1 of B)
            # V2 corresponds to B2 (rows m1..m-1 of B)
            for j in range(m1):
                for k in range(p):
                    V[j*p + k, 0:r1] = V1[j*p + k, :]
            for j in range(m2):
                for k in range(p):
                    V[(m1+j)*p + k, r1:r1+r2] = V2[j*p + k, :]
                    
            # Place W1, W2
            # W is the same for both sub-problems in terms of indices, but we concatenate ranks
            for k in range(p):
                for i in range(n):
                    W[k*n + i, 0:r1] = W1[k*n + i, :]
                    W[k*n + i, r1:r1+r2] = W2[k*n + i, :]
                    
            return U, V, W
            
        elif p > 2:
            p1 = p // 2
            p2 = p - p1
            # Split B (m x p) into B1 (m x p1), B2 (m x p2)
            # C = C1 (n x p1) + C2 (n x p2)
            
            U1, V1, W1 = generate_strassen(n, m, p1)
            U2, V2, W2 = generate_strassen(n, m, p2)
            
            rank = U1.shape[1] + U2.shape[1]
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            
            # Place U1, U2
            # U is the same for both sub-problems (input A is same)
            for i in range(n):
                for j in range(m):
                    U[i*m + j, 0:r1] = U1[i*m + j, :]
                    U[i*m + j, r1:r1+r2] = U2[i*m + j, :]
                    
            # Place V1, V2
            # V1 corresponds to B1 (cols 0..p1-1 of B)
            # V2 corresponds to B2 (cols p1..p-1 of B)
            for j in range(m):
                for k in range(p1):
                    V[j*p + k, 0:r1] = V1[j*p1 + k, :]
                for k in range(p2):
                    V[j*p + p1 + k, r1:r1+r2] = V2[j*p2 + k, :]
                    
            # Place W1, W2
            # W1 corresponds to C1 (cols 0..p1-1 of C)
            # W2 corresponds to C2 (cols p1..p-1 of C)
            for k in range(p1):
                for i in range(n):
                    W[k*n + i, 0:r1] = W1[k*n + i, :]
            for k in range(p2):
                for i in range(n):
                    W[(p1+k)*n + i, r1:r1+r2] = W2[k*n + i, :]
                    
            return U, V, W
            
        elif n > 2:
            # Similar logic for n, though not needed for 2,4,8
            n1 = n // 2
            n2 = n - n1
            U1, V1, W1 = generate_strassen(n1, m, p)
            U2, V2, W2 = generate_strassen(n2, m, p)
            
            rank = U1.shape[1] + U2.shape[1]
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            r1 = U1.shape[1]
            r2 = U2.shape[1]
            
            # Place U1, U2
            for i in range(n1):
                for j in range(m):
                    U[i*m + j, 0:r1] = U1[i*m + j, :]
            for i in range(n2):
                for j in range(m):
                    U[(n1+i)*m + j, r1:r1+r2] = U2[i*m + j, :]
                    
            # Place V1, V2
            for j in range(m):
                for k in range(p):
                    V[j*p + k, 0:r1] = V1[j*p + k, :]
                    V[j*p + k, r1:r1+r2] = V2[j*p + k, :]
                    
            # Place W1, W2
            for k in range(p):
                for i in range(n1):
                    W[k*n + i, 0:r1] = W1[k*n + i, :]
                for i in range(n2):
                    W[k*n + (n1+i), r1:r1+r2] = W2[k*n + i, :]
                    
            return U, V, W
            
        else:
            # Should not reach here for 2,4,8
            raise ValueError("Base case not handled")

    U, V, W = generate_strassen(n, m, p)
    rank = U.shape[1]
    
    # Verification
    if time.time() < deadline:
        np.random.seed(42)
        A = np.random.randint(-10, 10, (n, m)).astype(float)
        B = np.random.randint(-10, 10, (m, p)).astype(float)
        C_expected = A @ B
        
        # Verify tensor decomposition
        # T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if not valid:
            # Fallback to trivial decomposition if recursive fails (should not happen)
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
