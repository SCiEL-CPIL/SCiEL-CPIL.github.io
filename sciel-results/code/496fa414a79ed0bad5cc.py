import time
import numpy as np

DIMENSIONS = [2, 3, 5]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def _verify_tensor(U, V, W, n, m, p):
    """Verify T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_bilinear(U, V, W, n, m, p):
    """Verify the bilinear form: A*B = sum_r (U_r:A)(V_r:B) W_r"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for t in range(r):
        u_scalar = np.dot(U[:, t], A.flatten())
        v_scalar = np.dot(V[:, t], B.flatten())
        w_mat = W[:, t].reshape(p, n).T
        C_computed += u_scalar * v_scalar * w_mat
    return np.allclose(C_computed, C_expected, atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy: Use Strassen on 2x2 sub-blocks to reduce rank
    # Partition: j={0,1} with k={0,1} (Strassen, rank 7)
    #             j={0,1} with k={2,3} (Strassen, rank 7)  
    #             j={0,1} with k={4} (schoolbook, rank 2)
    #             j={2} with k={0,1,2,3,4} (schoolbook, rank 10)
    # Total: 7+7+2+10 = 26
    
    def build_decomposition(use_strassen_blocks):
        if use_strassen_blocks:
            rank = 26
        else:
            rank = 30
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        r = 0
        
        if use_strassen_blocks:
            # Strassen block 1: rows {0,1} of A, cols {0,1} of A (=rows of B), cols {0,1} of B (=cols of C)
            # Standard Strassen for 2x2x2x2:
            # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> contributes to C[0,0] and C[1,1]
            U[0*m+0, r] = 1; U[1*m+1, r] = 1
            V[0*p+0, r] = 1; V[1*p+1, r] = 1
            W[0*n+0, r] = 1; W[1*n+1, r] = 1
            r += 1
            
            # M2 = (A[1,0]+A[1,1])B[0,0] -> C[1,0]
            U[1*m+0, r] = 1; U[1*m+1, r] = 1
            V[0*p+0, r] = 1
            W[0*n+1, r] = 1
            r += 1
            
            # M3 = A[0,0](B[1,1]-B[0,1]) -> C[0,1]
            U[0*m+0, r] = 1
            V[1*p+1, r] = 1; V[0*p+1, r] = -1
            W[1*n+0, r] = 1
            r += 1
            
            # M4 = A[1,1](B[0,0]-B[1,0]) -> C[1,0]
            U[1*m+1, r] = 1
            V[0*p+0, r] = 1; V[1*p+0, r] = -1
            W[0*n+1, r] = 1
            r += 1
            
            # M5 = (A[0,0]+A[0,1])B[1,1] -> C[0,1]
            U[0*m+0, r] = 1; U[0*m+1, r] = 1
            V[1*p+1, r] = 1
            W[1*n+0, r] = 1
            r += 1
            
            # M6 = (A[1,0]-A[0,0])(B[0,1]+B[1,1]) -> C[1,1]
            U[1*m+0, r] = 1; U[0*m+0, r] = -1
            V[0*p+1, r] = 1; V[1*p+1, r] = 1
            W[1*n+1, r] = 1
            r += 1
            
            # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1]) -> C[0,0]
            U[0*m+1, r] = 1; U[1*m+1, r] = -1
            V[1*p+0, r] = 1; V[1*p+1, r] = 1
            W[0*n+0, r] = 1
            r += 1
            
            # Strassen block 2: rows {0,1} of A, cols {0,1} of A, cols {2,3} of B
            # M1 = (A[0,0]+A[1,1])(B[0,2]+B[1,3]) -> C[0,2] and C[1,3]
            U[0*m+0, r] = 1; U[1*m+1, r] = 1
            V[0*p+2, r] = 1; V[1*p+3, r] = 1
            W[2*n+0, r] = 1; W[3*n+1, r] = 1
            r += 1
            
            # M2 = (A[1,0]+A[1,1])B[0,2] -> C[1,2]
            U[1*m+0, r] = 1; U[1*m+1, r] = 1
            V[0*p+2, r] = 1
            W[2*n+1, r] = 1
            r += 1
            
            # M3 = A[0,0](B[1,3]-B[0,3]) -> C[0,3]
            U[0*m+0, r] = 1
            V[1*p+3, r] = 1; V[0*p+3, r] = -1
            W[3*n+0, r] = 1
            r += 1
            
            # M4 = A[1,1](B[0,2]-B[1,2]) -> C[1,2]
            U[1*m+1, r] = 1
            V[0*p+2, r] = 1; V[1*p+2, r] = -1
            W[2*n+1, r] = 1
            r += 1
            
            # M5 = (A[0,0]+A[0,1])B[1,3] -> C[0,3]
            U[0*m+0, r] = 1; U[0*m+1, r] = 1
            V[1*p+3, r] = 1
            W[3*n+0, r] = 1
            r += 1
            
            # M6 = (A[1,0]-A[0,0])(B[0,3]+B[1,3]) -> C[1,3]
            U[1*m+0, r] = 1; U[0*m+0, r] = -1
            V[0*p+3, r] = 1; V[1*p+3, r] = 1
            W[3*n+1, r] = 1
            r += 1
            
            # M7 = (A[0,1]-A[1,1])(B[1,2]+B[1,3]) -> C[0,2]
            U[0*m+1, r] = 1; U[1*m+1, r] = -1
            V[1*p+2, r] = 1; V[1*p+3, r] = 1
            W[2*n+0, r] = 1
            r += 1
            
            # Schoolbook: i={0,1}, j={0,1}, k={4}
            for i in [0, 1]:
                for j in [0, 1]:
                    U[i*m+j, r] = 1
                    V[j*p+4, r] = 1
                    W[4*n+i, r] = 1
                    r += 1
            
            # Schoolbook: i={0,1}, j={2}, k={0,1,2,3,4}
            for i in [0, 1]:
                for k in [0, 1, 2, 3, 4]:
                    U[i*m+2, r] = 1
                    V[2*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
        else:
            # Pure schoolbook
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, r] = 1
                        V[j*p+k, r] = 1
                        W[k*n+i, r] = 1
                        r += 1
        
        assert r == rank, f"Expected {rank} terms, got {r}"
        return U, V, W
    
    # Try rank 26 first
    U, V, W = build_decomposition(True)
    if _verify_tensor(U, V, W, n, m, p) and _verify_bilinear(U, V, W, n, m, p):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Fallback: schoolbook rank 30
    U, V, W = build_decomposition(False)
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
