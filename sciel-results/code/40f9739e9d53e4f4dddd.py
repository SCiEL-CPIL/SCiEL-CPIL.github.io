import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 120.0

def construct():
    n, m, p = 2, 4, 8
    
    # We'll use a tensor rank decomposition approach
    # The matrix multiplication tensor has indices (i, j, k) where:
    # i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    # T[i,j,k] = delta for C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # A term in the decomposition is a rank-1 tensor u * v * w
    # where u has shape (n*m,), v has shape (m*p,), w has shape (p*n,)
    # and the reconstruction is: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    
    # We'll try to find a low-rank decomposition
    # Start with schoolbook and try to reduce
    
    rank = 64
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.zeros((p*n, rank), dtype=np.int32)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # We can use the fact that matrix multiplication can be done with fewer multiplications
    # via Strassen-like methods
    
    # For 2x4x8, let's think about it differently
    # We can group the j index
    
    # Actually, let me try a different approach:
    # Use the standard tensor rank algorithm or try to find a lower rank decomposition
    
    # The tensor rank of (2,4,8) matrix multiplication
    # Lower bound: ceil(np.max([n*m, m*p, p*n]) * max(n,m,p) / something)
    # Actually lower bound is at least n*m, m*p, p*n
    
    # Let me try to implement a search for lower rank using the fact that
    # we can combine terms
    
    # For now, let's try the Strassen decomposition on blocks
    # Split j into two groups of 2, and k into four groups of 2
    
    # This gives us 2 * 4 = 8 blocks of (2, 2, 2) tensor
    # Each (2,2,2) tensor has rank 7 (Strassen)
    # Total rank = 8 * 7 = 56
    
    rank = 56
    U = np.zeros((n*m, rank), dtype=np.int32)
    V = np.zeros((m*p, rank), dtype=np.int32)
    W = np.zeros((p*n, rank), dtype=np.int32)
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Strassen 7 multiplications for 2x2 matrix mult
            # Block A: rows 0,1; cols j0, j0+1
            # Block B: rows j0, j0+1; cols k0, k0+1
            # Block C: rows 0,1; cols k0, k0+1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[0*m + j0, col] += 1; U[1*m + j0+1, col] += 1
            V[j0*p + k0, col] += 1; V[j0+1*p + k0+1, col] += 1
            W[k0*n + 0, col] += 1; W[k0+1*n + 1, col] += 1
            col += 1
            
            # M2 = (A10 + A11)B00
            U[1*m + j0, col] += 1; U[1*m + j0+1, col] += 1
            V[j0*p + k0, col] += 1
            W[k0*n + 1, col] += 1; W[k0+1*n + 1, col] -= 1
            col += 1
            
            # M3 = A00(B01 - B11)
            U[0*m + j0, col] += 1
            V[j0*p + k0+1, col] += 1; V[j0+1*p + k0+1, col] -= 1
            W[k0+1*n + 0, col] += 1; W[k0+1*n + 1, col] += 1
            col += 1
            
            # M4 = A11(B10 - B00)
            U[1*m + j0+1, col] += 1
            V[j0+1*p + k0, col] += 1; V[j0*p + k0, col] -= 1
            W[k0*n + 0, col] += 1; W[k0*n + 1, col] += 1
            col += 1
            
            # M5 = (A00 + A01)B11
            U[0*m + j0, col] += 1; U[0*m + j0+1, col] += 1
            V[j0+1*p + k0+1, col] += 1
            W[k0*n + 0, col] -= 1; W[k0+1*n + 0, col] += 1
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[1*m + j0, col] += 1; U[0*m + j0, col] -= 1
            V[j0*p + k0, col] += 1; V[j0*p + k0+1, col] += 1
            W[k0+1*n + 1, col] += 1
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[0*m + j0+1, col] += 1; U[1*m + j0+1, col] -= 1
            V[j0+1*p + k0, col] += 1; V[j0+1*p + k0+1, col] += 1
            W[k0*n + 0, col] += 1
            col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    def verify(U, V, W, n, m, p):
        for _trial in range(100):
            A = np.random.randint(-3, 3, size=(n, m))
            B = np.random.randint(-3, 3, size=(m, p))
            C_expected = A @ B
            C_actual = np.zeros((n, p), dtype=float)
            Ua = np.array(U, dtype=float)
            Va = np.array(V, dtype=float)
            Wa = np.array(W, dtype=float)
            rank = len(U[0])
            for r in range(rank):
                u = Ua[:, r].reshape(n, m)
                v = Va[:, r].reshape(m, p)
                w = Wa[:, r].reshape(p, n)
                # Term: (u * v * w) contributes to C
                # C[i,k] += sum_j u[i,j] * v[j,k] * w[k,i]
                temp = u @ v  # shape (n,p)
                C_actual += temp * w.T  # element-wise multiply by w[k,i] for each k
            # Actually need to be careful with the trace convention
            # W[k*n+i, r] so w indexed as w[k,i]
            C_actual = np.zeros((n, p))
            for r in range(rank):
                u = Ua[:, r].reshape(n, m)
                v = Va[:, r].reshape(m, p)
                w = Wa[:, r].reshape(p, n)
                # C[i,k] += sum_j u[i,j] * v[j,k] * w[k,i]
                for i in range(n):
                    for k in range(p):
                        s = 0.0
                        for j in range(m):
                            s += u[i,j] * v[j,k] * w[k,i]
                        C_actual[i,k] += s
            if not np.allclose(C_actual, C_expected):
                return False
        return True
    
    if not verify(U_list, V_list, W_list, n, m, p):
        # Fall back to schoolbook
        rank = 64
        U = np.zeros((n*m, rank), dtype=np.int32)
        V = np.zeros((m*p, rank), dtype=np.int32)
        W = np.zeros((p*n, rank), dtype=np.int32)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
