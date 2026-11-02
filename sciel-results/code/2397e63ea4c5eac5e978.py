import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20 second safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # n=2 (rows of A), m=2 (cols of A, rows of B), p=12 (cols of B)
    
    def build_tensor():
        """Build the matrix multiplication tensor of shape (n*m, m*p, p*n)."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                u = i * m + j
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        return T
    
    T = build_tensor()
    
    def strassen_decomposition():
        """Construct the Strassen-based rank-42 decomposition."""
        num_blocks = p // 2  # 6 blocks
        rank = num_blocks * 7  # 42
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strassen 7-product decomposition for 2x2x2
        # u_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) for A[i,j]
        # v_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) for B[j,k]
        # w_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) for C[i,k]
        strassen_terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1 = (a+d)(e+h)
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2 = (c+d)e
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3 = a(f-h)
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4 = d(g-e)
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5 = (a+b)h
            ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),             # M6 = (c-a)(e+f)
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # M7 = (b-d)(g+h)
        ]
        
        for t in range(num_blocks):
            for s in range(7):
                col = t * 7 + s
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
                
                for a_idx, val in u_coeffs.items():
                    U[a_idx, col] = val
                
                for local_b, val in v_coeffs.items():
                    # local_b: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
                    if local_b == 0:
                        j_local, k_local = 0, 0
                    elif local_b == 1:
                        j_local, k_local = 0, 1
                    elif local_b == 2:
                        j_local, k_local = 1, 0
                    else:
                        j_local, k_local = 1, 1
                    # global v index: j_global * p + k_global
                    j_global = j_local
                    k_global = 2 * t + k_local
                    V[j_global * p + k_global, col] = val
                
                for local_c, val in w_coeffs.items():
                    # local_c: 0=(i=0,k=0), 1=(i=0,k=1), 2=(i=1,k=0), 3=(i=1,k=1)
                    if local_c == 0:
                        i_local, k_local = 0, 0
                    elif local_c == 1:
                        i_local, k_local = 0, 1
                    elif local_c == 2:
                        i_local, k_local = 1, 0
                    else:
                        i_local, k_local = 1, 1
                    # global w index: k_global * n + i_global
                    i_global = i_local
                    k_global = 2 * t + k_local
                    W[k_global * n + i_global, col] = val
        
        return U, V, W
    
    U, V, W = strassen_decomposition()
    
    def verify(U, V, W):
        """Check if U, V, W correctly decompose the tensor."""
        # Reconstruct tensor from decomposition
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Compare with original tensor
        return np.allclose(T_recon, T, atol=1e-10)
    
    if not verify(U, V, W):
        # Fallback: try to fix or use schoolbook
        rank = n * m * p  # 48
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, r] = 1
                    V[v, r] = 1
                    W[w, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
