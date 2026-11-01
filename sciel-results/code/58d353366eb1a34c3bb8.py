import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    # n=2 (rows of A), m=2 (cols of A, rows of B), p=12 (cols of B)
    # A is (n,m) = (2,2), B is (m,p) = (2,12), C is (n,p) = (2,12)
    
    # The tensor T has shape (n*m, m*p, p*n)
    # T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for valid i,j,k
    # This is the matrix multiplication tensor.
    
    # Strategy: Use Strassen-based decomposition.
    # Partition the 12 columns of B (and C) into 6 blocks of 2.
    # Each block is a 2x2 matrix multiplication (since A is 2x2).
    # Use Strassen's 7-product algorithm for each 2x2x2 block.
    # Total rank = 6 * 7 = 42.
    
    # This is better than schoolbook rank 48.
    # Can we do better? 
    # The rank of 2x2x12 matrix multiplication is at least 42 (since it contains 6 disjoint 2x2x2 subproblems? No, that's not right).
    # Actually, the lower bound is max(rank(2x2x2), ...) but that's not a direct sum.
    # However, 42 is a known good value.
    # Is there a rank < 42 algorithm?
    # For 2x2xp, the rank is known to be 7p/2 for p even? No, that would be 42.
    # Let's check if we can share terms between blocks.
    # Strassen's algorithm is local to each 2x2 block.
    # There are no obvious cross-block savings for arbitrary 2x2x12.
    # So 42 is likely optimal or near-optimal.
    
    # Let's verify the Strassen decomposition carefully.
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 7-product decomposition for 2x2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # Local indices:
    # A: 0=a, 1=b, 2=c, 3=d
    # B: 0=e, 1=f, 2=g, 3=h
    # C: 0=C00, 1=C01, 2=C10, 3=C11
    
    # The 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's define the coefficients for each term (M1..M7)
    # Each term is a triple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict from local A index to coefficient
    # V_coeffs: dict from local B index to coefficient
    # W_coeffs: dict from local C index to coefficient
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (c-a)(e+f)
        ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            # U: A indices are global (0..3), no offset
            for a_idx, val in u_coeffs.items():
                U[a_idx, col] = val
            
            # V: B indices are local to the block
            # Local B index 0 -> B[0, 2t] -> global index 0*p + 2t = 2t
            # Local B index 1 -> B[0, 2t+1] -> global index 0*p + 2t+1 = 2t+1
            # Local B index 2 -> B[1, 2t] -> global index 1*p + 2t = 12+2t
            # Local B index 3 -> B[1, 2t+1] -> global index 1*p + 2t+1 = 12+2t+1
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t  # 12 + 2t
                else:  # local_b == 3
                    global_b = m * p // 2 + 2 * t + 1  # 12 + 2t + 1
                V[global_b, col] = val
            
            # W: C indices are local to the block
            # Local C index 0 -> C[0, 2t] -> W index k*n+i = (2t)*2 + 0 = 4t
            # Local C index 1 -> C[0, 2t+1] -> W index (2t+1)*2 + 0 = 4t+2
            # Local C index 2 -> C[1, 2t] -> W index (2t)*2 + 1 = 4t+1
            # Local C index 3 -> C[1, 2t+1] -> W index (2t+1)*2 + 1 = 4t+3
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0  # 4t
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0  # 4t+2
                elif local_c == 2:
                    global_w = (2 * t) * n + 1  # 4t+1
                else:  # local_c == 3
                    global_w = (2 * t + 1) * n + 1  # 4t+3
                W[global_w, col] = val
    
    # Verify the decomposition
    def verify(U, V, W):
        """Check if U, V, W correctly decompose the tensor."""
        # Reconstruct tensor from decomposition
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(n*m, m*p, p*n)
        
        # Build original tensor
        T_orig = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                u = i * m + j
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    T_orig[u, v, w] = 1.0
        
        return np.allclose(T_recon, T_orig, atol=1e-10)
    
    if not verify(U, V, W):
        # If verification fails, fall back to schoolbook
        # This should not happen with correct Strassen implementation
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
