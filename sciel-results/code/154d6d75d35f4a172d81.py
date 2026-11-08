import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T has shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find U, V, W of rank R such that
    # sum_r U[:, r] V[:, r] W[:, r] == T (flattened)
    
    # Schoolbook rank
    R_school = n * m * p
    
    # Try to find a lower rank
    # For 2x3x7, we can try to use the structure.
    # A simple improvement: 
    # The multiplication is C = A B.
    # C is 2x7.
    # We can compute C using 2x3x7 schoolbook.
    
    # Let's try a rank slightly lower, e.g., 35?
    # Or just return schoolbook to be safe, as search might timeout.
    
    # However, the directive says "stronger algorithm".
    # I will implement a random search for a valid decomposition.
    
    start_time = time.time()
    deadline = 100  # 100s limit
    
    def validate(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check if sum_r U[:,r] V[:,r] W[:,r] matches T
        # T is sparse.
        # T[i*m+j, j*p+k, k*n+i] = 1
        # We can check a few random entries.
        # Or reconstruct the full tensor if R is small.
        # For R=42, full tensor is 42 * 6 * 21 * 14 ~ 70k ops. Fast.
        
        # Construct T
        T = np.zeros((n*m, m*p, p*n), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Reconstruct
        # S = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # This is a 3-way sum.
        # Efficiently:
        # S[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # We can do this with matrix multiplications.
        # Let M1 = U @ V.T -> (nm, mp)
        # Then S[i,j,k] = sum_r M1[i,j] W[k,r]? No.
        # S[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # = sum_r (U[i,r] V[j,r]) W[k,r]
        # Let A = U @ V.T (nm x mp)
        # Then S[i,j,k] = sum_r A[i,j] W[k,r]? No.
        # A[i,j] = sum_r U[i,r] V[j,r].
        # Then S[i,j,k] is NOT A[i,j] W[k,r].
        # It is a 3-way contraction.
        
        # We can reshape U, V, W.
        # S = np.einsum('ir,jr,kr->ijk', U, V, W)
        # This might be slow for large R.
        # But R is small.
        
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(S, T, atol=1e-9)

    # Try ranks from R_school down to some limit
    for R in range(R_school, 20, -1):
        if time.time() > deadline:
            break
            
        # Random initialization
        # Use small integers
        U = np.random.randint(-2, 3, (n*m, R))
        V = np.random.randint(-2, 3, (m*p, R))
        W = np.random.randint(-2, 3, (p*n, R))
        
        # Try a few random restarts
        for _ in range(10):
            if validate(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Simple gradient descent? No, discrete.
            # Just try another random init.
            U = np.random.randint(-2, 3, (n*m, R))
            V = np.random.randint(-2, 3, (m*p, R))
            W = np.random.randint(-2, 3, (p*n, R))
            
            if time.time() > deadline:
                break

    # Fallback to schoolbook
    R = R_school
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
