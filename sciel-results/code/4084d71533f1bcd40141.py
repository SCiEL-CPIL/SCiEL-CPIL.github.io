import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = DIMENSIONS
    
    # We'll construct a verified rank-35 decomposition
    # First, find the correct Strassen decomposition for 2x2
    
    def verify_strassen(terms):
        """Verify that the given Strassen terms correctly decompose 2x2 matrix mult."""
        # terms: list of (u, v, w) where u,v,w are length-4 vectors
        # u[i] for A[i//2, i%2], v[j] for B[j//2, j%2], w[k] for C[k//2, k%2]
        # Check: sum_r u_r[i]*v_r[j]*w_r[k] = T[i,j,k]
        # T[i,j,k] = 1 if (i,j,k) is one of the 8 valid (i,j,k) combos, else 0
        # Valid combos: (i,j,k) where C[i,k] gets contribution from A[i,j]*B[j,k]
        # i in {0,1}, j in {0,1}, k in {0,1}
        # T[i,j,k] = 1 if the product A[i,j]*B[j,k] appears in C[i,k]
        # Which is always true for valid i,j,k
        
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randn(2, 2)
            B = np.random.randn(2, 2)
            C_exp = A @ B
            C_comp = np.zeros((2, 2))
            for u, v, w in terms:
                # u[i] for A[i], v[j] for B[j], w[k] for C[k]
                # A flat: [A00, A01, A10, A11]
                # B flat: [B00, B01, B10, B11]
                # C flat: [C00, C01, C10, C11]
                for i in range(2):
                    for k in range(2):
                        for j in range(2):
                            C_comp[i, k] += u[i*2+j] * v[j*2+k] * w[k*2+i] * A[i, j] * B[j, k]
            if np.max(np.abs(C_comp - C_exp)) > 1e-10:
                return False
        return True
    
    # Try the standard Strassen decomposition
    # Using the formulation where:
    # u vectors index A as [A00, A01, A10, A11]
    # v vectors index B as [B00, B01, B10, B11]  
    # w vectors index C as [C00, C01, C10, C11]
    
    # From various sources, a correct rank-7 decomposition:
    strassen_terms = [
        # (u, v, w)
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, -1])),
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0])),
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 0])),
        (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([0, 0, 0, 1])),
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([1, 0, 0, 0])),
        (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, -1])),
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 0])),
    ]
    
    if not verify_strassen(strassen_terms):
        # Try another set
        strassen_terms = [
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1])),
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1])),
            (np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0]), np.array([1, 1, 0, 0])),
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0])),
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])),
        ]
    
    if not verify_strassen(strassen_terms):
        # Just use schoolbook for safety
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for idx in range(rank):
            i = idx // (m*p)
            j = (idx // p) % m
            k = idx % p
            U[i*m+j, idx] = 1
            V[j*p+k, idx] = 1
            W[k*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build rank-35 decomposition
    rank = 35
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for block in range(5):
        for u, v, w in strassen_terms:
            # U: A indices [A00, A01, A10, A11] -> indices 0,1,2,3
            U[0, col] = u[0]
            U[1, col] = u[1]
            U[2, col] = u[2]
            U[3, col] = u[3]
            
            # V: B indices for block (columns 2*block, 2*block+1)
            # B[0, 2*block] -> index 0*10 + 2*block = 2*block
            # B[0, 2*block+1] -> index 0*10 + 2*block+1 = 2*block+1
            # B[1, 2*block] -> index 1*10 + 2*block = 10 + 2*block
            # B[1, 2*block+1] -> index 1*10 + 2*block+1 = 10 + 2*block+1
            V[2*block, col] = v[0]
            V[2*block+1, col] = v[1]
            V[10+2*block, col] = v[2]
            V[10+2*block+1, col] = v[3]
            
            # W: C indices for block
            # C[0, 2*block] -> index (2*block)*2 + 0 = 4*block
            # C[0, 2*block+1] -> index (2*block+1)*2 + 0 = 4*block+2
            # C[1, 2*block] -> index (2*block)*2 + 1 = 4*block+1
            # C[1, 2*block+1] -> index (2*block+1)*2 + 1 = 4*block+3
            # w indices: [C00, C01, C10, C11]
            # C00 = C[0, 2*block] -> W[4*block]
            # C01 = C[0, 2*block+1] -> W[4*block+2]
            # C10 = C[1, 2*block] -> W[4*block+1]
            # C11 = C[1, 2*block+1] -> W[4*block+3]
            W[4*block, col] = w[0]
            W[4*block+2, col] = w[1]
            W[4*block+1, col] = w[2]
            W[4*block+3, col] = w[3]
            
            col += 1
    
    # Final verification
    np.random.seed(99)
    valid = True
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_exp = A @ B
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
                C_comp[i,k] = s
                if abs(C_comp[i,k] - C_exp[i,k]) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Schoolbook fallback
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for idx in range(rank):
            i = idx // (m*p)
            j = (idx // p) % m
            k = idx % p
            U[i*m+j, idx] = 1
            V[j*p+k, idx] = 1
            W[k*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
