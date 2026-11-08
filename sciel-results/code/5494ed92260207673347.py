"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 54
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[u,v,w] = 1 iff u=i*m+j, v=j'*p+k, w=k'*n+i' with i=i', j=j', k=k'
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank = n*m*p)
    U = np.zeros((n*m, rank_school), dtype=np.float64)
    V = np.zeros((m*p, rank_school), dtype=np.float64)
    W = np.zeros((p*n, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Attempt to find lower-rank exact decompositions
    # For each candidate rank r, check if a valid decomposition exists
    # This is a hard nonlinear problem; use randomized search with validation
    
    def validate(Ur, Vr, Wr, tol=1e-10):
        """Check if Ur, Vr, Wr form a valid decomposition of T."""
        # Efficient check: T[u,v,w] = sum_t Ur[u,t] * Vr[v,t] * Wr[w,t]
        # We can compute the reconstruction and compare.
        # Using einsum for clarity and correctness.
        # Ur: (n*m, r), Vr: (m*p, r), Wr: (p*n, r)
        # Reconstructed tensor: (n*m, m*p, p*n)
        recon = np.einsum('ut,vt,wt->uvw', Ur, Vr, Wr)
        return np.allclose(recon, T, atol=tol)
    
    # Try ranks from rank_school-1 down to a reasonable minimum
    min_rank_to_try = max(30, rank_school - 20)
    for r in range(rank_school - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            # Random initialization with half-integer-friendly values
            Ur = np.random.randint(-2, 3, size=(n*m, r)) * 0.5
            Vr = np.random.randint(-2, 3, size=(m*p, r)) * 0.5
            Wr = np.random.randint(-2, 3, size=(p*n, r)) * 0.5
            
            if validate(Ur, Vr, Wr):
                best_rank = r
                best_U, best_V, best_W = Ur, Vr, Wr
                break
        
        if best_rank < rank_school:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
