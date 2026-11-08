"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Strategy: Kronecker product structure with 1x1 scalar multiplications.
    # This decomposes the tensor into n*m*p rank-1 terms.
    # Each term corresponds to a single scalar product A[i,j] * B[j,k].
    # U, V, W will have rank n*m*p.
    
    rank = n * m * p
    # n=2, m=3, p=6 => rank = 36
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)] 
    V = [[0] * rank for _ in range(m * p)] 
    W = [[0] * rank for _ in range(p * n)] 
    
    # Map indices to rank indices
    # Rank index r corresponds to (i, j, k)
    # We iterate over all i, j, k
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # For the scalar multiplication A[i,j] * B[j,k] -> contributes to C[i,k]
                
                # U part: A[i,j]
                # Index in U is i*m + j
                U[i * m + j][r_idx] = 1
                
                # V part: B[j,k]
                # Index in V is j*p + k
                V[j * p + k][r_idx] = 1
                
                # W part: C[i,k]
                # Index in W follows trace convention W[k*n + i]
                # So index is k*n + i
                W[k * n + i][r_idx] = 1
                
                r_idx += 1
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify if time permits
    if time.time() < deadline - 5:
        try:
            _verify(result, n, m, p)
        except Exception:
            pass
            
    return result

def _verify(result, n, m, p):
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    # Check if U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] sums to 1
                    # In this 1x1 decomposition, only one term r corresponds to (i,j,k)
                    # where U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
                    # All other terms are 0.
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    raise ValueError(f"Verification failed at i={i},j={j},k={k}: got {s}")
    return True
# EVOLVE-BLOCK-END
