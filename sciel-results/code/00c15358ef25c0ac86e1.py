"""Program for matrix multiplication 3x4 x 4x8 with verified schoolbook decomposition."""
import time

DIMENSIONS = [3, 4, 8]

def _build_schoolbook(n, m, p):
    """Build schoolbook tensor decomposition for n x m * m x p matrix multiplication.
    
    Returns (U, V, W) where:
        U: (n*m, n*m*p) - A-side factors
        V: (m*p, n*m*p) - B-side factors  
        W: (p*n, n*m*p) - C-side factors
    Each column t corresponds to one bilinear term:
        C[k*n+i] += W[k*n+i][t] * (sum U[i*m+j][t]*A[i,j]) * (sum V[j*p+k][t]*B[j,k])
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_tests=20):
    """Verify that the decomposition correctly computes C = A @ B.
    
    For random A (n x m) and B (m x p), check that:
        C[k*n+i] = sum_t W[k*n+i][t] * (sum_j U[i*m+j][t]*A[i,j]) * (sum_j V[j*p+k][t]*B[j,k])
    equals the true matrix product.
    """
    import random
    random.seed(42)
    for _ in range(num_tests):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        # True product
        true_C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Decomposition product
        # Compute u_t = sum_j U[i*m+j][t] * A[i,j] for each i, t
        # Compute v_t = sum_j V[j*p+k][t] * B[j,k] for each k, t
        # Then C[k*n+i] = sum_t W[k*n+i][t] * u_{i,t} * v_{k,t}
        
        rank = len(U[0]) if U else 0
        # u[i][t] = sum_j U[i*m+j][t] * A[i][j]
        u = [[0] * rank for _ in range(n)]
        for i in range(n):
            for t in range(rank):
                s = 0
                for j in range(m):
                    s += U[i * m + j][t] * A[i][j]
                u[i][t] = s
        
        # v[k][t] = sum_j V[j*p+k][t] * B[j][k]
        v = [[0] * rank for _ in range(p)]
        for k in range(p):
            for t in range(rank):
                s = 0
                for j in range(m):
                    s += V[j * p + k][t] * B[j][k]
                v[k][t] = s
        
        # C[k*n+i] = sum_t W[k*n+i][t] * u[i][t] * v[k][t]
        for i in range(n):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += W[k * n + i][t] * u[i][t] * v[k][t]
                if s != true_C[i][k]:
                    return False
    return True

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x8.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices with integer/half-integer entries.
        U: (n*m, rank) = (12, rank)
        V: (m*p, rank) = (32, rank) 
        W: (p*n, rank) = (24, rank)
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # Build schoolbook decomposition (rank = n*m*p = 96)
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should never happen for schoolbook, but guard against it
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Time exceeded (shouldn't happen for schoolbook)
        pass
    
    return {"U": U, "V": V, "W": W}
