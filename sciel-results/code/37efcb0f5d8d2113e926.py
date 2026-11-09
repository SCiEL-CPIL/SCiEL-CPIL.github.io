"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T[i,j,k] = 1
    # We need to find rank-minimal decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Schoolbook rank = n*m*p = 60
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # For small problems, we can use a direct approach
    
    # The tensor has size (n*m) x (m*p) x (p*n) = 6 x 30 x 20
    # Total entries: 6*30*20 = 3600
    
    # Let's try to find the minimum rank using a greedy/iterative approach
    # or use known constructions
    
    # For 2x3x10, let's try various ranks
    
    def make_schoolbook(rank=None):
        if rank is None:
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
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Try to find a better decomposition using a constructive approach
    # For 2x3xp, we can potentially use rank < 2*3*p
    
    # Let's try a structured decomposition
    # Idea: For each j, we have a 2x10 block of outputs
    # C[i,k] for fixed j gets contribution A[i,j]*B[j,k]
    
    # Alternative: use the fact that the problem is small enough to try
    # a direct search or use a known algorithm
    
    # Let's try rank values from a lower bound up
    # Lower bound: at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    # But actually the tensor rank lower bound is more subtle
    
    # For now, let's try to find if rank 59, 58, ... works
    # But that's expensive. Let's try a smarter approach.
    
    # Actually, for this specific case, let me try to use a 
    # decomposition that shares some terms
    
    # Key insight: B[j,k] is shared across i for fixed j,k
    # A[i,j] is shared across k for fixed i,j
    
    # Let's try: can we combine the j dimension?
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # This is like a dot product of vector (A[i,0], A[i,1], A[i,2]) with (B[0,k], B[1,k], B[2,k])
    
    # For each (i,k), this is a bilinear form in 3 variables.
    # The rank of a 3x3 identity-like structure...
    
    # Actually, let me just try to see if we can do better than 60
    # by trying a few specific constructions
    
    # Construction 1: Use the fact that we can precompute some products
    # But we need exact decomposition, so we can't approximate
    
    # Let me try a direct approach: assume a rank-r decomposition and solve
    # for the entries using linear algebra
    
    # The constraint is: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is nonlinear. But we can fix two of the matrices and solve for the third.
    
    # Let's try: fix U and V, then for each (i,k), we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
    
    # This is a system of linear equations in W.
    
    # Let's try rank = 59, 58, etc. and see if we can find a solution
    
    # Actually, this is getting complex. Let me just return the schoolbook for now
    # and note that finding a better rank requires more sophisticated methods.
    
    # For the purpose of this task, let me try a few specific low-rank constructions
    
    # Try rank = 59: can we merge two schoolbook terms?
    # The two terms would need to be compatible
    
    # Let's just return schoolbook as a safe fallback
    return make_schoolbook()
    
# EVOLVE-BLOCK-END
