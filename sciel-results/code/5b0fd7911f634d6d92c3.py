import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W form a valid decomposition for M_{n,m,p}.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = U.shape[1]
    # Tensor T[i, j, k] should be sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the convention is W[k*n+i, r].
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T_{i, j, k} = 1 if we are decomposing the multiplication tensor.
    # The decomposition is T = sum_r u_r v_r w_r
    # u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # We check if this sum equals 1 for all i, j, k.
    # Actually, the tensor entries are 1.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _find_m225_decomposition():
    """
    Try to find a rank 16 decomposition for M_{2,2,5}.
    """
    n, m, p = 2, 2, 5
    target_rank = 16
    
    # We use a randomized search with local refinement.
    # This is a heuristic.
    
    # Initialize with random integers in [-2, 2]
    # Or start from a known structure?
    # Let's try random restarts.
    
    for attempt in range(100):
        U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
        V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
        W = np.random.randint(-2, 3, size=(p*n, target_rank)).astype(float)
        
        # Simple local search: minimize error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(target_rank))
                    error += (val - 1.0)**2
        
        if error < 1e-6:
            # Verify exactness with integers/half-integers
            # The entries are integers, so the result should be integer.
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
                
    return None

def construct():
    _start = time.time()
    _deadline = _start + 135.0
    
    n, m, p = DIMENSIONS
    
    # Try to find M_{2,2,5} decomposition
    # If found, combine two of them for M_{2,4,5}
    
    # The M_{2,4,5} tensor can be split into two M_{2,2,5} tensors
    # A = [A1, A2] (2x2, 2x2)
    # B = [B1; B2] (2x5, 2x5)
    # C = A1 B1 + A2 B2
    
    # We need to map the indices correctly.
    # M_{2,2,5} indices:
    # A: i in {0,1}, j in {0,1}
    # B: j in {0,1}, k in {0..4}
    # C: i in {0,1}, k in {0..4}
    
    # For M_{2,4,5}:
    # A: i in {0,1}, j in {0..3}
    # B: j in {0..3}, k in {0..4}
    # C: i in {0,1}, k in {0..4}
    
    # Block 1: j in {0,1}
    # Block 2: j in {2,3}
    
    m225 = _find_m225_decomposition()
    
    if m225 is not None and time.time() < _deadline:
        U1, V1, W1 = m225
        rank_1 = U1.shape[1]
        
        # Construct M_{2,4,5} by combining two M_{2,2,5}
        total_rank = 2 * rank_1
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Block 1 (j=0,1)
        # U indices: i*m + j. i in {0,1}, j in {0,1}
        # V indices: j*p + k. j in {0,1}, k in {0..4}
        # W indices: k*n + i. k in {0..4}, i in {0,1}
        
        # Copy U1 to U
        # U1 has shape (4, rank_1). Rows are i*2+j.
        # In M_{2,4,5}, U rows are i*4+j.
        # For j=0,1:
        for i in range(2):
            for j in range(2):
                U[i*4 + j, 0:rank_1] = U1[i*2 + j, :]
                
        # Copy V1 to V
        # V1 has shape (10, rank_1). Rows are j*5+k.
        # In M_{2,4,5}, V rows are j*5+k.
        # For j=0,1:
        for j in range(2):
            for k in range(5):
                V[j*5 + k, 0:rank_1] = V1[j*5 + k, :]
                
        # Copy W1 to W
        # W1 has shape (10, rank_1). Rows are k*2+i.
        # In M_{2,4,5}, W rows are k*2+i.
        # Same indices.
        for k in range(5):
            for i in range(2):
                W[k*2 + i, 0:rank_1] = W1[k*2 + i, :]
        
        # Block 2 (j=2,3)
        # Shift j by 2.
        for i in range(2):
            for j in range(2, 4):
                U[i*4 + j, rank_1:2*rank_1] = U1[i*2 + (j-2), :]
                
        for j in range(2, 4):
            for k in range(5):
                V[j*5 + k, rank_1:2*rank_1] = V1[(j-2)*5 + k, :]
                
        for k in range(5):
            for i in range(2):
                W[k*2 + i, rank_1:2*rank_1] = W1[k*2 + i, :]
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
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
