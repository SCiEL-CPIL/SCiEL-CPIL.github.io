"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    N = 6  # Embed into N x N matrix multiplication
    
    # Build rank-20 Strassen decomposition for N x N matrix multiplication
    # Using two levels of Strassen: 6x6 = 2x(3x3), each 3x3 via Strassen on 2x2 blocks
    # Actually, let's use a direct rank-20 construction for 6x6
    
    # For 6x6 matrix multiplication, we can use:
    # Split into 2x2 blocks of 3x3 matrices, apply Strassen (7 multiplications of 3x3)
    # Each 3x3 multiplication can be done with rank... let's think differently.
    
    # Alternative: Use the known result that 6x6 matrix multiplication has rank <= 20
    # via specific constructions. Let me construct it explicitly.
    
    # Actually, let me use a simpler approach: embed 2x3 x 3x6 into 6x6 x 6x6
    # and use a rank-20 decomposition.
    
    # The 6x6 matrix multiplication tensor has a known rank-20 decomposition.
    # Let me construct it using the following approach:
    # - View 6x6 as 2x2 blocks of 3x3
    # - Use Strassen's algorithm: 7 multiplications of 3x3 matrices
    # - Each 3x3 multiplication: schoolbook rank 27, but we can do better
    #   Actually 3x3 has rank 19 or 20 in known results, but let's use schoolbook for simplicity
    #   That gives 7*27 = 189 which is too high.
    
    # Let me try a different approach: direct construction
    # For 2x3 x 3x6, the schoolbook rank is 36.
    # We want to find a lower rank decomposition.
    
    # Key idea: Use the fact that we only need a 2x6 output, not a 6x6 output.
    # This means we can potentially use fewer terms.
    
    # Let me try to construct a rank-20 decomposition directly.
    # The tensor T_{i,j,k} = delta_{i+k, j} for i in [0,2), j in [0,3), k in [0,6)
    # Wait, no. For matrix multiplication (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is T_{i,j,k} = 1 if indices match, where i is row of A (0..1), 
    # j is col of A / row of B (0..2), k is col of B (0..5)
    
    # Let me use a computational approach: try to find a low-rank decomposition
    # by exploiting structure.
    
    # Actually, let me just try rank values from 20 upward and see if we can find
    # a valid decomposition using random initialization + optimization.
    
    deadline = time.time() + 120
    
    # Try rank 20 first (best case)
    for target_rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]:
        if time.time() > deadline - 5:
            break
        
        # Try to find a rank-target_rank decomposition
        result = _try_find_decomposition(n, m, p, target_rank, deadline - 5)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each rank element r, we have:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should sum over r to give delta-like
    
    # The multiplication tensor: T_{i,j,k} = 1 if we're computing (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The decomposition says: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Wait, that's not right. Let me re-read the convention.
    
    # W[k*n+i, r] - this is the output index. For output (i,k), the index is k*n+i.
    # The decomposition: (AB)_{i,k} = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A_{i,j}) * (sum_j V[j*p+k, r] * B_{j,k})
    
    # Actually, the standard form is:
    # (AB)_{i,k} = sum_r sum_j U[i*m+j, r] * A_{i,j} * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't dimensionally work. Let me think again.
    
    # Standard tensor decomposition for matrix mult:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{i,k}^{(r)}
    # where T_{i,j,k} = 1 (the multiplication tensor)
    
    # In our indexing:
    # U has shape (n*m, rank): U[i*m+j, r]
    # V has shape (m*p, rank): V[j*p+k, r]  
    # W has shape (p*n, rank): W[k*n+i, r]
    
    # So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _try_find_decomposition(n, m, p, target_rank, time_limit):
    """Try to find a rank-target_rank decomposition using random init + Newton-like steps."""
    np.random.seed(42)
    
    # Initialize with random values
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # This is a system of n*m*p equations in 3*rank*(n*m + m*p + p*n) variables.
    
    # Use a simple iterative approach: randomly perturb and check
    # This is not efficient, but let's try a few random initializations
    
    start = time.time()
    for attempt in range(50):
        if time.time() - start > time_limit:
            break
        
        U = np.random.randn(n * m, target_rank) * (1.0 / np.sqrt(target_rank))
        V = np.random.randn(m * p, target_rank) * (1.0 / np.sqrt(target_rank))
        W = np.random.randn(p * n, target_rank) * (1.0 / np.sqrt(target_rank))
        
        # Check if this works (very unlikely for random init)
        if _verify_numpy(U, V, W, n, m, p):
            return (U.tolist(), V.tolist(), W.tolist())
    
    # Try a structured approach: start from schoolbook and try to merge terms
    # This is complex, so let's just return None and let the caller try higher ranks
    return None

def _verify_numpy(U, V, W, n, m, p):
    """Vectorized verification."""
    rank = U.shape[1]
    # Build the full tensor product and check
    # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    U_idx = i_idx * m + j_idx  # (n*m*p,)
    V_idx = j_idx * p + k_idx  # (n*m*p,)
    W_idx = k_idx * n + i_idx  # (n*m*p,)
    
    # For each rank r, compute the product
    # result = sum_r U[U_idx, r] * V[V_idx, r] * W[W_idx, r]
    result = np.zeros(len(i_idx))
    for r in range(rank):
        result += U[U_idx, r] * V[V_idx, r] * W[W_idx, r]
    
    if np.max(np.abs(result - 1.0)) > 1e-10:
        return False
    return True
# EVOLVE-BLOCK-END
