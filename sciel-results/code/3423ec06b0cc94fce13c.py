"""
Matrix multiplication 3x3x6 rank-minimal decomposition.

Strategy:
1. Try to find a rank-46 decomposition via block structure (two 3x3x3 blocks).
2. The key insight: 3x3x6 = 3x3x3 ⊗ (1x3x2) where the second tensor has rank 2.
   If rank(3x3x3) = 23, then rank(3x3x6) <= 23*2 = 46.
3. We use a structured approach: decompose into two independent 3x3x3 blocks.
4. For each 3x3x3 block, we use a known rank-23 decomposition pattern.
5. If that fails, fall back to schoolbook rank 54.

The 3x3x3 matrix multiplication tensor has rank 23 (Strassen-like).
We construct the 3x3x6 decomposition as the tensor product of:
- A rank-23 decomposition of 3x3x3 (for the matrix part)
- A rank-2 decomposition of the "row extraction" tensor 1x3x2

Actually, let me think more carefully. The 3x3x6 multiplication:
- Input: A (3x3), B (3x6)
- Output: C (3x6)
- C[i,k] = sum_j A[i,j] * B[j,k]

This is equivalent to: for each k in 0..5, C[:,k] = A * B[:,k]
So it's 6 independent 3x3 matrix-vector multiplications, but sharing A.

The tensor T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor).
The rank of this tensor is what we want to minimize.

Key observation: T_{3,3,6} can be written as T_{3,3,3} ⊗ T_{1,3,2} in a certain sense.
More precisely, if we group k into two groups of 3 (k=0,1,2 and k=3,4,5),
then for each group, the structure is 3x3x3.

But the two groups share the same A, so they're not independent.

Let me try a different approach: use the fact that rank(3,3,6) <= 2 * rank(3,3,3) = 46.
This is because we can write the 3x3x6 tensor as a sum of two 3x3x3 tensors (one for k=0,1,2 and one for k=3,4,5), each of which has the structure of 3x3x3 multiplication.

Wait, that's not quite right either. Let me think about this more carefully.

The multiplication tensor for 3x3x6 is:
T[i,j,k] = 1 for all i in 0..2, j in 0..2, k in 0..5

This tensor can be decomposed as:
T = T1 + T2
where T1[i,j,k] = 1 if k in {0,1,2} else 0, and T2[i,j,k] = 1 if k in {3,4,5} else 0.

Each of T1 and T2 is a 3x3x3 tensor (when restricted to its k-range).
T1 restricted to k=0,1,2 is the standard 3x3x3 multiplication tensor.
T2 restricted to k=3,4,5 is also the standard 3x3x3 multiplication tensor.

So rank(T) <= rank(T1) + rank(T2) = 23 + 23 = 46.

Now I need to construct an explicit rank-23 decomposition of 3x3x3.
This is non-trivial. The Strassen algorithm for 3x3 matrices uses rank 23.

Let me try to construct this. The standard Strassen algorithm for 3x3 uses 23 multiplications.
I'll need to hardcode or generate the U, V, W factors for rank 23.

Actually, let me try a simpler approach first: use random search with the block structure
constraint to see if I can find a valid rank-46 decomposition.

For the block structure:
- U has 9 rows (3*3) and 46 columns
- V has 18 rows (3*6) and 46 columns  
- W has 18 rows (6*3) and 46 columns

For the first 23 columns (block 1, k=0,1,2):
- V rows for k=3,4,5 should be 0
- W rows for k=3,4,5 should be 0

For the last 23 columns (block 2, k=3,4,5):
- V rows for k=0,1,2 should be 0
- W rows for k=0,1,2 should be 0

Let me implement this with random initialization and optimization.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 135.0  # seconds, safety margin before 150s kill


def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Fast verification using numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Check T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # For all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_block_decomposition(n, m, p, target_rank, max_iters=2000, tol=1e-6):
    """
    Try to find a decomposition with block structure.
    For 3x3x6, split k into two groups of 3.
    Each group uses target_rank//2 columns.
    """
    nm, mp, pn = n * m, m * p, p * n
    half = target_rank // 2
    rem = target_rank % 2
    
    U = np.zeros((nm, target_rank))
    V = np.zeros((mp, target_rank))
    W = np.zeros((pn, target_rank))
    
    # Block 1: k = 0,1,2 (first 3 k values)
    # Block 2: k = 3,4,5 (last 3 k values)
    
    for block in range(2):
        start = block * half + min(block, rem)
        end = start + half + (1 if block < rem else 0)
        block_size = end - start
        
        if block_size <= 0:
            continue
        
        # Random initialization
        U[:, start:end] = np.random.randn(nm, block_size) * 0.1
        V[:, start:end] = np.random.randn(mp, block_size) * 0.1
        W[:, start:end] = np.random.randn(pn, block_size) * 0.1
        
        # Zero out irrelevant V and W rows
        k_start = block * 3
        k_end = k_start + 3
        
        # V rows: j*p + k, zero out k not in [k_start, k_end)
        for j in range(m):
            for k in range(p):
                if k < k_start or k >= k_end:
                    V[j*p+k, start:end] = 0
        
        # W rows: k*n + i, zero out k not in [k_start, k_end)
        for k in range(p):
            for i in range(n):
                if k < k_start or k >= k_end:
                    W[k*n+i, start:end] = 0
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
        
        # Compute loss: sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    loss += (val - 1.0) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < tol:
            break
        
        # Simple gradient-free update
        U += lr * np.random.randn(nm, target_rank)
        V += lr * np.random.randn(mp, target_rank)
        W += lr * np.random.randn(pn, target_rank)
        
        # Re-apply zero constraints
        for block in range(2):
            start = block * half + min(block, rem)
            end = start + half + (1 if block < rem else 0)
            k_start = block * 3
            k_end = k_start + 3
            for j in range(m):
                for k in range(p):
                    if k < k_start or k >= k_end:
                        V[j*p+k, start:end] = 0
            for k in range(p):
                for i in range(n):
                    if k < k_start or k >= k_end:
                        W[k*n+i, start:end] = 0
        
        # Decay learning rate
        lr *= 0.999
    
    if best_loss < 1e-4:
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        if _verify_fast(U_list, V_list, W_list, n, m, p, tol=1e-4):
            return {"U": U_list, "V": V_list, "W": W_list, "rank": target_rank}
    
    return None


def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Try block decomposition with rank 46
    result = _try_block_decomposition(n, m, p, 46, max_iters=1500, tol=1e-6)
    if result is not None:
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Try rank 48
    if time.time() < DEADLINE - 20:
        result = _try_block_decomposition(n, m, p, 48, max_iters=1000, tol=1e-6)
        if result is not None:
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Try rank 50
    if time.time() < DEADLINE - 15:
        result = _try_block_decomposition(n, m, p, 50, max_iters=800, tol=1e-6)
        if result is not None:
            return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fall back to schoolbook
    sb = _schoolbook(n, m, p)
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}
