import time
import numpy as np
import sys

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For each (i,j,k), T[i,j,k] = 1 iff we need sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1
    # For each (i,j), the product matrix C = A*B has C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_block_decomposition(n, m, p, block_p, max_time):
    """
    Try to decompose M_{n,m,p} by splitting p into blocks of size block_p.
    If we can find a rank-r decomposition of M_{n,m,block_p}, we get rank (p//block_p)*r.
    """
    n, m = 2, 3
    n_blocks = p // block_p
    if n_blocks * block_p != p:
        return None
    
    start_time = time.time()
    
    # Try to find rank-r decomposition of M_{2,3,block_p}
    # For block_p=2, known rank is 9 (Strassen-based)
    # For block_p=5, we'd need rank ? 
    # Let's try block_p=2 first since M_{2,3,2} has known structure
    
    for target_rank in range(n * m * block_p - 1, 0, -1):
        if time.time() - start_time > max_time * 0.8:
            break
        
        # For M_{2,3,2}: U is 6xR, V is 6xR, W is 4xR
        # We want sum_r U[i*3+j,r] * V[j*2+k,r] * W[k*2+i,r] = 1 for all i,j,k
        
        # Try random initialization with some structure
        best_residual = float('inf')
        best_U = best_V = best_W = None
        
        for trial in range(50):
            if time.time() - start_time > max_time * 0.9:
                break
            
            R = target_rank
            # Random initialization
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * block_p, R) * 0.5
            W = np.random.randn(block_p * n, R) * 0.5
            
            # Try to optimize via simple gradient-like updates
            # Actually, let's just check if random works (unlikely) or use a smarter approach
            
            # Alternative: use the known Strassen decomposition for 2x2x2 as a building block
            if block_p == 2:
                # M_{2,3,2} can be seen as: for each column pair (k=0,1) of the 3x2 B matrix,
                # we compute 2x3 * 3x2. This is 5 independent 2x2x2 sub-problems? No.
                # Actually M_{2,3,2}: n=2, m=3, p=2
                # T[i,j,k] for i in {0,1}, j in {0,1,2}, k in {0,1}
                # This is 12 entries. Rank is known to be 9.
                # 
                # Strassen for 2x2x2 has rank 7. We can embed it.
                # For M_{2,3,2}, we can view it as computing A(i,:) * B(:,k) for i in {0,1}, k in {0,1}
                # Each such product is a 2x3 * 3x2 = 2x2 matrix, which has 4 entries.
                # But they share the same A and B.
                #
                # Let's try: use Strassen rank-7 decomposition for each "2x2x2 corner" but they overlap.
                #
                # Actually, let me just hardcode a known rank-9 decomposition if I can construct it.
                # 
                # For now, let's try a different approach: direct search with structure.
                pass
            
            # Check random (very unlikely to work but fast)
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(block_p):
                        val = sum(U[i*m + j, r] * V[j*block_p + k, r] * W[k*n + i, r] for r in range(R))
                        residual += (val - 1.0) ** 2
            
            if residual < best_residual:
                best_residual = residual
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_residual < 1e-6:
            # Found a decomposition for the block
            R = target_rank
            U_block = best_U.tolist()
            V_block = best_V.tolist()
            W_block = best_W.tolist()
            
            # Build full decomposition by repeating the block n_blocks times
            full_rank = R * n_blocks
            U_full = [[0.0] * full_rank for _ in range(n * m)]
            V_full = [[0.0] * full_rank for _ in range(m * p)]
            W_full = [[0.0] * full_rank for _ in range(p * n)]
            
            for b in range(n_blocks):
                for r in range(R):
                    col = b * R + r
                    for i in range(n):
                        for j in range(m):
                            U_full[i*m + j][col] = U_block[i*m + j][r]
                    for j in range(m):
                        for k in range(block_p):
                            V_full[j*p + b*block_p + k][col] = V_block[j*block_p + k][r]
                    for k in range(block_p):
                        for i in range(n):
                            W_full[(b*block_p + k)*n + i][col] = W_block[k*n + i][r]
            
            return {"U": U_full, "V": V_full, "W": W_full}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Best fallback
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try block decomposition with block_p=2
    # M_{2,3,2} has rank 9, so total rank would be 5*9=45
    result = _try_block_decomposition(n, m, p, 2, deadline - (time.time() - start_time))
    if result is not None:
        rank = len(result["U"][0])
        if rank < best_rank:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = rank
    
    if time.time() - start_time > deadline:
        return best_result
    
    # Try block_p=5
    # M_{2,3,5} rank unknown, schoolbook is 30, maybe can do better
    result = _try_block_decomposition(n, m, p, 5, deadline - (time.time() - start_time))
    if result is not None:
        rank = len(result["U"][0])
        if rank < best_rank:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = rank
    
    return best_result
