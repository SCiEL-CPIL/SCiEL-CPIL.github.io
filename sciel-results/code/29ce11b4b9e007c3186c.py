"""Matrix multiplication 3x4x4 tensor decomposition."""
import time

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4

def _verify(U, V, W, n=N, m=M, p=P, tol=1e-9):
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                ai, aj = divmod(a, m)
                bj, bk = divmod(b, p)
                ck, ci = divmod(c, n)
                expected = 1.0 if (aj == bj and bk == ck and ai == ci) else 0.0
                val = 0.0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if abs(val - expected) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = N, M, P
    
    # Strategy: Decompose 3x4x4 as 3x3x3 core + 3x1x4 strip
    # Core (j=0,1,2): Use a rank-23 decomposition of 3x3x3
    # Strip (j=3): Rank-12 schoolbook
    # Total: 35
    
    # For the 3x3x3 core, we use a known rank-23 decomposition.
    # The 3x3x3 tensor in flat indexing (n=m=p=3):
    # a = i*3+j, b = j*3+k, c = k*3+i
    # 
    # We'll use a systematic approach: start with schoolbook (rank 27) and
    # try to merge terms using algebraic identities.
    # 
    # Actually, let's use a different approach: construct the full 3x4x4
    # decomposition directly using a combination of Strassen-like ideas.
    # 
    # For now, let's use a practical approach:
    # 1. Start with schoolbook rank 48
    # 2. Try to reduce by identifying redundant terms
    
    # Schoolbook decomposition
    rank_sb = n * m * p  # 48
    U = [[0.0] * rank_sb for _ in range(n * m)]
    V = [[0.0] * rank_sb for _ in range(m * p)]
    W = [[0.0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Try to find a lower-rank decomposition using the block structure
    # Split by j: j=0,1,2 (core, 3x3x3) and j=3 (strip, 3x1x4)
    
    # Core: j in {0,1,2}, i in {0,1,2}, k in {0,1,2,3}
    # Wait, the core is actually 3x3x4 (i:3, j:3, k:4) not 3x3x3!
    # Let me reconsider.
    
    # A is 3x4: rows i=0,1,2; cols j=0,1,2,3
    # B is 4x4: rows j=0,1,2,3; cols k=0,1,2,3
    # C is 3x4: rows i=0,1,2; cols k=0,1,2,3
    # 
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # Split: C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k] + A[i,3]*B[3,k]
    # 
    # First term: 3x3x4 tensor (i:3, j:3, k:4)
    # Second term: 3x1x4 tensor (i:3, j:1, k:4)
    
    # The 3x3x4 tensor rank: at most 3*3*4=36 schoolbook
    # The 3x1x4 tensor rank: 3*4=12 (outer product type)
    # Total schoolbook: 48
    
    # Can we do better? The 3x3x4 tensor can potentially be done in less than 36.
    # But finding an explicit low-rank decomposition is non-trivial.
    
    # Let's try a different split: split by k instead
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # For fixed i: C[i,:] = A[i,:] * B (a row vector times a matrix)
    # This is just matrix-vector multiplication for each row.
    
    # Alternative: use the fact that 4 = 2+2
    # Split j into j_low={0,1} and j_high={2,3}
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    # Each part is 3x2x4, schoolbook rank 24 each, total 48. No improvement.
    
    # Let's try to use optimization with a time limit.
    # We'll use a simple gradient-based approach with the full tensor constraint.
    
    def optimize_rank(rank, max_time=30.0):
        """Try to find a rank-r decomposition using projected gradient descent."""
        import random
        rng = random.Random(rank * 1000)
        t0 = time.time()
        
        U = [[rng.gauss(0, 0.5) for _ in range(rank)] for _ in range(n * m)]
        V = [[rng.gauss(0, 0.5) for _ in range(rank)] for _ in range(m * p)]
        W = [[rng.gauss(0, 0.5) for _ in range(rank)] for _ in range(p * n)]
        
        lr = 0.01
        best_loss = float('inf')
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        
        for it in range(50000):
            if time.time() - t0 > max_time:
                break
            
            # Compute gradient of loss = sum_{a,b,c} (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
            # Only need to consider (a,b,c) where T[a,b,c]=1 (the 48 nonzero entries)
            # and a few where T=0 to prevent leakage
            
            dU = [[0.0] * rank for _ in range(n * m)]
            dV = [[0.0] * rank for _ in range(m * p)]
            dW = [[0.0] * rank for _ in range(p * n)]
            loss = 0.0
            
            # Nonzero entries of T
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = 0.0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        err = val - 1.0
                        loss += err * err
                        for r in range(rank):
                            dU[a][r] += 2 * err * V[b][r] * W[c][r]
                            dV[b][r] += 2 * err * U[a][r] * W[c][r]
                            dW[c][r] += 2 * err * U[a][r] * V[b][r]
            
            # Sample some zero entries to keep them at 0
            zero_entries = []
            for a in range(n * m):
                for b in range(m * p):
                    aj = a % m
                    bj = b // p
                    if aj != bj:
                        for c in range(p * n):
                            zero_entries.append((a, b, c))
                            if len(zero_entries) > 200:
                                break
                if len(zero_entries) > 200:
                    break
            if len(zero_entries) > 200:
                zero_entries = rng.sample(zero_entries, 200)
            
            for a, b, c in zero_entries:
                val = 0.0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                loss += 0.1 * val * val  # Weight zero entries less
                for r in range(rank):
                    dU[a][r] += 0.2 * val * V[b][r] * W[c][r]
                    dV[b][r] += 0.2 * val * U[a][r] * W[c][r]
                    dW[c][r] += 0.2 * val * U[a][r] * V[b][r]
            
            # Update with adaptive learning rate
            if loss > best_loss * 1.5:
                lr *= 0.5
            elif loss < best_loss * 0.9 and lr < 0.1:
                lr *= 1.1
            
            for a in range(n * m):
                for r in range(rank):
                    U[a][r] -= lr * dU[a][r]
            for b in range(m * p):
                for r in range(rank):
                    V[b][r] -= lr * dV[b][r]
            for c in range(p * n):
                for r in range(rank):
                    W[c][r] -= lr * dW[c][r]
            
            if loss < best_loss:
                best_loss = loss
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            
            if best_loss < 1e-14:
                break
        
        return best_U, best_V, best_W, best_loss
    
    # Try ranks from 35 to 47
    for rank in range(35, 48):
        if time.time() > deadline - 15:
            break
        
        U_opt, V_opt, W_opt, loss = optimize_rank(rank, max_time=25.0)
        
        if loss < 1e-10:
            if _verify(U_opt, V_opt, W_opt):
                return {"U": U_opt, "V": V_opt, "W": W_opt}
    
    # Fallback: schoolbook
    return {"U": U, "V": V, "W": W}
