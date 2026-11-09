"""
Matrix multiplication 4x5 x 5x5: rank-minimal exact tensor decomposition.

Strategy:
- Use Strassen's rank-7 decomposition for 2x2x2.
- Split the 5x5 dimension into 2+3 and 3+2 blocks.
- For 2x3 and 3x2 multiplications, use schoolbook (rank 6) since no better
  exact decomposition is readily available.
- For 3x3x3, use the known rank-23 decomposition (or schoolbook rank 27).
- Combine blocks to get the full 4x5 x 5x5 decomposition.

The key insight is that we can decompose:
  4x5 * 5x5 = (4x2 * 2x5) + (4x3 * 3x5) for the A dimension split
  and similarly for B.

Actually, let me think more carefully. The tensor for (n,m,p) matrix multiplication
can be decomposed by splitting any dimension. If we split m into m1+m2:
  r(n, m1+m2, p) <= r(n, m1, p) + r(n, m2, p)

Similarly for n and p.

So for (4,5,5):
  Split m=5 into 2+3: r(4,5,5) <= r(4,2,5) + r(4,3,5)
  
  For r(4,2,5): split n=4 into 2+2: r(4,2,5) <= r(2,2,5) + r(2,2,5) = 2*r(2,2,5)
  For r(2,2,5): split p=5 into 2+3: r(2,2,5) <= r(2,2,2) + r(2,2,3) = 7 + r(2,2,3)
  For r(2,2,3): schoolbook gives 12, but maybe we can do better.
  
  For r(4,3,5): split n=4 into 2+2: r(4,3,5) <= 2*r(2,3,5)
  For r(2,3,5): split p=5 into 2+3: r(2,3,5) <= r(2,3,2) + r(2,3,3)
  For r(2,3,2): schoolbook = 12
  For r(2,3,3): schoolbook = 18, or maybe better.

This is getting complex. Let me just use a direct approach:

1. Build the tensor T of shape (20, 25, 20) with the correct entries.
2. Use a known low-rank decomposition if available.
3. Otherwise, use optimization to find a good decomposition.

Given the time constraints, let me focus on a practical approach:
- Use Strassen for 2x2x2 blocks
- Use schoolbook for the rest
- Combine them

Let me implement a block-based decomposition that uses Strassen where possible.
"""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit

    # Build the target tensor T of shape (n*m, m*p, n*p)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k*n+i) with matching indices
    T = np.zeros((n * m, m * p, n * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i  # W[k*n+i, r] convention
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def to_output(U, V, W):
        """Convert numpy arrays to the required dict format."""
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Schoolbook decomposition
    def schoolbook(nn, mm, pp):
        rank = nn * mm * pp
        U = np.zeros((nn * mm, rank))
        V = np.zeros((mm * pp, rank))
        W = np.zeros((pp * nn, rank))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1.0
                    V[j * pp + k, col] = 1.0
                    W[k * nn + i, col] = 1.0
                    col += 1
        return U, V, W

    # Strassen decomposition for 2x2x2
    def strassen_2x2x2():
        """
        Strassen's algorithm for 2x2 matrix multiplication.
        Returns U, V, W matrices of shape (4, 7) each.
        
        The 7 products are:
        M1 = (a00+a11)*(b00+b11)
        M2 = (a10+a11)*b00
        M3 = a00*(b01-b11)
        M4 = a11*(b10-b00)
        M5 = (a00+a01)*b11
        M6 = (a10-a00)*(b00+b10)
        M7 = (a01-a11)*(b01+b11)
        
        Then:
        c00 = M1 - M4 + M5 + M7
        c01 = M3 + M4
        c10 = M2 + M5
        c11 = M1 - M2 + M3 + M6
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Indexing: a[i,j] -> i*2+j, b[i,j] -> i*2+j, c[i,j] -> i*2+j
        # But we need to use the W[k*n+i] convention for the output.
        # For 2x2x2, n=m=p=2, so c index = k*2+i where c[i,k] is the (i,k) entry.
        
        # Let's define the factors for each of the 7 Strassen products.
        # Product s: U[:,s] * V[:,s] gives the "input" combination, W[:,s] gives the "output" contribution.
        
        # M1 = (a00+a11)*(b00+b11)
        U[:, 0] = [1, 0, 0, 1]  # a00 + a11
        V[:, 0] = [1, 0, 0, 1]  # b00 + b11
        # c00 += M1, c11 += M1
        W[:, 0] = [1, 0, 0, 1]  # c00 + c11 (using c[i,k] indexing: c00=idx0, c11=idx3)
        
        # M2 = (a10+a11)*b00
        U[:, 1] = [0, 0, 1, 1]  # a10 + a11
        V[:, 1] = [1, 0, 0, 0]  # b00
        # c10 += M2, c11 -= M2
        W[:, 1] = [0, 0, 1, -1]  # c10 - c11
        
        # M3 = a00*(b01-b11)
        U[:, 2] = [1, 0, 0, 0]  # a00
        V[:, 2] = [0, 1, 0, -1]  # b01 - b11
        # c01 += M3, c11 += M3
        W[:, 2] = [0, 1, 0, 1]  # c01 + c11
        
        # M4 = a11*(b10-b00)
        U[:, 3] = [0, 0, 0, 1]  # a11
        V[:, 3] = [0, -1, 1, 0]  # b10 - b00
        # c00 -= M4, c01 += M4
        W[:, 3] = [-1, 1, 0, 0]  # -c00 + c01
        
        # M5 = (a00+a01)*b11
        U[:, 4] = [1, 1, 0, 0]  # a00 + a01
        V[:, 4] = [0, 0, 0, 1]  # b11
        # c00 += M5, c10 += M5
        W[:, 4] = [1, 0, 1, 0]  # c00 + c10
        
        # M6 = (a10-a00)*(b00+b10)
        U[:, 5] = [-1, 0, 1, 0]  # a10 - a00
        V[:, 5] = [1, 0, 1, 0]  # b00 + b10
        # c11 += M6
        W[:, 5] = [0, 0, 0, 1]  # c11
        
        # M7 = (a01-a11)*(b01+b11)
        U[:, 6] = [0, 1, 0, -1]  # a01 - a11
        V[:, 6] = [0, 1, 0, 1]  # b01 + b11
        # c00 += M7, c01 += M7
        W[:, 6] = [1, 1, 0, 0]  # c00 + c01
        
        return U, V, W

    # Verify Strassen decomposition
    def verify_strassen():
        U, V, W = strassen_2x2x2()
        # Build the 2x2x2 tensor
        T2 = np.zeros((4, 4, 4), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i * 2 + j
                    b = j * 2 + k
                    c = k * 2 + i
                    T2[a, b, c] = 1.0
        T2_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.max(np.abs(T2_recon - T2)) < 1e-10

    # Try to use a combination of Strassen and schoolbook blocks
    # Split 5 into 2+3 for both m and p dimensions
    
    # For the 2x2x2 part, use Strassen (rank 7)
    # For the 2x2x3 part, use schoolbook (rank 12)
    # For the 2x3x2 part, use schoolbook (rank 12)
    # For the 3x3x3 part, use schoolbook (rank 27)
    
    # But we need to be careful about how we combine these.
    
    # Let me try a different approach: use optimization to find a good decomposition.
    # Start with a reasonable rank and use gradient descent.
    
    def try_optimization(rank, num_iters=500, lr=0.05, seed=42):
        """Try to find a rank-r decomposition using gradient descent with restarts."""
        if time.time() > deadline - 20:
            return None
        
        rng = np.random.RandomState(seed)
        best_err = float('inf')
        best_UVW = None
        
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(n * p, rank) * 0.5
            
            for it in range(num_iters):
                if time.time() > deadline - 5:
                    break
                
                # Compute reconstruction
                T_recon = np.einsum('as,bs,cs->abc', U, V, W)
                residual = T_recon - T
                
                # Gradients
                dU = np.einsum('abc,bs,cs->as', residual, V, W)
                dV = np.einsum('abc,as,cs->bs', residual, U, W)
                dW = np.einsum('abc,as,bs->cs', residual, U, V)
                
                # Adaptive learning rate
                err = np.max(np.abs(residual))
                if err < 1e-12:
                    return U, V, W
                
                # Scale learning rate based on error
                eff_lr = lr * min(1.0, 1.0 / (1.0 + err))
                
                U -= eff_lr * dU
                V -= eff_lr * dV
                W -= eff_lr * dW
            
            err = np.max(np.abs(np.einsum('as,bs,cs->abc', U, V, W) - T))
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
        
        if best_err < 1e-10:
            return best_UVW
        return None

    # Try various ranks
    best_result = None
    best_rank = n * m * p  # 100
    
    # Try ranks from 40 to 99
    for rank in range(40, best_rank):
        if time.time() > deadline - 30:
            break
        
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            
            result = try_optimization(rank, num_iters=300, lr=0.1, seed=seed + rank * 7)
            if result is not None:
                U, V, W = result
                if verify(U, V, W):
                    best_rank = rank
                    best_result = to_output(U, V, W)
                    break
        if best_result is not None and best_rank <= rank:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    U, V, W = schoolbook(n, m, p)
    return to_output(U, V, W)
