"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x6 matrix.
    
    Strategy:
    1. Start with schoolbook rank 72.
    2. Try to find a lower rank decomposition using random search + local optimization.
    3. The tensor T[i,j,k] = 1 if (i,j) in A, (j,k) in B, (k,i) in C, else 0.
       We want U, V, W such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    Key insight: We can use the fact that the matrix multiplication tensor has structure.
    Split j=0..3 into j=0..2 and j=3.
    - For j=0..2: 3x3x6 sub-tensor. Split k=0..5 into k=0..2 and k=3..5.
      * 3x3x3 (j=0..2, k=0..2): schoolbook rank 27
      * 3x3x3 (j=0..2, k=3..5): schoolbook rank 27
    - For j=3: 3x1x6 sub-tensor. Rank 18 (outer product).
    Total: 27+27+18 = 72. Same as schoolbook.
    
    Better approach: Use Strassen-like decompositions where possible.
    For 3x3x3, the border rank is 23 (Landsberg). But we need exact rank.
    
    Let me try a different approach: use the fact that we can share computations.
    
    Actually, let me try to find a rank-54 or lower decomposition by exploiting
    the structure more carefully.
    
    Approach: Use the "ladder" method with Strassen for 2x2x2 blocks.
    
    Split A into A_top (3x2, j=0,1) and A_bottom (3x2, j=2,3).
    Split B into B_top (2x3, k=0,1,2) and B_bottom (2x3, k=3,4,5).
    Wait, this doesn't work because j is shared between A and B.
    
    Let me try yet another approach: random search with gradient descent.
    """
    start_time = time.time()
    DEADLINE = 110
    
    def check_time():
        return time.time() - start_time < DEADLINE
    
    def verify_full(U, V, W, tol=1e-6):
        """Full verification of all entries."""
        rank = len(U[0]) if U else 0
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val - 1) > tol:
                        return False
        # Check all zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = 0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if abs(val) > tol:
                            return False
        return True
    
    def schoolbook_full():
        """Return the schoolbook decomposition with rank n*m*p = 72."""
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
    
    def build_ladder_decomposition():
        """
        Build the ladder decomposition with rank 72:
        - Part 1: 3x3x3 (j=0..2, k=0..2) rank 27
        - Part 2: 3x3x3 (j=0..2, k=3..5) rank 27
        - Part 3: 3x1x3 (j=3, k=0..2) rank 9
        - Part 4: 3x1x3 (j=3, k=3..5) rank 9
        Total: 72
        """
        total_rank = 72
        U = np.zeros((12, total_rank))
        V = np.zeros((24, total_rank))
        W = np.zeros((18, total_rank))
        
        offset = 0
        # Part 1: 3x3x3
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*4 + j, offset] = 1
                    V[j*6 + k, offset] = 1
                    W[k*3 + i, offset] = 1
                    offset += 1
        # Part 2: 3x3x3
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*4 + j, offset] = 1
                    V[j*6 + 3 + k, offset] = 1
                    W[(3+k)*3 + i, offset] = 1
                    offset += 1
        # Part 3: 3x1x3
        for i in range(3):
            for k in range(3):
                U[i*4 + 3, offset] = 1
                V[3*6 + k, offset] = 1
                W[k*3 + i, offset] = 1
                offset += 1
        # Part 4: 3x1x3
        for i in range(3):
            for k in range(3):
                U[i*4 + 3, offset] = 1
                V[3*6 + 3 + k, offset] = 1
                W[(3+k)*3 + i, offset] = 1
                offset += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook
    best_result = schoolbook_full()
    if not verify_full(best_result["U"], best_result["V"], best_result["W"]):
        raise RuntimeError("Schoolbook verification failed")
    best_rank = n * m * p
    
    # Try ladder decomposition
    if check_time():
        ladder_result = build_ladder_decomposition()
        if verify_full(ladder_result["U"], ladder_result["V"], ladder_result["W"]):
            ladder_rank = len(ladder_result["U"][0])
            if ladder_rank < best_rank:
                best_result = ladder_result
                best_rank = ladder_rank
    
    # Try random search for lower rank
    # We'll try ranks from 60 down to 50
    for target_rank in range(60, 49, -1):
        if not check_time():
            break
        if target_rank >= best_rank:
            continue
        
        # Try multiple random initializations
        for attempt in range(10):
            if not check_time():
                break
            
            np.random.seed(attempt + target_rank * 1000)
            
            # Initialize U, V, W randomly
            U = np.random.randn(12, target_rank) * 0.1
            V = np.random.randn(24, target_rank) * 0.1
            W = np.random.randn(18, target_rank) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(1000):
                if not check_time():
                    break
                
                # Compute gradients
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                # Nonzero entries: T[i,j,k] = 1
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            # Compute current value
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            err = val - 1.0
                            for r in range(target_rank):
                                U_grad[a, r] += err * V[b, r] * W[c, r]
                                V_grad[b, r] += err * U[a, r] * W[c, r]
                                W_grad[c, r] += err * U[a, r] * V[b, r]
                
                # Zero entries: T[i,j,k] = 0
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            i = a // m
                            j = a % m
                            j2 = b // p
                            k = b % p
                            k2 = c // n
                            i2 = c % n
                            if i != i2 or j != j2 or k != k2:
                                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                err = val
                                for r in range(target_rank):
                                    U_grad[a, r] += err * V[b, r] * W[c, r]
                                    V_grad[b, r] += err * U[a, r] * W[c, r]
                                    W_grad[c, r] += err * U[a, r] * V[b, r]
                
                # Update
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                # Check convergence
                if step % 100 == 0:
                    # Quick check
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                c = k * n + i
                                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-4:
                        break
            
            # Verify
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if verify_full(U_list, V_list, W_list, tol=1e-4):
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if verify_full(U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), tol=1e-6):
                    best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    best_rank = target_rank
                    break
                else:
                    # Use the un-snapped version
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
        
        if best_rank <= target_rank:
            break
    
    return best_result
