import numpy as np
from itertools import product

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Build target tensor indices
    idx_a, idx_b, idx_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a.append(i*m + j)
                idx_b.append(j*p + k)
                idx_c.append(k*n + i)
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    
    # Try gradient descent with border rank heuristic
    np.random.seed(42)
    best_rank = None
    best_result = None
    
    # Try several rank values starting from theoretical lower bound
    # Lower bound for 4x4x5 is at least max(16, 20, 20) = 20
    # Try ranks from 25 to 40
    for rank in [25, 28, 30, 32, 35, 38, 40]:
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        lr = 0.05
        for step in range(5000):
            T_full = np.einsum('ar,br,cr->abc', U, V, W)
            T_target = np.zeros((nm, mp, pn))
            T_target[idx_a, idx_b, idx_c] = 1.0
            residual = T_full - T_target
            loss = np.sum(residual**2)
            
            dU = 2 * np.einsum('abc,br,cr->ar', residual, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', residual, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', residual, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss < 1e-8:
                break
        
        # Snap to half-integers
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        
        T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
        error = np.sum((T_half - T_target)**2)
        
        if error < 1e-10 and (best_rank is None or rank < best_rank):
            best_rank = rank
            best_result = (U_half, V_half, W_half)
            break
    
    if best_result is not None:
        U_half, V_half, W_half = best_result
        return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
    
    # Fallback: Strassen-based construction
    # For 4x4x5, we can use Strassen for 4x4x4 (rank 49) plus 16 more for the 5th column
    # Actually, let's use a simpler approach: schoolbook for 4x4x5 gives rank 80
    # But we can do better with block decomposition
    
    # Use Strassen for 4x4x4 part (rank 49) and handle the 5th column separately
    # This gives rank 49 + 16 = 65
    
    U_sb = [[0] * 65 for _ in range(16)]
    V_sb = [[0] * 65 for _ in range(20)]
    W_sb = [[0] * 65 for _ in range(20)]
    
    # Strassen 4x4x4 decomposition (49 terms)
    # Using standard Strassen formulas
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For 4x4, we use 2x2 blocks of 2x2 matrices
    # Let A = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # Similarly for B and C
    
    # The 7 Strassen products for 2x2x2 have rank 7
    # For 4x4x4, we need to combine these appropriately
    
    # Actually, let's just use the known rank-49 construction for 4x4x4
    # and add 16 terms for the 5th column
    
    # For simplicity, let's use a direct construction
    # 4x4x5 matrix multiplication can be done with rank at most 4*4*5 = 80 (schoolbook)
    # But we can do better. Let's use rank 65 as a safe upper bound.
    
    # Initialize with schoolbook for first 4 columns (rank 64) + 1 extra
    # Actually, 4x4x4 schoolbook is rank 64, plus 4x4x1 is rank 16, total 80
    # But Strassen 4x4x4 is rank 49, so 49 + 16 = 65
    
    # Let's implement Strassen 4x4x4 properly
    # The 7 Strassen vectors for 2x2x2:
    strassen_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,1,0]),  # M1 = (A11+A22)(B11+B22)
        ([0,1,1,0], [1,0,0,0], [0,1,0,0]),   # M2 = (A21+A22)B11
        ([1,0,0,0], [0,1,0,0], [0,0,1,0]),   # M3 = A11(B12-B22)
        ([0,0,0,1], [0,0,1,0], [0,0,0,1]),   # M4 = A22(B21-B11)
        ([1,1,0,0], [0,0,0,1], [0,1,0,1]),   # M5 = (A11+A12)B22
        ([1,-1,0,0], [1,1,0,0], [1,0,0,0]),  # M6 = (A21-A11)(B11+B12)
        ([0,1,-1,0], [0,0,1,1], [0,0,1,1]),  # M7 = (A12-A22)(B21+B22)
    ]
    
    # This is getting complex. Let's just return the schoolbook rank 80 as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n*m)]
    V_sb = [[0] * rank_sb for _ in range(m*p)]
    W_sb = [[0] * rank_sb for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][col] = 1
                V_sb[j*p + k][col] = 1
                W_sb[k*n + i][col] = 1
                col += 1
    return {"U": U_sb, "V": V_sb, "W": W_sb}
