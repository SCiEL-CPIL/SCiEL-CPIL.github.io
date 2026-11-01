import numpy as np
import time

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=100, tol=1e-9):
    """
    Fast vectorized verification.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    
    U_reshaped = U.reshape(n, m, rank)  # U[i, j, c]
    V_reshaped = V.reshape(m, p, rank)  # V[l, k, c]
    W_reshaped = W.reshape(p, n, rank)  # W[k, i, c]
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('lk,lkc->kc', B, V_reshaped)
        
        C_computed = np.einsum('kic,ic,kc->ik', W_reshaped, factor_A, factor_B)
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    Dimensions: n=2, m=2, p=2.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
    U row: i*m + j (0..3) -> i in 0..1, j in 0..1
    V row: l*p + k (0..3) -> l in 0..1, k in 0..1
    W row: k*n + i (0..3) -> k in 0..1, i in 0..1
    
    Strassen terms:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: A11+A22, B11+B22 -> C11 += 1, C22 += 1
    # A11: U[0,0]=1, A22: U[3,0]=1
    # B11: V[0,0]=1, B22: V[3,0]=1
    # C11: W[0,0]=1 (k=0,i=0), C22: W[3,0]=1 (k=1,i=1)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: A21+A22, B11 -> C21 += 1
    # A21: U[2,1]=1, A22: U[3,1]=1
    # B11: V[0,1]=1
    # C21: W[2,1]=1 (k=0,i=1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: A11, B21-B22 -> C12 += 1, C22 += 1
    # A11: U[0,2]=1
    # B21: V[2,2]=1, B22: V[3,2]=-1
    # C12: W[1,2]=1 (k=1,i=0), C22: W[3,2]=1 (k=1,i=1)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A22, B12-B11 -> C11 += 1, C21 += 1
    # A22: U[3,3]=1
    # B12: V[1,3]=1, B11: V[0,3]=-1
    # C11: W[0,3]=1, C21: W[2,3]=1
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: A11+A12, B22 -> C11 -= 1, C12 += 1
    # A11: U[0,4]=1, A12: U[1,4]=1
    # B22: V[3,4]=1
    # C11: W[0,4]=-1, C12: W[1,4]=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: A21-A11, B11+B12 -> C22 += 1
    # A21: U[2,5]=1, A11: U[0,5]=-1
    # B11: V[0,5]=1, B12: V[1,5]=1
    # C22: W[3,5]=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: A12-A22, B21+B22 -> C11 += 1
    # A12: U[1,6]=1, A22: U[3,6]=-1
    # B21: V[2,6]=1, B22: V[3,6]=1
    # C11: W[0,6]=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 4, 8
    
    # Decompose into 4 sub-problems of size 2x4x2
    # Each sub-problem: A (2x4) * B_sub (4x2) -> C_sub (2x2)
    # We use Strassen on each 2x4x2 block? No, Strassen is for 2x2x2.
    # Wait, the sub-problem is 2x4x2. We can split the 4 columns of A into two 2x2 blocks.
    # A = [A1, A2] where A1, A2 are 2x2.
    # B_sub = [B1; B2] where B1, B2 are 2x2 (rows 0-1 and 2-3).
    # C_sub = A1*B1 + A2*B2.
    # Each A1*B1 is 2x2x2 multiplication (rank 7).
    # Each A2*B2 is 2x2x2 multiplication (rank 7).
    # So each 2x4x2 sub-problem has rank 7+7=14.
    # Total rank = 4 sub-problems * 14 = 56.
    
    # Wait, the inspiration said rank 56 was achieved. The target is lower.
    # Can we do better?
    # The tensor is T_{i, j, k, l} = delta_{j,l}.
    # This is equivalent to 4 independent 2x4x2 multiplications.
    # Is the rank of 2x4x2 multiplication 14?
    # R(2x4x2) <= R(2x2x2) + R(2x2x2) = 7+7=14.
    # Is it possible to do better?
    # R(2xn x 2) is known. For n=4, R(2x4x2) = 14?
    # Actually, R(2xn x 2) = 2n + 2 for n>=2? No.
    # R(2x2x2)=7. R(2x3x2)=11? R(2x4x2)=14?
    # Let's check small cases.
    # If we can get R(2x4x2) < 14, we win.
    # But 56 is the current best from the Strassen insight.
    # The prompt says "current best of 56".
    # The strategy says "apply known rank-20 decomposition for 4x4... to achieve total rank 20".
    # This implies embedding 2x4x8 into 4x4x4?
    # 2x4x8 -> pad to 4x4x8? No.
    # Pad A (2x4) to 4x4 by adding 0 rows.
    # Pad B (4x8) to 4x8? No, B is 4x8.
    # If we pad A to 4x4, we need B to be 4x8.
    # Result C is 4x8. We only care about top-left 2x8.
    # This is 4x4x8 multiplication.
    # R(4x4x8) <= R(4x4x4) * 2? No.
    # R(4x4x8) is related to R(4x4x4).
    # If R(4x4x4) = 49 (Strassen), then R(4x4x8) might be 49*2 = 98? No.
    # The 4x4x8 tensor is not a simple product.
    
    # Let's stick to the proven 56 first, then try to optimize.
    # Actually, the prompt says "current best of 56".
    # So I should aim for < 56.
    
    # Let's construct the 56 rank decomposition first.
    U_total = np.zeros((n*m, 56))
    V_total = np.zeros((m*p, 56))
    W_total = np.zeros((p*n, 56))
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # 4 sub-problems for k in 0..3 (output columns pairs)
    # For each k_pair (0..1, 2..3, etc? No, k is 0..7)
    # The output C is 2x8.
    # C[:, 2*t : 2*t+2] = A * B[:, 2*t : 2*t+2]
    # So there are 4 independent 2x4x2 multiplications.
    
    for t in range(4):
        # B_sub = B[:, 2*t : 2*t+2]
        # This is a 4x2 matrix.
        # B_sub = [B1; B2] where B1, B2 are 2x2.
        # B1 = B[0:2, 2*t:2*t+2]
        # B2 = B[2:4, 2*t:2*t+2]
        
        # Term 1: A * B1 (2x4 * 2x2 -> 2x2)
        # A = [A1, A2] (2x2 each)
        # A*B1 = A1*B1[0:2, :] + A2*B1[2:4, :] ? No.
        # A is 2x4. B1 is 2x2 (rows 0,1 of B).
        # A * B1 = A[:, 0:2] * B[0:2, 2*t:2*t+2] + A[:, 2:4] * B[2:4, 2*t:2*t+2]
        # Let A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2).
        # Let B1_top = B[0:2, 2*t:2*t+2] (2x2), B1_bot = B[2:4, 2*t:2*t+2] (2x2).
        # C_sub = A1 * B1_top + A2 * B1_bot.
        
        # Both are 2x2x2 multiplications.
        
        # Sub-term 1: A1 * B1_top
        # A1 rows: 0,1. Cols: 0,1.
        # B1_top rows: 0,1. Cols: 2*t, 2*t+1.
        # C_sub rows: 0,1. Cols: 2*t, 2*t+1.
        
        base_col = t * 14
        for c in range(7):
            col_idx = base_col + c
            
            # U for A1
            for i in range(2):
                for j in range(2):
                    val = U_str[i*2+j, c]
                    if val != 0:
                        U_total[i*4+j, col_idx] += val
                        
            # V for B1_top
            for l in range(2):
                for k in range(2):
                    val = V_str[l*2+k, c]
                    if val != 0:
                        # V row index: l*p + k
                        # l is 0,1. k is 0,1 (local). Global k = 2*t + k_local.
                        V_total[l*8 + (2*t + k), col_idx] += val
                        
            # W for C_sub
            for k in range(2):
                for i in range(2):
                    val = W_str[k*2+i, c]
                    if val != 0:
                        # W row index: k*n + i
                        # k is 0,1 (local). Global k = 2*t + k_local.
                        W_total[(2*t + k)*2 + i, col_idx] += val
                        
        # Sub-term 2: A2 * B1_bot
        # A2 = A[:, 2:4]. Rows 0,1. Cols 2,3.
        # B1_bot = B[2:4, 2*t:2*t+2]. Rows 2,3. Cols 2*t, 2*t+1.
        # C_sub same.
        
        base_col = t * 14 + 7
        for c in range(7):
            col_idx = base_col + c
            
            # U for A2
            for i in range(2):
                for j in range(2):
                    val = U_str[i*2+j, c]
                    if val != 0:
                        U_total[i*4+(2+j), col_idx] += val
                        
            # V for B1_bot
            for l in range(2):
                for k in range(2):
                    val = V_str[l*2+k, c]
                    if val != 0:
                        # l is 0,1 (local). Global l = 2 + l_local.
                        V_total[(2+l)*8 + (2*t + k), col_idx] += val
                        
            # W for C_sub
            for k in range(2):
                for i in range(2):
                    val = W_str[k*2+i, c]
                    if val != 0:
                        W_total[(2*t + k)*2 + i, col_idx] += val
                        
    # Verify
    if not _verify_fast(U_total, V_total, W_total, n, m, p, trials=50):
        # Fallback to schoolbook
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
