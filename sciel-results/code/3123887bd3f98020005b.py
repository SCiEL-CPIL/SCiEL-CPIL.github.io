import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def _verify(U, V, W, n=2, m=3, p=6, tol=1e-9):
    """Verify that U, V, W give exact matrix multiplication."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i,j,k form a valid product, else 0
    # Actually the tensor for matrix multiplication C = AB where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # The decomposition is: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    
    # For efficiency, check a random subset or all
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Also check that non-product entries are 0
    # T[a,b,c] should be 0 unless b's first index (j from a) matches b's second index (j from b)
    # a = i*m+j1, b = j2*p+k, c = k*n+i
    # Product entry requires j1 == j2 and the k in b matches k in c
    for i in range(n):
        for j1 in range(m):
            for j2 in range(m):
                if j1 == j2:
                    continue
                for k in range(p):
                    a = i * m + j1
                    b = j2 * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
    # Check k mismatch
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(p):
                    if k1 == k2:
                        continue
                    a = i * m + j
                    b = j * p + k1
                    c = k2 * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s to leave margin
    
    # Start with schoolbook
    best_result = _schoolbook()
    best_rank = n * m * p  # 36
    
    # Strategy: Exploit block structure. 2x3x6 can be seen as 2x(3x2)x3 + 2x(3x2)x3
    # i.e., split the 6 columns into two blocks of 3.
    # C[:, 0:3] = A * B[:, 0:3] and C[:, 3:6] = A * B[:, 3:6]
    # Each is a 2x3x3 multiplication.
    # If rank(2x3x3) = R233, then rank(2x3x6) <= 2*R233.
    # Known: rank(2x3x3) is between 14 and 15. Schoolbook is 18.
    # Let's try to find a rank-15 or lower decomposition for 2x3x3,
    # then combine two copies for 2x3x6 with rank 2*R233.
    
    # For 2x3x3: n=2, m=3, p=3, schoolbook rank = 18
    # We'll search for a lower rank decomposition of 2x3x3.
    # Then for 2x3x6, we use two independent copies:
    #   U_total = [U233, U233]  (shape 6 x 2R233)
    #   V_total: first 9 rows (j=0..2, k=0..2) use V233, next 9 rows (j=0..2, k=3..5) use V233
    #   W_total: first 6 rows (k=0..2, i=0..1) use W233, next 6 rows (k=3..5, i=0..1) use W233
    
    def find_2x3x3_low_rank():
        """Try to find a low-rank decomposition for 2x3x3 matrix multiplication."""
        n2, m2, p2 = 2, 3, 3
        sb_rank = n2 * m2 * p2  # 18
        
        # Try random search with small integer/half-integer values
        for target_rank in range(14, sb_rank):
            if time.time() > deadline - 10:
                break
            max_attempts = 2000
            for attempt in range(max_attempts):
                if time.time() > deadline - 10:
                    break
                # Random initialization with small values
                scale = random.choice([0.5, 1.0])
                U233 = np.random.randint(0, 3, size=(n2*m2, target_rank)) * scale
                V233 = np.random.randint(0, 3, size=(m2*p2, target_rank)) * scale
                W233 = np.random.randint(0, 3, size=(p2*n2, target_rank)) * scale
                
                # Verify
                if _verify(U233, V233, W233, n=n2, m=m2, p=p2, tol=1e-6):
                    return U233, V233, W233, target_rank
        return None
    
    # Try to find a 2x3x3 decomposition
    result_233 = find_2x3x3_low_rank()
    
    if result_233 is not None:
        U233, V233, W233, R233 = result_233
        R = 2 * R233
        
        # Build 2x3x6 from two copies of 2x3x3
        U_total = np.zeros((n * m, R))  # 6 x 2R233
        V_total = np.zeros((m * p, R))  # 18 x 2R233
        W_total = np.zeros((p * n, R))  # 12 x 2R233
        
        # Copy 1: k in 0..2
        U_total[:, 0:R233] = U233
        V_total[0:9, 0:R233] = V233
        W_total[0:6, 0:R233] = W233
        
        # Copy 2: k in 3..5
        U_total[:, R233:R] = U233
        V_total[9:18, R233:R] = V233
        W_total[6:12, R233:R] = W233
        
        # Verify the full 2x3x6 decomposition
        if _verify(U_total, V_total, W_total, n=n, m=m, p=p, tol=1e-6):
            if R < best_rank:
                best_rank = R
                best_result = {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
    
    # Also try direct random search for 2x3x6 with rank below 36
    if time.time() < deadline - 20:
        for target_rank in range(30, best_rank):
            if time.time() > deadline - 15:
                break
            max_attempts = 500
            for attempt in range(max_attempts):
                if time.time() > deadline - 15:
                    break
                U = np.random.randint(-2, 3, size=(n*m, target_rank)) * 0.5
                V = np.random.randint(-2, 3, size=(m*p, target_rank)) * 0.5
                W = np.random.randint(-2, 3, size=(p*n, target_rank)) * 0.5
                if _verify(U, V, W, n=n, m=m, p=p, tol=1e-6):
                    best_rank = target_rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
    
    return best_result
