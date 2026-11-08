"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for ip in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                            if abs(s - expected) > 1e-9:
                                return False
    return True

def _is_half_int(x):
    return abs(x * 2 - round(x * 2)) < 1e-9

def _round_to_half_int(x):
    return round(x * 2) / 2.0

def _schoolbook(n, m, p):
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

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    A is 2x2, B is 2x2, C = A*B is 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_strassen_2x3x6():
    """
    Build a Strassen-based decomposition for 2x3x6.
    
    Strategy: Split the 6 columns of B into three 2-column blocks.
    For each 2-column block, we have A(2x3) * B_block(3x2) = C_block(2x2).
    
    For each block, split A into two parts:
      - A_left (2x2): columns 0,1 of A
      - A_right (2x1): column 2 of A
    
    Use Strassen (rank 7) for A_left * B_block_left (2x2 * 2x2 part)
    Use schoolbook (rank 2) for A_right * B_block_right (2x1 * 1x2 part)
    
    But wait - B_block is 3x2, so we need to split B_block too:
      - B_block_left (2x2): rows 0,1 of B_block
      - B_block_right (1x2): row 2 of B_block
    
    So for each of the 3 blocks:
      - Strassen on A[:,0:2] * B_block[0:2,:] : rank 7
      - Schoolbook on A[:,2:3] * B_block[2:3,:] : rank 2 (2x1 * 1x2)
    
    Total per block: 7 + 2 = 9
    Total: 3 * 9 = 27
    
    Actually, let me reconsider. The 2x3x2 problem:
    A is 2x3, B is 3x2, C is 2x2.
    
    Split A = [A1 | A2] where A1 is 2x2 (cols 0,1) and A2 is 2x1 (col 2)
    Split B = [B1; B2] where B1 is 2x2 (rows 0,1) and B2 is 1x2 (row 2)
    
    C = A1*B1 + A2*B2
    
    A1*B1 is 2x2x2, rank 7 (Strassen)
    A2*B2 is (2x1)*(1x2), rank 2 (schoolbook: 2 scalar products)
    
    Total per block: 7 + 2 = 9
    For 3 blocks: 3 * 9 = 27
    
    But we can potentially share the A2*B2 computation across blocks if we're clever.
    Actually no, each block has different B2 values.
    
    Let me also consider: can we do better than 9 per block?
    The 2x3x2 tensor has rank at most 10 (known), and at least 10 (known lower bound for some cases).
    Actually the rank of 2x3x2 matrix multiplication is known to be 10.
    
    Wait, let me reconsider. With Strassen + schoolbook split, I get 9 per block.
    But is that correct? Let me verify the tensor structure.
    
    For block b (columns 2b, 2b+1 of the full 6-column B):
    A is still 2x3 (all 3 columns)
    B_block is 3x2 (rows 0,1,2 of B, columns 2b, 2b+1)
    C_block is 2x2 (rows 0,1 of C, columns 2b, 2b+1)
    
    C_block[i,k] = sum_j A[i,j] * B_block[j,k]
    
    Split: A = [A1 | A2], B_block = [B1; B2]
    C_block = A1*B1 + A2*B2
    
    A1*B1: 2x2 * 2x2, rank 7 via Strassen
    A2*B2: 2x1 * 1x2, rank 2 via schoolbook
    
    Total: 9 per block, 27 total.
    """
    n, m, p = 2, 3, 6
    U_dim = n * m  # 6
    V_dim = m * p  # 18
    W_dim = p * n  # 12
    
    rank_per_block = 9
    total_rank = 3 * rank_per_block  # 27
    
    U_total = np.zeros((U_dim, total_rank))
    V_total = np.zeros((V_dim, total_rank))
    W_total = np.zeros((W_dim, total_rank))
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for b in range(3):
        base_col = b * rank_per_block
        
        # Strassen part: A[:, 0:2] * B_block[0:2, :]
        # U: rows i*3+j for i in 0,1 and j in 0,1 -> rows 0,1,3,4
        # In U_total, these are the same rows (A is shared across all blocks)
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U_total[i*3+j, base_col + c] = U_s[i*2+j, c]
        
        # V: rows j*6 + (2b+k) for j in 0,1 and k in 0,1
        # In V_s, rows are j*2+k for j in 0,1 and k in 0,1
        for c in range(7):
            for j in range(2):
                for k in range(2):
                    V_total[j*6 + (2*b + k), base_col + c] = V_s[j*2+k, c]
        
        # W: rows (2b+k)*2 + i for k in 0,1 and i in 0,1
        # In W_s, rows are k*2+i for k in 0,1 and i in 0,1
        for c in range(7):
            for k in range(2):
                for i in range(2):
                    W_total[(2*b + k)*2 + i, base_col + c] = W_s[k*2+i, c]
        
        # Schoolbook part: A[:, 2] * B_block[2, :]
        # This is 2x1 * 1x2, rank 2
        # A2[i] = A[i,2], B2[k] = B_block[2,k] = B[2, 2b+k]
        # C_block[i,k] += A2[i] * B2[k]
        for c in range(2):
            # c=0: k=0, c=1: k=1
            k = c
            # U: A[i,2] for i in 0,1 -> rows i*3+2
            for i in range(2):
                U_total[i*3+2, base_col + 7 + c] = 1
            # V: B[2, 2b+k] -> row 2*6 + (2b+k)
            V_total[2*6 + (2*b + k), base_col + 7 + c] = 1
            # W: C[2b+k, i] for i in 0,1 -> row (2b+k)*2 + i
            for i in range(2):
                W_total[(2*b + k)*2 + i, base_col + 7 + c] = 1
    
    return U_total, V_total, W_total

def _build_strassen_2x3x6_v2():
    """
    Alternative: try to get rank 26 or lower by being smarter about the tail.
    
    Idea: Instead of 3 separate blocks of rank 9, try to share some computations.
    
    Actually, let me try a different split:
    Split the 6 columns into 2 blocks of 3 columns each.
    Each block: A(2x3) * B_block(3x3) = C_block(2x3)
    
    For 2x3x3:
    Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    Split B_block = [B1 | B2] where B1 is 3x2, B2 is 3x1
    
    C_block = A1*B1 + A2*B2
    
    A1*B1: 2x2 * 2x3... this doesn't decompose cleanly.
    
    Let me try yet another approach:
    Split B into 6 individual columns.
    Each column: A(2x3) * B_col(3x1) = C_col(2x1)
    This is just 2 scalar products per column, rank 2 per column.
    Total: 6 * 2 = 12... but that's worse than 27.
    
    OK so the Strassen approach with 3 blocks of 2 is good.
    Let me try to see if I can reduce below 27.
    
    What if I use a 2x3x3 block decomposition?
    2 blocks of 3 columns each.
    
    For 2x3x3: A(2x3) * B(3x3) = C(2x3)
    Split A = [A1 | A2], A1 is 2x2, A2 is 2x1
    Split B = [B1 | B2], B1 is 3x2, B2 is 3x1
    
    C = A1*B1 + A2*B2
    
    A1*B1: 2x2 * 3x2... not standard.
    
    Hmm, let me think differently.
    
    What about: A(2x3) * B(3x6) = C(2x6)
    Split B into 2 blocks of 3: B = [B1 | B2], each 3x3
    C = A*B1 + A*B2
    
    For each A*B_block (2x3 * 3x3 = 2x3):
    Split A = [A1 | A2], A1 is 2x2 (cols 0,1), A2 is 2x1 (col 2)
    C_block = A1*B_block + A2*B_block
    
    A1*B_block: 2x2 * 3x3... still not clean.
    
    Let me stick with the 3x(2x3x2) approach giving rank 27, and see if I can
    find a rank-26 or lower decomposition through search.
    """
    return None  # Not implemented yet

def _try_rank_reduction(U, V, W, n, m, p, current_rank, target_rank, deadline):
    """
    Try to reduce rank from current_rank to target_rank by finding
    linear dependencies among the rank-1 terms.
    
    If we can find a set of coefficients such that some combination of
    rank-1 terms gives zero, we can eliminate those terms.
    """
    # This is complex; for now, just try direct optimization
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_U = np.array(result['U'], dtype=float)
    best_V = np.array(result['V'], dtype=float)
    best_W = np.array(result['W'], dtype=float)
    best_rank = n * m * p  # 36
    
    # Try Strassen-based decomposition: rank 27
    U_s, V_s, W_s = _build_strassen_2x3x6()
    if _verify(U_s.tolist(), V_s.tolist(), W_s.tolist(), n, m, p, 27):
        best_U = U_s
        best_V = V_s
        best_W = W_s
        best_rank = 27
    
    # Try to find even lower rank through structured search
    # Try rank 26, 25, 24, ...
    for target_rank in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a simple optimization approach
        found = False
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(attempt * 12345 + target_rank)
            
            U = np.random.randn(n*m, target_rank) * 0.2
            V = np.random.randn(m*p, target_rank) * 0.2
            W = np.random.randn(p*n, target_rank) * 0.2
            
            lr = 0.003
            for it in range(500):
                if time.time() > deadline - 30:
                    break
                
                # Sample some constraints
                num_samples = 50
                i = np.random.randint(0, n, num_samples)
                ip = np.random.randint(0, n, num_samples)
                j = np.random.randint(0, m, num_samples)
                jp = np.random.randint(0, m, num_samples)
                k = np.random.randint(0, p, num_samples)
                kp = np.random.randint(0, p, num_samples)
                
                expected = np.where((i == ip) & (j == jp) & (k == kp), 1.0, 0.0)
                actual = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :], axis=1)
                err = np.mean((actual - expected) ** 2)
                
                if err < 1e-10:
                    break
                
                # Gradient-free perturbation
                dU = np.random.randn(n*m, target_rank) * lr
                dV = np.random.randn(m*p, target_rank) * lr
                dW = np.random.randn(p*n, target_rank) * lr
                
                U_new = U + dU
                V_new = V + dV
                W_new = W + dW
                
                actual_new = np.sum(U_new[i*m+j, :] * V_new[jp*p+k, :] * W_new[kp*n+ip, :], axis=1)
                err_new = np.mean((actual_new - expected) ** 2)
                
                if err_new < err:
                    U, V, W = U_new, V_new, W_new
                    err = err_new
                else:
                    lr *= 0.95
            
            if err < 1e-8:
                U_r = np.array([[_round_to_half_int(x) for x in row] for row in U])
                V_r = np.array([[_round_to_half_int(x) for x in row] for row in V])
                W_r = np.array([[_round_to_half_int(x) for x in row] for row in W])
                
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, target_rank):
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
