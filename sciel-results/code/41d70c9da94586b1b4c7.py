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
    U[0, 0] = 1
    U[3, 0] = 1
    V[0, 0] = 1
    V[3, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1
    U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1
    V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 = A11(B01-B10)
    U[3, 3] = 1
    V[1, 3] = 1
    V[2, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1
    U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1
    U[0, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1
    U[3, 6] = -1
    V[2, 6] = 1
    V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decompose_2x3x2():
    """
    Decompose 2x3x2 matrix multiplication: A(2x3) * B(3x2) = C(2x2).
    Rank 10 decomposition.
    U: (6, 10), V: (6, 10), W: (4, 10)
    """
    # We can view this as: C = A * B where A is 2x3, B is 3x2
    # Split A into columns: A = [A_col0, A_col1, A_col2]
    # C = A_col0 * B_row0 + A_col1 * B_row1 + A_col2 * B_row2
    # Each term is outer product of a 2x1 and 1x2 = 2x2 matrix
    # But we can do better using Strassen-like ideas.
    
    # Actually, let's use a direct construction.
    # The 2x3x2 tensor has 12 nonzero entries in the multiplication table.
    # We need at least 10 terms (known result).
    
    # Let me construct it using the fact that we can embed 2x2x2 (rank 7) 
    # and handle the extra column.
    
    # Alternative: use the decomposition directly.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Let's try to find a rank-10 decomposition by optimization or known result.
    # For now, let's use schoolbook rank 12 and see if we can improve.
    
    # Actually, let me just return a schoolbook decomposition for 2x3x2 for now
    # and focus on the overall structure.
    
    U = np.zeros((6, 12))
    V = np.zeros((6, 12))
    W = np.zeros((4, 12))
    
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*3+j, col] = 1
                V[j*2+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    
    return U, V, W

def _build_2x3x6_decomposition():
    """
    Build a decomposition for 2x3x6 by splitting into three 2x3x2 blocks.
    Each block handles 2 columns of the 6-column output.
    Total rank: 3 * 12 = 36 (schoolbook) or potentially less.
    """
    n, m, p = 2, 3, 6
    U_dim = n * m  # 6
    V_dim = m * p  # 18
    W_dim = p * n  # 12
    
    # Split into 3 blocks of 2 columns each
    # Block b handles columns 2b, 2b+1 of B and C
    U_total = np.zeros((U_dim, 36))
    V_total = np.zeros((V_dim, 36))
    W_total = np.zeros((W_dim, 36))
    
    for b in range(3):
        U_block, V_block, W_block = _decompose_2x3x2()
        # U_block: (6, 12), V_block: (6, 12), W_block: (4, 12)
        col_offset = b * 12
        
        U_total[:, col_offset:col_offset+12] = U_block
        
        # V_block: row j*2+k corresponds to B[j, 2b+k]
        # In V_total, row jp*p+k = jp*6 + (2b+k)
        for j in range(3):
            for k in range(2):
                V_total[j*6 + (2*b + k), col_offset:col_offset+12] = V_block[j*2+k, :]
        
        # W_block: row k*2+i corresponds to C[2b+k, i]
        # In W_total, row kp*n+ip = (2b+kp)*2 + ip = 2b*2 + kp*2 + ip = 4b + kp*2 + ip
        for k in range(2):
            for i in range(2):
                W_total[(2*b + k)*2 + i, col_offset:col_offset+12] = W_block[k*2+i, :]
    
    return U_total, V_total, W_total

def _optimize_tensor_decomposition(n, m, p, target_rank, deadline, max_iterations=3000):
    """
    Optimize a rank-target decomposition for the n x m x p matrix multiplication tensor.
    """
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute all index combinations
    indices = list(product(range(n), range(n), range(m), range(m), range(p), range(p)))
    
    for attempt in range(10):
        if time.time() > deadline - 20:
            break
        
        np.random.seed(attempt * 98765 + 42)
        
        U = np.random.randn(U_dim, target_rank) * 0.3
        V = np.random.randn(V_dim, target_rank) * 0.3
        W = np.random.randn(W_dim, target_rank) * 0.3
        
        lr = 0.005
        for it in range(max_iterations):
            if time.time() > deadline - 20:
                break
            
            # Compute error on a random subset
            num_samples = min(100, len(indices))
            sample_indices = np.random.choice(len(indices), num_samples, replace=False)
            
            errs = []
            for idx in sample_indices:
                i, ip, j, jp, k, kp = indices[idx]
                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                prod = U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :]
                s = np.sum(prod)
                errs.append((s - expected) ** 2)
            
            err = np.mean(errs)
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                break
            
            # Perturbation
            dU = np.random.randn(U_dim, target_rank) * lr
            dV = np.random.randn(V_dim, target_rank) * lr
            dW = np.random.randn(W_dim, target_rank) * lr
            
            U_new = U + dU
            V_new = V + dV
            W_new = W + dW
            
            # Quick check
            num_check = 30
            check_indices = np.random.choice(len(indices), num_check, replace=False)
            errs_new = []
            for idx in check_indices:
                i, ip, j, jp, k, kp = indices[idx]
                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                prod = U_new[i*m+j, :] * V_new[jp*p+k, :] * W_new[kp*n+ip, :]
                s = np.sum(prod)
                errs_new.append((s - expected) ** 2)
            
            new_err = np.mean(errs_new)
            
            if new_err < err:
                U, V, W = U_new, V_new, W_new
                err = new_err
            else:
                lr *= 0.95
    
    if best_err < 1e-8:
        U_r = np.array([[_round_to_half_int(x) for x in row] for row in best_U])
        V_r = np.array([[_round_to_half_int(x) for x in row] for row in best_V])
        W_r = np.array([[_round_to_half_int(x) for x in row] for row in best_W])
        
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, target_rank):
            return U_r, V_r, W_r
    
    return None

def _build_strassen_2x3x6():
    """
    Build a decomposition for 2x3x6 using Strassen on 2x2x2 blocks.
    
    A is 2x3, B is 3x6.
    Split A = [A0 A1 A2] where each Ai is 2x1.
    Split B into 3 blocks B0, B1, B2 each 3x2.
    
    C = A0*B0 + A1*B1 + A2*B2
    
    For each term Ai*Bj, this is (2x1)*(3x2) = 2x2.
    We can use Strassen-like ideas for each 2x2 block.
    
    Actually, let's think differently.
    A is 2x3, B is 3x6.
    We can write:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    For each pair of columns (2b, 2b+1), we compute a 2x2 block.
    Let's use Strassen for each 2x2 block, but we need to handle the 3-row structure.
    
    Alternative approach:
    Group B's rows into pairs. But B has 3 rows, so we can't cleanly pair them.
    
    Let's try:
    C = A * B
    Write A as [A_left, A_right] where A_left is 2x2 (columns 0,1) and A_right is 2x1 (column 2).
    Write B as [B_top; B_bottom] where B_top is 2x6 (rows 0,1) and B_bottom is 1x6 (row 2).
    
    C = A_left * B_top + A_right * B_bottom
    
    A_left * B_top is (2x2)*(2x6). We can use Strassen for the 2x2 part.
    A_right * B_bottom is (2x1)*(1x6), which is just 2 outer products.
    
    For A_left * B_top:
    A_left = [[a00, a01], [a10, a11]]
    B_top = [[b00, b01, ..., b05], [b10, b11, ..., b15]]
    
    C_top = A_left * B_top = 2x6 matrix.
    Split B_top into 3 blocks of 2 columns: B_top0, B_top1, B_top2 each 2x2.
    For each block, use Strassen (rank 7).
    Total for A_left * B_top: 3 * 7 = 21 rank.
    
    For A_right * B_bottom:
    A_right = [a02, a12]^T (2x1)
    B_bottom = [b20, b21, ..., b25] (1x6)
    C_bottom = A_right * B_bottom = 2x6 matrix.
    This is just 6 outer products, rank 6.
    
    Total: 21 + 6 = 27 rank.
    
    But wait, we can potentially share some computations.
    Let's just implement this and see.
    """
    n, m, p = 2, 3, 6
    U_dim = n * m  # 6
    V_dim = m * p  # 18
    W_dim = p * n  # 12
    
    total_rank = 27
    U = np.zeros((U_dim, total_rank))
    V = np.zeros((V_dim, total_rank))
    W = np.zeros((W_dim, total_rank))
    
    # Part 1: A_left * B_top using Strassen on 3 blocks of 2 columns
    # A_left uses A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,3,4 in U (i*m+j)
    # B_top uses B[0,k], B[1,k] for k=0..5 -> indices 0..5, 6..11 in V (j*p+k)
    
    col_offset = 0
    for block in range(3):
        k0 = block * 2  # starting column for this block
        
        # Get Strassen decomposition
        U_s, V_s, W_s = _strassen_2x2x2()
        # U_s: (4,7), V_s: (4,7), W_s: (4,7)
        # U_s[i*2+j, c] for i,j in {0,1}
        # V_s[j*2+k, c] for j,k in {0,1}
        # W_s[k*2+i, c] for k,i in {0,1}
        
        for c in range(7):
            # Map U_s indices to U indices
            # U_s row i*2+j corresponds to A[i, j] for j in {0,1}
            # In U, A[i,j] is at index i*m+j = i*3+j
            for i in range(2):
                for j in range(2):
                    U[i*3+j, col_offset+c] = U_s[i*2+j, c]
            
            # Map V_s indices to V indices
            # V_s row j*2+k corresponds to B[j, k] for j in {0,1}, k in {0,1}
            # In V, B[j, k0+k] is at index j*p + (k0+k) = j*6 + k0 + k
            for j in range(2):
                for k in range(2):
                    V[j*6 + (k0+k), col_offset+c] = V_s[j*2+k, c]
            
            # Map W_s indices to W indices
            # W_s row k*2+i corresponds to C[k, i] for k in {0,1}, i in {0,1}
            # In W, C[k0+k, i] is at index (k0+k)*n + i = (k0+k)*2 + i
            for k in range(2):
                for i in range(2):
                    W[(k0+k)*2 + i, col_offset+c] = W_s[k*2+i, c]
        
        col_offset += 7
    
    # Part 2: A_right * B_bottom
    # A_right = [A[0,2], A[1,2]] -> indices 2, 5 in U
    # B_bottom = B[2, k] for k=0..5 -> indices 12..17 in V
    # C_bottom = A_right * B_bottom -> C[k, i] for k=0..5, i=0,1
    # In W, C[k, i] is at index k*2 + i
    
    for k in range(6):
        c = col_offset + k
        # A[0,2] * B[2,k] -> C[k, 0]
        U[2, c] = 1  # A[0,2] at index 0*3+2=2
        V[12+k, c] = 1  # B[2,k] at index 2*6+k=12+k
        W[k*2+0, c] = 1  # C[k,0] at index k*2+0
        
        # A[1,2] * B[2,k] -> C[k, 1]
        # Wait, this doesn't work. We need separate terms for each i.
        # Let me redo this.
    
    # Actually, A_right * B_bottom gives:
    # C[k, 0] += A[0,2] * B[2,k]
    # C[k, 1] += A[1,2] * B[2,k]
    # These are 6 + 6 = 12 separate terms, not 6.
    # So rank is 21 + 12 = 33, not 27.
    
    # Let me fix this.
    col_offset = 21
    for k in range(6):
        # Term for C[k, 0]: A[0,2] * B[2,k]
        c = col_offset + k
        U[2, c] = 1  # A[0,2]
        V[12+k, c] = 1  # B[2,k]
        W[k*2+0, c] = 1  # C[k,0]
        
        # Term for C[k, 1]: A[1,2] * B[2,k]
        c = col_offset + 6 + k
        U[5, c] = 1  # A[1,2] at index 1*3+2=5
        V[12+k, c] = 1  # B[2,k]
        W[k*2+1, c] = 1  # C[k,1]
    
    # Total rank: 21 + 12 = 33
    # Let's verify this works.
    
    # Actually, let me just return the full 33-rank decomposition.
    U_final = U[:, :33]
    V_final = V[:, :33]
    W_final = W[:, :33]
    
    return U_final, V_final, W_final

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 36
    
    # Try Strassen-based decomposition (rank 33)
    try:
        U_strassen, V_strassen, W_strassen = _build_strassen_2x3x6()
        if _verify(U_strassen.tolist(), V_strassen.tolist(), W_strassen.tolist(), n, m, p, 33):
            return {"U": U_strassen.tolist(), "V": V_strassen.tolist(), "W": W_strassen.tolist()}
    except Exception:
        pass
    
    # Try block-based decomposition (3 x 2x3x2 blocks)
    try:
        U_block, V_block, W_block = _build_2x3x6_decomposition()
        if _verify(U_block.tolist(), V_block.tolist(), W_block.tolist(), n, m, p, 36):
            # This is just schoolbook rearranged, rank 36
            pass
    except Exception:
        pass
    
    # Try to find lower rank decompositions
    # Try ranks from 32 down to 24
    for target_rank in [32, 31, 30, 29, 28, 27, 26, 25, 24]:
        if time.time() > deadline - 30:
            break
        
        result_opt = _optimize_tensor_decomposition(n, m, p, target_rank, deadline, max_iterations=2000)
        if result_opt is not None:
            U, V, W = result_opt
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If optimization didn't find anything better, return schoolbook
    return result
# EVOLVE-BLOCK-END
