import time
import numpy as np

DIMENSIONS = [2, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                T_recon[i, j, k] = s
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9

def construct_tensor_slice_decomposition():
    """
    Construct a decomposition using tensor slicing method.
    
    For 2x4x4 matrix multiplication, we use the fact that we can split
    the 4x4 matrix into two 4x2 blocks or use other rectangular splits.
    
    Key insight: We can use a hybrid approach combining:
    1. Direct multiplication for some parts
    2. Strassen for 2x2 sub-blocks where beneficial
    
    The minimal rank for <2,4,4> is known to be at most 28 (Strassen),
    but we can potentially achieve 26 or 27 with clever construction.
    
    Approach: Split the 4x4 matrix B into columns [B1, B2] where each is 4x2.
    Then C = A * [B1, B2] = [A*B1, A*B2].
    Each A*Bi is 2x2, which can be done with rank 7 (Strassen) or 4 (direct).
    
    But we can do better by sharing computation between the two 2x2 multiplications.
    """
    n, m, p = 2, 4, 4
    
    # Let's try a direct construction with rank 26
    # Using the structure: we have 2 rows of A, 4 columns of A, 4 rows of B, 4 cols of B
    
    # Strategy: Use 4 Strassen decompositions for the four 2x2x2 sub-problems
    # but try to merge some of the rank-1 terms.
    
    # Actually, let's use a known better bound. The rank of <2,4,4> 
    # can be achieved via the following construction:
    
    # Split A into A = [A_left, A_right] where each is 2x2
    # Split B into B = [[B11, B12], [B21, B22]] where each is 2x2
    # Then C = A*B = [A_left*B11 + A_right*B21, A_left*B12 + A_right*B22]
    #              = [C_left, C_right]
    
    # Each Ci is 2x2, computed as sum of two 2x2 products.
    # We can use Strassen for each pair of 2x2 multiplications.
    
    # For C_left = A_left*B11 + A_right*B21:
    # This is like multiplying two "augmented" matrices.
    # We can use the identity: (X+Y)(Z+W) = XZ + XW + YZ + YW
    # But we want X*Z + Y*W, not the full product.
    
    # Alternative: use 7 multiplications for each 2x2 result,
    # but share some between left and right.
    
    # Let me construct explicitly with rank 27:
    rank = 27
    U = [[0] * rank for _ in range(n * m)]  # 8 x 27
    V = [[0] * rank for _ in range(m * p)]  # 16 x 27
    W = [[0] * rank for _ in range(p * n)]  # 8 x 27
    
    # Strassen's 7 multiplications for a 2x2 matrix product:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A10)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our problem:
    # A is 2x4: A = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
    # B is 4x4: B = [[b00, b01, b02, b03],
    #                 [b10, b11, b12, b13],
    #                 [b20, b21, b22, b23],
    #                 [b30, b31, b32, b33]]
    # C is 2x4: C = A*B
    
    # C[0,0] = a00*b00 + a01*b10 + a02*b20 + a03*b30
    # C[0,1] = a00*b01 + a01*b11 + a02*b21 + a03*b31
    # C[0,2] = a00*b02 + a01*b12 + a02*b22 + a03*b32
    # C[0,3] = a00*b03 + a01*b13 + a02*b23 + a03*b33
    # C[1,0] = a10*b00 + a11*b10 + a12*b20 + a13*b30
    # C[1,1] = a10*b01 + a11*b11 + a12*b21 + a13*b31
    # C[1,2] = a10*b02 + a11*b12 + a12*b22 + a13*b32
    # C[1,3] = a10*b03 + a11*b13 + a12*b23 + a13*b33
    
    # Let's use the 4-block Strassen approach but try to optimize.
    # Block 1: A_left (cols 0-1) * B_left (rows 0-1) -> contributes to C[:, 0:2]
    # Block 2: A_left (cols 0-1) * B_right (rows 2-3) -> contributes to C[:, 2:4]
    # Block 3: A_right (cols 2-3) * B_left (rows 0-1) -> contributes to C[:, 0:2]
    # Block 4: A_right (cols 2-3) * B_right (rows 2-3) -> contributes to C[:, 2:4]
    
    # Each block is a 2x2 * 2x2 -> 2x2 multiplication (rank 7 each)
    # Total: 4 * 7 = 28
    
    # To reduce below 28, we need to share some multiplications.
    
    # Let's try a different split: treat it as (2x4)*(4x4) = 2x4
    # Use the fact that we only have 2 rows in A.
    
    # New approach: Use 4 multiplications for each output column.
    # For column k of C: C[:,k] = A * B[:,k] where B[:,k] is 4x1.
    # This is a 2x4 * 4x1 = 2x1 multiplication, which has rank 4 (direct) or can be done with fewer.
    
    # Actually, 2x4 * 4x1 can be done with rank 4 (it's just 8 scalar mults reduced to 4).
    # Wait, that's not right. The bilinear complexity of <2,4,1> is 4.
    
    # So for 4 output columns, we need 4 * 4 = 16 multiplications? No, that's not how it works.
    
    # Let me think differently. The tensor <2,4,4> can be viewed as:
    # T[i,j,k] = sum_l A[i,l] B[l,k]
    # This is equivalent to the tensor product of <2,4,1> and <1,4,4> in some sense.
    
    # Known result: rank(<2,4,4>) <= 28 (Strassen)
    # Lower bound: max(2*4, 4*4, 4*2) = 16 (but this is weak)
    # Better lower bound from Strassen: at least 2*4 + 4*4 - 4 = 24? Not sure.
    
    # Let me just implement the 28-rank Strassen and try to find linear dependencies.
    
    # For now, return the standard Strassen construction with rank 28
    # and let the reduction step find any improvements.
    
    return None  # Fall back to Strassen

def construct_strassen():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen_steps = [
        { 'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1} },
        { 'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1} },
        { 'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1} },
        { 'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1} },
        { 'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1} },
        { 'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1} },
        { 'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1} }
    ]

    def place_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gi = lr
                gj = a_col_start + lc
                U[gi * m + gj][col] += coeff
            for local_idx, coeff in step['v'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gi = b_row_start + lr
                gj = b_col_start + lc
                V[gi * p + gj][col] += coeff
            for local_idx, coeff in step['w'].items():
                lr = local_idx // 2
                lc = local_idx % 2
                gk = c_col_start + lc
                gi = lr
                W[gk * n + gi][col] += coeff

    place_block(0, 0, 0, 0, 0)
    place_block(0, 0, 2, 2, 7)
    place_block(2, 2, 0, 0, 14)
    place_block(2, 2, 2, 2, 21)

    return U, V, W

def try_reduce_rank(U, V, W, n=2, m=4, p=4, max_time=10):
    """
    Attempt to reduce rank by removing columns.
    """
    start_t = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = len(U_arr[0])
    
    # Heuristic: Try removing columns with smallest L2 norm
    norms = []
    for r in range(current_rank):
        norm = np.linalg.norm(U_arr[:, r]) + np.linalg.norm(V_arr[:, r]) + np.linalg.norm(W_arr[:, r])
        norms.append((norm, r))
    norms.sort()
    
    for _, r_to_remove in norms:
        if time.time() - start_t > max_time:
            break
        if current_rank <= 24: # Don't go below 24, unlikely to succeed and save time
            break
            
        mask = [i for i in range(current_rank) if i != r_to_remove]
        new_U = U_arr[:, mask]
        new_V = V_arr[:, mask]
        new_W = W_arr[:, mask]
        
        # Verify
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(mask)):
                        s += new_U[i*m+j, r] * new_V[j*p+k, r] * new_W[k*n+i, r]
                    T_recon[i, j, k] = s
        
        T_target = np.ones((n, m, p), dtype=float)
        if np.max(np.abs(T_recon - T_target)) < 1e-9:
            U_arr = new_U
            V_arr = new_V
            W_arr = new_W
            current_rank = len(mask)
            # Recalculate norms
            norms = []
            for r in range(current_rank):
                norm = np.linalg.norm(U_arr[:, r]) + np.linalg.norm(V_arr[:, r]) + np.linalg.norm(W_arr[:, r])
                norms.append((norm, r))
            norms.sort()
            
    # Convert back to list of lists
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()
    return U_out, V_out, W_out

def try_tensor_slice_optimization(U, V, W, n=2, m=4, p=4, max_time=15):
    """
    Use tensor slicing to find a better decomposition.
    
    Fix A to each of the 8 basis vectors e_i (i=0..7).
    For each such A, the output C = A*B is a linear function of B.
    The rank-1 terms must span the image of this linear map.
    
    This gives constraints on the possible decompositions.
    """
    start_t = time.time()
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = len(U_arr[0])
    
    # For each basis vector e_a (a = 0..7) in the A-space:
    # C = e_a * B
    # The (i,k) entry of C is: sum_j U[a*?]... 
    
    # Actually, let's use the slicing method more carefully.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Fix i=0, j=0: T[0,0,k] = sum_r U[0, r] * V[k, r] * W[k, r]  (for k=0..3)
    # This should equal sum_l A[0,l] B[l,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    
    # The slicing gives us 8*4 = 32 equations (for each of 8 A-basis and 4 output entries per slice)
    # that must be satisfied.
    
    # Let's try to find if there's a rank-26 decomposition by solving for the coefficients.
    # This is a nonlinear system, so we'll use a heuristic approach.
    
    # Try random perturbations of the existing decomposition and see if we can reduce rank.
    np.random.seed(42)
    
    best_U, best_V, best_W = U_arr.copy(), V_arr.copy(), W_arr.copy()
    best_rank = current_rank
    
    # Try to find a rank-26 solution by projecting out one dimension
    for attempt in range(50):
        if time.time() - start_t > max_time:
            break
        
        # Try removing a random column and see if we can adjust the others to compensate
        r_to_remove = np.random.randint(current_rank)
        mask = [i for i in range(current_rank) if i != r_to_remove]
        
        new_U = U_arr[:, mask].copy()
        new_V = V_arr[:, mask].copy()
        new_W = W_arr[:, mask].copy()
        
        # Try to find small adjustments to new_U, new_V, new_W to make the decomposition work
        # This is a hard nonlinear problem, so we'll use a simple heuristic:
        # Scale the remaining terms to compensate
        
        # Compute the error if we just remove the column
        T_recon = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(mask)):
                        s += new_U[i*m+j, r] * new_V[j*p+k, r] * new_W[k*n+i, r]
                    T_recon[i, j, k] = s
        
        T_target = np.ones((n, m, p), dtype=float)
        error = np.max(np.abs(T_recon - T_target))
        
        if error < 1e-9:
            best_U, best_V, best_W = new_U, new_V, new_W
            best_rank = len(mask)
            break
        
        # If error is small, try to fix it by scaling
        if error < 0.1:
            # Scale all terms by 1/(1-error) approximately
            scale = 1.0 / (1.0 - error * 0.5)
            new_U_scaled = new_U * scale
            new_V_scaled = new_V * scale  
            new_W_scaled = new_W * scale
            
            T_recon2 = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(len(mask)):
                            s += new_U_scaled[i*m+j, r] * new_V_scaled[j*p+k, r] * new_W_scaled[k*n+i, r]
                        T_recon2[i, j, k] = s
            
            if np.max(np.abs(T_recon2 - T_target)) < 1e-9:
                best_U, best_V, best_W = new_U_scaled, new_V_scaled, new_W_scaled
                best_rank = len(mask)
                break
    
    if best_rank < current_rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    else:
        return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

def construct():
    n, m, p = DIMENSIONS
    # Get the base Strassen decomposition
    U, V, W = construct_strassen()
    
    # Verify base
    if not verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition failed verification")
        
    # Attempt to reduce rank with greedy method
    U, V, W = try_reduce_rank(U, V, W, n, m, p, max_time=10)
    
    # Try tensor slice optimization
    U, V, W = try_tensor_slice_optimization(U, V, W, n, m, p, max_time=15)
    
    return {"U": U, "V": V, "W": W}
