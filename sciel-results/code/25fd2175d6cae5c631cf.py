"""Combined program: structured Strassen-based decomposition for 4x4x7 matrix multiplication."""

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all valid (i,j,k)
    
    # Verify by computing the actual multiplication for random matrices
    for _ in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # Reconstruct A and B factors
            A_c = np.zeros((n, m))
            for idx in range(n * m):
                i, j = idx // m, idx % m
                A_c[i, j] = uc[idx]
            B_c = np.zeros((m, p))
            for idx in range(m * p):
                j, k = idx // p, idx % p
                B_c[j, k] = vc[idx]
            # The contribution to C from this rank-1 term
            C_c = np.zeros((n, p))
            for idx in range(p * n):
                k, i = idx // n, idx % n
                C_c[i, k] = wc[idx]
            C_computed += C_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
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

def build_strassen_222():
    """Build Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    U[i*2+j, r] for A[i,j]
    V[j*2+k, r] for B[j,k]  
    W[k*2+i, r] for C[i,k]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products:
    # p1 = (a11 + a22)(b11 + b22)
    # p2 = (a21 + a22)b11
    # p3 = a11(b12 - b22)
    # p4 = a22(b21 - b11)
    # p5 = (a11 + a12)b22
    # p6 = (a21 - a11)(b11 + b12)
    # p7 = (a12 - a22)(b21 + b22)
    
    # c11 = p1 + p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    
    # p1 = (a11 + a22)(b11 + b22)
    U[0][0] = 1; U[3][0] = 1  # a11, a22
    V[0][0] = 1; V[3][0] = 1  # b11, b22
    W[0][0] = 1; W[3][0] = 1  # c11, c22
    
    # p2 = (a21 + a22)b11
    U[2][1] = 1; U[3][1] = 1  # a21, a22
    V[0][1] = 1  # b11
    W[1][1] = 1; W[3][1] = -1  # c21, -c22
    
    # p3 = a11(b12 - b22)
    U[0][2] = 1  # a11
    V[1][2] = 1; V[3][2] = -1  # b12, -b22
    W[2][2] = 1; W[3][2] = 1  # c12, c22
    
    # p4 = a22(b21 - b11)
    U[3][3] = 1  # a22
    V[2][3] = 1; V[0][3] = -1  # b21, -b11
    W[0][3] = 1; W[1][3] = 1  # c11, c21
    
    # p5 = (a11 + a12)b22
    U[0][4] = 1; U[1][4] = 1  # a11, a12
    V[3][4] = 1  # b22
    W[0][4] = -1; W[2][4] = 1  # -c11, c12
    
    # p6 = (a21 - a11)(b11 + b12)
    U[2][5] = 1; U[0][5] = -1  # a21, -a11
    V[0][5] = 1; V[1][5] = 1  # b11, b12
    W[3][5] = 1  # c22
    
    # p7 = (a12 - a22)(b21 + b22)
    U[1][6] = 1; U[3][6] = -1  # a12, -a22
    V[2][6] = 1; V[3][6] = 1  # b21, b22
    W[0][6] = 1  # c11
    
    return U, V, W

def build_444_decomposition():
    """Build rank-49 decomposition for 4x4x4 matrix multiplication using Kronecker product of Strassen."""
    U2, V2, W2 = build_strassen_222()
    
    U4 = [[0]*49 for _ in range(16)]  # 4*4 = 16 rows
    V4 = [[0]*49 for _ in range(16)]  # 4*4 = 16 rows
    W4 = [[0]*49 for _ in range(16)]  # 4*4 = 16 rows
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            # The U part: U4[i*4+j, r] = U2[i_hi*2+i_lo, r1] * U2[j_hi*2+j_lo, r2]
            for i_hi in range(2):
                for j_hi in range(2):
                    for i_lo in range(2):
                        for j_lo in range(2):
                            i = i_hi * 2 + i_lo
                            j = j_hi * 2 + j_lo
                            val = U2[i_hi*2 + i_lo][r1] * U2[j_hi*2 + j_lo][r2]
                            if val != 0:
                                U4[i*4 + j][r] += val
            
            # The V part: V4[j*4+k, r] = V2[j_hi*2+j_lo, r1] * V2[k_hi*2+k_lo, r2]
            for j_hi in range(2):
                for k_hi in range(2):
                    for j_lo in range(2):
                        for k_lo in range(2):
                            j = j_hi * 2 + j_lo
                            k = k_hi * 2 + k_lo
                            val = V2[j_hi*2 + j_lo][r1] * V2[k_hi*2 + k_lo][r2]
                            if val != 0:
                                V4[j*4 + k][r] += val
            
            # The W part: W4[k*4+i, r] = W2[k_hi*2+i_hi, r1] * W2[k_lo*2+i_lo, r2]
            for k_hi in range(2):
                for i_hi in range(2):
                    for k_lo in range(2):
                        for i_lo in range(2):
                            k = k_hi * 2 + k_lo
                            i = i_hi * 2 + i_lo
                            val = W2[k_hi*2 + i_hi][r1] * W2[k_lo*2 + i_lo][r2]
                            if val != 0:
                                W4[k*4 + i][r] += val
    
    return U4, V4, W4

def build_443_decomposition():
    """Build schoolbook decomposition for 4x4x3 matrix multiplication. Rank = 48."""
    rank = 4 * 4 * 3
    U = [[0]*rank for _ in range(16)]  # 4*4
    V = [[0]*rank for _ in range(12)]  # 4*3
    W = [[0]*rank for _ in range(12)]  # 3*4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4 + j][col] = 1
                V[j*3 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    return U, V, W

def _try_strassen_style(n, m, p):
    """
    Try to find a lower-rank decomposition using structured patterns.
    We use a basis of "structured" rank-1 tensors where the factors have small integer entries
    and solve for coefficients.
    """
    start_time = time.time()
    best_result = _build_schoolbook(n, m, p)
    
    # For 4x4x7, the schoolbook rank is 112. Known bounds suggest we can do better.
    # Let's try ranks from 112 down to see what's feasible.
    
    # Generate a pool of structured rank-1 tensors
    # Each tensor is defined by (u, v, w) where:
    #   u: vector of length n*m with entries in {-1, 0, 1}
    #   v: vector of length m*p with entries in {-1, 0, 1}  
    #   w: vector of length p*n with entries in {-1, 0, 1}
    
    # This is a huge search space, so we'll use a greedy approach:
    # 1. Start with the full tensor
    # 2. Iteratively find the best rank-1 approximation and subtract
    
    # Actually, let's use a different approach: use known results and try to verify
    # For 4x4x4, the tensor rank is known to be 49 (Strassen-like)
    # For 4x4x7, we can pad or use a direct approach
    
    # Let's try a direct numerical approach: 
    # Set up the system and use iterative refinement
    
    # The constraint is: for all (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p equations in rank*3 unknowns (but bilinear/cubic)
    
    # Alternative: fix U and V, solve for W (linear in W)
    # For fixed U, V: for each (k,i):
    #   W[k*n+i, c] appears in equations for all j:
    #   sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #   => W[k*n+i, c] * sum_j U[i*m+j, c] * V[j*p+k, c] = 1
    #   => W[k*n+i, c] = 1 / sum_j U[i*m+j, c] * V[j*p+k, c]
    
    # So if we fix U and V, W is determined (where the denominator is nonzero)
    
    # Let's try: parameterize U and V with random small entries, compute W, check if it works
    # and if W has nice values (integers or half-integers)
    
    np.random.seed(42)
    
    # Try various ranks
    for target_rank in range(112, 50, -1):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        for attempt in range(20):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            # Generate random U and V with small integer entries
            U_rand = np.random.randint(-2, 3, size=(n*m, target_rank))
            V_rand = np.random.randint(-2, 3, size=(m*p, target_rank))
            
            # Compute W
            W_rand = np.zeros((p*n, target_rank))
            valid = True
            for c in range(target_rank):
                for k in range(p):
                    for i in range(n):
                        denom = 0.0
                        for j in range(m):
                            denom += U_rand[i*m+j, c] * V_rand[j*p+k, c]
                        if abs(denom) < 1e-12:
                            valid = False
                            break
                        W_rand[k*n+i, c] = 1.0 / denom
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W has nice values (close to integers or half-integers)
            W_check = W_rand * 2
            if not np.allclose(W_check, np.round(W_check), atol=1e-6):
                continue
            
            # Verify the decomposition
            U_list = [[int(round(U_rand[i, c])) for c in range(target_rank)] for i in range(n*m)]
            V_list = [[int(round(V_rand[i, c])) for c in range(target_rank)] for i in range(m*p)]
            W_list = [[int(round(W_rand[i, c] * 2)) / 2.0 for c in range(target_rank)] for i in range(p*n)]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # First, try to build the Strassen-based decomposition
    try:
        # Build the 4x4x4 decomposition
        U_444, V_444, W_444 = build_444_decomposition()
        
        # Build the 4x4x3 decomposition
        U_443, V_443, W_443 = build_443_decomposition()
        
        # Now combine them:
        # The full problem is 4x4x7, where B is 4x7.
        # Split B into B_left (columns 0-3) and B_right (columns 4-6)
        # C_left = A * B_left (4x4)
        # C_right = A * B_right (4x3)
        
        # The U matrix for the full problem:
        # U_full has 16 rows (4*4) and 97 columns (49 + 48)
        # For the 4x4x4 part, U uses all 16 rows
        # For the 4x4x3 part, U also uses all 16 rows (same A matrix)
        
        rank_total = 49 + 48
        U_full = [[0]*rank_total for _ in range(16)]
        V_full = [[0]*rank_total for _ in range(28)]  # 4*7
        W_full = [[0]*rank_total for _ in range(28)]  # 7*4
        
        # Fill in the 4x4x4 part (columns 0-48)
        for row in range(16):
            for col in range(49):
                U_full[row][col] = U_444[row][col]
        
        for row in range(16):
            for col in range(49):
                V_full[row][col] = V_444[row][col]
        
        for row in range(16):
            for col in range(49):
                W_full[row][col] = W_444[row][col]
        
        # Fill in the 4x4x3 part (columns 49-96)
        # The B_right part corresponds to columns 4-6 of B, which are rows 12-27 of V (since V has 4*7=28 rows, and j*7+k where j is 0-3 and k is 0-6)
        # Wait, V has shape (m*p, rank) = (4*7, rank) = (28, rank)
        # V[j*p+k, col] for B[j,k]
        # For the 4x4x3 part, we're computing A * B_right where B_right is 4x3
        # B_right[j, k'] = B[j, 4+k'] for k' in 0-2
        # So V_right[j*3+k', col] = B[j, 4+k']
        # In the full V, this corresponds to V[j*7 + (4+k'), col]
        
        for row in range(16):
            for col in range(48):
                U_full[row][49 + col] = U_443[row][col]
        
        for j in range(4):
            for k_prime in range(3):
                k_full = 4 + k_prime  # column index in the full 4x7 B
                v_row = j * 7 + k_full
                for col in range(48):
                    # Find which column in V_443 corresponds to this (j, k_prime)
                    # V_443[j*3+k_prime, col]
                    v_443_row = j * 3 + k_prime
                    V_full[v_row][49 + col] = V_443[v_443_row][col]
        
        # For the W part of the 4x4x3 decomposition:
        # W_443[k_prime*4+i, col] for C[i, 4+k_prime]
        # In the full W, this corresponds to W[(4+k_prime)*4+i, col]
        for k_prime in range(3):
            for i in range(4):
                w_full_row = (4 + k_prime) * 4 + i
                w_443_row = k_prime * 4 + i
                for col in range(48):
                    W_full[w_full_row][49 + col] = W_443[w_443_row][col]
        
        # Verify the decomposition
        if _verify_decomposition(U_full, V_full, W_full, n, m, p):
            return {"U": U_full, "V": V_full, "W": W_full}
        
    except Exception as e:
        print(f"Error in Strassen-based construction: {e}")
    
    # If the Strassen-based construction fails or verification fails, try the random search approach
    if time.time() - start_time < DEADLINE_SECONDS - 10:
        result = _try_strassen_style(n, m, p)
        if result["U"] and len(result["U"][0]) < 112:
            return result
    
    # Fallback to schoolbook
    return _build_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
