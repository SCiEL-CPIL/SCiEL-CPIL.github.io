"""Program for 2x3x8 matrix multiplication tensor decomposition."""
import time

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p, rank, max_trials=20):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    import random
    random.seed(12345)
    for _ in range(max_trials):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # Expected C
        C_expected = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_expected[i][k] = s
        # Computed C via decomposition
        C_comp = [[0]*p for _ in range(n)]
        for r in range(rank):
            u_val = 0
            for i in range(n):
                for j in range(m):
                    u_val += U[i*m + j][r] * A[i][j]
            v_val = 0
            for j in range(m):
                for k in range(p):
                    v_val += V[j*p + k][r] * B[j][k]
            prod = u_val * v_val
            for k in range(p):
                for i in range(n):
                    C_comp[i][k] += prod * W[k*n + i][r]
        if C_comp != C_expected:
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W, rank

def _strassen_2x2x2():
    """Return rank-7 Strassen factors for 2x2x2.
    U is 4x7 (rows indexed by i*2+j), V is 4x7 (rows indexed by j*2+k), W is 4x7 (rows indexed by k*2+i)
    """
    # Strassen factors
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    U_str = [
        [1, 0, 1, 0, 1, -1, 0],  # A11
        [0, 0, 0, 0, 1, 0, -1],  # A12
        [0, 1, 0, 0, 0, 1, 0],   # A21
        [1, 1, 0, 1, 0, 0, 1],   # A22
    ]
    V_str = [
        [1, 1, 0, 0, 0, 1, 0],   # B11
        [0, 0, -1, -1, 0, 1, 0], # B12
        [1, 0, 1, 0, 0, 0, 1],   # B21
        [1, 0, 0, 0, 1, 0, 1],   # B22
    ]
    W_str = [
        [-1, 0, 1, 1, 0, 0, 0],  # C11 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],   # C12 (k=1,i=0)
        [0, 1, 0, 0, 0, 1, 0],   # C21 (k=0,i=1)
        [1, -1, 0, -1, 1, 0, 1], # C22 (k=1,i=1)
    ]
    return U_str, V_str, W_str

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 130  # leave margin before 150s kill
    
    # First try the block decomposition approach
    # Decompose 2x3x8 as:
    # A (2x3) * B (3x8) = C (2x8)
    # Split B into B1 (3x4) and B2 (3x4)
    # Split A into rows A_top (1x3) and A_bottom (1x3) -- actually keep 2x3
    # 
    # Strategy: treat the 3x8 product as two 3x4 products.
    # For each 3x4 product, we can further decompose.
    #
    # Better approach: Use the known result that 2x3x3 has rank 11 (not 12).
    # Actually, let's use a more systematic approach.
    #
    # Key insight: 2x3x8 can be seen as 2x(3x2)x(2x4) if we group appropriately.
    # Or: decompose p=8 as 4+4, giving two independent 2x3x4 problems.
    # 2x3x4: can we do better than schoolbook 24?
    # 
    # Let's try: 2x3x4 = combine a 2x2x2 Strassen (rank 7) with remaining parts.
    # Split m=3 into m1=2, m2=1. Split p=4 into p1=2, p2=2.
    # 
    # Block form:
    # A = [A11 (2x2) | A12 (2x1)]
    # B = [B11 (2x2) ; B12 (2x2) ; B21 (1x2) ; B22 (1x2)]
    # C = A*B = [A11*B11 + A12*B21 | A11*B12 + A12*B22]
    #
    # For the top-left 2x2 block of C: use Strassen on A11*B11 (rank 7)
    # Plus additional terms for A12*B21
    #
    # This gets complex. Let me try a cleaner approach.
    #
    # Alternative: Use the fact that we can apply Strassen to the 2x2x2 core
    # and handle the extra column of A and extra columns of B separately.
    #
    # Let's define:
    # A (2x3) = [A_core (2x2) | a (2x1)]
    # B (3x8) = [B_top (2x8); b (1x8)]
    # C = A_core * B_top + a * b
    #
    # a*b is outer product: rank 1 per output element, but it's a single rank-1
    # contribution to each C[i,k]. So total rank contribution is 1 (shared).
    #
    # For A_core * B_top (2x2x8): split B_top into four 2x2 blocks.
    # Each 2x2x2 multiplication uses Strassen rank 7.
    # Total: 4*7 = 28, plus 1 for the outer product = 29.
    #
    # But wait - we can be smarter. Let's split B_top (2x8) into 2x2x4 shape.
    # 2x2x4: split the 4 columns into two 2-column groups.
    # 2x2x2 + 2x2x2 = 7 + 7 = 14 for the 2x2x4 part? No, that's if we do them separately.
    # Actually 2x2x4 means multiplying 2x2 by 2x4.
    # Split into two 2x2x2: rank 7+7=14. Or can we do better?
    # The rank of 2x2x4 is at most 14, and it's known to be 14 (since 2x2x2 is 7).
    #
    # So total: 2x2x8 = 4 * (2x2x2) = 4*7 = 28
    # Plus a*b outer product = 1
    # Total = 29
    #
    # But we also have cross terms! A_core * B_top where B_top is 2x8.
    # If we split B_top = [B1 (2x2) | B2 (2x2) | B3 (2x2) | B4 (2x2)],
    # then C = A_core*B1 + A_core*B2 + A_core*B3 + A_core*B4 + a*b
    # Each A_core*Bi is a 2x2x2 multiply, rank 7 each, total 28.
    # a*b adds rank 1.
    # Grand total: 29.
    #
    # Let's build this decomposition.
    
    if time.time() > deadline:
        U, V, W, rank = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Build the 29-rank decomposition
    # U rows: n*m = 2*3 = 6, indexed by i*3+j
    # V rows: m*p = 3*8 = 24, indexed by j*8+k
    # W rows: p*n = 8*2 = 16, indexed by k*2+i
    
    U_list = []
    V_list = []
    W_list = []
    
    U_str, V_str, W_str = _strassen_2x2x2()
    
    # For each of the 4 blocks of B (columns 0-1, 2-3, 4-5, 6-7):
    for block in range(4):
        col_offset = block * 2
        # Strassen for A_core (2x2) * B_block (2x2)
        for r in range(7):
            u = [0.0] * 6
            v = [0.0] * 24
            w = [0.0] * 16
            
            # U: A_core entries are A[i][j] for i in {0,1}, j in {0,1}
            # In our 2x3 matrix, A[i][j] is at row i*3+j
            for i in range(2):
                for j in range(2):
                    u[i*3 + j] = U_str[i*2 + j][r]
            
            # V: B_block entries are B[j][col_offset + k] for j in {0,1}, k in {0,1}
            for j in range(2):
                for k in range(2):
                    v[j*8 + (col_offset + k)] = V_str[j*2 + k][r]
            
            # W: Output C[i][col_offset + k] for i in {0,1}, k in {0,1}
            for k in range(2):
                for i in range(2):
                    w[(col_offset + k)*2 + i] = W_str[k*2 + i][r]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
    
    # Outer product a * b: a is A[:,2] (2x1), b is B[2,:] (1x8)
    # C[i][k] += A[i][2] * B[2][k]
    u_outer = [0.0] * 6
    v_outer = [0.0] * 24
    w_outer = [0.0] * 16
    for i in range(2):
        u_outer[i*3 + 2] = 1.0
    for k in range(8):
        v_outer[2*8 + k] = 1.0
        for i in range(2):
            w_outer[k*2 + i] = 1.0
    U_list.append(u_outer)
    V_list.append(v_outer)
    W_list.append(w_outer)
    
    rank = len(U_list)
    
    # Verify
    if time.time() > deadline:
        # Can't verify, return schoolbook to be safe
        U, V, W, rank = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    is_valid = _verify_decomposition(U_list, V_list, W_list, n, m, p, rank, max_trials=5)
    
    if is_valid:
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
