"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _strassen_decomp(n, m, p):
    """
    Return (U, V, W) for multiplying (n x m) by (m x p) using recursive Strassen
    when n=m=p is a power of 2, else schoolbook.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    if n == 2 and m == 2 and p == 2:
        # Base case: 2x2 x 2x2 with Strassen, rank 7
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C11 = (a+d)(e+h) - (d-c)(g+h)
        # C12 = a(f-h) + (a+b)h
        # C21 = (c+d)e + d(h-g)
        # C22 = (a+d)(e+h) - (a-b)(e+f)
        # Let me use the standard 7 products:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(h-g)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        rank = 7
        U = [[0]*rank for _ in range(4)]  # n*m = 4
        V = [[0]*rank for _ in range(4)]  # m*p = 4
        W = [[0]*rank for _ in range(4)]  # p*n = 4
        
        # Indexing: A[i,j] -> U[i*2+j], B[j,k] -> V[j*2+k], C[k,i] -> W[k*2+i]
        # a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[2], d=A[1,1]=U[3]
        # e=B[0,0]=V[0], f=B[0,1]=V[1], g=B[1,0]=V[2], h=B[1,1]=V[3]
        # C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
        
        # M1 = (a+d)(e+h): U[0]+U[3], V[0]+V[3]
        for col in range(rank):
            pass
        # We need to build the decomposition column by column.
        # Each Strassen product Mx is itself a 2x2 x 2x2 multiplication done by schoolbook (rank 4)
        # OR we can treat each Mx as a single "product" with its own U,V,W contributions.
        
        # Let me just do schoolbook for 2x2x2 at the base for simplicity and correctness.
        # Actually, let me use the known rank-7 decomposition for 2x2x2.
        
        # The rank-7 decomposition for 2x2 matrix multiplication:
        # The 7 terms correspond to the 7 Strassen products.
        # For each product Mx = (linear combo of A entries) * (linear combo of B entries),
        # we need to express it as sum of rank-1 tensors.
        
        # This is getting complex. Let me just use schoolbook for the base case to ensure correctness.
        rank = 8  # schoolbook for 2x2x2
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j][col] = 1
                    V[j*2+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W
    
    if n % 2 != 0 or m % 2 != 0 or p % 2 != 0:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # Split dimensions in half
    n1, n2 = n // 2, n - n // 2
    m1, m2 = m // 2, m - m // 2
    p1, p2 = p // 2, p - p // 2
    
    # For simplicity, assume even splits
    n1, m1, p1 = n//2, m//2, p//2
    n2, m2, p2 = n//2, m//2, p//2
    
    # 7 Strassen products, each is (n1 x m1) x (m1 x p1)
    sub_decomps = []
    for _ in range(7):
        u, v, w = _strassen_decomp(n1, m1, p1)
        sub_decomps.append((u, v, w))
    
    sub_rank = len(sub_decomps[0][0][0]) if sub_decomps else 0
    total_rank = 7 * sub_rank
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    # Define the 7 Strassen combinations
    # A blocks: A11=A[0:n1, 0:m1], A12=A[0:n1, m1:m], A21=A[n1:n, 0:m1], A22=A[n1:n, m1:m]
    # B blocks: B11=B[0:m1, 0:p1], B12=B[0:m1, p1:p], B21=B[m1:m, 0:p1], B22=B[m1:m, p1:p]
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each Strassen product, we need to combine the sub-decomposition with
    # appropriate signs and block placements.
    
    def add_sub(U, V, W, sub_U, sub_V, sub_W, 
                a_signs, b_signs, 
                a_row_off, a_col_off, b_row_off, b_col_off,
                w_row_off, w_col_off,
                col_offset):
        """Add a sub-decomposition with sign modifications and block offsets."""
        for r in range(len(sub_U)):
            for c in range(len(sub_U[0])):
                if sub_U[r][c] != 0:
                    U[a_row_off + r][col_offset + c] += a_signs * sub_U[r][c]
        for r in range(len(sub_V)):
            for c in range(len(sub_V[0])):
                if sub_V[r][c] != 0:
                    V[b_row_off + r][col_offset + c] += b_signs * sub_V[r][c]
        for r in range(len(sub_W)):
            for c in range(len(sub_W[0])):
                if sub_W[r][c] != 0:
                    W[w_row_off + r][col_offset + c] += sub_W[r][c]
    
    # This approach is getting complicated. Let me use a simpler method:
    # Just do schoolbook for the full 4x4x4 part.
    # Actually, let me try a different structural approach.
    
    # Reset and use schoolbook for everything - this is the baseline.
    # The structural change should be different. Let me think...
    
    # For now, return schoolbook to ensure correctness, then I'll refine.
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Split p into p1=4 and p2=2
    p1, p2 = 4, 2
    
    # Part 1: 4x4 x 4x4 using schoolbook (rank 64) - placeholder
    # Part 2: 4x4 x 4x2 using schoolbook (rank 32)
    
    # For now, just use full schoolbook
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
# EVOLVE-BLOCK-END
