"""Rewritten program for Matrix multiplication 2,2,13"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Approach: decompose B into 13 rank-1 column vectors, use bilinear identity
    # A * B_k = (A * ones) * b_k^T ... actually more carefully:
    # For each column k of B, we need A * (column k of B) = (A @ col_k)
    # This is a 2x1 matrix product: A (2x2) times col (2x1)
    # We can use a rank-4 decomposition for 2x2x1 (schoolbook)
    # But we can do better by sharing structure across columns.
    
    # Actually, the cleanest approach for 2x2x13:
    # The tensor is T[i,j,k] = C[i,k] = sum_j A[i,j] * B[j,k]
    # We need rank r such that T = sum_t U_t ⊗ V_t ⊗ W_t
    # where U_t is 2x2, V_t is 2x13, W_t is 13x2
    
    # Schoolbook rank for 2x2x13: for each (i,j,k) we have a term A[i,j]*B[j,k]*C[i,k]
    # That's 2*2*13 = 52 terms. But we can share: for fixed j, all k terms share A[i,j].
    # So rank = 2*2*13 = 52 for naive, but we can do 4*13 = 52 still.
    
    # Better: use Strassen-like on the 2x2 part for pairs of k columns.
    # 13 = 6*2 + 1, so 6 Strassen blocks (rank 7 each) + 1 schoolbook column (rank 4)
    # Total: 6*7 + 4 = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46
    
    col = 0
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1
    
    # Process 6 blocks of 2 columns using Strassen
    for blk in range(6):
        k0 = blk * 2
        k1 = blk * 2 + 1
        
        # U indices: i*m+j -> 00:0, 01:1, 10:2, 11:3
        # V indices: j*p+k -> j=0: k, j=1: 13+k
        # W indices: k*n+i -> i=0: 2k, i=1: 2k+1
        
        # M1 = (a00+a11)(b00+b11)
        add_term(
            {0: 1, 3: 1},
            {k0: 1, 13 + k1: 1},
            {2 * k0: 1, 2 * k1 + 1: 1}
        )
        
        # M2 = (a10+a11)b00
        add_term(
            {2: 1, 3: 1},
            {k0: 1},
            {2 * k0 + 1: 1, 2 * k1 + 1: -1}
        )
        
        # M3 = a00(b01-b11)
        add_term(
            {0: 1},
            {k1: 1, 13 + k1: -1},
            {2 * k1: 1, 2 * k1 + 1: 1}
        )
        
        # M4 = a11(b10-b00)
        add_term(
            {3: 1},
            {13 + k0: 1, k0: -1},
            {2 * k0: 1, 2 * k0 + 1: 1}
        )
        
        # M5 = (a00+a01)b11
        add_term(
            {0: 1, 1: 1},
            {13 + k1: 1},
            {2 * k0: -1, 2 * k1: 1}
        )
        
        # M6 = (a10-a00)(b00+b01)
        add_term(
            {2: 1, 0: -1},
            {k0: 1, k1: 1},
            {2 * k1 + 1: 1}
        )
        
        # M7 = (a01-a11)(b10+b11)
        add_term(
            {1: 1, 3: -1},
            {13 + k0: 1, 13 + k1: 1},
            {2 * k0: 1}
        )
    
    # Process last column k=12 with schoolbook (4 terms)
    k = 12
    add_term({0: 1}, {k: 1}, {2 * k: 1})
    add_term({1: 1}, {13 + k: 1}, {2 * k: 1})
    add_term({2: 1}, {k: 1}, {2 * k + 1: 1})
    add_term({3: 1}, {13 + k: 1}, {2 * k + 1: 1})
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
