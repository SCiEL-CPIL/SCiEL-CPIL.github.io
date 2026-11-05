"""Tensor decomposition for 3x4x5 matrix multiplication using optimized Strassen blocks."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _build_decomposition():
    """Build a low-rank decomposition using Strassen 2x2x2 blocks with explicit (i,j,k) tracking."""
    n, m, p = 3, 4, 5
    terms = []  # list of (u_dict, v_dict, w_dict)

    def add_term(u, v, w):
        terms.append((u, v, w))

    def strassen222(a_i0, a_i1, b_j0, b_j1, b_k0, b_k1, c_i0, c_i1):
        """2x2x2 Strassen block."""
        # M1
        add_term({(a_i0,b_j0):1, (a_i1,b_j0):1}, {(b_j0,b_k0):1, (b_j1,b_k1):1}, {(b_k0,c_i0):1, (b_k1,c_i1):1})
        # M2
        add_term({(a_i1,b_j0):1, (a_i0,b_j0):-1}, {(b_j0,b_k0):1, (b_j1,b_k1):1}, {(b_k0,c_i1):1, (b_k1,c_i1):1})
        # M3
        add_term({(a_i0,b_j0):1, (a_i1,b_j0):1}, {(b_j1,b_k1):1, (b_j0,b_k0):-1}, {(b_k0,c_i1):1, (b_k1,c_i1):-1})
        # M4
        add_term({(a_i0,b_j0):1}, {(b_j1,b_k1):1, (b_j0,b_k1):-1}, {(b_k1,c_i0):1, (b_k1,c_i1):1})
        # M5
        add_term({(a_i1,b_j1):1}, {(b_j0,b_k1):1, (b_j1,b_k1):-1}, {(b_k1,c_i0):-1, (b_k1,c_i1):1})
        # M6
        add_term({(a_i0,b_j1):1, (a_i1,b_j1):-1}, {(b_j1,b_k1):1, (b_j1,b_k0):1}, {(b_k1,c_i0):1, (b_k1,c_i1):-1})
        # M7
        add_term({(a_i0,b_j1):1, (a_i1,b_j1):1}, {(b_j0,b_k1):1, (b_j1,b_k1):-1}, {(b_k1,c_i0):1, (b_k0,c_i0):-1})

    def elementary(a_i, b_j, b_k, c_i):
        add_term({(a_i,b_j):1}, {(b_j,b_k):1}, {(b_k,c_i):1})

    # Optimize: use Strassen for 2x2x2 blocks, handle 1x2x2 and 2x1x2 more efficiently
    # Key insight: For row 2 of C, we can combine some elementary terms
    
    # C[0:2, 0:2]
    strassen222(0, 1, 0, 1, 0, 1, 0, 1)
    strassen222(0, 1, 2, 3, 0, 1, 0, 1)

    # C[0:2, 2:4]
    strassen222(0, 1, 0, 1, 2, 3, 0, 1)
    strassen222(0, 1, 2, 3, 2, 3, 0, 1)

    # C[0:2, 4] - handle column 4 with combined terms
    # A[0:2, 0:4] * B[0:4, 4:5] -> C[0:2, 4:5]
    # This is a 2x4x1 block. We can use 2 Strassen-like operations
    # M1 = (A[0][0]+A[0][1]+A[0][2]+A[0][3]) * (B[0][4]+B[1][4]+B[2][4]+B[3][4])
    # But that's not quite right. Let's use a different approach.
    # Actually, for 2x4x1, we can split into 2x2x1 + 2x2x1
    # For 2x2x1: use 3 multiplications instead of 4
    # M1 = (a00+a10)(b00+b10) -> c0+=1, c1+=1
    # M2 = (a10-a00)(b00+b10) -> c1+=1
    # M3 = a00(b10-b00) -> c0+=1, c1+=1
    # Wait, that's not standard. Let me just use elementary for now but try to combine.
    
    # Actually, let's try a different partitioning to maximize Strassen usage.
    # Split A cols as {0,1,2} and {3}, B rows as {0,1,2} and {3}
    # Split B cols as {0,1,2} and {3,4}
    
    # Let me restart with a cleaner approach:
    # Use the fact that we can do 2x2x2 Strassen (7 mults) vs 2x2x2 schoolbook (4 mults)
    # For 3x4x5, let's try:
    # C[0:3, 0:3] = A[0:3, 0:3]*B[0:3, 0:3] + A[0:3, 3:4]*B[3:4, 0:3]
    # C[0:3, 3:5] = A[0:3, 0:3]*B[0:3, 3:5] + A[0:3, 3:4]*B[3:4, 3:5]
    
    # For 3x3x3: can we do better than 27?
    # Known: 3x3x3 has rank at most 23 (Bini et al.)
    # But implementing that is complex. Let's use Strassen blocks.
    
    # 3x3x3 = 2x2x2 (Strassen, 7) + 2x2x2 (Strassen, 7) + 2x2x2 (Strassen, 7) + 1x1x1 (1) + 1x1x1 (1) + 1x1x1 (1) + 1x1x2 (2) + 1x2x1 (2) + 2x1x1 (2)
    # That's 7+7+7+1+1+1+2+2+2 = 30, which is worse than 27.
    
    # Better: 3x3x3 = 2x2x2 (7) + 2x2x2 (7) + 2x2x2 (7) + 1x1x3 (3) + 3x1x1 (3) + 1x3x1 (3) = 30
    # Still worse.
    
    # Let's try: 3x3x3 = 2x2x2 (7) + 2x2x2 (7) + 2x2x2 (7) + 1x1x1 (1) + 1x1x1 (1) + 1x1x1 (1) + 1x1x2 (2) + 1x2x1 (2) + 2x1x1 (2) = 30
    
    # Actually, the simplest is to just use elementary for the 3x3x3 part: 27 terms
    # Or use one 2x2x2 Strassen (7) and handle the rest: 7 + 1*2*2 + 2*1*2 + 2*2*1 = 7+4+4+4 = 19? No that's not right.
    
    # Let me just stick with the original approach but try to reduce elementary terms.
    
    # Original count: 4*7 (Strassen) + 4 + 4 + 8 + 12 = 28 + 28 = 56
    # Let's see if we can reduce the elementary terms.
    
    # For C[2:3, 0:5] = A[2:3, 0:4]*B[0:4, 0:5]
    # This is 1x4x5 = 20 elementary terms
    # Can we do better? 1x4x5 = 1x2x2 (4) + 1x2x2 (4) + 1x2x1 (2) + 1x2x1 (2) = 12? No, that's still 12.
    # Actually 1x4x5 = 4*5 = 20 elementary. We can't really reduce this with standard methods.
    
    # But wait, we can combine with the Strassen blocks above!
    # The key is that C[2:3, 0:2] overlaps with the Strassen blocks for C[0:2, 0:2].
    # We can't directly combine them because they're different output positions.
    
    # Let me try a completely different approach: use a 4x4x4 embedding.
    # Pad 3x4x5 to 4x4x4 by adding zero rows/cols.
    # Then use the known rank-23 decomposition for 4x4x4? No, 4x4x4 has rank at most 26.
    # Actually, 4x4x4 multiplication has rank at most 26 (Bini et al.)
    # But that's for 4x4x4, and we need 3x4x5.
    
    # Let me just optimize the current approach by being smarter about the elementary terms.
    
    # Reset and start fresh with a better strategy
    terms.clear()
    
    # Strategy: Use Strassen for 2x2x2 blocks where possible
    # Partition: A rows {0,1}|{2}, A cols {0,1}|{2,3}, B rows {0,1}|{2,3}, B cols {0,1}|{2,3}|{4}
    
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    strassen222(0, 1, 0, 1, 0, 1, 0, 1)  # 7 terms
    strassen222(0, 1, 2, 3, 0, 1, 0, 1)  # 7 terms
    
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    strassen222(0, 1, 0, 1, 2, 3, 0, 1)  # 7 terms
    strassen222(0, 1, 2, 3, 2, 3, 0, 1)  # 7 terms
    
    # C[0:2, 4] = A[0:2, 0:4]*B[0:4, 4:5]
    # This is 2x4x1. Use 2x2x1 + 2x2x1
    # For 2x2x1, we can use 3 multiplications:
    # M1 = (a00+a10)(b00+b10) -> c0+=1, c1+=1
    # M2 = (a10-a00)(b00+b10) -> c1+=1  
    # M3 = a00(b10-b00) -> c0+=1, c1+=1
    # Wait, let me verify: c0 = a00*b00 + a10*b10, c1 = a00*b10 + a10*b00
    # M1 = (a00+a10)(b00+b10) = a00*b00 + a00*b10 + a10*b00 + a10*b10
    # M2 = (a10-a00)(b00+b10) = a10*b00 + a10*b10 - a00*b00 - a00*b10
    # M3 = a00(b10-b00) = a00*b10 - a00*b00
    # c0 = M1 - M2 - M3 = (a00*b00 + a00*b10 + a10*b00 + a10*b10) - (a10*b00 + a10*b10 - a00*b00 - a00*b10) - (a00*b10 - a00*b00)
    #    = a00*b00 + a00*b10 + a10*b00 + a10*b10 - a10*b00 - a10*b10 + a00*b00 + a00*b10 - a00*b10 + a00*b00
    #    = 3*a00*b00 + a00*b10 - a00*b10 + a10*b00 - a10*b00 + a10*b10 - a10*b10
    #    = 3*a00*b00. That's wrong.
    
    # Let me use the standard 2x2x1 decomposition:
    # c0 = a00*b00 + a10*b10
    # c1 = a00*b10 + a10*b00
    # M1 = (a00+a10)(b00+b10) = a00*b00 + a00*b10 + a10*b00 + a10*b10
    # M2 = (a10-a00)(b00+b10) = a10*b00 + a10*b10 - a00*b00 - a00*b10
    # M3 = a00(b10-b00) = a00*b10 - a00*b00
    # c0 = M1 - M2 - M3? Let me recalculate.
    # Actually, the standard is:
    # c0 = M1 - M2 - M3 where M3 = a00*b00? No.
    
    # I'll just use elementary for the 2x4x1 part: 8 terms
    elementary(0, 0, 4, 0)
    elementary(0, 1, 4, 0)
    elementary(0, 2, 4, 0)
    elementary(0, 3, 4, 0)
    elementary(1, 0, 4, 1)
    elementary(1, 1, 4, 1)
    elementary(1, 2, 4, 1)
    elementary(1, 3, 4, 1)
    
    # C[2:3, 0:5] = A[2:3, 0:4]*B[0:4, 0:5]
    # This is 1x4x5 = 20 elementary terms
    for j in range(4):
        for k in range(5):
            elementary(2, j, k, 2)
    
    # Total: 28 + 8 + 20 = 56 terms. Same as before.
    
    # Let me try to reduce the 1x4x5 part.
    # 1x4x5: can we use any structure?
    # A[2, 0:4] * B[0:4, 0:5] = C[2, 0:5]
    # This is a vector-matrix product. We can't really reduce it below 20 with standard methods.
    
    # But wait, we can combine with the Strassen blocks!
    # The issue is that C[2, 0:2] is computed separately from C[0:2, 0:2].
    # We can't combine them directly.
    
    # Let me try a different partition: include row 2 in the Strassen blocks.
    # A rows {0,1,2}, but Strassen is for 2x2.
    # We can do: A[0:2, ...] and A[2:3, ...] separately.
    
    # Alternative: use a 3x3x3 block for the main part.
    # 3x3x3 has rank at most 23, but implementing that is complex.
    
    # Let me try to use the fact that we can do 2x2x2 Strassen for multiple output positions.
    # For example, A[0:2, 0:2] * B[0:2, 0:5] -> C[0:2, 0:5]
    # This is 2x2x5. We can do it as 2x2x2 + 2x2x2 + 2x2x1
    # 2x2x2: 7 terms
    # 2x2x2: 7 terms  
    # 2x2x1: 3 terms (if we can find a 3-term decomposition)
    # Total: 17 terms for 2x2x5, vs 20 elementary.
    
    # Let me verify the 2x2x1 3-term decomposition:
    # c0 = a00*b00 + a10*b10
    # c1 = a00*b10 + a10*b00
    # M1 = (a00+a10)(b00+b10)
    # M2 = (a10-a00)(b00+b10)
    # M3 = a00(b10-b00)
    # c0 = M1 - M2 - M3
    # c1 = M1 + M2 - M3
    # Let me verify c0:
    # M1 - M2 - M3 = (a00*b00 + a00*b10 + a10*b00 + a10*b10) - (a10*b00 + a10*b10 - a00*b00 - a00*b10) - (a00*b10 - a00*b00)
    # = a00*b00 + a00*b10 + a10*b00 + a10*b10 - a10*b00 - a10*b10 + a00*b00 + a00*b10 - a00*b10 + a00*b00
    # = 3*a00*b00 + a00*b10 - a00*b10 + a10*b00 - a10*b00 + a10*b10 - a10*b10
    # = 3*a00*b00. Still wrong.
    
    # I think the issue is that 2x2x1 requires 4 multiplications in general.
    # The lower bound for 2x2x1 is 4 (it's just 4 scalar products).
    
    # So 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1 = 7 + 7 + 4 = 18 terms.
    # Vs 20 elementary. Saves 2 terms.
    
    # Let me redo with this optimization:
    terms.clear()
    
    # C[0:2, 0:2]
    strassen222(0, 1, 0, 1, 0, 1, 0, 1)  # 7
    strassen222(0, 1, 2, 3, 0, 1, 0, 1)  # 7
    
    # C[0:2, 2:4]
    strassen222(0, 1, 0, 1, 2, 3, 0, 1)  # 7
    strassen222(0, 1, 2, 3, 2, 3, 0, 1)  # 7
    
    # C[0:2, 4] = A[0:2, 0:4]*B[0:4, 4:5]
    # Split into A[0:2, 0:2]*B[0:2, 4:5] + A[0:2, 2:4]*B[2:4, 4:5]
    # Each is 2x2x1 = 4 elementary terms
    elementary(0, 0, 4, 0)
    elementary(0, 1, 4, 0)
    elementary(1, 0, 4, 1)
    elementary(1, 1, 4, 1)
    elementary(0, 2, 4, 0)
    elementary(0, 3, 4, 0)
    elementary(1, 2, 4, 1)
    elementary(1, 3, 4, 1)
    
    # C[2:3, 0:5] = A[2:3, 0:4]*B[0:4, 0:5]
    # 1x4x5 = 20 elementary
    for j in range(4):
        for k in range(5):
            elementary(2, j, k, 2)
    
    # Total: 28 + 8 + 20 = 56. Still the same.
    
    # The only way to reduce is to find a better decomposition for 1x4x5 or to combine terms.
    
    # Let me try: can we use a 2x2x2 Strassen block that includes row 2?
    # For example, A[1:3, 0:2] * B[0:2, 0:2] -> C[1:3, 0:2]
    # But C[1, 0:2] is already computed by the previous Strassen block.
    # We'd be double-counting.
    
    # I think 56 is close to optimal for this approach.
    # Let me just return this and see if we can find a better method.
    
    # Actually, let me try one more thing: use a 3x2x2 block.
    # A[0:3, 0:2] * B[0:2, 0:2] -> C[0:3, 0:2]
    # This is 3x2x2. Can we do it in less than 12 terms?
    # 3x2x2 = 2x2x2 (7) + 1x2x2 (4) = 11 terms. Saves 1 term.
    
    terms.clear()
    
    # C[0:3, 0:2] = A[0:3, 0:2]*B[0:2, 0:2] + A[0:3, 2:4]*B[2:4, 0:2]
    # A[0:3, 0:2]*B[0:2, 0:2]:
    strassen222(0, 1, 0, 1, 0, 1, 0, 1)  # A[0:2, 0:2]*B[0:2, 0:2] -> C[0:2, 0:2]
    elementary(2, 0, 0, 2)
    elementary(2, 0, 1, 2)
    elementary(2, 1, 0, 2)
    elementary(2, 1, 1, 2)
    # A[0:3, 2:4]*B[2:4, 0:2]:
    strassen222(0, 1, 2, 3, 0, 1, 0, 1)  # A[0:2, 2:4]*B[2:4, 0:2] -> C[0:2, 0:2]
    elementary(2, 2, 0, 2)
    elementary(2, 2, 1, 2)
    elementary(2, 3, 0, 2)
    elementary(2, 3, 1, 2)
    
    # C[0:3, 2:4] = A[0:3, 0:2]*B[0:2, 2:4] + A[0:3, 2:4]*B[2:4, 2:4]
    # A[0:3, 0:2]*B[0:2, 2:4]:
    strassen222(0, 1, 0, 1, 2, 3, 0, 1)  # A[0:2, 0:2]*B[0:2, 2:4] -> C[0:2, 2:4]
    elementary(2, 0, 2, 2)
    elementary(2, 0, 3, 2)
    elementary(2, 1, 2, 2)
    elementary(2, 1, 3, 2)
    # A[0:3, 2:4]*B[2:4, 2:4]:
    strassen222(0, 1, 2, 3, 2, 3, 0, 1)  # A[0:2, 2:4]*B[2:4, 2:4] -> C[0:2, 2:4]
    elementary(2, 2, 2, 2)
    elementary(2, 2, 3, 2)
    elementary(2, 3, 2, 2)
    elementary(2, 3, 3, 2)
    
    # C[0:3, 4] = A[0:3, 0:4]*B[0:4, 4:5]
    for i in range(3):
        for j in range(4):
            elementary(i, j, 4, i)
    
    # Count: 7+4+7+4 + 7+4+7+4 + 12 = 28+28+12 = 68. Worse!
    
    # The problem is that when we include row 2 in the Strassen blocks, we add 4 elementary terms per block.
    # Original: 4 Strassen blocks (28) + 8 + 20 = 56
    # New: 4 Strassen blocks (28) + 16 + 12 = 56. Same!
    
    # I'm stuck at 56. Let me just return the original approach.
    terms.clear()
    
    # C[0:2, 0:2]
    strassen222(0, 1, 0, 1, 0, 1, 0, 1)
    strassen222(0, 1, 2, 3, 0, 1, 0, 1)
    
    # C[0:2, 2:4]
    strassen222(0, 1, 0, 1, 2, 3, 0, 1)
    strassen222(0, 1, 2, 3, 2, 3, 0, 1)
    
    # C[0:2, 4]
    elementary(0, 0, 4, 0)
    elementary(0, 1, 4, 0)
    elementary(0, 2, 4, 0)
    elementary(0, 3, 4, 0)
    elementary(1, 0, 4, 1)
    elementary(1, 1, 4, 1)
    elementary(1, 2, 4, 1)
    elementary(1, 3, 4, 1)
    
    # C[2:3, 0:5]
    for j in range(4):
        for k in range(5):
            elementary(2, j, k, 2)
    
    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i*m + j][r] = val
        for (j, k), val in v.items():
            V[j*p + k][r] = val
        for (k, i), val in w.items():
            W[k*n + i][r] = val

    return U, V, W, rank

def _build_schoolbook():
    n, m, p = 3, 4, 5
    rank = n*m*p
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
    return U, V, W, rank

def construct():
    deadline = time.time() + 140.0
    U, V, W, rank = _build_decomposition()
    
    import random
    def check():
        for seed in range(20):
            rng = random.Random(seed)
            n, m, p = 3, 4, 5
            A = [[rng.randint(-5,5) for _ in range(m)] for _ in range(n)]
            B = [[rng.randint(-5,5) for _ in range(p)] for _ in range(m)]
            C = [[0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += A[i][j]*B[j][k]
                    C[i][k] = s
            Cd = [[0]*p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        for r in range(rank):
                            s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    Cd[i][k] = s
            if not all(Cd[i][k] == C[i][k] for i in range(n) for k in range(p)):
                return False
        return True
    
    if not check():
        n, m, p = 3, 4, 5
        rank = n*m*p
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
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
