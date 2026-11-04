"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # First, get a valid schoolbook decomposition as baseline
    schoolbook = _schoolbook_decomposition(n, m, p)
    
    # Strategy: Use a structured algebraic extension of Strassen's decomposition.
    # We partition the 4x4x5 tensor into:
    # 1. A 3x3x3 core (indices i=0..2, j=0..2, k=0..2) - can use rank-7 Strassen
    # 2. A 1x4x4 row strip (i=3, j=0..3, k=0..3) - schoolbook rank 16
    # 3. A 4x1x5 column strip (i=0..3, j=3, k=0..4) - schoolbook rank 20
    # 4. A 1x1x5 corner (i=3, j=3, k=0..4) - schoolbook rank 5
    # 
    # However, the interactions between these blocks are not independent.
    # The full tensor has rank at most 80 (schoolbook).
    # 
    # Let's try to find a decomposition that exploits the block structure.
    # We'll use the Strassen decomposition for the 3x3x3 core and add
    # additional terms for the boundary interactions.
    
    # For now, let's try a more direct approach: use a randomized search
    # to find a low-rank decomposition.
    
    # Actually, let's try to use the known results for small matrix multiplication.
    # For 4x4 x 4x5, the optimal rank is not known exactly.
    # 
    # Let's try to find a decomposition with rank < 80 using a structured search.
    
    # Approach: Use the DFT over Z_5 to create a block-circulant structure.
    # The DFT diagonalizes circulant matrices, which can reduce the rank.
    
    omega = complex(math.cos(2 * math.pi / 5), math.sin(2 * math.pi / 5))
    DFT = [[omega ** (j * k) for k in range(5)] for j in range(5)]
    
    # For each frequency component f (0 to 4), we can create a rank-4 block
    # (since the input is 4x4). Total rank would be 5 * 4 = 20.
    
    # Let's try to construct this decomposition.
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 16 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 20 x 20
    
    # For each frequency component f, assign a rank-4 block
    for f in range(5):
        base = f * 4
        for j in range(m):  # j = 0..3
            for i in range(n):  # i = 0..3
                for k in range(p):  # k = 0..4
                    # U[i*m + j][base + j] = 1
                    # V[j*p + k][base + j] = DFT[f][k]
                    # W[k*n + i][base + j] = 1
                    # 
                    # But this doesn't quite work because the indices don't align.
                    # Let me think more carefully.
                    pass
    
    # This approach is getting complicated. Let's try a different strategy.
    
    # Strategy: Use a known low-rank decomposition for small matrix multiplication.
    # For 4x4 x 4x5, we can try to use a decomposition based on the
    # algebraic structure of the problem.
    
    # Actually, let's try a more direct approach: use a greedy algorithm
    # to find a decomposition with rank < 80.
    
    # For now, let's return the schoolbook decomposition as a valid baseline.
    # A more advanced search would be needed to find a lower-rank decomposition.
    
    return schoolbook


def _schoolbook_decomposition(n, m, p):
    """Create a schoolbook (trivial) tensor decomposition for n x m by m x p matrix multiplication."""
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


def _verify_decomposition(decomp, n, m, p, num_checks=200):
    """Verify the decomposition by checking random entries."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check some invalid positions give 0
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        # Check if (a,b,c) corresponds to a valid (i,j,k)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue  # Valid position, skip
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True
# EVOLVE-BLOCK-END
