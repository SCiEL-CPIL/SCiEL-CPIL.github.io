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
    
    # Try to find a better decomposition using structured search
    # We'll use a DFT-based approach over Z_5 to exploit the cyclic structure
    
    # The key insight: for matrix multiplication (A * B) where A is n x m and B is m x p,
    # we can view this as a bilinear map. The DFT over Z_p can diagonalize certain
    # convolution-like structures.
    
    # Let's try a rank reduction approach using the fact that p=5 is prime.
    # We can use the discrete Fourier transform over Z_5 to transform the problem.
    
    # Strategy: Use the DFT basis to create a decomposition that exploits
    # the algebraic structure of Z_5.
    
    # First, let's try to find a decomposition with rank < 80 using structured methods.
    
    # Approach 1: Use the fact that we can embed the problem in a larger algebraic structure
    # and use the DFT to diagonalize it.
    
    # For now, let's try a more sophisticated search that uses the DFT structure.
    
    # The DFT matrix over Z_5
    omega = complex(math.cos(2 * math.pi / 5), math.sin(2 * math.pi / 5))
    
    # Try to construct a decomposition using the DFT basis
    # This is a heuristic approach - we'll try to find a low-rank decomposition
    # that exploits the cyclic symmetry.
    
    # Let's try a different approach: use the known results for small matrix multiplication.
    # For 4x4 x 4x5, the optimal rank is not known exactly, but we can try to find
    # a decomposition with rank significantly less than 80.
    
    # Approach: Use a structured tensor decomposition based on the DFT.
    # We'll try to find a rank-20 decomposition (which would be a 4x improvement).
    
    # For this specific problem, let's try to use the algebraic structure of Z_5.
    
    # The DFT over Z_5 gives us 5 frequency components. We can use these to
    # create a decomposition that has a block structure.
    
    # Let's try to construct a decomposition using the following idea:
    # 1. Decompose the output dimension (size 5) using the DFT
    # 2. For each frequency component, solve a smaller rank-1 problem
    # 3. Combine the results
    
    # However, this is non-trivial. Let's try a more direct approach.
    
    # Approach 2: Use a greedy algorithm to find a low-rank decomposition.
    # Start with the schoolbook decomposition and try to merge terms.
    
    # Approach 3: Use the fact that the problem has a specific structure.
    # For 4x4 x 4x5, we can try to find a decomposition that exploits
    # the rectangular nature of the problem.
    
    # Let's try a structured search using the DFT basis.
    
    # The DFT matrix D over Z_5:
    # D[j][k] = omega^(j*k) for j,k in 0..4
    
    # We can use this to transform the B matrix (4x5) into the frequency domain.
    # Then the multiplication becomes a set of independent multiplications.
    
    # However, this doesn't directly give us a lower rank decomposition.
    
    # Let's try a different approach: use the known lower bounds and try to
    # find a decomposition that achieves them.
    
    # For now, let's try to find a decomposition with rank < 80 using a
    # more sophisticated search.
    
    # Approach 4: Use a randomized search to find a low-rank decomposition.
    # This is unlikely to work well for this problem, but let's try.
    
    # Actually, let's try a more structured approach.
    
    # The key insight for this problem is that we can use the DFT to create
    # a decomposition that has a block-circulant structure.
    
    # Let's try to construct a decomposition using the following method:
    # 1. Create a 5x5 DFT matrix
    # 2. Use this to transform the problem into the frequency domain
    # 3. Solve the transformed problem
    # 4. Transform back
    
    # This is a complex approach. For now, let's return the schoolbook
    # decomposition as a valid baseline.
    
    # However, let's try one more thing: use a structured decomposition
    # based on the algebraic properties of Z_5.
    
    # The group Z_5 has 5 elements and is cyclic. We can use this to create
    # a decomposition that exploits the cyclic symmetry.
    
    # For a 4x4 x 4x5 multiplication, we can try to find a decomposition
    # that has rank 20 (which is 4*5, the product of the input and output
    # dimensions).
    
    # Let's try to construct such a decomposition.
    
    # Actually, let's try a simpler approach: use the fact that we can
    # embed the 4x5 output into a 5x5 matrix and use the DFT.
    
    # For now, let's return the schoolbook decomposition.
    
    # But wait - let's try to use a more sophisticated method.
    
    # The idea is to use the DFT over Z_5 to create a decomposition that
    # has a lower rank. This is based on the fact that the DFT diagonalizes
    # circulant matrices, and we can use this to create a block structure.
    
    # Let's try to implement this.
    
    # Step 1: Create the DFT matrix over Z_5
    DFT = [[omega ** (j * k) for k in range(5)] for j in range(5)]
    
    # Step 2: Use this to create a decomposition
    # This is a heuristic - we'll try to find a decomposition that uses
    # the DFT basis vectors.
    
    # For each of the 5 frequency components, we can create a rank-4
    # decomposition (since the input is 4x4).
    # Total rank would be 5 * 4 = 20.
    
    # Let's try to construct this.
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 16 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 20 x 20
    
    # For each frequency component f (0 to 4), create a rank-4 block
    for f in range(5):
        base = f * 4
        for i in range(n):  # i = 0..3 (row of A)
            for j in range(m):  # j = 0..3 (column of A, row of B)
                # U[i*m + j][base + ?]
                # V[j*p + k][base + ?]
                # W[k*n + i][base + ?]
                
                # For frequency f, we use the DFT coefficient
                # Let's assign:
                # U[i*m + j][base + j] = 1  (for all i, k)
                # V[j*p + k][base + j] = DFT[f][k]
                # W[k*n + i][base + j] = 1  (for all j)
                
                # Wait, this doesn't quite work. Let me think more carefully.
                
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
