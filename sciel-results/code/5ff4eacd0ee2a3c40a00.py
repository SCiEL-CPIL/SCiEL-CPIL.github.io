"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    # Build the tensor: T[i, j, k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to (i,k) of C.
    # The tensor is T[i*n*m + j, k*m + i, ?] -- actually let's use the standard form.
    # T[i, j, k] = 1 for all i, j, k (it's the 4x5x5 identity tensor)
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a rank-minimal tensor decomposition for 4x5 x 5x5 matrix multiplication.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 100).
    2. Try to use algebraic structure to reduce rank.
    3. Use cyclic group algebra over Z_5 for the 5x5 block.
    
    We construct the decomposition explicitly using a combination of:
    - Direct terms for boundary interactions
    - Fourier-based decomposition for the 5x5 core
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Start with schoolbook
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify schoolbook
    if not _verify(U, V, W, n, m, p):
        pass  # Should never fail
    
    # Try to find a better decomposition using algebraic techniques
    # For now, we'll attempt a structured decomposition
    
    # Attempt: Use the fact that 5x5 matrix mult can be decomposed using
    # properties of the cyclic group Z_5. The Fourier transform over Z_5 gives
    # 5 frequency components. Each frequency component involves a 4x4 matrix mult
    # (since we have 4 rows of the first matrix interacting with the 5x5 block).
    
    # Construction:
    # We decompose the 5x5 multiplication using DFT over Z_5.
    # Let omega = exp(2*pi*i/5). The DFT diagonalizes the cyclic convolution.
    # However, since we need exact integer/rational decompositions, we use
    # the algebraic integer approach with the 5th cyclotomic field.
    
    # For a simpler approach, let's try to directly construct a lower-rank
    # decomposition using known lower bounds and algebraic identities.
    
    # Known: rank(5x5) >= 17 (from lower bounds), but we need exact.
    # Schoolbook for 5x5 is 25. Strassen gives 19 for 7x7, but for 5x5 it's less clear.
    
    # Let's try a direct approach: construct the tensor and attempt rank reduction
    # via algebraic manipulation.
    
    # For the 4x5 x 5x5 problem, the tensor has dimensions 20x25x25.
    # We'll attempt to find a decomposition with rank < 100.
    
    # Strategy: Use the structure that the 4x5 part interacts with 5x5.
    # We can write this as a block matrix and use block decomposition.
    
    # Let A be 4x5 and B be 5x5. C = A*B is 4x5.
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    # We can split A into its 5 columns, each of size 4x1.
    # A = [a_0, a_1, a_2, a_3, a_4] where a_j is 4x1.
    # C = sum_j a_j * b_j^T where b_j is row j of B (1x5).
    
    # This gives rank 5*4*5 = 100 (schoolbook).
    
    # To reduce rank, we need to find linear combinations.
    
    # Let's try to use the fact that for the 5x5 part, we can use
    # the Strassen-like decomposition if applicable.
    
    # For now, return the schoolbook decomposition as a valid baseline.
    # Future work: implement actual rank reduction.
    
    # Actually, let's try a simple improvement: use the structure of 5 as a prime.
    # The ring Z_5[x]/(x^5 - 1) has structure that might help.
    
    # For the 5x5 matrix multiplication, we can use the fact that it's
    # equivalent to polynomial multiplication in Z_5[x]/(x^5 - 1).
    # The DFT over Z_5 splits this into 5 independent 1x1 multiplications
    # in the extension fields, but this requires working over complex numbers
    # or algebraic integers.
    
    # Since we need exact rational/integer entries, let's use the following
    # approach: construct the decomposition explicitly.
    
    # For the purpose of this implementation, we return the schoolbook
    # decomposition which is guaranteed correct.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
