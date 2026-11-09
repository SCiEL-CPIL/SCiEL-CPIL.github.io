"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Split 4x4 into 2x2 blocks of size 2x2
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] where each block is 2x7
    # C = A * B, each block of C is 2x7
    
    # Strassen's algorithm for 2x2 block multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen multiplication, we need to handle the fact that
    # the "blocks" are 2x2 * 2x7 = 2x7
    # So each M_i is a sum of products of 2x2 and 2x7 matrices
    
    # Let's define the tensor decomposition directly
    # Each rank-1 term corresponds to one scalar multiplication
    
    # For a 2x2 * 2x2 multiplication, schoolbook needs 4 multiplications
    # Strassen needs 7 multiplications
    
    # For 2x2 * 2x7, we can think of it as 2x2 * (2x7)
    # Each column of the 2x7 is a 2x1 vector
    # So we need 7 independent 2x2 * 2x1 multiplications? No...
    
    # Actually, let's think of it differently.
    # A block is 2x2, B block is 2x7
    # The product is 2x7
    # We can decompose the 2x7 B block into 7 columns, each 2x1
    # But that doesn't help with Strassen directly
    
    # Alternative: treat the 7 as a parameter
    # For each of the 7 output columns, we have a 2x2 * 2x1 multiplication
    # Strassen doesn't directly apply to rectangular matrices in the standard way
    
    # Let's use a different approach: exploit the structure more directly
    # 
    # The 4x4 * 4x7 multiplication can be seen as:
    # 4 independent 4x1 * 4x7 multiplications? No, that's not right either.
    
    # Let's go back to basics and use a hybrid approach:
    # Use Strassen for the 4x4 block structure, but handle the 7 dimension
    # by treating each column of B separately
    
    # Actually, the cleanest approach:
    # Split A into 2x2 blocks (each 2x2)
    # Split B into 2x4 blocks (each 2x7)  
    # Use Strassen's 7 multiplications for the block structure
    # Each "multiplication" is a 2x2 * 2x7 product
    
    # A 2x2 * 2x7 product has rank at most 2*7 = 14 (schoolbook)
    # Can we do better? The tensor 2x2*2x7 has rank... 
    # Actually for 2x2*2x7, the rank is exactly 4 (since it's equivalent to 4 independent 2x1*2x1)
    # Wait no, that's not right either.
    
    # Let me just use schoolbook for the inner 2x2*2x7 and Strassen for the outer 2x2 structure
    # This gives: 7 (Strassen) * 4 (schoolbook for 2x2*2x7) = 28 multiplications
    # vs schoolbook: 4*4*7 = 112
    
    # But wait, can we do better on the 2x2*2x7?
    # The multiplication tensor for 2x2*2x7:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a (2,2,7) tensor
    # The rank of this tensor... for m=2, n=2, p=7
    # It's known that the rank of matrix multiplication tensor M_{m,n,p} 
    # For 2x2*2x7, since p=7 > 4, the rank is likely 4 (each output column is independent)
    # Actually no, the rank should be at most 2*7=14, and at least 4
    
    # Let's just use schoolbook for the inner products and Strassen for outer
    # Total rank: 7 * (2*2*7) / something... 
    
    # I think I'm overcomplicating this. Let me just implement Strassen properly.
    
    # Define Strassen's 7 multiplications for 2x2 blocks
    # Each multiplication involves 2x2 * 2x7 matrices
    # For each such multiplication, use schoolbook: 2*2*7 = 28 scalar mults
    # Total: 7 * 28 = 196... that's worse than schoolbook!
    
    # The issue: Strassen reduces the number of block multiplications from 4 to 7,
    # but each block multiplication is more expensive when rectangular.
    
    # For square 2x2*2x2: schoolbook=4, Strassen=7 (worse for small n)
    # Strassen wins for 4x4*4x4: schoolbook=16, Strassen=7*4=28... still worse!
    # Strassen wins for 8x8*8x8: schoolbook=64, Strassen=7*16=112... 
    # Actually Strassen's asymptotic win starts showing for larger matrices
    
    # For our 4x4*4x7 case, let's try a different decomposition.
    
    # Key insight: 4x4*4x7 can be seen as 4 independent 4x7*1x4? No.
    # Or as computing 4*7=28 outputs, each a dot product of length 4.
    
    # Let's use the fact that 7 = 4 + 3 and try to exploit structure.
    
    # Actually, let's try a simpler improvement: use the known rank for 4x4*4x4
    # which is at most 49 (Winograd) or use Strassen variants.
    
    # For 4x4*4x4, the best known rank is 49 (Winograd's algorithm)
    # For 4x4*4x7, we can extend this.
    
    # Winograd's 4x4*4x4 algorithm uses 49 multiplications.
    # For 4x4*4x7, we can use a similar approach.
    
    # Let me implement Winograd's algorithm for 4x4*4x4 and extend to 4x7.
    
    # Winograd 4x4*4x4:
    # The algorithm uses 49 scalar multiplications.
    # The key idea is to use the tensor rank of matrix multiplication.
    
    # For 4x4*4x7, one approach:
    # Split the 7 columns into groups and use Winograd for each group.
    # 7 = 4 + 3, so use Winograd for 4x4*4x4 (49 mults) and schoolbook for 4x4*4x3 (48 mults)
    # Total: 49 + 48 = 97, which is better than 112.
    
    # Or: 7 = 2 + 2 + 3, use Winograd for two 4x4*4x2 and schoolbook for 4x4*4x3
    # 4x4*4x2: rank is at most 4*2*4=32, but can we do better?
    # Actually for 4x4*4x2, the rank is at most 8 (since it's like two independent 4x4*4x1)
    # No wait, 4x4*4x1 has rank 4 (it's just a linear map)
    # So 4x4*4x2 has rank at most 8
    # 4x4*4x3 has rank at most 12
    # Total: 8 + 8 + 12 = 28? That seems too good.
    
    # I think I'm confusing myself. Let me just implement a correct decomposition.
    
    # The safest approach: use schoolbook but with some optimization.
    # Schoolbook rank = 112.
    
    # Let's try: split B into B1 (4x4) and B2 (4x3)
    # Compute C1 = A*B1 using Winograd (49 mults)
    # Compute C2 = A*B2 using schoolbook (4*4*3 = 48 mults)
    # Total: 49 + 48 = 97
    
    # Now I need to implement Winograd's 4x4*4x4 algorithm.
    
    # Winograd's algorithm for 4x4*4x4:
    # This is complex. Let me use a simpler approach first.
    
    # Alternative: use the fact that for 4x4*4x7, we can use
    # a tensor decomposition based on the structure.
    
    # Let's just go with the 97-rank approach if I can implement Winograd.
    # If not, fall back to schoolbook.
    
    # Winograd 4x4*4x4 implementation:
    # The algorithm is based on the tensor rank-49 decomposition.
    # This is quite involved to code from scratch.
    
    # Let me try a different structural approach:
    # Use the fact that 4x4*4x7 = (2x2 block) structure
    # And use a custom decomposition.
    
    # Actually, let me just implement a correct schoolbook and see if I can find
    # a simple improvement.
    
    # For now, let's implement the 97-rank approach with a simplified Winograd.
    
    # Winograd 4x4*4x4:
    # I'll use the known rank-49 decomposition.
    # The coefficients are based on the tensor decomposition.
    
    # This is getting too complex for a quick implementation.
    # Let me just use schoolbook for now and try to find a simple improvement.
    
    # Simple improvement: notice that for 4x4*4x7, we can write
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # If we fix i, this is a linear combination of the rows of B.
    # Not immediately helpful.
    
    # Let me try: use the 4x4*4x4 Winograd for the first 4 columns,
    # and schoolbook for the last 3 columns.
    
    # I need to implement Winograd 4x4*4x4.
    # Let me look up the algorithm.
    
    # Winograd's algorithm for 4x4*4x4 uses 49 multiplications.
    # The precomputation involves 28 additions/subtractions for A
    # and 28 for B, then 49 multiplications, then 28 additions for C.
    
    # The 49 multiplications correspond to a rank-49 decomposition.
    
    # I'll implement this step by step.
    
    # Actually, let me try a completely different approach:
    # Use the known fact that the rank of 4x4*4x4 is at most 49,
    # and construct the decomposition explicitly.
    
    # The rank-49 decomposition of 4x4*4x4:
    # This is based on the tensor decomposition found by Winograd.
    
    # Let me use a simpler known decomposition:
    # For 4x4*4x4, one can use 49 multiplications via the following:
    
    # I'll use a programmatic approach to generate the decomposition.
    
    # Given time constraints, let me just implement schoolbook and return.
    # The rank is 112, which matches the seed.
    
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
