import time
import numpy as np

DIMENSIONS = [3, 3, 7]
_START_TIME = time.time()
_DEADLINE = 140  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    # Try to use structured algebraic construction
    # For 3x3 matrix multiplication, Strassen gives rank 7
    # For 3x7, we can potentially use rank 7 for the "core" and extend
    
    # Strategy: Use the fact that C[i,k] = sum_j A[i,j] * B[j,k]
    # We can try to find a lower-rank decomposition using algebraic identities
    
    # First, let's try the schoolbook as baseline
    rank_school = n * m * p  # 63
    
    # Try to find a better decomposition using structured search
    # Use the Kronecker product structure: (3x3) * (3x7) -> (3x7)
    
    # For the 3x3 part, use Strassen's algorithm (rank 7)
    # Strassen's 7 multiplications for 3x3 * 3x3 -> 3x3
    # Then extend to 3x7 by processing 7 columns
    
    # Actually, let's try a direct approach: use rank-1 terms cleverly
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta[i,k] for the tensor
    
    # Let's try to use the structure where we group operations
    # For each i, we have: C[i,0:7] = A[i,0:3] * B[0:3, 0:7]
    # This is a row of A times B
    
    # Try using 3x3 Strassen for the core and extend
    # Strassen's 7 products for 3x3:
    # M1 = (A00+A11+A22) * (B00+B11+B22)
    # M2 = (A10+A20) * B00
    # M3 = A00 * (B10+B20)
    # M4 = A22 * (B01+B11)
    # M5 = (A02+A12) * B20
    # M6 = A20 * (B02+B12)
    # M7 = (A11+A21) * (B20+B21)
    # ... this is getting complex
    
    # Let's try a simpler structured approach: use the fact that
    # we can share computations across k indices
    
    # For fixed i, C[i,k] for k=0..6 all use the same A[i,0:3]
    # So we can compute A[i,0:3] * B[0:3, 0:7] as a 1x3 * 3x7 multiplication
    # This has rank 3 (schoolbook) but might be reducible
    
    # Actually, 1x3 * 3x7 -> 1x7 has rank 3 (it's just 3 multiplications)
    # So for 3 rows of i, total rank = 3*3 = 9? No, that's not right.
    
    # Let me think differently: the tensor is T[i,j,k] where the multiplication is
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has indices (i,j,k) with i in 0..2, j in 0..2, k in 0..6
    # Rank is the minimum number of rank-1 tensors to sum to T
    
    # T[i,j,k] = 1 if we're computing the (i,k) output using (i,j) and (j,k) inputs
    # Actually T is the structure tensor: T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # where the multiplication pairs (i,j) with (j,k) to give (i,k)
    
    # So T[i,j,k] = 1 if the indices are consistent for the multiplication
    # Specifically, for output (i,k), we use input (i,j) from A and (j,k) from B
    # So T[i,j,k] = 1 for all valid (i,j,k) where j is the shared index
    
    # Wait, I need to be more careful. The multiplication tensor is:
    # T_{i,j,k} = 1 if we're multiplying A[i,j] by B[j,k] to contribute to C[i,k]
    # So T[i,j,k] = 1 for all i in 0..2, j in 0..2, k in 0..6
    # That's just the all-ones tensor? No, that's not right either.
    
    # The correct structure: the tensor T has entries T[i,j,k] where
    # i indexes the row of A (and C), j indexes the column of A (and row of B),
    # k indexes the column of B (and C)
    # T[i,j,k] = 1 means that A[i,j] and B[j,k] multiply to contribute to C[i,k]
    # So T[i,j,k] = 1 for all valid combinations, which is all of them
    # But that would make the tensor rank 1, which is wrong
    
    # I'm confusing the structure. Let me reconsider.
    # The matrix multiplication can be written as:
    # C = A * B, where C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined such that the bilinear form is:
    # <T, A, B> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k]
    # Wait, that's not right. The tensor should encode the multiplication structure.
    
    # Actually, the standard definition: the matrix multiplication tensor T_{n,m,p} has
    # T[i,j,k] = 1 if the indices form a valid multiplication path
    # For C[i,k] = sum_j A[i,j] * B[j,k], we have:
    # T[i,j,k] = 1 for all i, j, k (since every combination is valid)
    # But that gives a rank-1 tensor, which is incorrect
    
    # I think I'm misunderstanding the tensor structure. Let me look at it differently.
    # The multiplication tensor is such that:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k] * D[i,k]
    # for some D, but that's not the standard definition
    
    # The correct definition: the tensor T satisfies
    # (T * A) * B = C, where the multiplication is:
    # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k
    # But that's not a tensor decomposition; that's just the multiplication formula
    
    # For tensor decomposition, we want to write:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where the sum is over r, and the indices are flattened
    
    # This is equivalent to finding rank-1 decompositions of the multiplication tensor
    
    # Let me just try to find a lower-rank decomposition by search
    
    # Use the schoolbook as initial solution
    rank = rank_school
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
    
    # Now try to find a better decomposition using algebraic structure
    # Key insight: we can use the fact that for fixed i, the computation is
    # C[i,0:7] = A[i,0:3] * B[0:3, 0:7]
    # This is a 1x3 vector times a 3x7 matrix, which can be done with 3 multiplications
    # But we can potentially share work across different i values
    
    # Try using the structure: for each j, we have A[:,j] (3x1) and B[j,:] (1x7)
    # C[:,k] += A[:,j] * B[j,k]
    # This is an outer product contribution
    
    # The tensor can be written as sum_j (A[:,j] outer B[j,:])
    # But A[:,j] is 3x1 and B[j,:] is 1x7, so their outer product is 3x7
    # This suggests rank 3 (one for each j), but that's not how tensor rank works
    
    # Let me try a concrete lower-rank construction
    # Use the fact that 7 = 3 + 3 + 1, and apply Strassen-like tricks
    
    # For the first 3 columns (k=0,1,2), use a 3x3 * 3x3 multiplication
    # For the next 3 columns (k=3,4,5), use another 3x3 * 3x3
    # For the last column (k=6), use a 3x3 * 3x1 multiplication
    
    # 3x3 * 3x3 has rank 7 (Strassen)
    # 3x3 * 3x1 has rank 3 (it's just 3 multiplications)
    # Total rank = 7 + 7 + 3 = 17
    
    # Let me implement this
    rank_new = 17
    U2 = [[0] * rank_new for _ in range(n * m)]
    V2 = [[0] * rank_new for _ in range(m * p)]
    W2 = [[0] * rank_new for _ in range(p * n)]
    
    # Strassen's 7 products for 3x3 * 3x3 -> 3x3
    # M1 = (A00+A11+A22) * (B00+B11+B22)
    # M2 = (A10+A20) * B00
    # M3 = A00 * (B10+B20)
    # M4 = A22 * (B01+B11)
    # M5 = (A02+A12) * B20
    # M6 = A20 * (B02+B12)
    # M7 = (A11+A21) * (B20+B21)
    # Then:
    # C00 = M1 + M4 - M7
    # C01 = M5 - M7
    # C02 = M6
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 - M4
    # C12 = M5 + M4
    # C20 = M3
    # C21 = M2 + M6
    # C22 = M1 - M3 + M4
    
    # This is getting very complex. Let me just return the schoolbook for now
    # and try a simpler improvement
    
    # Actually, let me try a different approach: use the fact that we can
    # combine terms that have the same U or V entries
    
    # For now, return the schoolbook decomposition
    return {"U": U, "V": V, "W": W}
