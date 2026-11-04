DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = 3, 5, 7
    # Block decomposition: split middle dimension into 3+2
    # C = A1 * B1 + A2 * B2 where A1 is 3x3, A2 is 3x2, B1 is 3x7, B2 is 2x7
    # rank = rank(M_{3,3,7}) + rank(M_{3,2,7})
    # For M_{3,3,7}: use Strassen-like for the 3x3x3 part + schoolbook for 3x3x4
    # Actually, let's just do schoolbook for both but that's 105.
    
    # Better: M_{3,3,7} - split 7 into 3+4
    # M_{3,3,3} has rank 23 (Strassen)
    # M_{3,3,4} has rank 36 (schoolbook) or can we do better?
    # M_{3,2,7} has rank 42 (schoolbook)
    
    # Total: 23 + 36 + 42 = 101
    # That's better than 105!
    
    rank = 101
    U = [[0]*rank for _ in range(15)]
    V = [[0]*rank for _ in range(35)]
    W = [[0]*rank for _ in range(21)]
    
    col = 0
    # Block 1: M_{3,3,3} using Strassen, indices i in 0..2, j in 0..2, k in 0..2
    # Strassen for 3x3x3:
    # M1 = A00*(B00+B11)
    # M2 = (A10+A20)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A11)*(B01+B11)
    # M6 = (A10+A11)*(B00-B10)
    # M7 = (A01+A11)*(B00+B10)
    # ... this is getting complicated. Let me just use schoolbook for all blocks.
    
    # Schoolbook for M_{3,3,7}: 3*3*7 = 63
    # Schoolbook for M_{3,2,7}: 3*2*7 = 42
    # Total: 105. Same as before.
    
    # Let me implement Strassen for M_{3,3,3} properly.
    # The 3x3x3 matrix multiplication tensor has rank 23.
    # I'll hardcode the Strassen decomposition.
    
    # For M_{3,3,3}, indices (i,j,k) with i,j,k in {0,1,2}
    # T[i,j,k] = 1 (the multiplication tensor)
    # U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    
    # Strassen 3x3x3 decomposition (rank 23):
    # This is complex. Let me use a known rank-23 decomposition.
    
    # Actually, let me just use the schoolbook but with the 5 split as 3+2
    # and hope for the best. The key insight is that I need to verify correctness.
    
    # Simplest correct approach: schoolbook, rank 105
    # But I want to do better.
    
    # Let me implement the Winograd algorithm for the size-5 middle dimension.
    # For a 1x5 x 5x1 dot product, the Winograd algorithm uses 5 multiplications (optimal).
    # But when we have 3x5 x 5x7, we can use the Winograd transform to reduce the
    # number of scalar multiplications from 3*5*7=105 to 3*7*5=105 (same).
    
    # The Winograd algorithm for size m uses m + 2*ceil(m/2) - 2 multiplications? No.
    # For a 1D convolution of size n with kernel size k, Winograd uses n+k-1 multiplications
    # which is the same as direct. The savings come from batching.
    
    # I think for this specific case, the best I can do without deep research is
    # to use a block decomposition. Let me implement M_{3,5,7} by padding to
    # M_{4,8,8} and using a known decomposition, or just do a careful block split.
    
    # Final decision: implement schoolbook (rank 105) as a fallback, but try
    # a block decomposition that should give rank 101.
    
    # M_{3,3,3}: Strassen rank 23
    # M_{3,3,4}: schoolbook rank 36  
    # M_{3,2,7}: schoolbook rank 42
    # Total: 101
    
    # Implement M_{3,3,3} Strassen:
    # The 23 rank-1 tensors for 3x3x3 matrix multiplication.
    # I'll use the known decomposition.
    
    # For simplicity and correctness, let me just do schoolbook for everything.
    # Rank 105 is guaranteed correct.
    
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
