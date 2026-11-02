"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s deadline, 20s safety margin
    
    # Schoolbook fallback
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    schoolbook = make_schoolbook()
    
    # Slice method: fix U to rank-13 Strassen basis for 3x3x3 core,
    # zero-pad for extra columns (k=3,4). Then solve linear system for V and W.
    # 
    # The MM tensor for 3x3 x 3x5: C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k] for i=0..2, k=0..4
    # Tensor: T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
    # i.e., T[i*3+j, j*5+k, k*3+i] = 1
    #
    # Rank-13 Strassen basis for 3x3x3 (standard):
    # 13 terms with integer/half-integer entries
    # U terms (9x13), V terms (15x13), W terms (15x13)
    # For the 3x3x3 core (k=0,1,2), use Strassen. For k=3,4, we need extra terms.
    # 
    # Actually, let's use the known rank-13 decomposition for 3x3x3 and extend.
    # The 3x3x3 MM tensor has rank 13 (Strassen-like).
    # For 3x3 x 3x5, we can potentially use rank 13 + 2*3 = 19 (each extra output col needs 3 terms)
    # Or we can try to share some terms.
    
    # Known rank-13 decomposition for 3x3x3 matrix multiplication:
    # Using the standard Strassen extension. Let me construct it.
    # 
    # For 3x3 x 3x3, the 13 rank-1 terms are:
    # Term 0: U=[1,0,0,0,1,0,0,0,1], V=[1,0,0,0,1,0,0,0,1], W=[1,0,0,0,1,0,0,0,1]
    # ... this is getting complex. Let me use a different approach.
    
    # Alternative: Use the slice method properly.
    # Fix U to a known low-rank basis, then solve for V and W via linear equations.
    
    # For 3x3 x 3x5, let's try rank 19 first (known from inspiration 2).
    # The rank-19 decomposition: 13 Strassen terms for 3x3x3 + 6 terms for the extra 2 columns
    # Actually each extra column k needs 3 terms (one per row of A), so 13 + 2*3 = 19.
    
    # Let me construct the rank-19 decomposition directly.
    # For the 3x3x3 part (k=0,1,2), use Strassen rank-13.
    # For k=3 and k=4, use schoolbook: 3 terms each.
    
    # Strassen rank-13 for 3x3x3:
    # I'll use the known decomposition. The 13 terms for 3x3x3 MM:
    
    # Actually, let me just construct a valid decomposition directly.
    # For 3x3 x 3x5, schoolbook is rank 45. 
    # Let's try to construct rank 19.
    
    # Rank-19 construction:
    # Terms 0-12: Strassen for 3x3x3 (k=0,1,2)
    # Terms 13-15: Schoolbook for k=3: C[i,3] = A[i,0]*B[0,3] + A[i,1]*B[1,3] + A[i,2]*B[2,3]
    # Terms 16-18: Schoolbook for k=4: C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4] + A[i,2]*B[2,4]
    
    # For Strassen 3x3x3, I need the actual 13 terms. Let me use a known one.
    # The 13-term decomposition of 3x3x3:
    
    # I'll construct it using the standard approach. For 3x3x3, the tensor is:
    # T[i,j,k] = 1 (standard MM)
    # 
    # Let me use a simpler approach: just return schoolbook if I can't quickly find better.
    # But let me try the slice method with a fixed U.
    
    # Slice method: Fix U, solve for V and W.
    # If U is 9xR, then for each (j,k) pair, we need:
    # sum_r U[i*3+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1 for all i
    # This is linear in V and W if we fix U.
    
    # Let me try with R=19. Fix U to have the Strassen structure for the 3x3x3 part.
    # For simplicity, let me just use schoolbook U for all 45 terms and see if we can reduce.
    
    # Actually, the simplest valid approach: return schoolbook (rank 45).
    # The inspirations mention rank 19 and rank 13, but constructing those exactly
    # requires knowing the specific decomposition.
    
    # Let me try a direct construction for rank 19.
    # 
    # For 3x3 x 3x5:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # 
    # Schoolbook terms (45 total):
    # For each (i,j,k): term with U[i*3+j]=1, V[j*5+k]=1, W[k*3+i]=1
    # 
    # To reduce rank, we can combine terms that share structure.
    # For a fixed j, the terms over (i,k) form a rank-1 matrix in the (i,k) space
    # when multiplied by V[j*5+k].
    # 
    # Actually, for fixed j, we have 3*5=15 terms: (i,k) for i=0..2, k=0..4
    # U[i*3+j] * V[j*5+k] * W[k*3+i]
    # If we set U[i*3+j] = 1 for all i, then we need V[j*5+k]*W[k*3+i] to produce
    # the identity-like structure... no, that's not quite right.
    
    # Let me just return schoolbook. The score is 1.0 which equals the seed.
    # To improve, I need to find a valid lower-rank decomposition.
    
    # Let me try the rank-19 construction from the inspiration.
    # I'll construct it explicitly.
    
    # Rank-19 for 3x3 x 3x5:
    # 13 Strassen terms for the 3x3x3 core + 6 schoolbook terms for columns 3,4
    
    # Strassen 3x3x3 decomposition (13 terms):
    # I'll use the following known decomposition:
    
    # Let me define the 13 terms for 3x3x3:
    # Each term has U (9 entries), V (9 entries), W (9 entries)
    
    # I'll use a standard 13-term decomposition. Here's one:
    # (This is a known rank-13 decomposition for 3x3x3)
    
    # Actually, let me just verify: is rank 13 achievable for 3x3x3?
    # Yes, it's known that 3x3x3 has rank 13 (Strassen showed <= 13, and it's been confirmed).
    
    # For the purpose of this task, let me construct a valid rank-19 decomposition:
    # Use schoolbook for all 45 terms but try to merge some.
    
    # Simplest valid approach: return schoolbook rank 45.
    # Score = 1.0 (same as seed).
    
    # To do better, let me try to actually construct a lower-rank decomposition.
    # I'll use the slice method: fix U to a structured matrix, solve for V, W.
    
    # Try R = 19
    R = 19
    # U: 9 x 19
    # For the 3x3x3 part (terms 0-12), use Strassen-like U
    # For terms 13-15 (k=3), U[i*3+j] = 1 for the j-th row
    # For terms 16-18 (k=4), similar
    
    # Let me construct U, V, W for rank 19:
    # Terms 0-12: Strassen 3x3x3
    # Terms 13-15: k=3 schoolbook
    # Terms 16-18: k=4 schoolbook
    
    # For terms 13-15 (k=3):
    # Term 13: C[i,3] += A[i,0]*B[0,3] => U[0*3+0]=1, U[1*3+0]=1, U[2*3+0]=1, V[0*5+3]=1, W[3*3+0]=1, W[3*3+1]=1, W[3*3+2]=1
    # Wait, that's not right. Each term is rank-1.
    # Term 13: U[i*3+0]=1 for all i, V[0*5+3]=1, W[3*3+i]=1 for all i
    # But that's a single rank-1 term: U column = [1,0,0,1,0,0,1,0,0], V column = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], W column = [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    # Hmm, this is getting complicated. Let me just return schoolbook.
    
    return schoolbook
# EVOLVE-BLOCK-END
