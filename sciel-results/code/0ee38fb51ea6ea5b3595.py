"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit

    # Schoolbook as guaranteed fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook

    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Strategy: Try to construct decompositions using block structure
    # Split A (3x5) into A_left (3x2) and A_right (3x3)
    # Split B (5x6) into B_top (2x6) and B_bottom (3x6)
    # C = A_left @ B_top + A_right @ B_bottom
    
    # For the 3x2x6 block: rank <= 3*2*6 = 36, but we can do better
    # For the 3x3x6 block: rank <= 3*3*6 = 54
    
    # Actually, let's try a more systematic approach:
    # Use the fact that for matrix mult (n x m) @ (m x p), 
    # we can group terms by the intermediate index j.
    
    # Key idea: For each value of the "intermediate" index j, 
    # the contribution is A[:,j] outer B[j,:] which is rank 1 in the (i,k) space.
    # But we can combine multiple j's if we use linear combinations.
    
    # Let's try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But actually a tighter bound: the tensor has rank at least max(n*p, m*p, n*m)
    
    # Try to find a decomposition with rank < 90
    # We'll use a greedy/iterative approach
    
    # First, let's try a structured decomposition:
    # For each pair (i,k), we need sum_j A[i,j]*B[j,k]
    # This is like computing an inner product for each (i,k)
    
    # Alternative: Use the identity that for fixed i, C[i,:] = A[i,:] @ B
    # This is a matrix-vector product, which can be done in rank n*p per row of A
    # But that gives rank n * (m*p) = same as schoolbook
    
    # Let's try a different approach: exploit linearity
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # If we can write this as sum_c f_c(i) * g_c(k) * h_c(j) where the j-sum is absorbed
    
    # Actually, the standard way: the tensor T[i,j,k] = delta_{jk} in the sense that
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] where T[i,j,k]=1 for all i,j,k
    # Wait no, T[i,j,k] is always 1, and the sum over j gives the matrix product.
    
    # The rank of the matrix multiplication tensor M_{n,m,p} is what we're looking for.
    # Known results:
    # - M_{2,2,2}: rank 7 (Strassen)
    # - M_{3,3,3}: rank 23 (Landsberg)
    # - M_{n,n,n}: rank O(n^2.79) asymptotically, but exact values unknown for most n
    
    # For 3x5x6, we can use:
    # 3x5x6 = 3x3x6 + 3x2x6
    # 3x3x6 can be done as 2 * (3x3x3) using rank 23 each? No, that's not right.
    # Actually, 3x3x6: we can split the 6 into 3+3, so 2 * (3x3x3) with rank 2*23=46
    # 3x2x6: split into 3x2x3 + 3x2x3, each has rank at most 3*2*3=18, 
    # but we can do better. 3x2x3 rank is at most 3*2*3=18, but likely less.
    # Actually for 3x2x3, the rank is at most 3*2*3=18, but we can try to find lower.
    
    # Let's just try to verify if we can construct something with rank < 90
    # by using known sub-decompositions.
    
    # For now, let's try a simple improvement: 
    # Instead of schoolbook rank 90, can we do rank 60?
    # 3x5x6: if we treat it as (3x5)@(5x6), and use the fact that
    # the "middle" dimension is 5, we have 5 independent matrix-vector products.
    # Each (3x1)@(1x6) product has rank 3*6=18 in the naive sense,
    # but since they share the same A and B structure, we might combine.
    
    # Actually, I realize the schoolbook IS optimal for the naive approach.
    # The question is whether there's a clever linear combination that reduces rank.
    
    # Let's try a random search for ranks below 90 with integer/half-integer entries.
    
    import random
    random.seed(12345)
    
    # Try ranks from 30 to 89
    for rank in range(30, min(90, best_rank)):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 3 if rank > 60 else 10
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random integers or half-integers
            U = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(rank)] for _ in range(p*n)]
            
            if verify(U, V, W, rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        
        if best_rank <= rank:
            break
    
    # Also try a structured approach: use the Strassen-like idea
    # For 3x3x3, there's a rank 23 decomposition. Let's try to build on that.
    # But hardcoding it is complex. Instead, let's try a different structured approach.
    
    # Key insight: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j (A[i,j] * B[j,k])
    # If we define for each j, the outer product A[:,j] @ B[j,:], we get m rank-1 terms.
    # But we can combine them: C = sum_j A[:,j] B[j,:]
    # This is already the schoolbook. To reduce rank, we need to find linear dependencies.
    
    # For now, return the best we found.
    return best_result
# EVOLVE-BLOCK-END
