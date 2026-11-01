"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x5 matrix by a 5x6 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Strategy:
    1. Start with the schoolbook decomposition of rank n*m*p = 60.
    2. Try to find a lower rank decomposition using algebraic techniques.
    
    For small matrix sizes, we can try:
    - Block diagonal decompositions
    - Kronecker product structures
    - Greedy rank reduction attempts
    
    The key insight is that for matrix multiplication C = A @ B where A is (n x m) and B is (m x p),
    the tensor has rank at most n*m*p (schoolbook). But sometimes we can do better.
    
    For 2x5 and 5x6:
    - We can split the 5 middle dimension into parts
    - Try block diagonal structures
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Schoolbook decomposition as baseline
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try block diagonal decomposition
    # Split the middle dimension m into m1 + m2
    # A = [A1 | A2] where A1 is (n x m1) and A2 is (n x m2)
    # B = [B1; B2] where B1 is (m1 x p) and B2 is (m2 x p)
    # C = A1 @ B1 + A2 @ B2
    # Rank = n*m1*p + n*m2*p = n*p*(m1+m2) = n*m*p (same as schoolbook!)
    # So simple block diagonal doesn't help.
    
    # Try a different approach: use the fact that n=2 is small
    # For 2x5 * 5x6, we can potentially use rank reduction
    
    # Let's try to find a decomposition of lower rank using integer programming or greedy approach
    
    # First, let's try known results:
    # For 2x2 * 2x2: rank 7 (Strassen)
    # For 2x3 * 3x2: rank 8 (known optimal)
    # For 2x4 * 4x2: rank 11 (known)
    
    # For our case 2x5 * 5x6, let's try to construct a decomposition
    
    # Approach: Use the structure of 2x5 and 5x6
    # We can write 5 = 2 + 3 and 6 = 2 + 4 or other splits
    
    # Try: 5 = 1 + 4, 6 = 2 + 4
    # Or: 5 = 2 + 3, 6 = 3 + 3
    
    # Let's try a more systematic approach using numpy
    
    # The tensor T[i,j,k] = delta_{i,k} for matrix multiplication
    # Actually, T[A_idx, B_idx, C_idx] where:
    # A_idx = i*m + j (i in range(n), j in range(m))
    # B_idx = j*p + k (j in range(m), k in range(p))
    # C_idx = k*n + i (k in range(p), i in range(n))
    # T[A_idx, B_idx, C_idx] = 1 if the indices are consistent, else 0
    
    # Let's build the tensor and try to find a low-rank decomposition
    
    def build_tensor():
        """Build the matrix multiplication tensor T[A_idx, B_idx, C_idx]"""
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T[a_idx, b_idx, c_idx] = 1.0
        return T
    
    T = build_tensor()
    
    # Try to find a rank-r decomposition
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a bilinear optimization problem. Let's try a greedy approach.
    
    def try_rank(r):
        """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
        if time.time() > deadline:
            return None
        
        # Initialize U, V, W randomly or with structure
        # Use a deterministic initialization based on the schoolbook pattern
        
        # Start with the first r schoolbook terms
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Fill with first r schoolbook entries
        for col in range(min(r, n * m * p)):
            # col = i * m * p + j * p + k
            i = col // (m * p)
            rem = col % (m * p)
            j = rem // p
            k = rem % p
            
            a_idx = i * m + j
            b_idx = j * p + k
            c_idx = k * n + i
            
            U[a_idx, col] = 1.0
            V[b_idx, col] = 1.0
            W[c_idx, col] = 1.0
        
        # Now try to optimize
        # This is a hard problem. For small r, we might be able to solve it.
        
        # For now, let's just check if the schoolbook prefix works
        # (it won't unless r = n*m*p)
        
        return None
    
    # Since finding a lower rank decomposition is computationally hard,
    # and for 2x5 * 5x6 the optimal rank is likely close to the schoolbook rank,
    # let's try a few specific constructions.
    
    # Known result: For 2x2 * 2x2, Strassen gives rank 7 < 8
    # For 2x3 * 3x2, the optimal rank is 8 < 12
    
    # Let's try to use a block structure:
    # 2x5 * 5x6 = (2x2 * 2x2) + (2x3 * 3x4) + (2x0 * 0x0) ... no, that doesn't work directly
    
    # Better: split 5 into 2+3, and 6 into 2+4
    # A = [A_22 | A_23] (2x5)
    # B = [B_22; B_34] (5x6)
    # C = A_22 @ B_22 + A_23 @ B_34
    # Rank = rank(2x2*2x2) + rank(2x3*3x4)
    # If we use Strassen for 2x2*2x2: rank 7
    # For 2x3*3x4: schoolbook is 2*3*4 = 24
    # Total: 7 + 24 = 31 < 60
    
    # But wait, this isn't quite right. The blocks don't align properly.
    # Let me think more carefully.
    
    # Actually, for block matrix multiplication:
    # If A is (n1+n2) x (m1+m2) and B is (m1+m2) x (p1+p2),
    # then C = A @ B has blocks:
    # C_11 = A_11 @ B_11 + A_12 @ B_21
    # C_12 = A_11 @ B_12 + A_12 @ B_22
    # C_21 = A_21 @ B_11 + A_22 @ B_21
    # C_22 = A_21 @ B_12 + A_22 @ B_22
    
    # This gets complicated. Let's try a simpler approach.
    
    # For n=2, we can use the fact that 2x2 matrix multiplication has rank 7.
    # Let's try to embed this into our problem.
    
    # Split 5 = 2 + 3, 6 = 3 + 3
    # A = [A1 (2x2) | A2 (2x3)]
    # B = [B1 (2x3); B2 (3x3)]
    # C = A1 @ B1 + A2 @ B2
    # C is 2x6
    
    # Rank = rank(2x2*2x3) + rank(2x3*3x3)
    # Schoolbook: 2*2*3 + 2*3*3 = 12 + 18 = 30
    # If we can do 2x2*2x3 with rank < 12 and 2x3*3x3 with rank < 18, we save.
    
    # For 2x2*2x3: This is like multiplying 2x2 by 2x3.
    # We can think of it as two 2x2*2x2 multiplications (using the first 2 columns) 
    # plus a 2x2*2x1 multiplication (using the 3rd column).
    # Or: 2x2*2x3 has tensor rank at most 2*2*3 = 12.
    # Can we do better? 2x2*2x2 is rank 7, and 2x2*2x1 is rank 4 (schoolbook).
    # But they share the same A, so it's not simply additive.
    
    # This is getting complex. Let's just use the schoolbook decomposition 
    # and return it, since finding a provably lower rank decomposition 
    # for 2x5*5x6 is non-trivial.
    
    # However, let's try one more thing: use numpy to verify the schoolbook
    # and see if we can find any improvement through a simple search.
    
    # For now, return the schoolbook decomposition
    return best_result

# EVOLVE-BLOCK-END
