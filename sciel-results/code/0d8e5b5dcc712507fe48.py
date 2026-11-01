import time
import numpy as np
import random

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm = n * m   # 8
    mp = m * p   # 24
    pn = p * n   # 12

    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W form a valid decomposition."""
        rank = len(U[0]) if U and U[0] else 0
        # Check all entries of the tensor
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    s = 0.0
                    for r in range(rank):
                        s += float(U[a][r]) * float(V[b][r]) * float(W[c][r])
                    if abs(s - expected) > tol:
                        return False
        return True

    # Schoolbook decomposition (rank 48)
    def schoolbook():
        rank = n * m * p  # 48
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()
    
    # Strategy: Construct a rank-42 decomposition using block structure.
    # We view A (2x4) as [A1 A2] (2x2 blocks) and B (4x6) as [B1; B2] (2x6 blocks).
    # C = A1 B1 + A2 B2.
    # B1 and B2 are 2x6. We can further split B1 = [B1a B1b] (2x3) and B2 = [B2a B2b] (2x3).
    # So C = A1 B1a + A1 B1b + A2 B2a + A2 B2b.
    # Each term is (2x2) * (2x3).
    # The tensor <2,2,3> has rank 7 (Strassen-like).
    # If we use rank 7 for each of the 4 terms, total rank = 4 * 7 = 28.
    # Wait, the terms are independent?
    # C = A1 B1a + A1 B1b + A2 B2a + A2 B2b.
    # The inputs A1, A2 are shared. B1a, B1b, B2a, B2b are disjoint parts of B.
    # The outputs are disjoint parts of C.
    # So we can just concatenate the decompositions?
    # Let's check the tensor structure.
    # T_{i,j,k} = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # If we split the problem into independent sub-problems, we can sum the ranks.
    # Sub-problem 1: Compute C1 = A1 B1a. A1 is 2x2, B1a is 2x3, C1 is 2x3.
    # Sub-problem 2: Compute C2 = A1 B1b. A1 is 2x2, B1b is 2x3, C2 is 2x3.
    # Sub-problem 3: Compute C3 = A2 B2a. A2 is 2x2, B2a is 2x3, C3 is 2x3.
    # Sub-problem 4: Compute C4 = A2 B2b. A2 is 2x2, B2b is 2x3, C4 is 2x3.
    
    # We need a rank-7 decomposition for <2,2,3>.
    # Standard Strassen for 2x2*2x2 is rank 7.
    # For 2x2*2x3, we can use Strassen on the first 2 columns of B, and schoolbook for the 3rd?
    # Or just use a known rank-7 algorithm for 2x2*2x3?
    # Actually, the rank of <2,2,3> is 7.
    
    # Let's construct the <2,2,3> decomposition explicitly.
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12, b13], [b21, b22, b23]]
    # C = A B = [[c11, c12, c13], [c21, c22, c23]]
    
    # We can use Strassen for the 2x2 part (cols 0,1) and schoolbook for col 2?
    # Strassen (rank 7) for A * B_2x2 gives C_2x2.
    # Schoolbook (rank 4) for A * B_1x3 (col 2) gives C_1x3 (2 rows).
    # Total rank 7 + 4 = 11. This is not 7.
    
    # Is there a rank 7 algorithm for 2x2*2x3?
    # Yes, because <2,2,3> is a sub-tensor of <2,2,4> which has rank 14? No.
    # Actually, the rank of <2,2,3> is known to be 7.
    # One way to get it: Use Strassen for 2x2*2x2 (rank 7) on the first 2 columns.
    # Then handle the 3rd column.
    # C[:, 2] = A * B[:, 2].
    # C12 = a11 b13 + a12 b23
    # C22 = a21 b13 + a22 b23
    # This requires 4 multiplications if done naively.
    # Can we reuse the Strassen products?
    # Strassen products:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # We need a11 b13, a12 b23, a21 b13, a22 b23.
    # We don't have b13, b23 in the Strassen products.
    # So we need 4 new multiplications? That would be 11.
    
    # Maybe a different algorithm?
    # Let's just use the schoolbook rank 48 for now and try to improve with a simple Strassen on 2x2*2x2 blocks?
    # If we split 2x4 * 4x6 into:
    # A = [A1 A2] (2x2 each)
    # B = [B1; B2] (2x6 each)
    # C = A1 B1 + A2 B2
    # B1 is 2x6. Split B1 = [B1a B1b] (2x3 each).
    # B2 is 2x6. Split B2 = [B2a B2b] (2x3 each).
    # C = A1 B1a + A1 B1b + A2 B2a + A2 B2b.
    
    # If we use rank 7 for each <2,2,3>, total rank is 28.
    # But we need to verify if rank 7 is correct for <2,2,3>.
    # According to literature, rank(<2,2,3>) = 7.
    
    # Let's implement a rank-7 algorithm for 2x2*2x3.
    # Reference: "The rank of the 2x2x3 matrix multiplication tensor is 7"
    # One such algorithm:
    # Use Strassen for the 2x2 part (cols 0,1).
    # For col 2, we can express it using linear combinations of the Strassen products + new terms?
    # Actually, a simpler approach:
    # Just use the schoolbook for <2,2,3> which is rank 6 (2*2*3).
    # 4 * 6 = 24.
    # This is better than 48.
    
    # Let's try to construct the rank-24 decomposition.
    # It's just the schoolbook decomposition for the 4 sub-problems.
    # Sub-problem 1: A1 (2x2) * B1a (2x3) -> C1 (2x3). Rank 6.
    # Sub-problem 2: A1 (2x2) * B1b (2x3) -> C2 (2x3). Rank 6.
    # Sub-problem 3: A2 (2x2) * B2a (2x3) -> C3 (2x3). Rank 6.
    # Sub-problem 4: A2 (2x2) * B2b (2x3) -> C4 (2x3). Rank 6.
    
    # Total rank 24.
    # This is a valid decomposition.
    
    def schoolbook_24():
        rank = 24
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        # Indices:
        # A: i in {0,1}, j in {0,1,2,3}.
        # B: j in {0,1,2,3}, k in {0,1,2,3,4,5}.
        # C: i in {0,1}, k in {0,1,2,3,4,5}.
        
        # Sub-problem 1: A1 (i, j in {0,1}) * B1a (j in {0,1}, k in {0,1,2}) -> C1 (i, k in {0,1,2})
        # Sub-problem 2: A1 (i, j in {0,1}) * B1b (j in {0,1}, k in {3,4,5}) -> C2 (i, k in {3,4,5})
        # Sub-problem 3: A2 (i, j in {2,3}) * B2a (j in {2,3}, k in {0,1,2}) -> C3 (i, k in {0,1,2})
        # Sub-problem 4: A2 (i, j in {2,3}) * B2b (j in {2,3}, k in {3,4,5}) -> C4 (i, k in {3,4,5})
        
        col = 0
        # Sub-problem 1
        for i in range(2):
            for j in range(2): # j=0,1
                for k in range(3): # k=0,1,2
                    U[i * 4 + j][col] = 1
                    V[j * 6 + k][col] = 1
                    W[k * 2 + i][col] = 1
                    col += 1
        # Sub-problem 2
        for i in range(2):
            for j in range(2): # j=0,1
                for k in range(3, 6): # k=3,4,5
                    U[i * 4 + j][col] = 1
                    V[j * 6 + k][col] = 1
                    W[k * 2 + i][col] = 1
                    col += 1
        # Sub-problem 3
        for i in range(2):
            for j in range(2, 4): # j=2,3
                for k in range(3): # k=0,1,2
                    U[i * 4 + j][col] = 1
                    V[j * 6 + k][col] = 1
                    W[k * 2 + i][col] = 1
                    col += 1
        # Sub-problem 4
        for i in range(2):
            for j in range(2, 4): # j=2,3
                for k in range(3, 6): # k=3,4,5
                    U[i * 4 + j][col] = 1
                    V[j * 6 + k][col] = 1
                    W[k * 2 + i][col] = 1
                    col += 1
                    
        return {"U": U, "V": V, "W": W}, rank

    result_24, rank_24 = schoolbook_24()
    if verify(result_24["U"], result_24["V"], result_24["W"]):
        best_result, best_rank = result_24, rank_24

    # Can we do better than 24?
    # Rank of <2,2,3> is 7.
    # If we use rank 7 for each sub-problem, total rank is 28.
    # Wait, 7 > 6. So schoolbook for <2,2,3> is better.
    # Is there a rank < 24 for the whole tensor?
    # Lower bound for <2,4,6> is not well known, but 24 is a strong candidate.
    
    # Let's try to find a rank-20 decomposition?
    # Unlikely to find one quickly with random search.
    
    return best_result
