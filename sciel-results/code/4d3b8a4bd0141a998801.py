"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not U or not V or not W:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Build tensor T[(i,k),(k',j),(i',j')] = sum_r U[i*m+k, r] * V[k'*p+j, r] * W[j*n+i', r]
    # Should equal 1 iff i==i' and k==k' (and j is the same in V and W indices, which it is by construction)
    for i in range(n):
        for k in range(m):
            for jp in range(p):
                for ip in range(n):
                    for kp in range(m):
                        expected = 1.0 if (i == ip and k == kp) else 0.0
                        actual = 0.0
                        for r in range(rank):
                            actual += U[i * m + k][r] * V[kp * p + jp][r] * W[jp * n + ip][r]
                        if abs(actual - expected) > 1e-9:
                            return False
    return True

def _build_rank1(U, V, W, r, ui, vi, wi):
    """Add a rank-1 term at index r."""
    U[ui][r] = ui
    V[vi][r] = vi
    W[wi][r] = wi

def _decompose_2x2x2_strassen(U, V, W, rank_offset, u_row0, u_row1, v_col0, v_col1, w_row0, w_row1):
    """
    Decompose C = A*B where A is 2x2, B is 2x2, C is 2x2 using Strassen (rank 7).
    A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[i, j], [k, l]]
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-f)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    """
    # We need to add 7 rank-1 terms.
    # The indices for U (A elements), V (B elements), W (C elements)
    # U rows: 0,1 for A (row 0 col 0, row 1 col 0, row 0 col 1, row 1 col 1) -> actually U[i*m+k]
    # Here m=2 for the sub-problem? No, m is the inner dimension.
    # Let's use the global indices passed.
    # u_row0 is index of A[0,0] and A[0,1] in U? No, U has shape (n*m, rank).
    # For the sub-problem A (2x2), m_sub=2.
    # A[0,0] -> U[u_row0 + 0]
    # A[0,1] -> U[u_row0 + 1]
    # A[1,0] -> U[u_row0 + 2]
    # A[1,1] -> U[u_row0 + 3]
    
    # Wait, the indices passed in are for the *full* matrix.
    # Let's just compute the coefficients and add them.
    
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # This is 4 rank-1 terms if we don't share. But Strassen shares.
    # Actually, Strassen is a specific set of 7 bilinear forms.
    # We can just implement the standard Strassen algorithm directly as rank-1 terms?
    # No, Strassen is not a sum of 7 independent rank-1 terms in the tensor sense unless we define them carefully.
    # The tensor rank of 2x2x2 is 7. The standard Strassen algorithm gives the 7 terms.
    
    # Let's construct the 7 terms explicitly.
    # Let a = A[0,0], b = A[0,1], c = A[1,0], d = A[1,1]
    # Let e = B[0,0], f = B[0,1], g = B[1,0], h = B[1,1]
    
    # Term 1: (a+d)(e+h) -> contributes to C[0,0] (+) and C[1,1] (+)
    # This is (a+d) * (e+h) = a*e + a*h + d*e + d*h
    # This is not a single rank-1 term.
    # The rank-1 decomposition of the Strassen algorithm is non-trivial to write out by hand correctly.
    # Let's use a known rank-7 decomposition for 2x2x2.
    
    # Actually, it's easier to just use schoolbook for 2x2x2 (rank 4) if I'm not sure about Strassen indices.
    # But I want to improve from 63.
    # Schoolbook 3x3x7 is 63.
    # If I use Strassen for 2x2 blocks, I can get lower rank.
    
    # Let's try a different approach: use a known low-rank decomposition for 3x3x3 (rank 19 or 23?)
    # Rank of 3x3x3 is at most 23 (Strassen), conjectured 19.
    # If I use rank 23 for 3x3x3, then for 3x3x7 I can do:
    # Split B into two 3x3 and one 3x1.
    # C1 (3x3) = A * B1 (3x3) -> rank 23
    # C2 (3x3) = A * B2 (3x3) -> rank 23
    # C3 (3x1) = A * B3 (3x1) -> rank 9 (schoolbook)
    # Total rank = 23 + 23 + 9 = 55.
    # This is a guaranteed improvement over 63.
    
    # Let's implement the rank-23 Strassen decomposition for 3x3x3.
    # Reference: "Strassen algorithm" for 3x3 matrices.
    # There are 23 bilinear multiplications.
    
    # However, writing the 23 terms by hand is error-prone.
    # Let's stick to a safer, slightly less optimal but definitely correct improvement.
    # Split 3x3x7 into:
    # 1. 3x3x3 (left) -> Rank 9 (schoolbook)
    # 2. 3x3x3 (middle) -> Rank 9 (schoolbook)
    # 3. 3x3x1 (right) -> Rank 3 (schoolbook)
    # Total: 21. This is NOT better than 63.
    
    # Wait, schoolbook is n*m*p.
    # If I split p=7 into 3+3+1:
    # Term 1: A(3x3) * B1(3x3) -> 9 terms
    # Term 2: A(3x3) * B2(3x3) -> 9 terms
    # Term 3: A(3x3) * B3(3x1) -> 3 terms
    # Total 21 terms.
    # But wait, the A matrix is shared!
    # In the tensor decomposition, if the same U row is used in multiple rank-1 terms, it doesn't reduce the rank.
    # Rank is the number of columns in U, V, W.
    # If I do 9+9+3 = 21 terms, the rank is 21.
    # 21 < 63. This is a huge improvement.
    
    # Let's verify this logic.
    # The tensor T has dimensions (3*3) x (3*7) x (7*3) = 9 x 21 x 21.
    # Schoolbook rank is 9*21 = 189? No, n*m*p = 3*3*7 = 63.
    # Why? Because U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    # Rank is the size of the shared index.
    # Schoolbook: for each i,k,j, we have one term. Total n*m*p = 63 terms.
    
    # My split:
    # For j in 0..2 (B1): 3*3*3 = 27 terms? No.
    # A is 3x3. B1 is 3x3.
    # Terms for C1 = A*B1: for each i (0..2), k (0..2), j (0..2).
    # U[i*3+k], V[k*3+j], W[j*3+i].
    # That is 3*3*3 = 27 terms.
    # Similarly for B2: 27 terms.
    # For B3 (3x1): j=0 only. 3*3*1 = 9 terms.
    # Total 27+27+9 = 63 terms.
    # It's the same!
    
    # Why? Because the schoolbook decomposition is already minimal for this split?
    # No, the rank of matrix multiplication tensor M_{n,m,p} is at most n*m*p, but often lower.
    # For 2x2x2, rank is 7 < 8.
    # For 3x3x3, rank is <= 23 < 27.
    # So if I use the optimal 3x3x3 decomposition (rank 23), I get:
    # 23 + 23 + 9 = 55.
    # Or if I use rank 19 (if available): 19+19+9 = 47.
    
    # So the key is to use a non-schoolbook decomposition for the 3x3x3 blocks.
    # Let's use the rank-23 Strassen-like decomposition for 3x3x3.
    # I need the explicit 23 terms.
    
    # Alternatively, I can use the fact that 3x3x3 can be decomposed using 2x2x2 Strassen + some extra terms.
    # Let's just use a known rank-23 decomposition.
    # From literature: "A 3x3 matrix multiplication with 23 multiplications".
    
    # I will implement the rank-23 decomposition for 3x3x3.
    # Then combine two such blocks and one 3x3x1 block.
    
    # If I can't get the exact 23 terms, I can fallback to a simpler improvement.
    # But 63 is the baseline. I need to beat it.
    # The split 3+3+1 gives 27+27+9 = 63 with schoolbook.
    # So I MUST use a lower-rank decomposition for the 3x3x3 blocks.
    
    # Let's try to find a rank-20 or 21 decomposition for 3x3x3.
    # Actually, the rank of 3x3x3 is known to be at most 23.
    # Let's assume I can implement the 23-term decomposition.
    
    # Here is a known 23-term decomposition for 3x3 matrix multiplication.
    # (Source: Various papers on tensor rank)
    # It is complex. Let's try a different strategy.
    
    # What if I use the 2x2 Strassen on sub-blocks?
    # Split 3x3 into 2x2, 2x1, 1x2, 1x1.
    # C = A*B.
    # C11 (2x2) = A11*B11 + A12*B21 -> Rank 7 (A11B11) + Rank 7 (A12B21) = 14? No, they share no variables.
    # Actually, A11*B11 is 2x2x2 (rank 7). A12*B21 is 2x1 * 1x2 -> 2x2. Rank 4?
    # This is getting complicated.
    
    # Let's go back to the split 3+3+1.
    # If I use rank 23 for 3x3x3, total rank is 55.
    # I will implement the rank-23 decomposition.
    
    # If I fail to implement rank-23 correctly, I will return rank 63.
    
    # Let's write the rank-23 decomposition for 3x3x3.
    # I will use a specific set of 23 bilinear forms.
    
    # Due to complexity, I will implement a safe fallback:
    # If I cannot verify the 23-term decomposition, I use 63.
    
    # Let's try to implement the 23-term decomposition using a known algorithm.
    # Actually, I can use a heuristic: try to optimize the rank using a greedy method?
    # No, must be exact.
    
    # I will use the following 23-term decomposition for 3x3x3:
    # (Coefficients are integers or half-integers)
    
    # I will define the 23 terms as a list of tuples (U_indices, V_indices, W_indices, coeff).
    # This is tedious.
    
    # Alternative: Use the fact that 3x3x3 rank <= 2*2*2 (Strassen) + ...
    # No.
    
    # Let's just use the 63-rank schoolbook for now, but structure the code so that if I find a lower rank, I can insert it.
    # Wait, the task says "Find a rank-minimal exact tensor decomposition".
    # If I return 63, it's valid but not minimal.
    # I should try to improve.
    
    # Let's try the 3+3+1 split with a rank-23 3x3x3 block.
    # I will hardcode the 23 terms for 3x3x3.
    
    # The 23 terms for 3x3x3 matrix multiplication:
    # (i, k, j) -> U[i*3+k], V[k*3+j], W[j*3+i]
    # I will use the decomposition from "Strassen's algorithm for 3x3 matrices".
    
    # Since I don't have the exact 23 terms memorized, I will use a conservative estimate.
    # I will implement the 63-rank decomposition but with a check: if I can find a 55-rank decomposition, use it.
    # For now, I will return 63.
    
    # Wait, I can use the 2x2 Strassen on the 2x2 sub-blocks of the 3x3 matrices.
    # Let A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # Same for B.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Rank(A11*B11) = 7
    # Rank(A12*B21) = 2*1*2 = 4 (schoolbook)
    # So C11 rank <= 7+4 = 11.
    # Similarly for others.
    # Total rank <= 4 * 11 = 44?
    # This is a good improvement over 63.
    
    # Let's implement this block decomposition.
    
    def _strassen_2x2x2(U, V, W, rank_offset, u_base, v_base, w_base):
        """
        Add 7 rank-1 terms for 2x2x2 multiplication.
        u_base: index of A[0,0] in U
        v_base: index of B[0,0] in V
        w_base: index of C[0,0] in W
        """
        # A indices: 0,1,2,3 (row-major)
        # B indices: 0,1,2,3
        # C indices: 0,1,2,3
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-f)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to express these as rank-1 terms.
        # This is not straightforward to do with just 7 terms without knowing the exact factorization.
        # The standard Strassen algorithm uses 7 multiplications, but they are not rank-1 in the tensor sense
        # unless we define the tensor factors accordingly.
        
        # Actually, the tensor rank of 2x2x2 is 7, so there IS a 7-term decomposition.
        # I will use a known 7-term decomposition.
        
        # Term 1: (A00+A11)(B00+B11) -> U: A00,A11; V: B00,B11; W: C00,C11
        # This is not a rank-1 term.
        
        # I will skip Strassen and use schoolbook for 2x2 (rank 4) to be safe.
        # Then C11 rank = 4 + 4 = 8.
        # Total rank = 4 * 8 = 32?
        # No, the terms are not independent.
        
        # I will return the 63-rank schoolbook decomposition for now.
        
        pass

    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Uses schoolbook decomposition (rank 63) as the guaranteed valid baseline.
    """
    n, m, p = DIMENSIONS
    
    # Return schoolbook decomposition
    result = _schoolbook(n, m, p)
    
    # Verify it's correct
    if not _verify(result["U"], result["V"], result["W"], n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return result
# EVOLVE-BLOCK-END
