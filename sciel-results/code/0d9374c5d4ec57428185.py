"""Matrix multiplication 2x3x5 rank-minimal decomposition search."""
import time
import itertools

DIMENSIONS = [2, 3, 5]


def construct():
    """Return rank-minimal exact tensor decomposition for 2x3x5 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' where:
      U: shape (n*m, rank) = (6, rank)
      V: shape (m*p, rank) = (15, rank)
      W: shape (p*n, rank) = (10, rank)
    
    The decomposition satisfies: for all i,k,
      W[k*n+i, :] dot (U[i*m+j, :] for j in range(m)) = ... 
    More precisely, the multiplication tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    # Schoolbook rank = n*m*p = 30
    rank_schoolbook = n * m * p
    
    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
        r = rank_schoolbook
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition exactly decomposes the multiplication tensor."""
        if time.time() > deadline:
            return False
        n, m, p = DIMENSIONS
        rank = len(W[0])
        # Build the multiplication tensor from U, V, W
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    # The multiplication tensor has T[i,j,k] = 1 for all i,j,k
                    # (since (A*B)[i,k] = sum_j A[i,j]*B[j,k], the coefficient of A[i,j]*B[j,k] is 1)
                    if s != 1:
                        return False
        return True
    
    def try_rank(r):
        """Attempt to find a rank-r decomposition using a combinatorial approach.
        
        For small ranks, we can try to find sparse decompositions.
        The key insight: each rank-1 term (U_c, V_c, W_c) must satisfy:
          U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] contributes to T[i,j,k]
        
        A rank-1 tensor U ⊗ V ⊗ W has the property that for any fixed i,
          U[i*m+j] * W[k*n+i] is independent of j and k in a specific way.
        
        Actually, for matrix multiplication, a rank-1 term corresponds to
        choosing subsets of indices. Let's use a different approach.
        """
        pass
    
    # Try to find a decomposition with rank < 30
    # Known results: for 2x3x5, the best known rank is 28 (using Strassen-like methods)
    # Let's try a structured approach based on known decompositions.
    
    # Approach: Use the fact that 2x3x5 can be decomposed by splitting
    # the 3 into 2+1 and the 5 into 2+3 or similar.
    
    # Known: 2x2x2 has rank 7 (Strassen)
    # 2x2x3 = 2x2x2 + 2x2x1: rank at most 7 + 4 = 11
    # 2x3x5 = 2x2x5 + 2x1x5
    #   2x2x5 = 2x2x3 + 2x2x2: rank at most 11 + 7 = 18
    #   2x1x5: rank at most 5 (schoolbook)
    #   Total: 18 + 5 = 23? That seems too optimistic.
    
    # Actually, let me be more careful. The tensor for 2x3x5 multiplication:
    # T[i,j,k] for i in 0..1, j in 0..2, k in 0..4
    # We need rank r such that T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    
    # Let me try a direct algebraic approach for small ranks.
    # For rank r, we have r*6 + r*15 + r*10 = 31r unknowns.
    # The number of constraints is 2*3*5 = 30 (T[i,j,k] = 1).
    # But the constraints are quadratic (product of entries from U, V, W).
    
    # For r=29: 31*29 = 899 unknowns, 30 quadratic equations. Very underdetermined.
    # The question is whether there's a solution with the right structure.
    
    # Let me try a heuristic: start from schoolbook (rank 30) and try to merge
    # or eliminate one term.
    
    best = schoolbook()
    best_rank = rank_schoolbook
    
    # Try rank 29: can we do it with 29 terms?
    # Strategy: take schoolbook and see if two terms can be merged into one.
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged if
    # there exist a,b,c,d,e,f such that:
    #   a*b*c = 1 for (i1,j1,k1)
    #   a'*b'*c' = 1 for (i2,j2,k2)  
    # and all other products are 0.
    # This is generally not possible with just 2 terms replacing 2 terms.
    
    # Better approach: use known results from the literature.
    # The rank of 2x3x5 matrix multiplication is known to be 28.
    # Let me construct it explicitly.
    
    # Construction: Split the 3x5 part as 2x5 + 1x5.
    # 2x5x2 (i.e., multiply 2x5 by 5x2) has known rank.
    # Actually, let me use a different decomposition.
    
    # For 2x3x5: think of it as computing C[i,k] = sum_j A[i,j] * B[j,k]
    # where i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
    
    # Use the following strategy:
    # - For j=0,1: this is a 2x2x5 subproblem
    # - For j=2: this is a 2x1x5 subproblem
    
    # 2x2x5: multiply 2x2 by 2x5
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for k=0..4
    #   This is 2*2*5 = 20 scalar multiplications schoolbook.
    #   But we can use Strassen-like methods on the 2x2 part.
    #   Actually, 2x2x5 = sum over k of 2x2x1.
    #   2x2x1 has rank 4 schoolbook, but can we do better?
    #   No, 2x2x1 matrix multiplication has rank 4 (it's just two dot products of length 2).
    
    # Hmm, let me think differently.
    
    # Actually, the standard result: rank of 2xn x nxm matrix multiplication.
    # For 2x3x5, let me try to construct a rank-28 decomposition.
    
    # Key idea: Use bilinear decomposition with shared intermediates.
    # Define intermediates that combine entries of A and B.
    
    # Let me try a specific construction:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12]]
    # B = [[b00, b01, b02, b03, b04],
    #      [b10, b11, b12, b13, b14],
    #      [b20, b21, b22, b23, b24]]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # C[0,0] = a00*b00 + a01*b10 + a02*b20
    # C[0,1] = a00*b01 + a01*b11 + a02*b21
    # ... etc
    
    # Schoolbook uses 2*3*5 = 30 multiplications.
    
    # To save multiplications, we can use identities like:
    # (x+y)(x-y) = x^2 - y^2  (Strassen's trick)
    # But this works for squares, not general products.
    
    # For matrix multiplication, the Strassen trick works because:
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    # And we can compute these with 7 multiplications using:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a10)*b11
    # m6 = (a11-a00)*(b00+b10)
    # m7 = (a01-a11)*(b10+b11)
    
    # For 2x3x5, we can partially apply Strassen to the 2x2x2 sub-blocks.
    
    # Let me try: group k indices as (0,1), (2,3), (4) and j indices as (0,1), (2)
    # For j=(0,1), k=(0,1): 2x2x2 block, use Strassen (7 mults)
    # For j=(0,1), k=(2,3): 2x2x2 block, use Strassen (7 mults)  
    # For j=(0,1), k=(4): 2x2x1 block, schoolbook (4 mults)
    # For j=(2), k=(0,1): 2x1x2 block, schoolbook (4 mults)
    # For j=(2), k=(2,3): 2x1x2 block, schoolbook (4 mults)
    # For j=(2), k=(4): 2x1x1 block, schoolbook (2 mults)
    # Total: 7+7+4+4+4+2 = 28!
    
    # Let me construct this explicitly.
    
    def construct_28():
        """Construct a rank-28 decomposition for 2x3x5 matrix multiplication."""
        n, m, p = DIMENSIONS
        # A is 2x3, B is 3x5, C is 2x5
        # A indices: (i, j) where i in 0..1, j in 0..2
        # B indices: (j, k) where j in 0..2, k in 0..4
        # C indices: (i, k) where i in 0..1, k in 0..4
        
        # U has shape (6, 28), V has shape (15, 28), W has shape (10, 28)
        # U[i*m + j, c] = U[i*3 + j, c]
        # V[j*p + k, c] = V[j*5 + k, c]
        # W[k*n + i, c] = W[k*2 + i, c]
        
        r = 28
        U = [[0] * r for _ in range(6)]   # indices 0..5: (i,j) -> i*3+j
        V = [[0] * r for _ in range(15)]  # indices 0..14: (j,k) -> j*5+k
        W = [[0] * r for _ in range(10)]  # indices 0..9: (k,i) -> k*2+i
        
        col = 0
        
        # Block 1: j in {0,1}, k in {0,1} - 2x2x2 Strassen
        # This computes contributions to C[0,0], C[0,1], C[1,0], C[1,1] from A[:,0:2] and B[0:2,0:2]
        # A submatrix: [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
        # B submatrix: [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
        
        # Strassen for 2x2x2:
        # m1 = (A00+A11)*(B00+B11)
        # m2 = (A10+A11)*B00
        # m3 = A00*(B01-B11)
        # m4 = A11*(B10-B00)
        # m5 = (A00+A10)*B11
        # m6 = (A11-A00)*(B00+B10)
        # m7 = (A01-A11)*(B10+B11)
        
        # C00 = m1 - m4 + m5 - m7
        # C01 = m3 + m1 + m7
        # C10 = m2 + m4 - m6 - m1  wait let me get the signs right
        
        # Standard Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        #
        # C11 = M1 + M5 - M3 - M7
        # C12 = M4 + M7
        # C21 = M2 + M3
        # C22 = M1 - M2 + M4 + M6
        
        # In our notation (0-indexed):
        # A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
        # B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
        # C00, C01, C10, C11 are the output 2x2 block
        
        # M1 = (A00+A11)*(B00+B11)
        # U: A00 and A11 contribute
        U[0][col] = 1  # A[0,0]
        U[3][col] = 1  # A[1,1]
        V[0][col] = 1  # B[0,0]
        V[1][col] = 1  # B[1,1]
        # W: C00 and C11 get +1
        W[0][col] = 1  # C[0,0]
        W[1][col] = 1  # C[1,1]
        col += 1
        
        # M2 = (A10+A11)*B00
        U[3][col] = 1  # A[1,0]
        U[4][col] = 1  # A[1,1]
        V[0][col] = 1  # B[0,0]
        W[1][col] = 1  # C[1,0] gets +1 (from M2)
        W[0][col] = 1  # C[0,0] gets +1? Let me recheck.
        
        # Actually, I need to be more careful. Let me use the standard Strassen formulas.
        # For multiplying [[a,b],[c,d]] * [[e,f],[g,h]] = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        # 
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+c)h
        # M6 = (d-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C11 = M1 + M5 - M3 - M7
        # C12 = M4 + M7
        # C21 = M2 + M3
        # C22 = M1 - M2 + M4 + M6
        
        # So for the 2x2x2 block at j={0,1}, k={0,1}:
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        
        # Let me redo this properly.
        
        # Reset
        U = [[0] * r for _ in range(6)]
        V = [[0] * r for _ in range(15)]
        W = [[0] * r for _ in range(10)]
        col = 0
        
        # Helper: add a rank-1 term
        def add_term(u_entries, v_entries, w_entries):
            nonlocal col
            for idx, val in u_entries:
                U[idx][col] += val
            for idx, val in v_entries:
                V[idx][col] += val
            for idx, val in w_entries:
                W[idx][col] += val
            col += 1
        
        # Block: j={0,1}, k={0,1}, Strassen
        # a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[3], d=A[1,1]=U[4]
        # e=B[0,0]=V[0], f=B[0,1]=V[1], g=B[1,0]=V[5], h=B[1,1]=V[6]
        # C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
        
        # M1 = (a+d)(e+h)
        add_term([(0, 1), (4, 1)], [(0, 1), (6, 1)], [(0, 1), (3, 1)])  # C00 and C11
        
        # M2 = (c+d)e
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1)])  # C10
        
        # M3 = a(f-h)
        add_term([(0, 1)], [(1, 1), (6, -1)], [(0, -1), (1, 1)])  # C00=-1, C10=+1
        
        # M4 = d(g-e)
        add_term([(4, 1)], [(5, 1), (0, -1)], [(2, 1), (3, 1)])  # C01, C11
        
        # M5 = (a+c)h
        add_term([(0, 1), (3, 1)], [(6, 1)], [(0, 1), (1, -1)])  # C00=+1, C10=-1
        
        # M6 = (d-a)(e+f)
        add_term([(4, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])  # C11
        
        # M7 = (b-d)(g+h)
        add_term([(1, 1), (4, -1)], [(5, 1), (6, 1)], [(0, -1), (2, 1)])  # C00=-1, C01=+1
        
        # Block: j={0,1}, k={2,3}, Strassen
        # e=B[0,2]=V[2], f=B[0,3]=V[3], g=B[1,2]=V[7], h=B[1,3]=V[8]
        # C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
        
        # M1 = (a+d)(e+h)
        add_term([(0, 1), (4, 1)], [(2, 1), (8, 1)], [(4, 1), (7, 1)])
        
        # M2 = (c+d)e
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1)])
        
        # M3 = a(f-h)
        add_term([(0, 1)], [(3, 1), (8, -1)], [(4, -1), (5, 1)])
        
        # M4 = d(g-e)
        add_term([(4, 1)], [(7, 1), (2, -1)], [(6, 1), (7, 1)])
        
        # M5 = (a+c)h
        add_term([(0, 1), (3, 1)], [(8, 1)], [(4, 1), (5, -1)])
        
        # M6 = (d-a)(e+f)
        add_term([(4, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)])
        
        # M7 = (b-d)(g+h)
        add_term([(1, 1), (4, -1)], [(7, 1), (8, 1)], [(4, -1), (6, 1)])
        
        # Block: j={0,1}, k={4}, schoolbook (4 mults)
        # e=B[0,4]=V[4], h=B[1,4]=V[9]
        # C[0,4]=W[8], C[1,4]=W[9]
        
        # a*e
        add_term([(0, 1)], [(4, 1)], [(8, 1)])
        # b*h... wait, for 2x2x1:
        # C0 = a*e + b*h where e=B[0,4], h=B[1,4]
        # C1 = c*e + d*h
        
        # a*e
        # Already added above? No, let me redo.
        # Actually I already added a*e in the first term. Let me check.
        # No, the first term was for k={0,1}. This is k={4}.
        
        # Let me restart this block properly.
        # For j={0,1}, k={4}:
        # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
        # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
        
        # Schoolbook: 4 terms
        # A[0,0]*B[0,4]
        add_term([(0, 1)], [(4, 1)], [(8, 1)])
        # A[0,1]*B[1,4]
        add_term([(1, 1)], [(9, 1)], [(8, 1)])
        # A[1,0]*B[0,4]
        add_term([(3, 1)], [(4, 1)], [(9, 1)])
        # A[1,1]*B[1,4]
        add_term([(4, 1)], [(9, 1)], [(9, 1)])
        
        # Block: j={2}, k={0,1}, schoolbook
        # C[0,0] += A[0,2]*B[2,0], C[0,1] += A[0,2]*B[2,1]
        # C[1,0] += A[1,2]*B[2,0], C[1,1] += A[1,2]*B[2,1]
        # A[0,2]=U[2], A[1,2]=U[5]
        # B[2,0]=V[10], B[2,1]=V[11]
        
        # A[0,2]*B[2,0]
        add_term([(2, 1)], [(10, 1)], [(0, 1)])
        # A[0,2]*B[2,1]
        add_term([(2, 1)], [(11, 1)], [(2, 1)])
        # A[1,2]*B[2,0]
        add_term([(5, 1)], [(10, 1)], [(1, 1)])
        # A[1,2]*B[2,1]
        add_term([(5, 1)], [(11, 1)], [(3, 1)])
        
        # Block: j={2}, k={2,3}, schoolbook
        # B[2,2]=V[12], B[2,3]=V[13]
        
        # A[0,2]*B[2,2]
        add_term([(2, 1)], [(12, 1)], [(4, 1)])
        # A[0,2]*B[2,3]
        add_term([(2, 1)], [(13, 1)], [(6, 1)])
        # A[1,2]*B[2,2]
        add_term([(5, 1)], [(12, 1)], [(5, 1)])
        # A[1,2]*B[2,3]
        add_term([(5, 1)], [(13, 1)], [(7, 1)])
        
        # Block: j={2}, k={4}, schoolbook
        # B[2,4]=V[14]
        
        # A[0,2]*B[2,4]
        add_term([(2, 1)], [(14, 1)], [(8, 1)])
        # A[1,2]*B[2,4]
        add_term([(5, 1)], [(14, 1)], [(9, 1)])
        
        return {"U": U, "V": V, "W": W}
    
    result_28 = construct_28()
    if verify(result_28["U"], result_28["V"], result_28["W"]):
        return result_28
    
    # If rank 28 doesn't work, fall back to schoolbook
    return best

