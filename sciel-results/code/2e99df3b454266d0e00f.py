import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen decomposition for 2x2x2
    # i0, i1 are row indices of A
    # j0, j1 are col indices of A (row indices of B)
    # k0, k1 are col indices of B (col indices of C)
    def add_strassen(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        # U[i0, j0] += 1, U[i1, j1] += 1
        # V[j0, k0] += 1, V[j1, k1] += 1
        # W[k0, i0] += 1, W[k1, i1] += 1
        # Wait, Strassen is a decomposition of the tensor.
        # The tensor T[i,j,k] = A[i,j] B[j,k].
        # Strassen expresses T as sum of 7 rank-1 tensors.
        
        # The 7 products are:
        # 1. (A00 + A11) * (B00 + B11)
        #    Term: U[i0,j0] += 1, U[i1,j1] += 1
        #           V[j0,k0] += 1, V[j1,k1] += 1
        #           W[k0,i0] += 1, W[k1,i1] += 1
        #    Wait, this is not a single rank-1 term.
        #    Strassen's algorithm computes 7 intermediate products M1..M7.
        #    Each M is a scalar product of two sums.
        #    M1 = (A00 + A11) * (B00 + B11)
        #    This corresponds to a rank-1 term in the decomposition?
        #    No, the decomposition is:
        #    T = sum_{c=1}^7 u_c v_c w_c
        #    where u_c, v_c, w_c are vectors.
        
        # Let's use the explicit rank-1 vectors for Strassen.
        # Reference: https://en.wikipedia.org/wiki/Strassen_algorithm
        # The 7 terms are:
        # 1. (e0+e1) (e0+e1) (e0 e0 + e1 e1)? No.
        
        # Let's use the standard vector forms.
        # Indices for A: 00, 01, 10, 11.
        # Indices for B: 00, 01, 10, 11.
        # Indices for C: 00, 01, 10, 11.
        
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        # Contributes to C00, C01, C10, C11?
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # This means each M is a scalar.
        # M1 = (A00 + A11) * (B00 + B11)
        # This can be written as a sum of 4 rank-1 terms? No, it's a single product of sums.
        # In tensor decomposition, a term like (A00 + A11)(B00 + B11) is NOT a single rank-1 tensor.
        # A rank-1 tensor is u_i v_j w_k.
        # (A00 + A11) is a vector in A-space. (B00 + B11) is a vector in B-space.
        # But we need a vector in C-space too.
        # The contribution to C is distributed.
        
        # Actually, Strassen's algorithm is a *algorithm*, not directly a *decomposition* into 7 rank-1 tensors of the multiplication tensor.
        # The multiplication tensor T has rank 7.
        # The 7 rank-1 tensors are:
        # 1. (e0+e1) (e0+e1) (e00+e11)? No.
        
        # Let's look up the explicit rank-1 decomposition for 2x2x2.
        # The 7 terms are:
        # 1. (1, 1, 0, 0) (1, 1, 0, 0) (1, 0, 1, 0)? No.
        
        # Correct Strassen rank-1 decomposition:
        # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]].
        # C = [[ae+cg, af+dh], [ce+dg, cf+gh]].
        
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(g-h)
        # M4 = d(f-e)
        # M5 = (a+c)h
        # M6 = (d-a)(e+g)
        # M7 = (b-d)(h+f)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Each M is a product of a linear form in A and a linear form in B.
        # M1 = (A00 + A11) * (B00 + B11)
        # This term contributes to C00 and C11.
        # Specifically, M1 * (1 at C00, 1 at C11).
        # So the rank-1 tensor for M1 is:
        # u = (1, 0, 0, 1) in A (indices 00, 01, 10, 11)
        # v = (1, 0, 0, 1) in B (indices 00, 01, 10, 11)
        # w = (1, 0, 0, 1) in C (indices 00, 01, 10, 11)
        # Check: u_i v_j w_k.
        # i=00, j=00, k=00: 1*1*1 = 1.
        # i=11, j=11, k=11: 1*1*1 = 1.
        # i=00, j=11, k=00: 1*1*0 = 0.
        # This matches M1 contributing to C00 and C11?
        # Wait, M1 = (a+d)(e+h).
        # a=A00, d=A11. e=B00, h=B11.
        # Term is A00 B00 + A00 B11 + A11 B00 + A11 B11.
        # In tensor T, T[00, 00, 00] gets A00 B00.
        # T[00, 11, 00] gets A00 B11.
        # T[11, 00, 00] gets A11 B00.
        # T[11, 11, 11] gets A11 B11.
        
        # The rank-1 tensor u v w with u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) gives:
        # T[00,00,00] += 1
        # T[00,11,00] += 0 (w[11] is 0? No, w is vector in C space. C indices 00,01,10,11.
        # w[00]=1, w[11]=1.
        # So T[i,j,k] += u[i]v[j]w[k].
        # i=00, j=00, k=00: 1*1*1=1.
        # i=00, j=11, k=00: 1*1*1=1. (A00 B11)
        # i=11, j=00, k=00: 1*1*1=1. (A11 B00)
        # i=11, j=11, k=00: 1*1*1=1. (A11 B11) -> Wait, w[00]=1. So this adds to C00.
        # But M1 contributes to C11 as well?
        # M1 = (a+d)(e+h).
        # In C00: M1 is added.
        # In C11: M1 is added.
        # So we need w[00]=1 and w[11]=1.
        # If w=(1,0,0,1), then:
        # i=11, j=11, k=11: 1*1*1=1.
        # i=11, j=11, k=00: 1*1*1=1.
        # This adds A11 B11 to C00 AND C11.
        # But A11 B11 is part of M1.
        # M1 contributes to C00 and C11.
        # So yes, u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) is correct for M1?
        # Let's check M1 expansion:
        # M1 = A00 B00 + A00 B11 + A11 B00 + A11 B11.
        # The tensor term adds:
        # (00,00,00): 1
        # (00,11,00): 1
        # (11,00,00): 1
        # (11,11,00): 1
        # (00,00,11): 1
        # (00,11,11): 1
        # (11,00,11): 1
        # (11,11,11): 1
        # This adds A11 B11 to C00 (index 00) and C11 (index 11).
        # But in the actual product, A11 B11 only appears in C11.
        # So this rank-1 tensor is NOT correct for M1.
        
        # The issue is that M1 is a scalar. The decomposition must distribute the contributions correctly.
        # The 7 rank-1 tensors of Strassen are:
        # 1. (1, 1, 0, 0) (1, 1, 0, 0) (1, 0, 0, 1) ? No.
        
        # Let's use a known correct set.
        # From "The rank of the matrix multiplication tensor":
        # The 7 terms are:
        # 1. (1, 0, 0, 1) (1, 0, 0, 1) (1, 0, 0, 1) -> No, this was wrong.
        
        # Actually, it is easier to just use the schoolbook for 2x2x2 if we can't get the Strassen vectors right quickly.
        # But 7 < 8.
        
        # Let's try to find the vectors online or derive them.
        # M1 = (A00+A11)(B00+B11).
        # We need a term that picks up A00 B00, A00 B11, A11 B00, A11 B11 and puts them in C00 and C11.
        # This is not a single rank-1 tensor.
        # Wait, the rank of the tensor is 7.
        # The 7 rank-1 tensors are:
        # 1. u1=(1,1,0,0), v1=(1,1,0,0), w1=(1,0,1,0)?
        
        # I will use a library or hardcode the Strassen vectors if I can find them.
        # Since I cannot search, I will fall back to schoolbook for 2x2x2 if I can't be sure.
        # Rank 8 for 2x2x2.
        # Then total rank = 4*8 + 16 + 16 + 8 = 32 + 40 = 72.
        # No improvement.
        
        # I need Strassen.
        # Let's assume the standard Strassen decomposition vectors.
        # I will use the following 7 vectors (u, v, w) in basis (00, 01, 10, 11):
        
        # Term 1: (1, 1, 0, 0) (1, 1, 0, 0) (1, 0, 0, 1)
        # Term 2: (0, 0, 1, 1) (1, 0, 0, 0) (0, 1, 0, 0)
        # Term 3: (1, 0, 0, 0) (0, 0, 1, -1) (0, 1, 0, 0)
        # Term 4: (0, 0, 0, 1) (0, 1, -1, 0) (0, 0, 1, 0)
        # Term 5: (1, 0, 1, 0) (0, 0, 0, 1) (0, 1, 0, 0)
        # Term 6: (0, 0, -1, 1) (1, 0, 1, 0) (0, 0, 1, 0)
        # Term 7: (0, 1, 0, -1) (0, 0, 1, 1) (1, 0, 0, 0)
        
        # I am not confident in these vectors.
        # Given the risk, I will stick to the schoolbook decomposition which is guaranteed correct.
        # The score will be 1.0.
        
        pass

    # Since I cannot guarantee the Strassen vectors, I will return the schoolbook decomposition.
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
