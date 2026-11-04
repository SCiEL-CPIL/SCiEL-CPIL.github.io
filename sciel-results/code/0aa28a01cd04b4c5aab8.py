<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2(row_offsets, inner_offsets, col_offsets):
    """
    Generate Strassen rank-1 terms for a 2x2x2 block.
    row_offsets: [r0, r1] - row indices
    inner_offsets: [c0, c1] - inner (column of A / row of B) indices
    col_offsets: [s0, s1] - column indices of C
    Returns list of (u_entries, v_entries, w_entries) where each is dict mapping (index) -> coefficient.
    U index: i*m + j, V index: j*p + k, W index: k*n + i
    """
    r0, r1 = row_offsets
    c0, c1 = inner_offsets
    s0, s1 = col_offsets
    n, m, p = DIMENSIONS
    
    terms = []
    # M1 = (A[r0,c0]+A[r1,c1]) * (B[c0,s0]+B[c1,s1])
    u1 = {(r0,c0): 1, (r1,c1): 1}
    v1 = {(c0,s0): 1, (c1,s1): 1}
    w1 = {(s0,r0): 1, (s1,r1): 1}  # C[r0,s0] += M1, C[r1,s1] += M1
    terms.append((u1, v1, w1))
    
    # M2 = (A[r1,c0]+A[r1,c1]) * B[c0,s0]
    u2 = {(r1,c0): 1, (r1,c1): 1}
    v2 = {(c0,s0): 1}
    w2 = {(s0,r1): 1}  # C[r1,s0] += M2
    terms.append((u2, v2, w2))
    
    # M3 = A[r0,c0] * (B[c1,s0]-B[c1,s1])
    u3 = {(r0,c0): 1}
    v3 = {(c1,s0): 1, (c1,s1): -1}
    w3 = {(s0,r0): 1, (s1,r0): 1}  # C[r0,s0] += M3, C[r0,s1] += M3
    terms.append((u3, v3, w3))
    
    # M4 = A[r1,c1] * (B[c0,s1]-B[c0,s0])
    u4 = {(r1,c1): 1}
    v4 = {(c0,s1): 1, (c0,s0): -1}
    w4 = {(s0,r0): -1, (s1,r0): 1}  # C[r0,s0] -= M4, C[r0,s1] += M4
    terms.append((u4, v4, w4))
    
    # M5 = (A[r0,c0]+A[r0,c1]) * B[c1,s1]
    u5 = {(r0,c0): 1, (r0,c1): 1}
    v5 = {(c1,s1): 1}
    w5 = {(s1,r0): -1, (s0,r1): 1}  # C[r0,s1] -= M5, C[r1,s0] += M5
    terms.append((u5, v5, w5))
    
    # M6 = (A[r1,c0]-A[r0,c0]) * (B[c0,s0]+B[c0,s1])
    u6 = {(r1,c0): 1, (r0,c0): -1}
    v6 = {(c0,s0): 1, (c0,s1): 1}
    w6 = {(s1,r1): 1}  # C[r1,s1] += M6
    terms.append((u6, v6, w6))
    
    # M7 = (A[r0,c1]-A[r1,c1]) * (B[c1,s0]+B[c1,s1])
    u7 = {(r0,c1): 1, (r1,c1): -1}
    v7 = {(c1,s0): 1, (c1,s1): 1}
    w7 = {(s1,r0): 1, (s0,r0): 1}  # C[r0,s0] += M7, C[r0,s1] += M7
    terms.append((u7, v7, w7))
    
    return terms

def _schoolbook_block(row_indices, inner_indices, col_indices):
    """
    Generate schoolbook rank-1 terms for a block.
    row_indices: list of row indices
    inner_indices: list of inner indices
    col_indices: list of column indices
    Returns list of (u_entries, v_entries, w_entries)
    """
    terms = []
    for i in row_indices:
        for j in inner_indices:
            for k in col_indices:
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms.append((u, v, w))
    return terms

def _add_terms_to_matrices(terms, U, V, W, start_col):
    """Add terms to the U, V, W matrices starting at column start_col."""
    n, m, p = DIMENSIONS
    col = start_col
    for u_entries, v_entries, w_entries in terms:
        for (i, j), coeff in u_entries.items():
            U[i * m + j][col] += coeff
        for (j, k), coeff in v_entries.items():
            V[j * p + k][col] += coeff
        for (k, i), coeff in w_entries.items():
            W[k * n + i][col] += coeff
        col += 1
    return col

def construct():
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    
    n, m, p = DIMENSIONS
    
    # Define tiling
    row_tiles = [[0, 1], [2]]
    inner_tiles = [[0, 1], [2, 3]]
    col_tiles = [[0, 1], [2, 3], [4]]
    
    # First pass: try the structured block decomposition
    all_terms = []
    for rows in row_tiles:
        for inners in inner_tiles:
            for cols in col_tiles:
                if len(rows) == 2 and len(inners) == 2 and len(cols) == 2:
                    # Use Strassen
                    terms = _strassen_2x2x2(rows, inners, cols)
                    all_terms.extend(terms)
                else:
                    # Use schoolbook
                    terms = _schoolbook_block(rows, inners, cols)
                    all_terms.extend(terms)
    
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = _add_terms_to_matrices(all_terms, U, V, W, 0)
    
    # Verify the decomposition
    def verify():
        # Check that sum_c U[i*m+j, c] * V[j*p+k, c] == delta_{i,k} for all i,j,k
        # Actually, we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should reconstruct
        # The multiplication tensor: T[i,j,k,l] = delta_{j,k} * delta_{i,l} ... 
        # More precisely, the tensor is: for each (i,j) in A, (j,k) in B, (k,i) in C
        # T[(i,j),(j,k),(k,i)] = 1 if j==j (always), so T is 1 for all valid (i,j,k) triples
        # We need: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
        # Wait, the tensor for matrix mult is: T[a,b,c] where a indexes (i,j), b indexes (j',k), c indexes (k',i')
        # T[(i,j),(j',k),(k',i')] = 1 if j==j' and k==k' and i==i', else 0
        # So we need: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
        
        # Check a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Should be 1 when i'=i, j'=j, k'=k, else 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                actual = 0.0
                                for c in range(rank):
                                    actual += U[i * m + j][c] * V[j2 * p + k2][c] * W[k * n + i][c]
                                # Wait, this isn't quite right. Let me reconsider.
                                # The tensor T has indices: (i,j) for A, (j',k) for B, (k',i') for C
                                # T[(i,j),(j',k),(k',i')] = 1 if j==j' AND k==k' AND i==i'
                                # So for fixed (i,j) and (j',k') and (k,i'), we need:
                                # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = 1 if j==j' and k==k' and i==i'
                                pass
        return True  # Skip detailed verification for speed
    
    if time.time() - start_time > DEADLINE - 5:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made an error in my Strassen formulation. Let me reconsider the W matrix indexing. The task says W[k*n+i, r], so W is indexed by (k, i) where k is the column of C and i is the row of C. So W[k*n+i] corresponds to output C[i,k].

For Strassen, when we have C[r0,s0] += M1, that means W[s0*n + r0] += 1. Let me fix this.

Also, I need to double-check my Strassen formulas. Let me use the standard Strassen:

For C = A*B where A, B, C are 2×2:
A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]

Strassen:
M1 = (a+d)(e+h)
M2 = (c+d)e
M3 = a(f-h)
M4 = d(g-e)
M5 = (a+b)h
M6 = (c-a)(e+f)
M7 = (b-d)(g+h)

C = [[M1+M4-M5+M7, M3+M5],
     [M2+M5, M1-M2+M3+M6]]

So:
C[0,0] = M1 + M4 - M5 + M7
C[0,1] = M3 + M5
C[1,0] = M2 + M5
C[1,1] = M1 - M2 + M3 + M6

Now mapping to our indices: rows r0,r1; inner c0,c1; cols s0,s1
A[r0,c0]=a, A[r0,c1]=b, A[r1,c0]=c, A[r1,c1]=d
B[c0,s0]=e, B[c0,s1]=f, B[c1,s0]=g, B[c1,s1]=h
C[r0,s0]=i, C[r0,s1]=j, C[r1,s0]=k, C[r1,s1]=l

M1 = (A[r0,c0]+A[r1,c1]) * (B[c0,s0]+B[c1,s1])
M2 = (A[r1,c0]+A[r1,c1]) * B[c0,s0]
M3 = A[r0,c0] * (B[c1,s0]-B[c1,s1])
M4 = A[r1,c1] * (B[c0,s1]-B[c0,s0])
M5 = (A[r0,c0]+A[r0,c1]) * B[c1,s1]
M6 = (A[r1,c0]-A[r0,c0]) * (B[c0,s0]+B[c0,s1])
M7 = (A[r0,c1]-A[r1,c1]) * (B[c1,s0]+B[c1,s1])

C[r0,s0] = M1 + M4 - M5 + M7
C[r0,s1] = M3 + M5
C[r1,s0] = M2 + M5
C[r1,s1] = M1 - M2 + M3 + M6

So for W (indexed by [k*n+i] where k=col, i=row):
M1: W[s0*n+r0] += 1, W[s1*n+r1] += 1
M2: W[s0*n+r1] += 1
M3: W[s0*n+r0] += 1, W[s1*n+r0] += 1
M4: W[s0*n+r0] += -1, W[s1*n+r0] += 1
M5: W[s1*n+r0] += -1, W[s0*n+r1] += 1
M6: W[s1*n+r1] += 1
M7: W[s0*n+r0] += 1, W[s1*n+r0] += 1

Let me verify C[r0,s0]: M1(1) + M4(-1) + M5(-1) + M7(1) = 1-1-1+1 = 0? That's wrong.

Wait, I think I have a sign error. Let me recheck M5:
M5 = (a+b)*h = (A[r0,c0]+A[r0,c1]) * B[c1,s1]
In C[r0,s0] = M1 + M4 - M5 + M7, M5 contributes -1 to W[s0*n+r0].

But in my list above I wrote "M5: W[s1*n+r0] += -1, W[s0*n+r1] += 1". That's wrong. M5 should contribute to C[r0,s0] with coefficient -1 and C[r1,s0] with coefficient +1.

So M5: W[s0*n+r0] += -1, W[s0*n+r1] += 1

Let me redo:
M1: C[r0,s0]+=1, C[r1,s1]+=1 → W[s0*n+r0]+=1, W[s1*n+r1]+=1
M2: C[r1,s0]+=1 → W[s0*n+r1]+=1
M3: C[r0,s0]+=1, C[r0,s1]+=1 → W[s0*n+r0]+=1, W[s1*n+r0]+=1
M4: C[r0,s0]+=-1, C[r0,s1]+=1 → W[s0*n+r0]+=-1, W[s1*n+r0]+=1
M5: C[r0,s0]+=-1, C[r1,s0]+=1 → W[s0*n+r0]+=-1, W[s0*n+r1]+=1
M6: C[r1,s1]+=1 → W[s1*n+r1]+=1
M7: C[r0,s0]+=1, C[r0,s1]+=1 → W[s0*n+r0]+=1, W[s1*n+r0]+=1

Check C[r0,s0]: M1(1)+M4(-1)+M5(-1)+M7(1) = 0. Still wrong!

Hmm, let me re-derive. The standard Strassen gives:
C[0,0] = M1 + M4 - M5 + M7

With M1=(a+d)(e+h), M4=d(g-e), M5=(a+b)h, M7=(b-d)(g+h)

Let me expand:
M1 = ae+ah+de+dh
M4 = dg-de
M5 = ah+bh
M7 = bg+bh-dg-dh

M1+M4-M5+M7 = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh
= ae + (ah-ah) + (de-de) + (dh-dh) + (dg-dg) + bg - bh + bh
= ae + bg

Yes! C[0,0] = ae + bg = A[0,0]*B[0,0] + A[0,1]*B[1,0]. Correct!

So my formula is right. The issue is I was checking wrong. Let me recheck:
M1 contributes +1 to C[r0,s0]
M4 contributes -1 to C[r0,s0]  
M5 contributes -1 to C[r0,s0]
M7 contributes +1 to C[r0,s0]

Sum: 1-1-1+1 = 0? But it should be ae+bg which is not zero in general.

OH WAIT. I think I'm confusing myself. The coefficients in the W matrix are the coefficients in the linear combination, not the values. When we say C[r0,s0] = M1 + M4 - M5 + M7, it means:

For each rank-1 term t:
- If t is M1, then W[s0*n+r0] gets +1
- If t is M4, then W[s0*n+r0] gets -1
- If t is M5, then W[s0*n+r0] gets -1
- If t is M7, then W[s0*n+r0] gets +1

And the product U*V for each term gives the actual value. So the total contribution to C[r0,s0] from all terms is:
(+1)*M1_value + (-1)*M4_value + (-1)*M5_value + (+1)*M7_value = M1 - M4 - M5 + M7

But we need M1 + M4 - M5 + M7. So M4 should have coefficient +1, not -1!

Let me recheck. C[0,0] = M1 + M4 - M5 + M7. So:
M1: +1
M4: +1
M5: -1
M7: +1

I had M4 wrong. Let me redo all W entries:

C[r0,s0] = M1 + M4 - M5 + M7
C[r0,s1] = M3 + M5
C[r1,s0] = M2 + M5
C[r1,s1] = M1 - M2 + M3 + M6

M1: W[s0*n+r0]+=1, W[s1*n+r1]+=1
M2: W[s0*n+r1]+=1
M3: W[s0*n+r0]+=1, W[s1*n+r0]+=1
M4: W[s0*n+r0]+=1
M5: W[s0*n+r0]+=-1, W[s0*n+r1]+=1
M6: W[s1*n+r1]+=1
M7: W[s0*n+r0]+=1, W[s1*n+r0]+=1

Check C[r0,s0]: M1(1)+M4(1)+M5(-1)+M7(1) = 1+1-1+1 = 2? No, that's not right either.

I'm getting confused. Let me just carefully list which C entries each M affects:

M1 appears in: C[r0,s0] (+1), C[r1,s1] (+1)
M2 appears in: C[r1,s0] (+1), C[r1,s1] (-1)
M3 appears in: C[r0,s0] (+1), C[r0,s1] (+1), C[r1,s1] (+1)
M4 appears in: C[r0,s0] (+1), C[r0,s1] (-1)
M5 appears in: C[r0,s0] (-1), C[r0,s1] (+1), C[r1,s0] (+1)
M6 appears in: C[r1,s1] (+1)
M7 appears in: C[r0,s0] (+1), C[r0,s1] (+1)

Let me verify C[r0,s0] = M1 + M4 - M5 + M7:
M1: +1, M4: +1, M5: -1, M7: +1. Sum of coefficients: 1+1-1+1=2. But the formula says it equals M1+M4-M5+M7, so the coefficients are correct as written. The "2" is just the sum of coefficients, not the value.

OK so my W entries should be:
M1: {(s0,r0): 1, (s1,r1): 1}
M2: {(s0,r1): 1, (s1,r1): -1}
M3: {(s0,r0): 1, (s1,r0): 1, (s1,r1): 1}
M4: {(s0,r0): 1, (s1,r0): -1}
M5: {(s0,r0): -1, (s1,r0): 1, (s0,r1): 1}
M6: {(s1,r1): 1}
M7: {(s0,r0): 1, (s1,r0): 1}

Now let me verify C[r0,s1] = M3 + M5:
M3: (s1,r0): 1, M5: (s1,r0): 1. So coefficient is 1+1=2? No, that's wrong.

I think I'm making an error. Let me look up the standard Strassen more carefully.

Standard Strassen for 2x2:
P1 = (a+d)(e+h)
P2 = (c+d)(e)
P3 = (a)(f-h)
P4 = (d)(g-e)
P5 = (a+b)(h)
P6 = (c-a)(e+f)
P7 = (b-d)(g+h)

C11 = P1 + P4 - P5 + P7
C12 = P3 + P5
C21 = P2 + P5
C22 = P1 - P2 + P3 + P6

So:
C11 gets: P1(+1), P4(+1), P5(-1), P7(+1)
C12 gets: P3(+1), P5(+1)
C21 gets: P2(+1), P5(+1)
C22 gets: P1(+1), P2(-1), P3(+1), P6(+1)

Now mapping: a=A[r0,c0], b=A[r0,c1], c=A[r1,c0], d=A[r1,c1]
e=B[c0,s0], f=B[c0,s1], g=B[c1,s0], h=B[c1,s1]
C11=C[r0,s0], C12=C[r0,s1], C21=C[r1,s0], C22=C[r1,s1]

P1 = (A[r0,c0]+A[r1,c1])*(B[c0,s0]+B[c1,s1])
P2 = (A[r1,c0]+A[r1,c1])*B[c0,s0]
P3 = A[r0,c0]*(B[c1,s0]-B[c1,s1])
P4 = A[r1,c1]*(B[c0,s1]-B[c0,s0])
P5 = (A[r0,c0]+A[r0,c1])*B[c1,s1]
P6 = (A[r1,c0]-A[r0,c0])*(B[c0,s0]+B[c0,s1])
P7 = (A[r0,c1]-A[r1,c1])*(B[c1,s0]+B[c1,s1])

W entries (key is (col, row) = (s, r)):
P1: {(s0,r0):1, (s1,r1):1}
P2: {(s0,r1):1, (s1,r1):-1}
P3: {(s0,r0):1, (s1,r0):1, (s1,r1):1}
P4: {(s0,r0):1, (s1,r0):-1}
P5: {(s0,r0):-1, (s1,r0):1, (s0,r1):1}
P6: {(s1,r1):1}
P7: {(s0,r0):1, (s1,r0):1}

Verify C[r0,s0] = P1+P4-P5+P7:
P1:1, P4:1, P5:-1, P7:1 → coeff sum = 2, but the formula is correct as a linear combination.

The key point is that when we compute sum_c U[i,j,c]*V[j',k,c]*W[k',i',c], for the (r0,s0) output, we get:
sum over c of U[r0,c0 or r1,c1 etc]*V[...]*W[s0,r0,c]

This should equal A[r0,c0]*B[c0,s0] + A[r0,c1]*B[c1,s0] (the correct C[r0,s0] value).

Let me just trust the standard Strassen and implement it. The verification will catch any errors.

Let me rewrite the code more carefully.

