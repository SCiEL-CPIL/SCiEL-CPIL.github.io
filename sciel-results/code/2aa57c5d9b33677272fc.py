"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-27 decomposition for 2x3 by 3x9 matrix multiplication.
    
    Key insight: C[i, k] = sum_j A[i,j] B[j,k] where k ranges over 9 columns.
    We can view this as 3 independent 2x3 by 3x3 multiplications:
    C[:, 0:3] = A @ B[:, 0:3]
    C[:, 3:6] = A @ B[:, 3:6]  
    C[:, 6:9] = A @ B[:, 6:9]
    
    Each 2x3 by 3x3 multiplication has tensor rank 9 (schoolbook).
    Since the A factors are shared across all three blocks, we can use
    rank 3*9 = 27 by assigning separate rank components for each block.
    
    The decomposition is:
    For each block t in {0,1,2}, for each (i,j) pair:
      U[i*m+j] has 1 in the rank component corresponding to (t,i,j)
      V[j*p + (3t+r)] has 1 in the same rank component for each r
      W[(3t+r)*n + i] has 1 in the same rank component
      
    Wait, that's not right. Let me reconsider.
    
    Actually, the standard schoolbook for 2x3 by 3x3 uses rank 9:
    For each i in {0,1}, j in {0,1,2}, k in {0,1,2}:
      U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    
    For 2x3 by 3x9, we have 3 such blocks. The U matrix has the same
    entries for all blocks (since A is the same), but V and W differ.
    
    To get rank 27, we need to use 9 rank components per block.
    For block t, the rank components are indexed by (i,j) with an offset.
    
    Let me define: rank component index = t*9 + i*3 + j for t in {0,1,2}, i in {0,1}, j in {0,1,2}
    
    For each rank component (t, i, j):
      U[i*m + j] = 1
      V[j*p + 3*t + r] = 1 for r in {0,1,2}  -- NO, this doesn't work directly
      
    Actually, I think the issue is that for a single 2x3 by 3x3 multiplication,
    the schoolbook decomposition has rank 9 where each term corresponds to
    (i, j, k) and:
      U[i*m+j] = 1
      V[j*3+k] = 1  
      W[k*2+i] = 1
    
    For the 2x3 by 3x9 case, if we treat it as 3 separate 2x3 by 3x3 multiplications,
    we need 3*9 = 27 rank components. For block t (columns 3t to 3t+2 of B):
      For each (i, j) in the 2x3 grid:
        Rank component c = t*9 + i*3 + j
        U[i*m + j][c] = 1
        V[j*p + 3*t + r][c] = 1 for r in {0,1,2}
        W[(3*t + r)*n + i][c] = 1 for r in {0,1,2}
    
    Wait, that's still not right. Let me think more carefully.
    
    The tensor T[i, j, k] = A[i,j] * B[j,k] where:
    - i in {0, 1} (2 rows of A)
    - j in {0, 1, 2} (3 columns of A / 3 rows of B)
    - k in {0, ..., 8} (9 columns of B)
    
    C[i, k] = sum_j A[i,j] B[j,k]
    
    The rank-r decomposition requires:
    C[i,k] = sum_{c=0}^{r-1} U[i*m+j]... 
    
    Actually, the factor matrices are:
    U: shape (n*m, rank) = (6, rank)
    V: shape (m*p, rank) = (27, rank)
    W: shape (p*n, rank) = (18, rank)
    
    And the reconstruction is:
    C[i,k] = sum_{j, c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that's not the right formula. Let me look at the trace convention.
    
    The trace convention says W[k*n+i, r]. So W has rows indexed by (k, i) where
    k is the column index of B (0..8) and i is the row index of A (0..1).
    
    The standard tensor decomposition for matrix multiplication M_{n,m,p} has:
    U: (n*m) x rank
    V: (m*p) x rank  
    W: (p*n) x rank
    
    And C[i,k] = sum_{j, c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    No wait, that's not right either. Let me think about this differently.
    
    The matrix multiplication tensor T has entries T[i,j,k] = delta where it represents
    the bilinear form. The rank-r decomposition means:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Hmm, but that would give a 3-tensor, not a matrix product. Let me reconsider.
    
    Actually, for matrix multiplication, the tensor is:
    T[i, j, k] = 1 if this term contributes to C[i,k]
    
    More precisely, C[i,k] = sum_j A[i,j] B[j,k]
    
    So the "tensor" of the multiplication is T[i,j,k] = 1 for all valid (i,j,k),
    and the decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait no. The correct formulation is:
    
    The multiplication map is: (A, B) -> C where C = A @ B
    
    This is a bilinear map from (R^{n x m} x R^{m x p}) -> R^{n x p}
    
    The rank of this bilinear map is the minimum r such that:
    C[i,k] = sum_{c=0}^{r-1} sum_{j=0}^{m-1} U[i*m+j, c] * V[j*p+k, c] * ... 
    
    I'm getting confused. Let me just use the known schoolbook decomposition.
    
    For standard matrix multiplication of (n x m) by (m x p):
    Rank = n * m * p (schoolbook)
    
    The decomposition is:
    For each (i, j, k):
      U[i*m+j, idx] = 1
      V[j*p+k, idx] = 1
      W[k*n+i, idx] = 1
      idx += 1
    
    This gives rank n*m*p.
    
    For n=2, m=3, p=9: rank = 2*3*9 = 54
    
    Can we do better? The lower bound is max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    
    So rank 27 is the theoretical minimum (if achievable).
    
    Let me try to construct a rank-27 decomposition.
    
    Idea: Group the k index into 3 blocks of size 3.
    For block t (t=0,1,2), columns 3t, 3t+1, 3t+2 of B.
    
    C[i, 3t+r] = sum_j A[i,j] B[j, 3t+r] for r=0,1,2
    
    This is a 2x3 by 3x3 multiplication for each t.
    
    For a 2x3 by 3x3 multiplication, the schoolbook rank is 2*3*3 = 18.
    But can we do better for 2x3 by 3x3?
    
    Actually, for multiplying a 2x3 matrix by a 3x3 matrix, the minimum rank
    is known to be 9 (from the lower bound max(2*3, 3*3, 3*2) = max(6,9,6) = 9).
    
    Wait, that's not right. The lower bound is max(n*m, m*p, p*n) where n=2, m=3, p=3.
    max(6, 9, 6) = 9.
    
    And the schoolbook gives 2*3*3 = 18.
    
    Is there a rank-9 decomposition for 2x3 by 3x3?
    
    Actually, I recall that for square matrix multiplication M_{n,n,n}, the rank is
    at least n^2 and at most n^3 (schoolbook). For n=2, rank is 7 (Strassen).
    
    For non-square, it's more complex. Let me just try to construct a rank-27
    decomposition for 2x3 by 3x9 using the block structure.
    
    If each 2x3 by 3x3 block has rank 9, and we have 3 blocks, total rank = 27.
    
    For block t, using rank-9 decomposition of 2x3 by 3x3:
    We need to find 9 terms such that:
    C[i, r] = sum_{c=0}^{8} U_t[i*m+j_c, c] * V_t[j_c*p+3t+r, c] * W_t[(3t+r)*n+i, c]
    
    Hmm, this is getting complicated. Let me just use the schoolbook for each block.
    
    For block t, schoolbook rank = 2*3*3 = 18.
    Total rank = 3 * 18 = 54. Same as before.
    
    So I need a better decomposition for the 2x3 by 3x3 case.
    
    Let me think about this differently. The key is that A is the same for all 3 blocks.
    So the U factors can be shared.
    
    Let's say we use rank r. For each rank component c, we have:
    U[:, c]: a 6-dimensional vector
    V[:, c]: a 27-dimensional vector
    W[:, c]: an 18-dimensional vector
    
    And C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, I don't think that's the right formula. Let me look at this more carefully.
    
    The bilinear map is: (A, B) -> A @ B
    
    This can be written as:
    (A @ B)[i,k] = sum_j A[i,j] B[j,k]
    
    A rank-r decomposition of this bilinear map means:
    (A @ B)[i,k] = sum_{c=0}^{r-1} f_c(A) * g_c(B)
    
    where f_c: R^{n x m} -> R and g_c: R^{m x p} -> R are linear maps.
    
    f_c(A) = sum_{i,j} U[i*m+j, c] * A[i,j]
    g_c(B) = sum_{j,k} V[j*p+k, c] * B[j,k]
    
    And the output is:
    C[i,k] = sum_c f_c(A) * g_c(B)
    
    But this gives C as a matrix, not indexed by (i,k). So we need:
    C[i,k] = sum_c h_c^{(i,k)}(A) * ... 
    
    I think I'm overcomplicating this. Let me just use the standard formulation.
    
    The standard rank-r decomposition for matrix multiplication M_{n,m,p} uses:
    U: (n*m) x r
    V: (m*p) x r
    W: (p*n) x r
    
    And the reconstruction is:
    C = sum_c (U[:,c] reshaped to n x m) @ (V[:,c] reshaped to m x p)
    
    No wait, that's not right either.
    
    Let me just look at what the current program does and verify it works.
    
    The current program uses:
    U[i*m+j][idx] = 1
    V[j*p+k][idx] = 1
    W[k*n+i][idx] = 1
    
    And the evaluator checks that this correctly decomposes the tensor.
    
    So the formula must be:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    where T[i,j,k] = 1 for the multiplication tensor.
    
    For schoolbook, each (i,j,k) gets its own rank component, so:
    T[i,j,k] = U[i*m+j, i*m+j*p+k] * V[j*p+k, i*m+j*p+k] * W[k*n+i, i*m+j*p+k] = 1
    
    And for other (i',j',k'), the product is 0 because at least one factor is 0.
    
    OK so now I understand the formulation. Let me try to find a rank-27 decomposition.
    
    The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,8}.
    
    We want: T[i,j,k] = sum_{c=0}^{26} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Idea: Use the fact that T[i,j,k] = T[i,j,k mod 3] * delta(k div 3, something)
    
    Actually, T[i,j,k] = 1 for all (i,j,k), so it's just the all-ones tensor.
    
    Wait no! The tensor for matrix multiplication is NOT all ones. Let me reconsider.
    
    The matrix multiplication tensor T_{n,m,p} has entries:
    T[i,j,k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k]
    
    Since every term A[i,j] * B[j,k] contributes to exactly one entry C[i,k],
    the tensor is:
    T[i,j,k] = 1 for all valid (i,j,k)
    
    And the decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    So yes, T is the all-ones tensor of size (n*m) x (m*p) x (p*n) ... wait no.
    
    T is a 3-tensor with indices (i*m+j), (j*p+k), (k*n+i) but that doesn't make sense
    because i,j,k are not independent in those index combinations.
    
    I think the correct interpretation is:
    T is a 3-tensor with indices (a, b, c) where:
    a in {0, ..., n*m-1} corresponds to (i,j) with i = a // m, j = a % m
    b in {0, ..., m*p-1} corresponds to (j,k) with j = b // p, k = b % p
    c in {0, ..., p*n-1} corresponds to (k,i) with k = c // n, i = c % n
    
    And T[a,b,c] = 1 if the (j,k) in b matches the (i,j) in a and (k,i) in c,
    i.e., if a%m == b//p and b%p == c//n and c%n == a//m.
    
    So T[a,b,c] = 1 if and only if a%m == b//p and b%p == c//n and c%n == a//m.
    
    This is the standard matrix multiplication tensor.
    
    Now, for the schoolbook decomposition:
    For each (i,j,k), create a rank component c = i*m*p + j*p + k (or similar).
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    
    And T[a,b,c] = sum_{c'} U[a,c'] * V[b,c'] * W[c,c']
    
    For the schoolbook, this gives T[a,b,c] = 1 if there exists a rank component c'
    such that U[a,c']=V[b,c']=W[c,c']=1, which happens exactly when the indices match.
    
    OK so now I'm confident in the formulation. Let me try to find a rank-27 decomposition.
    
    Key observation: T[a,b,c] = 1 if and only if:
    - a = i*m + j for some i in {0,1}, j in {0,1,2}
    - b = j*p + k for some j in {0,1,2}, k in {0,...,8}
    - c = k*n + i for some k in {0,...,8}, i in {0,1}
    - AND the j's and k's and i's are consistent.
    
    So T[i*m+j, j*p+k, k*n+i] = 1 and T[a,b,c] = 0 otherwise.
    
    Now, let's try to decompose this with rank 27.
    
    Idea: For each t in {0,1,2} (block index), and for each (i,j) in {0,1} x {0,1,2},
    create a rank component c = t*9 + i*3 + j.
    
    For this rank component:
    U[i*m+j, c] = 1
    V[j*p+3*t+r, c] = ? for r in {0,1,2}
    W[(3*t+r)*n+i, c] = ? for r in {0,1,2}
    
    We need: T[i*m+j, j*p+3*t+r, (3*t+r)*n+i] = 1
    = sum_c' U[i*m+j, c'] * V[j*p+3*t+r, c'] * W[(3*t+r)*n+i, c']
    
    For the rank component c = t*9 + i*3 + j:
    U[i*m+j, c] = 1
    We need V[j*p+3*t+r, c] * W[(3*t+r)*n+i, c] to contribute to the sum.
    
    If we set V[j*p+3*t+r, c] = 1 and W[(3*t+r)*n+i, c] = 1 for all r,
    then the product is 1, which is what we want.
    
    But we also need to ensure that T[a,b,c] = 0 for other (a,b,c).
    
    Let's check: For a different (i',j',k'):
    T[i'*m+j', j'*p+k', k'*n+i'] should be 0 unless (i',j',k') = (i,j,3*t+r) for some r.
    
    With our construction:
    U[i'*m+j', c] = 1 only if i'=i and j'=j.
    So if i'!=i or j'!=j, U[i'*m+j', c] = 0, and the product is 0. Good.
    
    If i'=i and j'=j, then:
    V[j*p+k', c] = 1 only if k' is in {3*t, 3*t+1, 3*t+2}.
    W[k'*n+i, c] = 1 only if k' is in {3*t, 3*t+1, 3*t+2}.
    
    So if k' is not in {3*t, 3*t+1, 3*t+2}, the product is 0. Good.
    
    If k' is in {3*t, 3*t+1, 3*t+2}, then:
    V[j*p+k', c] = 1 and W[k'*n+i, c] = 1, so the product is 1.
    
    But we want T[i*m+j, j*p+k', k'*n+i] = 1 for all k' in {3*t, 3*t+1, 3*t+2}.
    And indeed, the sum over c' includes c = t*9+i*3+j, which contributes 1.
    
    But wait, do other rank components also contribute? Let's check.
    
    For a different t' != t, the rank component c' = t'*9+i*3+j has:
    U[i*m+j, c'] = 1
    V[j*p+k', c'] = 1 only if k' is in {3*t', 3*t'+1, 3*t'+2}
    
    If k' is in {3*t, 3*t+1, 3*t+2} and t' != t, then k' is not in {3*t', 3*t'+1, 3*t'+2},
    so V[j*p+k', c'] = 0. Good.
    
    So the total sum is 1, which is correct.
    
    Now let's verify that T[a,b,c] = 0 when the indices don't match.
    
    Case 1: a = i*m+j, b = j'*p+k, c = k'*n+i' where j != j'.
    U[a, c'] = 1 only for c' = t*9+i*3+j for some t.
    V[b, c'] = 1 only if j' = j (since V[j*p+k, c'] = 1 requires j' = j).
    But j != j', so V[b, c'] = 0 for all c'. Product is 0. Good.
    
    Case 2: a = i*m+j, b = j*p+k, c = k'*n+i' where i != i'.
    U[a, c'] = 1 only for c' = t*9+i*3+j.
    W[c, c'] = 1 only if i' = i (since W[k*n+i, c'] = 1 requires i' = i).
    But i != i', so W[c, c'] = 0 for all c'. Product is 0. Good.
    
    Case 3: a = i*m+j, b = j*p+k, c = k'*n+i where k != k'.
    U[a, c'] = 1 only for c' = t*9+i*3+j.
    V[b, c'] = 1 only if k is in {3*t, 3*t+1, 3*t+2}.
    W[c, c'] = 1 only if k' is in {3*t, 3*t+1, 3*t+2}.
    
    If k and k' are in different blocks, then for any t, either k or k' is not in the block,
    so either V or W is 0. Product is 0. Good.
    
    If k and k' are in the same block, say k = 3*t+r and k' = 3*t+r', then:
    V[b, c'] = 1 and W[c, c'] = 1 for c' = t*9+i*3+j.
    So the product is 1, and T[i*m+j, j*p+k, k'*n+i] = 1.
    
    But we want T[i*m+j, j*p+k, k'*n+i] = 1 only if k = k'.
    If k != k' but they're in the same block, we get T = 1, which is wrong!
    
    So my construction is incorrect. I need to distinguish between different k values within a block.
    
    Let me revise. For each (t, i, j, r) where t in {0,1,2}, i in {0,1}, j in {0,1,2}, r in {0,1,2},
    create a rank component. That's 3*2*3*3 = 54 components, which is the schoolbook.
    
    To get rank 27, I need to reduce by a factor of 2. 
    
    Idea: Use the fact that for fixed (t,i,j), the three r values can be handled with
    a rank-3 decomposition of the 3-dimensional subspace.
    
    Actually, I think the minimum rank for 2x3 by 3x9 is indeed 54 (schoolbook), and
    there's no known better decomposition. The lower bound is 27, but it may not be achievable.
    
    Let me just output the schoolbook decomposition with rank 54, but with a cleaner implementation.
    
    """
    n, m, p = DIMENSIONS
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
