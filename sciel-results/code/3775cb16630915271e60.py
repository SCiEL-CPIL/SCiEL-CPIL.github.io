"""
Rank-minimal exact tensor decomposition for 2x3 * 3x5 matrix multiplication.

We search for the minimum rank R such that there exist integer/half-integer
matrices U (6 x R), V (15 x R), W (10 x R) satisfying:
    for all i in [0,2), j in [0,3), k in [0,5):
        sum_{c=0}^{R-1} U[i*3+j, c] * V[j*5+k, c] * W[k*2+i, c] = 1
    for all other (i,j,k) triples (where the product is not part of the
    multiplication tensor), the sum = 0.

The multiplication tensor T_{i,j,k} = delta_{i,j,k} (1 if (i,j,k) is a valid
matrix multiplication index, 0 otherwise). We need a rank decomposition.

Lower bound: rank >= max(n, m, p) = 5. Also rank >= n*m = 6, m*p = 15, p*n = 10.
Actually the rank must be at least max(n*m, m*p, p*n) = 15? No, that's not right.
The rank of the tensor is at least the rank of any slice. The i-slice is a 3x5 matrix
of rank at most 3, so that doesn't help. 

Known results: For 2x3 x 3x5, the rank is known to be 15 (the schoolbook).
Actually, let me think more carefully. The trivial rank is n*m*p = 30.
Can we do better?

For 2x2 x 2x2, rank is 7 (Strassen). For 2x3 x 3x3, I'm not sure of the exact rank.

Let me try a computational approach: search for low-rank decompositions using
random initialization and iterative refinement, trying ranks from a lower bound upward.

Lower bound on rank: The tensor T has "slice rank" considerations. 
- For each i, the slice T_i is a 3x5 matrix (entries V and W contracted). 
  Actually, the slice along the first mode: for fixed i, we have a 3x5 matrix 
  M_i where M_i[j,k] = T[i,j,k]. These are all either all-1 or all-0 matrices
  depending on whether the (i,j,k) combination is valid. Wait, no - in matrix
  multiplication, T[i,j,k] = 1 for ALL valid (i,j,k) combinations. So every
  slice is either all-1 (if i is valid) or... wait, all i,j,k combinations are
  valid in the sense that the tensor entry is 1 for every (i,j,k) in [0,n)x[0,m)x[0,p).
  
  Actually no! In standard matrix multiplication, the tensor is:
  T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
  So T[i,j,k] = 1 for ALL i in [0,n), j in [0,m), k in [0,p).
  
  That means the tensor is just the all-ones tensor! Its rank is 1!
  
  Wait, that can't be right. Let me re-read the problem.
  
  "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix 
   by a 3 by 5 matrix."
   
  The multiplication tensor for (n,m,p) has entries T_{i,j,k} where:
  - i indexes rows of A (and C): 0..n-1
  - j indexes columns of A / rows of B: 0..m-1  
  - k indexes columns of B (and C): 0..p-1
  
  And T_{i,j,k} = 1 for all valid (i,j,k). So yes, it's the all-ones tensor.
  
  But wait, that would mean rank 1 always, which is clearly wrong for the 
  purpose of this problem. Let me re-read the interface:
  
  "W[k*n+i, r]" - this is the trace convention.
  
  The decomposition is: C[i,k] = sum_j A[i,j] * B[j,k]
  
  In tensor form: C = sum_r U_r * V_r * W_r where:
  - U_r is an n x m matrix
  - V_r is an m x p matrix  
  - W_r is a p x n matrix (but stored as k*n+i, i.e., W[k*n+i])
  
  And the product is: sum_r (sum_{i,j} U_r[i,j] * A[i,j]) * (sum_{j,k} V_r[j,k] * B[j,k]) * ... 
  
  No wait, that's not right either. Let me think about this more carefully.
  
  The standard tensor decomposition for matrix multiplication:
  C[i,k] = sum_{j} A[i,j] B[j,k]
  
  This can be written as a tensor contraction. The multiplication tensor M has
  M[i,j,k] = 1, and:
  C[i,k] = sum_{i,j,k} M[i,j,k] A[i,j] B[j,k]
  
  A rank-r decomposition of M means:
  M[i,j,k] = sum_{c=1}^r u_c[i,j] v_c[j,k] w_c[k,i]
  
  where u_c is n x m, v_c is m x p, w_c is p x n.
  
  Then: C[i,k] = sum_c (sum_{i,j} u_c[i,j] A[i,j]) (sum_{j,k} v_c[j,k] B[j,k]) w_c[k,i]
  
  Hmm, that doesn't quite work dimensionally. Let me look at the interface again:
  
  "U (n*m, rank), V (m*p, rank), W (p*n, rank)"
  
  So U has n*m rows, indexed by (i,j) -> i*m+j
  V has m*p rows, indexed by (j,k) -> j*p+k
  W has p*n rows, indexed by (k,i) -> k*n+i
  
  The decomposition condition is:
  For the multiplication tensor, we need:
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
  
  This is exactly saying the all-ones tensor has rank r. But the all-ones tensor
  T[i,j,k] = 1 for all i,j,k does have rank 1! Just take U[i*m+j, 0] = 1, 
  V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
  
  That gives rank 1, which is trivially correct. But the current program uses
  rank 30 (schoolbook), so something is off.
  
  OH WAIT. I think I'm misunderstanding the tensor. In matrix multiplication,
  the tensor is NOT all ones. Let me reconsider.
  
  Actually, I think the issue is that the "multiplication tensor" in the context
  of tensor rank for matrix multiplication is defined differently. The standard
  definition is:
  
  M_{n,m,p} is the tensor such that:
  C[i,k] = sum_{i,j,k} M[i,j,k] A[i,j] B[j,k]
  
  where M[i,j,k] = 1 if the indices are "compatible" for the multiplication.
  In standard matrix multiplication, ALL combinations (i,j,k) contribute, so
  M[i,j,k] = 1 for all i,j,k.
  
  But that gives rank 1, which is absurd for a problem about finding minimal rank.
  
  Let me look at this from a different angle. Maybe the tensor is defined as:
  M[i,j,k] where the multiplication is:
  C[i,k] = sum_j A[i,j] B[j,k]
  
  And the tensor decomposition represents this as:
  C = sum_c (A * U_c) * (V_c * B) * W_c  or something similar.
  
  Actually, I think the correct interpretation is:
  
  The bilinear complexity of multiplying an (n x m) matrix by an (m x p) matrix
  is the rank of a specific tensor. The tensor is:
  
  T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (where e_i are basis vectors)
  
  No wait, that's still the all-ones tensor.
  
  I think I need to look at this differently. In the standard formulation:
  
  We want to compute C[i,k] = sum_j A[i,j] B[j,k] for all i,k.
  
  A rank-r algorithm uses r scalar multiplications. Each scalar multiplication
  computes some bilinear form of A and B, and then linear combinations give C.
  
  The tensor for this is:
  M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)}
  
  Wait no. The correct tensor is:
  M = sum_{j=0}^{m-1} (sum_i e_i^{(n)}) ⊗ e_j^{(m)} ⊗ (sum_k e_k^{(p)})
  
  No, that's not right either.
  
  Let me be very careful. The matrix multiplication map is:
  (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
  
  This is a bilinear map from (R^{n*m} x R^{m*p}) to R^{n*p}.
  
  The associated tensor M in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} is defined by:
  M(a, b, c) = sum_{i,j,k} M[i,j,k] a[i,j] b[j,k] c[i,k]
  
  For this to represent matrix multiplication, we need:
  sum_{i,j,k} M[i,j,k] a[i,j] b[j,k] c[i,k] = sum_{i,k} (sum_j a[i,j] b[j,k]) c[i,k]
  
  This holds when M[i,j,k] = 1 for all i,j,k (with the understanding that the
  sum over i,j,k on the left must equal the sum over i,k,j on the right).
  
  So M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
  
  The rank of this tensor... let me compute it for small cases.
  
  For n=m=p=1: M = 1, rank = 1.
  For n=2,m=1,p=1: M is a 2x1x1 tensor with all entries 1. 
    M[i,0,0] = 1 for i=0,1.
    This is e_0 + e_1 in the first mode, so rank = 1 (it's e_0 ⊗ e_0 ⊗ e_0 + e_1 ⊗ e_0 ⊗ e_0 = (e_0+e_1) ⊗ e_0 ⊗ e_0).
  
  Hmm, so for any n,m,p, the all-ones tensor has rank 1?
  
  M[i,j,k] = 1 = 1*1*1 for all i,j,k.
  So M = (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k).
  
  Yes! The all-ones tensor always has rank 1!
  
  But that can't be the intended problem, because the current program uses rank 30
  and the problem says "score is the rank" with lower being better, and 1.0 is the
  seed score.
  
  I must be misunderstanding the problem. Let me re-read the interface:
  
  "U (n*m, rank), V (m*p, rank), W (p*n, rank) with integer/half-integer entries 
   that exactly decompose the matrix multiplication tensor"
  
  And the W indexing: "W[k*n+i, r]"
  
  Maybe the tensor is NOT the all-ones tensor. Maybe it's a different tensor that
  encodes matrix multiplication in a non-trivial way.
  
  Actually, I think I've been confusing two different things. The "multiplication
  tensor" in the context of tensor rank for matrix multiplication is indeed the
  all-ones tensor, and its rank is 1. But that's the rank of the tensor as a
  multilinear form.
  
  The BILINEAR COMPLEXITY of matrix multiplication is different. It's the minimum
  number of scalar multiplications needed, which equals the rank of a DIFFERENT
  tensor, or more precisely, it's related to the rank but with a specific structure.
  
  Wait, I think I'm overcomplicating this. Let me look at the actual condition
  that must be satisfied:
  
  The decomposition must satisfy: for all i,j,k,
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
  
  If this is the only condition, then rank 1 works with all entries = 1.
  
  But maybe there are additional constraints? Like the decomposition must actually
  compute the matrix product, meaning:
  
  C[i,k] = sum_c (sum_{i',j'} U[i'*m+j', c] * A[i',j']) * (sum_{j',k'} V[j'*p+k', c] * B[j',k']) * W[k*n+i, c]
  
  No, that doesn't make sense either because the sums would be over all indices.
  
  I think the correct interpretation is:
  
  C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i'*m+j', c] * A[i',j']) * (sum_{j',k'} V[j'*p+k', c] * B[j',k'])
  
  No, that's a trilinear form, not bilinear.
  
  Let me think about this differently. In a rank-r matrix multiplication algorithm:
  
  1. Compute r intermediate values: z_c = (A * u_c) · (v_c * B) or similar
  2. Combine: C[i,k] = sum_c w_c[i,k] * z_c
  
  Where u_c is n x m, v_c is m x p, w_c is n x p.
  
  The condition is:
  sum_c u_c[i',j'] * v_c[j',k'] * w_c[i,k] = delta_{i,i'} * delta_{k,k'} * sum_j delta_{j,j'} ... 
  
  No, let's be precise:
  C[i,k] should equal sum_j A[i,j] B[j,k].
  
  Using the algorithm:
  C[i,k] = sum_c w_c[i,k] * (sum_{i',j'} u_c[i',j'] A[i',j']) * (sum_{j',k'} v_c[j',k'] B[j',k'])
  
  This is trilinear in A and B, which is wrong. Matrix multiplication is bilinear.
  
  I think the correct formulation is:
  C[i,k] = sum_c w_c[i,k] * z_c
  where z_c = sum_{j} (sum_i u_c[i,j] A[i,j]) * (sum_k v_c[j,k] B[j,k])
  
  No, that's still not right.
  
  OK let me look at this from the Strassen algorithm perspective. Strassen computes:
  M1 = (A00+A11)(B00+B11)
  M2 = (A10+A11)B00
  ...
  Then C00 = M1 - M4 + M6, etc.
  
  Each M_c is a product of a linear form in A and a linear form in B.
  So: M_c = (u_c · A) * (v_c · B) where u_c is n*m and v_c is m*p.
  Then: C[i,k] = sum_c w_c[i,k] * M_c = sum_c w_c[i,k] * (u_c · A) * (v_c · B)
  
  For this to equal sum_j A[i,j] B[j,k]:
  sum_c w_c[i,k] * u_c[i',j'] * v_c[j',k'] = delta_{i,i'} * delta_{k,k'} * (sum over j of delta_{j,j'} for the j that connects i and k)
  
  More precisely:
  sum_c w_c[i,k] * u_c[i',j'] * v_c[j',k'] = 1 if i=i' and k=k' and j'=j for some j... 
  
  No. The condition is:
  sum_c w_c[i,k] * u_c[i',j'] * v_c[j',k'] = delta_{i,i'} * delta_{k,k'} * sum_j delta_{j,j'}
  
  Wait, that's not right. Let me be very explicit:
  
  We want: C[i,k] = sum_j A[i,j] B[j,k]
  
  The algorithm computes: C'[i,k] = sum_c w_c[i,k] * (sum_{i',j'} u_c[i',j'] A[i',j']) * (sum_{j',k'} v_c[j',k'] B[j',k'])
  
  For C' = C for all A, B, we need:
  sum_c w_c[i,k] * u_c[i',j'] * v_c[j',k'] = sum_j delta_{i,i'} delta_{j,j'} delta_{k,k'} delta_{j,j'}
  
  Hmm, this is getting confusing. Let me use index notation more carefully.
  
  C[i,k] = sum_j A[i,j] B[j,k]
  
  C'[i,k] = sum_c w_c[i,k] * (sum_{i',j'} u_c[i',j'] A[i',j']) * (sum_{j',k'} v_c[j',k'] B[j',k'])
  
  = sum_c w_c[i,k] * sum_{i',j'} u_c[i',j'] A[i',j'] * sum_{j',k'} v_c[j',k'] B[j',k']
  
  = sum_{i',j',j',k'} (sum_c w_c[i,k] u_c[i',j'] v_c[j',k']) A[i',j'] B[j',k']
  
  For this to equal sum_j A[i,j] B[j,k] for all A,B:
  sum_c w_c[i,k] u_c[i',j'] v_c[j',k'] = sum_j delta_{i,i'} delta_{j,j'} delta_{k,k'} delta_{j,j'}
  
  The RHS is: for given (i,k,i',j',j',k'), it's 1 if i=i' and k=k' and j'=j for the j that appears... 
  
  Actually, sum_j delta_{i,i'} delta_{j,j'} delta_{k,k'} delta_{j,j'} = delta_{i,i'} delta_{k,k'} * (sum_j delta_{j,j'} delta_{j,j'}) = delta_{i,i'} delta_{k,k'} * 1 (if j' is a valid index) = delta_{i,i'} delta_{k,k'}
  
  Wait no. sum_j delta_{j,j'} delta_{j,j'} = sum_j delta_{j,j'} = 1 (since j' is a specific index and the sum over j picks out j=j').
  
  So the condition is:
  sum_c w_c[i,k] u_c[i',j'] v_c[j',k'] = delta_{i,i'} * delta_{k,k'}
  
  for all i,k,i',j',j',k'.
  
  Now, in the interface, we have:
  U[i*m+j, c] = u_c[i,j]  (flattened)
  V[j*p+k, c] = v_c[j,k]  (flattened)
  W[k*n+i, c] = w_c[i,k]  (note: indexed by k*n+i, so W[k*n+i] = w_c[i,k])
  
  And the condition becomes:
  sum_c U[i'*m+j', c] * V[j'*p+k', c] * W[k*n+i, c] = delta_{i,i'} * delta_{k,k'}
  
  for all i,k,i',j',j',k'.
  
  This is NOT the all-ones tensor! It's a tensor that is 1 when i=i' and k=k' (regardless of j'), and 0 otherwise.
  
  So the tensor T[i,k,i',j',j',k'] has a specific structure. But in the interface,
  the indices are (i*m+j, j*p+k, k*n+i), so the tensor is:
  
  T[a, b, c] where a = i*m+j, b = j*p+k, c = k'*n+i'
  
  And T[a,b,c] = 1 if i=i' and k=k' (where a=i*m+j, b=j'*p+k', c=k'*n+i')
  
  Wait, I need to be more careful. Let me redefine:
  
  a = i*m + j  (row of U)
  b = j'*p + k'  (row of V)  
  c = k*n + i  (row of W)
  
  The condition is:
  sum_c U[a, c] V[b, c] W[c, c] = 1 if (i from a) = (i from c) and (k from b) = (k from c)
  
  Where i from a = a // m, j from a = a % m
  j' from b = b // p, k' from b = b % p
  k from c = c // n, i from c = c % n
  
  So: sum_c U[a,c] V[b,c] W[c,c] = delta_{a//m, c%n} * delta_{b%p, c//n}
  
  This is the actual tensor we need to decompose!
  
  For n=2, m=3, p=5:
  - U is 6 x r
  - V is 15 x r
  - W is 10 x r
  
  And the tensor T has T[a,b,c] = 1 if a//3 == c%2 and b%5 == c//2, else 0.
  
  Let me verify with the schoolbook: U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
  for c = i*m*p + j*p + k (or some ordering).
  
  For a given (a,b,c): a = i*m+j, b = j'*p+k', c = k''*n+i''
  U[a,c] = 1 if c corresponds to (i,j) = (i,j), i.e., c = i*m*p + j*p + k for some k... 
  
  Actually in the schoolbook, each term (i,j,k) gets its own column. So:
  U[i*m+j, c_{i,j,k}] = 1
  V[j*p+k, c_{i,j,k}] = 1
  W[k*n+i, c_{i,j,k}] = 1
  
  And for a given (a,b,c):
  sum_c U[a,c] V[b,c] W[c,c] = number of columns c where U[a,c]=1, V[b,c]=1, W[c,c]=1
  
  U[a,c]=1 means c is a column for (i,j) = (a//m, a%m)
  V[b,c]=1 means c is a column for (j',k') = (b//p, b%p)
  W[c,c]=1 means c is a column for (k'',i'') = (c//n, c%n)
  
  So we need: a//m = i, a%m = j, b//p = j, b%p = k, c//n = k, c%n = i
  i.e., a//m = c%n (i matches), a%m = b//p (j matches), b%p = c//n (k matches)
  
  So the sum is 1 if a//m == c%n and a%m == b//p and b%p == c//n, else 0.
  
  But the condition we derived was: a//m == c%n and b%p == c//n.
  The schoolbook additionally requires a%m == b//p (the j indices match).
  
  So the schoolbook gives: T[a,b,c] = delta_{a//m, c%n} * delta_{a%m, b//p} * delta_{b%p, c//n}
  
  But we need: T[a,b,c] = delta_{a//m, c%n} * delta_{b%p, c//n}
  
  These are different! The schoolbook has an extra delta_{a%m, b//p} factor.
  
  So the schoolbook does NOT satisfy the condition? That can't be right.
  
  Let me re-derive. I think I made an error in the bilinear complexity formulation.
  
  OK I think the issue is that in the standard matrix multiplication tensor decomposition,
  the condition is different from what I derived. Let me look at this from first principles.
  
  We have A (n x m), B (m x p), C = A * B (n x p).
  C[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k]
  
  A rank-r decomposition means:
  C[i,k] = sum_{c=0}^{r-1} w_c[i,k] * z_c
  where z_c = sum_{i',j'} u_c[i',j'] A[i',j'] * sum_{j',k'} v_c[j',k'] B[j',k']
  
  NO. z_c must be a product of a linear form in A and a linear form in B:
  z_c = (sum_{i',j'} u_c[i',j'] A[i',j']) * (sum_{j',k'} v_c[j',k'] B[j',k'])
  
  Then: C[i,k] = sum_c w_c[i,k] * (sum_{i',j'} u_c[i',j'] A[i',j']) * (sum_{j',k'} v_c[j',k'] B[j',k'])
  
  For this to equal sum_j A[i,j] B[j,k]:
  
  Coefficient of A[i',j'] B[j'',k''] in C[i,k] must be delta_{i',i} delta_{j',j''} delta_{k'',k}
  
  Coefficient is: sum_c w_c[i,k] u_c[i',j'] v_c[j'',k'']
  
  So: sum_c w_c[i,k] u_c[i',j'] v_c[j'',k''] = delta_{i',i} delta_{j',j''} delta_{k'',k}
  
  In the flattened indices:
  a = i'*m + j'
  b = j''*p + k''
  c = k*n + i  (note: W is indexed by k*n+i, so the row index is k*n+i)
  
  Condition: sum_c U[a,c] V[b,c] W[c,c] = delta_{i',i} delta_{j',j''} delta_{k'',k}
  
  Where i' = a//m, j' = a%m, j'' = b//p, k'' = b%p, k = c//n, i = c%n.
  
  So: sum_c U[a,c] V[b,c] W[c,c] = delta_{a//m, c%n} * delta_{a%m, b//p} * delta_{b%p, c//n}
  
  THIS is the correct tensor! And the schoolbook satisfies it with rank n*m*p = 30.
  
  Now, can we do better? This is the standard matrix multiplication tensor rank problem.
  For 2x3 x 3x5, the minimum rank is what we need to find.
  
  Lower bounds:
  - rank >= n*m = 6 (from the U matrix having 6 rows)
  - rank >= m*p = 15 (from V having 15 rows)
  - rank >= n*p = 10 (from W having 10 rows)
  So rank >= 15.
  
  Upper bound: 30 (schoolbook).
  
  Known results for small matrix multiplication:
  - 2x2 x 2x2: rank 7 (Strassen)
  - 2x3 x 3x3: I'm not sure
  - 3x3 x 3x3: rank 21 (Laderman)
  
  For 2x3 x 3x5, I suspect the rank might be 15 or slightly above.
  
  Let me try to find a rank-15 decomposition. If rank >= 15 is a lower bound,
  maybe 15 is achievable.
  
  For rank 15, V must be 15x15 and have full rank (since V has 15 rows and rank 15
  columns, V must be invertible if it's square and full rank).
  
  Let me try a computational search. I'll use random initialization and gradient-free
  optimization, or I'll try to construct a decomposition analytically.
  
  Actually, let me try a different approach. I'll set up the equations and solve them.
  
  For rank r, we have:
  U: 6 x r
  V: 15 x r
  W: 10 x r
  
  The condition is: for all a in [0,6), b in [0,15), c in [0,10):
  sum_{t=0}^{r-1} U[a,t] V[b,t] W[c,t] = T[a,b,c]
  
  where T[a,b,c] = 1 if a//3 == c%2 and a%3 == b//5 and b%5 == c//2, else 0.
  
  Let me enumerate T:
  a = i*3+j, i in [0,2), j in [0,3)
  b = j'*5+k', j' in [0,3), k' in [0,5)
  c = k*2+i, k in [0,5), i in [0,2)
  
  T[a,b,c] = 1 if i == i (from c) and j == j' (from b) and k' == k (from c)
  
  So for each (i,j,k) with i in [0,2), j in [0,3), k in [0,5):
  a = i*3+j, b = j*5+k, c = k*2+i
  T[a,b,c] = 1
  
  And T[a,b,c] = 0 otherwise.
  
  There are 2*3*5 = 30 ones in T, and the total size is 6*15*10 = 900 entries.
  
  Now, the number of variables is r*(6+15+10) = 31r.
  The number of equations is 900 (but only 30 are non-trivial, the rest are = 0).
  
  Actually, all 900 equations must be satisfied. For the 30 non-zero entries, the sum
  must equal 1. For the 870 zero entries, the sum must equal 0.
  
  This is a system of 900 cubic equations in 31r variables. For r=15, that's 465 variables.
  
  This is a hard nonlinear system. Let me try a different approach.
  
  I'll try to use the fact that for rank 15, V is 15x15. If V is invertible, we can
  write U = T_contracted * V^{-1} * ... hmm, this is getting complicated.
  
  Let me just try a numerical search with random restarts and see if I can find a
  low-rank solution. I'll use the fact that the entries should be integers or half-integers.
  
  Actually, for a practical approach within the time limit, let me try:
  1. Start with the schoolbook (rank 30)
  2. Try to reduce the rank by merging terms
  3. Use random search for low-rank solutions
  
  Given the time constraints, let me try a simple approach: random initialization with
  small integer/half-integer values, and check if the decomposition is valid.
  
  Actually, let me think about this more carefully. The problem is asking for the
  MINIMUM rank. The current best is 30. Can we do better?
  
  For 2x3 x 3x5, I'll try to find if rank 15, 16, ..., 29 are achievable.
  
  Let me write a program that:
  1. Tries ranks from 15 upward
  2. For each rank, does a random search for a valid decomposition
  3. Returns the best (lowest rank) decomposition found within the time limit
  
  The search will use random integer/half-integer values in a small range.
  
  Actually, this is unlikely to work for rank > 15 because the search space is huge.
  
  Let me try a more structured approach. I'll use the fact that the tensor has a
  specific structure and try to exploit it.
  
  Alternatively, let me just try to verify if rank 15 is achievable by solving the
  equations directly.
  
  For rank 15:
  U: 6 x 15
  V: 15 x 15
  W: 10 x 15
  
  If V is invertible, then for each (a,c):
  sum_t U[a,t] V[b,t] W[c,t] = T[a,b,c] for all b
  
  This is: (U_row_a * V^T * W_row_c^T) = T_row_a_c  (as a vector of length 15)
  
  Or: U_row_a * (V^T * W_row_c^T) = T_row_a_c
  
  Hmm, this is getting complicated. Let me just try a brute-force random search with
  a time limit.
  
  Given the 150-second time limit, I'll do a quick random search for low ranks and
  fall back to the schoolbook if nothing is found.
"""

import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook
    rank_school = n * m * p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    def check(U, V, W):
        T_rec = np.einsum('at,bt,ct->abc', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)
    
    def snap(arr):
        return np.round(arr * 2) / 2.0
    
    def is_half_int(arr, tol=1e-9):
        return np.allclose(arr * 2, np.round(arr * 2), atol=tol)
    
    min_rank = max(n*m, m*p, n*p)  # 15
    
    # Precompute the linear system structure
    # For fixed U,V: W @ (U[a,:]*V[b,:])^T = T[a,b,:] for all (a,b)
    # Unknown: W (p*n x rank)
    # Eq for (a,b,c): sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
    
    # Try fixing U and V, solve for W
    for rank in range(min_rank, best_rank):
        if time.time() > deadline - 15:
            break
        
        max_attempts = 300 if rank <= 20 else 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random U, V with small half-integer entries
            vals = np.array([-1, -0.5, 0, 0.5, 1])
            U = np.random.choice(vals, size=(n*m, rank))
            V = np.random.choice(vals, size=(m*p, rank))
            
            # Build linear system: A @ W_flat = b
            # A has shape (n*m * m*p * p*n, rank * p*n)
            # b has shape (n*m * m*p * p*n,)
            num_eq = n*m * m*p * p*n  # 900
            num_unk = rank * p*n
        
            # Efficient construction
            # For eq index ((a * m*p + b) * p*n + c), unk index (c * rank + t):
            # A[eq, unk] = U[a,t] * V[b,t]
            
            # Build using broadcasting
            # U_exp: (n*m, 1, 1, rank)
            # V_exp: (1, m*p, 1, rank)
            # We need A[((a*mp+b)*pn+c), (c*rank+t)] = U[a,t]*V[b,t]
            
            # Let's build it in chunks for memory efficiency
            W_flat, residuals, rk, sv = np.linalg.lstsq(
                _build_A_UV(U, V, n, m, p), T.flatten(), rcond=None
            )
            W_sol = W_flat.reshape(p*n, rank)
            
            if np.allclose(np.einsum('at,bt,ct->abc', U, V, W_sol), T, atol=1e-9):
                if is_half_int(W_sol):
                    W_snapped = snap(W_sol)
                    if np.allclose(np.einsum('at,bt,ct->abc', U, V, W_snapped), T, atol=1e-9):
                        best_rank = rank
                        best_U, best_V, best_W = U, V, W_snapped
                        break
        
        if best_rank == rank:
            break
    
    # Try fixing V and W, solve for U
    for rank in range(min_rank, best_rank):
        if time.time() > deadline - 15:
            break
        
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            vals = np.array([-1, -0.5, 0, 0.5, 1])
            V = np.random.choice(vals, size=(m*p, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            U_flat, _, _, _ = np.linalg.lstsq(
                _build_A_VW(V, W, n, m, p), T.flatten(), rcond=None
            )
            U_sol = U_flat.reshape(n*m, rank)
            
            if np.allclose(np.einsum('at,bt,ct->abc', U_sol, V, W), T, atol=1e-9):
                if is_half_int(U_sol):
                    U_snapped = snap(U_sol)
                    if np.allclose(np.einsum('at,bt,ct->abc', U_snapped, V, W), T, atol=1e-9):
                        best_rank = rank
                        best_U, best_V, best_W = U_snapped, V, W
                        break
        
        if best_rank == rank:
            break
    
    # Try fixing U and W, solve for V
    for rank in range(min_rank, best_rank):
        if time.time() > deadline - 15:
            break
        
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            vals = np.array([-1, -0.5, 0, 0.5, 1])
            U = np.random.choice(vals, size=(n*m, rank))
            W = np.random.choice(vals, size=(p*n, rank))
            
            V_flat, _, _, _ = np.linalg.lstsq(
                _build_A_UW(U, W, n, m, p), T.flatten(), rcond=None
            )
            V_sol = V_flat.reshape(m*p, rank)
            
            if np.allclose(np.einsum('at,bt,ct->abc', U, V_sol, W), T, atol=1e-9):
                if is_half_int(V_sol):
                    V_snapped = snap(V_sol)
                    if np.allclose(np.einsum('at,bt,ct->abc', U, V_snapped, W), T, atol=1e-9):
                        best_rank = rank
                        best_U, best_V, best_W = U, V_snapped, W
                        break
        
        if best_rank == rank:
            break
    
    def clean_list(lst):
        result = []
        for row in lst:
            clean_row = []
            for val in row:
                snapped = round(val * 2) / 2.0
                if abs(snapped - round(snapped)) < 1e-10:
                    clean_row.append(int(round(snapped)))
                else:
                    clean_row.append(snapped)
            result.append(clean_row)
        return result
    
    return {
        "U": clean_list(best_U.tolist()),
        "V": clean_list(best_V.tolist()),
        "W": clean_list(best_W.tolist())
    }

def _build_A_UV(U, V, n, m, p):
    """Build A matrix for solving W given U,V.
    A[((a*mp+b)*pn+c), (c*rank+t)] = U[a,t]*V[b,t]
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    num_eq = nm * mp * pn
    num_unk = rank * pn
    
    A = np.zeros((num_eq, num_unk))
    
    # U: (nm, rank), V: (mp, rank)
    # For each (a,b): q = U[a,:]*V[b,:] (element-wise, shape rank)
    # For each c: A[((a*mp+b)*pn+c), c*rank:(c+1)*rank] = q
    
    for a in range(nm):
        for b in range(mp):
            q = U[a] * V[b]  # (rank,)
            base_eq = (a * mp + b) * pn
            for c in range(pn):
                A[base_eq + c, c*rank:(c+1)*rank] = q
    
    return A

def _build_A_VW(V, W, n, m, p):
    """Build A matrix for solving U given V,W.
    A[((a*mp+b)*pn+c), (a*rank+t)] = V[b,t]*W[c,t]
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = V.shape[1]
    num_eq = nm * mp * pn
    num_unk = rank * nm
    
    A = np.zeros((num_eq, num_unk))
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                q = V[b] * W[c]  # (rank,)
                A[(a * mp + b) * pn + c, a*rank:(a+1)*rank] = q
    
    return A

def _build_A_UW(U, W, n, m, p):
    """Build A matrix for solving V given U,W.
    A[((a*mp+b)*pn+c), (b*rank+t)] = U[a,t]*W[c,t]
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    num_eq = nm * mp * pn
    num_unk = rank * mp
    
    A = np.zeros((num_eq, num_unk))
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                q = U[a] * W[c]  # (rank,)
                A[(a * mp + b) * pn + c, b*rank:(b+1)*rank] = q
    
    return A
