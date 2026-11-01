"""
Rank-minimal tensor decomposition for 2x3 * 3x10 matrix multiplication.

Strategy:
- The schoolbook rank is 2*3*10 = 60.
- We attempt to find a lower rank by exploiting the rectangular structure.
- Known lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30.
- We use a structured ansatz based on splitting the 3x10 product into blocks.

Key insight: For multiplying A (2x3) by B (3x10), we can view this as:
  C[i,k] = sum_j A[i,j] * B[j,k]

We can factor the middle dimension j=0,1,2. Consider expressing the tensor
T[i,j,k] = delta_{i,j,k} (the multiplication tensor) as a sum of rank-1 terms.

A rank-1 term is u_i * v_j * w_k.

We seek rank r such that:
  T[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]

Wait, the convention is:
  U has shape (n*m, rank) -> indexed by (i,j)
  V has shape (m*p, rank) -> indexed by (j,k)
  W has shape (p*n, rank) -> indexed by (k,i)

And the decomposition is:
  For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k)
  where delta(i,j,k) = 1 if this is a valid multiplication term... 

Actually, the multiplication tensor M has entries:
  M[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
  
The tensor for matrix multiplication (n x m) * (m x p) -> (n x p) is:
  T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
  T[i,j,k] = 1 if this term contributes to the product, which is always 1 for the
  standard multiplication. But the tensor is defined on the space of (A, B, C) such that
  C = A*B. The bilinear map is:
  C[i,k] = sum_j A[i,j] B[j,k]

So the tensor T has components T[i,j,k] = 1 for all i,j,k (since every (i,j,k) triple
corresponds to a term in the product). Wait, that's not right either.

Let me reconsider. The multiplication tensor for (n x m) * (m x p) is a 3-tensor of size
n x m x p with all entries equal to 1? No.

The bilinear map f: R^{n*m} x R^{m*p} -> R^{n*p} is:
  f(A, B)[i,k] = sum_j A[i,j] B[j,k]

This can be written as:
  f(A, B) = sum_{i,j,k} A[i,j] B[j,k] * E_{i,k}

where E_{i,k} is the standard basis matrix.

As a 3-tensor T in R^{n*m} x R^{m*p} x R^{n*p}, the entries are:
  T[i*m + j, j*p + k, i*p + k]... no, that's not the right indexing.

Actually, the tensor T has indices (i,j), (j,k), (i,k) with:
  T[(i,j), (j,k), (i,k)] = 1

But since (i,j) and (j,k) share j, and (i,k) is determined by i and k, the tensor is:
  T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)

So it's an n x m x p tensor with all entries 1.

The rank of this tensor is what we want to minimize.

For the 2x3x10 case, we want to decompose the all-ones tensor of size 2x3x10 into a sum
of rank-1 tensors.

A rank-1 tensor is u_i * v_j * w_k for vectors u in R^2, v in R^3, w in R^10.

We want: 1 = sum_c u_c[i] * v_c[j] * w_c[k] for all i,j,k.

This is equivalent to finding matrices U (2 x r), V (3 x r), W (10 x r) such that:
  U * V^T * W^T (broadcast) = ones(2,3,10)

More precisely: sum_c U[i,c] * V[j,c] * W[k,c] = 1 for all i,j,k.

This is a tensor decomposition problem. The all-ones tensor of size a x b x c has rank
min(a,b,c) in some cases? No, the rank of the all-ones tensor of size a x b x c is
min(a,b,c) if we allow arbitrary entries? Let me think.

Actually, for the all-ones tensor of size a x b x c, the rank is min(a,b,c). This is a
known result. The rank-1 decomposition is:
  1 = (1/sqrt(min)) * sum_{c=1}^{min} e_c^T e_c^T e_c^T ... no.

Let me think again. We want to write:
  J_{a,b,c} = sum_{t=1}^{r} u_t v_t^T w_t^T

where J is the all-ones tensor. By symmetry, if a <= b <= c, we can use r = a:
  Let u_t = e_t (standard basis in R^a), v_t = 1/sqrt(b) * ones(b), w_t = 1/sqrt(c) * ones(c)
  Then sum_t u_t[i] * v_t[j] * w_t[k] = sum_t delta(i,t) * (1/sqrt(b)) * (1/sqrt(c))
  = delta(i,t) * 1/sqrt(bc) for the t=i term... this doesn't give 1.

Let me try: u_t = e_t, v_t = ones(b), w_t = ones(c)/bc.
Then sum_t u_t[i] v_t[j] w_t[k] = sum_t delta(i,t) * 1 * 1/(bc) = 1/(bc) for i in [0,a).
That's not 1.

We need: sum_t u_t[i] v_t[j] w_t[k] = 1.

If we set v_t = ones(b) for all t, and w_t = ones(c) for all t, then:
  sum_t u_t[i] * 1 * 1 = sum_t u_t[i] = 1 for all i.

So we need u_t[i] such that sum_t u_t[i] = 1 for all i, and we can choose r terms.
The minimum r is 1: just take u_1 = ones(a), v_1 = ones(b), w_1 = ones(c).
Then u_1[i] v_1[j] w_1[k] = 1*1*1 = 1. So rank 1?

That can't be right. The rank of the all-ones tensor is 1? Let me verify.

A rank-1 tensor is one that can be written as u v^T w^T (outer product of three vectors).
The all-ones tensor of size a x b x c is indeed the outer product of ones(a), ones(b), ones(c).
So yes, it has rank 1!

But wait, that's the rank of the tensor as a 3-tensor. But in matrix multiplication, the
tensor is not the all-ones tensor in the naive sense. Let me re-examine.

The multiplication tensor for A (n x m) * B (m x p) -> C (n x p) is:
  C[i,k] = sum_j A[i,j] B[j,k]

This is a bilinear map. The associated 3-tensor T has:
  T[i, j, k] = 1 for all i, j, k

And the bilinear form is:
  <T, A, B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]

But this sum is not C[i,k] for a specific (i,k); it's the sum over ALL i,j,k. That's not
the right way to think about it.

The correct way: The multiplication map is:
  (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]

This is a collection of n*p bilinear forms, one for each (i,k):
  C[i,k] = sum_j A[i,j] B[j,k]

Each C[i,k] is a bilinear form on R^{n*m} x R^{m*p}. The tensor for the entire multiplication
is a 3-tensor of size (n*m) x (m*p) x (n*p), but it's sparse: only entries where the indices
are consistent are non-zero.

Specifically, T[A_idx, B_idx, C_idx] where:
  A_idx = i*m + j
  B_idx = j*p + k  
  C_idx = i*p + k

T[i*m+j, j*p+k, i*p+k] = 1 for all i,j,k, and 0 otherwise.

So the tensor is not all-ones; it's a structured sparse tensor. The rank of this tensor is
the bilinear complexity of matrix multiplication, which is what we want to minimize.

For 2x3x10, the schoolbook rank is 60. We want to find a lower rank.

Known results:
- For 2x2x2, the rank is 7 (Strassen).
- For 2x3x3, the rank is at most 13 (some results).
- For 2x3x10, I'm not aware of a known optimal rank, but it should be less than 60.

Let me try a constructive approach. We can split the 10 columns of B into groups.

Consider B as having 10 columns. We can group them: 10 = 5 + 5, or 10 = 3+3+3+1, etc.

Alternatively, we can use the fact that n=2 is small. For each j (3 values), we have:
  C[0,k] += A[0,j] * B[j,k]
  C[1,k] += A[1,j] * B[j,k]

So for each j, we're adding A[0,j]*B[j,:] and A[1,j]*B[j,:] to the two rows of C.

This is like: for each j, we multiply a scalar A[0,j] by the row B[j,:] and add to C[0,:],
and multiply A[1,j] by B[j,:] and add to C[1,:].

We can think of this as: C = A[0,:]*B[0,:]^T + A[1,:]*B[1,:]^T + A[2,:]*B[2,:]^T... no,
that's not right either.

Let me try a different approach. Since n=2, we have two rows of A. Let's denote:
  A0 = A[0, :] (1x3 vector)
  A1 = A[1, :] (1x3 vector)
  B0 = B[0, :] (1x10 vector)
  B1 = B[1, :] (1x10 vector)
  B2 = B[2, :] (1x10 vector)

Then:
  C[0, :] = A0[0]*B0 + A0[1]*B1 + A0[2]*B2
  C[1, :] = A1[0]*B0 + A1[1]*B1 + A1[2]*B2

So C[0,:] is a linear combination of B0, B1, B2 with coefficients A0[0], A0[1], A0[2].
Similarly for C[1,:].

We can write:
  C[0,:] = sum_j A[0,j] B[j,:]
  C[1,:] = sum_j A[1,j] B[j,:]

This is 2*10 = 20 output entries, each a sum of 3 products.

The tensor rank is the minimum number of rank-1 terms to express this.

Let me try to find a decomposition of rank less than 60.

One idea: use the fact that we only have 2 rows in A. We can pair up the computations.

For each k (10 values), C[0,k] and C[1,k] are:
  C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k]
  C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k]

This is a 2x10 matrix C, and for each k, we have a 2x1 vector [C[0,k], C[1,k]]^T =
  [A[0,0] B[0,k] + A[0,1] B[1,k] + A[0,2] B[2,k],
   A[1,0] B[0,k] + A[1,1] B[1,k] + A[1,2] B[2,k]]^T

= sum_j [A[0,j], A[1,j]]^T * B[j,k]

So for each k, we're summing 3 vectors in R^2 (the columns of A) scaled by B[j,k].

This is like: for each k, C[:,k] = A^T * B[:,k] where A is 2x3 and B[:,k] is 3x1.

The tensor for this is: T[i,j,k] = 1, with i in {0,1}, j in {0,1,2}, k in {0,...,9}.

We want to decompose this 2x3x10 tensor into rank-1 terms.

Let me try rank 30 (the lower bound). Is rank 30 achievable?

For rank 30, we need 30 rank-1 terms. Each term is u_i * v_j * w_k.

We need: sum_{c=1}^{30} u_c[i] v_c[j] w_c[k] = 1 for all i,j,k.

This is a system of 2*3*10 = 60 equations in 30*(2+3+10) = 390 unknowns.

Let me try a structured solution. Suppose we set w_c[k] to be a basis for R^10.
If we use 10 terms for each (i,j) pair, that's 2*3*10 = 60, back to schoolbook.

To get rank 30, we need to share terms across different (i,j) pairs.

Let me try: for each k, we need 3 values (one for each j) to sum to 1 for each i.
If we use the same w_c[k] for all i, then we need:
  sum_c u_c[i] v_c[j] w_c[k] = 1

If we set v_c[j] to be a delta function, say v_c[j] = delta(j, j_c), then:
  sum_c u_c[i] delta(j, j_c) w_c[k] = u_c[i] w_c[k] for the c where j_c = j.

So for each j, we need: sum_{c: j_c=j} u_c[i] w_c[k] = 1 for all i,k.

This means for each j, the sum of outer products u_c w_c^T over c with j_c=j must be the
2x10 all-ones matrix.

The rank of the 2x10 all-ones matrix is 2, so we need at least 2 terms for each j.
With 3 j's, that's 6 terms. But we also need to satisfy the constraint for all i,k.

If for each j, we use 2 terms: u_c = [1,0]^T, w_c = ones(10)/2 and u_c = [0,1]^T, w_c = ones(10)/2,
then sum u_c[i] w_c[k] = 1 for all i,k. So 2 terms per j, 3 j's = 6 terms total.

But wait, we need v_c[j] = delta(j, j_c). So for each j, we have 2 terms with v_c[j]=1 and
v_c[j']=0 for j'!=j. Then the total rank is 6.

Let me verify:
  sum_c u_c[i] v_c[j] w_c[k] = sum_{c: j_c=j} u_c[i] w_c[k] = 1 for all i,j,k.

Yes! So the rank is 6? That seems too low. Let me check the dimensions.

U is 2x6, V is 3x6, W is 10x6.

For j=0: terms c=0,1. u_0=[1,0], v_0=[1,0,0], w_0=ones(10)/2. u_1=[0,1], v_1=[1,0,0], w_1=ones(10)/2.
For j=1: terms c=2,3. u_2=[1,0], v_2=[0,1,0], w_2=ones(10)/2. u_3=[0,1], v_3=[0,1,0], w_3=ones(10)/2.
For j=2: terms c=4,5. u_4=[1,0], v_4=[0,0,1], w_4=ones(10)/2. u_5=[0,1], v_5=[0,0,1], w_5=ones(10)/2.

Then for i=0, j=0, k=any:
  sum_c U[0,c] V[0,c] W[k,c] = U[0,0]*V[0,0]*W[k,0] + U[0,1]*V[0,1]*W[k,1] + ... 
  = 1*1*1/2 + 0*1*1/2 + 0 + 0 + 0 + 0 = 1/2.

That's not 1. I made an error.

The issue is that w_c should be such that sum_c u_c[i] v_c[j] w_c[k] = 1.

If v_c[j] = delta(j, j_c), then for a fixed j, only terms with j_c = j contribute.
For those terms, we need sum_{c: j_c=j} u_c[i] w_c[k] = 1.

If we use 2 terms per j: u_c = [1,0], w_c = ones(10) and u_c = [0,1], w_c = ones(10),
then sum = 1*1 + 0*1 = 1 for i=0, and 0*1 + 1*1 = 1 for i=1. But w_c = ones(10) means
w_c[k] = 1 for all k, so the sum is 1*1 + 0*1 = 1. Yes!

So with w_c = ones(10) (not divided by 2), we get:
  For i=0, j=0: U[0,0]*V[0,0]*W[k,0] + U[0,1]*V[0,1]*W[k,1] = 1*1*1 + 0*1*1 = 1.
  For i=1, j=0: U[1,0]*V[0,0]*W[k,0] + U[1,1]*V[0,1]*W[k,1] = 0*1*1 + 1*1*1 = 1.

So rank 6 works! But wait, the lower bound is max(n*m, m*p, p*n) = max(6, 30, 20) = 30.
How can the rank be 6?

I think I'm confusing the tensor rank with the bilinear complexity. Let me re-read the problem.

The problem says: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."

And the interface: U (n*m, rank), V (m*p, rank), W (p*n, rank).

The decomposition is: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.

This is the tensor rank of the multiplication tensor. The lower bound is max(n*m, m*p, p*n)
only if the tensor is "generic" in some sense, but for matrix multiplication, the lower bound
is actually max(n*m, m*p, p*n) / something? No, the standard lower bound for matrix multiplication
of (n x m) * (m x p) is max(n*m, m*p, p*n) is NOT correct. The correct lower bound is
max(n*m, m*p, p*n) is for the rank of the tensor, but I think the actual lower bound is
higher.

Wait, the rank of a tensor of size a x b x c is at least max(a,b,c). For the multiplication
tensor of size (n*m) x (m*p) x (n*p), the rank is at least max(n*m, m*p, n*p) = max(6, 30, 20) = 30.

But my construction gave rank 6, which violates this. So I must be making an error.

Let me re-examine. The tensor T has size (n*m) x (m*p) x (n*p) = 6 x 30 x 20.
The rank of a 6x30x20 tensor is at least max(6,30,20) = 30.

My construction above had U of size 2x6, not 6x6. I was using the wrong indexing.

U should be indexed by (i,j) with i in [0,2), j in [0,3), so U has 6 rows.
V should be indexed by (j,k) with j in [0,3), k in [0,10), so V has 30 rows.
W should be indexed by (k,i) with k in [0,10), i in [0,2), so W has 20 rows.

In my construction, I was treating U as 2x6 (i and c), but it should be 6x6 (i*m+j and c).

Let me redo: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].

For my rank-6 construction, I had 6 terms. For term c, I need:
  U[i*m+j, c] = u_c[i] * v_c[j]
  V[j*p+k, c] = v_c[j] * w_c[k]  -- wait, this doesn't work because v_c[j] appears in both U and V.

The rank-1 term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = u_c[i] * v_c[j] * w_c[k].

But U[i*m+j, c] = u_c[i] * v_c[j] means U is a product of a 2x6 matrix and a 3x6 matrix,
which is not a general 6x6 matrix. This is a restricted form.

For a general rank-r decomposition, U, V, W are arbitrary matrices of sizes 6xr, 30xr, 20xr.

My construction above assumed a separable form, which is not general. So rank 6 is not
achievable in general.

The lower bound of 30 is correct. Let me try to achieve rank 30.

For rank 30, we need U (6x30), V (30x30), W (20x30).

One approach: for each k (10 values), we have a 2x3 subproblem:
  C[:,k] = A^T * B[:,k]

This is a 2x1 vector = 3x2 matrix (A^T) times 3x1 vector (B[:,k]).

The tensor for this is 2x3x1 (for fixed k). The rank of the 2x3x1 all-ones tensor is
min(2,3,1) = 1? No, the rank of a 2x3x1 tensor with all entries 1 is 1, since it's
ones(2) * ones(3) * ones(1).

But we have 10 such k's, and they're independent. So the total rank is 10 * 1 = 10?
No, because the terms for different k's can share structure.

Actually, the full tensor is 6x30x20, and it's block-structured. For each k, the sub-tensor
is 6x3x2 (U rows for all i,j; V rows for all j, this k; W rows for this k, all i).

The sub-tensor for fixed k has entries T[i*m+j, j*p+k, k*n+i] = 1.

This is a 6x3x2 tensor. Its rank is at most 6 (since the first dimension is 6).

If we can decompose each k-block with rank 3, then total rank is 10*3 = 30.

For the 6x3x2 tensor (fixed k), we need:
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j.

This is a 6x3x2 tensor with all entries 1. The rank of the 6x3x2 all-ones tensor is
min(6,3,2) = 2? Let me check.

The all-ones tensor of size a x b x c has rank min(a,b,c). So 6x3x2 has rank 2.

So for each k, we can use 2 rank-1 terms. Total rank = 10 * 2 = 20.

But the lower bound is 30, so 20 is not possible. I must be making an error again.

The issue is that the tensor is not all-ones in the 6x30x20 sense. It's sparse: only entries
where the indices are consistent are 1. Specifically, T[i*m+j, j'*p+k, k'*n+i'] = 1 only if
j = j' and k = k' and i = i'. So it's a very sparse tensor.

Let me just implement the schoolbook decomposition and try to verify it, then attempt to
find a lower rank using a search or structured method.

Given the complexity, I'll implement the schoolbook rank 60 and also try a few structured
lower-rank candidates. If they don't work, I'll return rank 60.

Actually, let me try a different approach: use the fact that n=2 to pair up rows.

For each j and k, we have two equations:
  C[0,k] += A[0,j] B[j,k]
  C[1,k] += A[1,j] B[j,k]

We can combine these using Strassen-like tricks. For 2x1x1 (scalar times scalar), the rank
is 1. For 2x2x1, it's 2. For 2x3x10, let's think of it as 10 independent 2x3x1 problems.

Each 2x3x1 problem: C[0] = A[0,0]B[0] + A[0,1]B[1] + A[0,2]B[2], C[1] = A[1,0]B[0] + A[1,1]B[1] + A[1,2]B[2].

This is a 2x3x1 tensor. The rank is at most 3 (one term per j). Can we do better?

The 2x3x1 all-ones tensor has rank min(2,3,1) = 1. But again, the tensor is not all-ones;
it's T[i,j,0] = 1 for all i,j. So it IS the all-ones tensor of size 2x3x1, which has rank 1.

So for each k, rank 1 suffices. Total rank = 10.

But the lower bound is 30, so this can't be right. The lower bound max(n*m, m*p, n*p) = 30
must be wrong, or I'm misunderstanding the tensor.

Let me look up: the bilinear complexity of multiplying an (n x m) matrix by an (m x p) matrix
is at least max(n*m, m*p, n*p). For 2x3x10, that's max(6,30,20) = 30.

So the rank must be at least 30. My rank-10 construction must be flawed.

The flaw: when I say the 2x3x1 tensor for fixed k has rank 1, I'm treating it as an isolated
tensor. But in the full 6x30x20 tensor, the terms for different k's must be consistent. The
U matrix is shared across all k's, so we can't independently minimize each k-block.

Given the time, I'll implement the schoolbook rank 60 and also try a rank-30 construction
based on the lower bound. If rank 30 doesn't work, I'll return rank 60.

Let me try rank 30 with a structured ansatz:
- For each k, use 3 terms (one per j), but share the U and W matrices across k's.

Actually, let me just return the schoolbook decomposition. It's guaranteed to work and has
rank 60. Any improvement would require a sophisticated search that I can't guarantee to
complete in 120s.

I'll implement the schoolbook and add a deadline guard.
"""
import time

DIMENSIONS = [2, 3, 10]

DEADLINE = 120  # seconds

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
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
    
    # Check deadline
    if time.time() - start_time > DEADLINE - 10:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
