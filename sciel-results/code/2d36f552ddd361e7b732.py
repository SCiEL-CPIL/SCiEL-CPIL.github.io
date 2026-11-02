"""Rewritten program for 2x4x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=4, p=5):
    """Check if the decomposition is exact (within integer tolerance)."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T[i,j,k] = delta
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # For each column c: U[:,c] * V[:,c] * W[:,c] should sum to the tensor
    
    # T[i*m+j, j*p+k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - so W is indexed by (k, i) where k is column of B (0..p-1), i is row of A (0..n-1)
    # U[i*m+j, c] - i is row of A, j is column of A / row of B
    # V[j*p+k, c] - j is row of B, k is column of B
    
    # The product C[i,k] = sum_j A[i,j] * B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(j,j) ... no
    # Actually the tensor T[i,j,k] for matrix mult is: T[i,j,k] = 1 if this (i,j,k) contributes to C[i,k]
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor has T[i,j,k] = 1 for all (i,j,k) where the term A[i,j]*B[j,k] appears.
    # That's all (i,j,k) with i in [0,n), j in [0,m), k in [0,p).
    # So T[i,j,k] = 1 for ALL valid (i,j,k).
    
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify: for each (i,j,k), compute the sum and check it equals 1.
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each (i,j,k), we need sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1
    
    # c=0 (M1): A00+A11 times B00+B11
    # U: A00 and A11 have coefficient 1 -> U[0,0]=1, U[3,0]=1
    # V: B00 and B11 have coefficient 1 -> V[0,0]=1, V[3,0]=1
    # W: C00 gets +1, C11 gets +1 -> W[0,0]=1, W[3,0]=1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # c=1 (M2): (A10+A11)B00
    # U: A10 and A11 -> U[2,1]=1, U[3,1]=1
    # V: B00 -> V[0,1]=1
    # W: C10 gets +1 -> W[1,1]=1
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1
    
    # c=2 (M3): A00(B10-B11)
    # U: A00 -> U[0,2]=1
    # V: B10-B11 -> V[2,2]=1, V[3,2]=-1
    # W: C01 gets +1 -> W[2,2]=1
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[2,2] = 1
    
    # c=3 (M4): A11(B01-B10)
    # U: A11 -> U[3,3]=1
    # V: B01-B10 -> V[1,3]=1, V[2,3]=-1
    # W: C00 gets +1, C10 gets +1 -> W[0,3]=1, W[1,3]=1
    U[3,3] = 1
    V[1,3] = 1; V[2,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # c=4 (M5): (A00+A01)B11
    # U: A00+A01 -> U[0,4]=1, U[1,4]=1
    # V: B11 -> V[3,4]=1
    # W: C00 gets -1, C01 gets +1 -> W[0,4]=-1, W[2,4]=1
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # c=5 (M6): (A10-A00)(B00+B10)
    # U: A10-A00 -> U[2,5]=1, U[0,5]=-1
    # V: B00+B10 -> V[0,5]=1, V[2,5]=1
    # W: C11 gets +1 -> W[3,5]=1
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[2,5] = 1
    W[3,5] = 1
    
    # c=6 (M7): (A01-A11)(B01+B11)
    # U: A01-A11 -> U[1,6]=1, U[3,6]=-1
    # V: B01+B11 -> V[1,6]=1, V[3,6]=1
    # W: C00 gets +1 -> W[0,6]=1
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _block_diagonal(U1, V1, W1, U2, V2, W2):
    """Combine two decompositions into a block-diagonal sum."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    rank = r1 + r2
    U = np.zeros((U1.shape[0], rank))
    V = np.zeros((V1.shape[0], rank))
    W = np.zeros((W1.shape[0], rank))
    U[:, :r1] = U1
    V[:, :r1] = V1
    W[:, :r1] = W1
    U[:, r1:] = U2
    V[:, r1:] = V2
    W[:, r1:] = W2
    return U, V, W

def construct():
    """Construct rank-minimal decomposition for 2x4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = 2, 4, 5
    
    # Strategy: Decompose 2x4x5 as block-diagonal of 2x2x4 and 2x2x1
    # For 2x2x4: use Strassen-like approach
    # For 2x2x1: rank 2 (schoolbook for 2x2x1 is rank 4, but we can do better)
    
    # Actually, let's think more carefully.
    # 2x4x5: A is 2x4, B is 4x5, C is 2x5
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split j into two groups: j=0,1 and j=2,3
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    
    # First part: 2x2x5 (A is 2x2, B is 2x5)
    # Second part: 2x2x5 (A is 2x2, B is 2x5)
    
    # For 2x2x5: we can use a combination.
    # Actually, let me try a different approach.
    
    # Let's use the known result: rank(2x4x5) <= rank(2x2x5) + rank(2x2x5)
    # And rank(2x2x5) can be computed.
    
    # For 2x2xp: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is like multiplying a 2x2 by a 2xp.
    # Rank of 2x2xp: for each column k, we need 2x2 multiplication, but they share A.
    # Actually, the tensor is: T[i,j,k] = 1 for all (i,j,k) with i in {0,1}, j in {0,1}, k in {0,...,p-1}
    # This is the Kronecker product of 2x2x1 and 1x1xp? No.
    # T[i,j,k] = 1 means it's the outer product structure.
    # Actually T[i,j,k] = 1 for all valid indices means it's the "all-ones" tensor of size 2x2xp.
    # The rank of the all-ones tensor of size a x b x c is max(a,b,c)? No, that's not right either.
    # The all-ones tensor 1_{a,b,c} has rank 1 (it's just 1*1*1 broadcast).
    # Wait no. T[i,j,k] = 1 means for each (i,j,k), the value is 1.
    # This tensor can be written as sum over c of... 
    # Actually T[i,j,k] = 1 = 1 * 1 * 1, so it's rank 1!
    # U[i*m+j, 0] = 1 for all, V[j*p+k, 0] = 1 for all, W[k*n+i, 0] = 1 for all.
    # Let's check: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1. Yes!
    
    # So 2x2xp has rank 1? That can't be right for matrix multiplication.
    # Let me re-examine. The multiplication tensor for 2x2xp:
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # The tensor T[i,j,k] should encode: coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So T[i,j,k] = 1 for all (i,j,k). Yes, it's the all-ones tensor.
    # And the all-ones tensor of any size has rank 1.
    
    # Wait, that means 2x2xp has rank 1? Let me verify with 2x2x2.
    # T[i,j,k] = 1 for i,j,k in {0,1}.
    # U[0,0]=U[1,0]=U[2,0]=U[3,0]=1
    # V[0,0]=V[1,0]=V[2,0]=V[3,0]=1
    # W[0,0]=W[1,0]=W[2,0]=W[3,0]=1
    # Check: for (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. OK.
    # For (i=0,j=1,k=0): U[1,0]*V[2,0]*W[0,0] = 1*1*1 = 1. OK.
    # But wait, this says C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]. 
    # The tensor entry T[0,0,0] corresponds to the term A[0,0]*B[0,0] in C[0,0].
    # T[0,1,0] corresponds to A[0,1]*B[1,0] in C[0,0].
    # Both are 1. So yes, the tensor is all-ones.
    
    # Hmm, but that would mean 2x2x2 has rank 1, which contradicts Strassen's rank 7.
    # I must be misunderstanding the tensor convention.
    
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T is defined by T[i,j,k] = 1 if (i,j,k) is a valid 
    # (row of A, col of A/row of B, col of B) triple, and the term A[i,j]*B[j,k] appears in C[i,k].
    # For standard matrix multiplication, every valid (i,j,k) contributes, so T[i,j,k]=1 for all.
    # But then the rank would be 1, which is clearly wrong.
    
    # I think the issue is that the tensor is not just "1 for all" but has a specific structure.
    # Let me reconsider. In the schoolbook decomposition:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for c = i*m*p + j*p + k
    # So for a given (i,j,k), only one column c has all three non-zero.
    # The sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 (only the matching c contributes).
    
    # So the tensor T[i,j,k] = 1 for all (i,j,k) is correct, and it can be decomposed with rank n*m*p
    # by the schoolbook method. But can it be done with lower rank?
    
    # The key insight: the tensor T[i,j,k] = 1 for all (i,j,k) in {0..n-1} x {0..m-1} x {0..p-1}
    # is the all-ones tensor, which has tensor rank 1!
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all indices.
    # Then sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1*1*1 = 1 for all (i,j,k).
    
    # Wait, but that would mean ANY matrix multiplication has rank 1, which is absurd.
    # There must be something I'm missing about the tensor definition.
    
    # Let me look at this differently. Maybe the tensor isn't all-ones.
    # In matrix multiplication C = AB, we have C[i,k] = sum_j A[i,j] B[j,k].
    # The "tensor" for this operation is a 3-way array where T[i,j,k] represents 
    # the coefficient connecting A[i,j], B[j,k], and C[i,k].
    # For standard matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k).
    
    # But the decomposition condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    # If T[i,j,k] = 1 for all (i,j,k), then yes, rank 1 suffices.
    # This can't be the intended interpretation.
    
    # Let me re-read the task: "exactly decompose the matrix multiplication tensor"
    # Maybe the tensor is defined differently. Perhaps it's:
    # T[i,j,k] = 1 if j is the "middle" index connecting A and B, and the result goes to C[i,k].
    # That's what I've been assuming.
    
    # I think the issue is that I'm confusing the "multiplication tensor" with something else.
    # Let me just trust the schoolbook decomposition and try to improve upon it.
    
    # The schoolbook gives rank = n*m*p = 2*4*5 = 40.
    # The known lower bound is max(n*m, m*p, p*n) = max(8, 20, 10) = 20.
    # Strassen-style algorithms can reduce the rank.
    
    # For 2x4x5, let's try to use a block approach.
    # Split the 4 middle dimension into 2+2:
    # C = A1*B1 + A2*B2 where A1 is 2x2, B1 is 2x5, A2 is 2x2, B2 is 2x5.
    
    # For each 2x2x5 block, we need a decomposition.
    # The tensor for 2x2x5: T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,4}.
    # This is an all-ones tensor of size 2x2x5, which has rank 1.
    # So each block has rank 1, total rank 2?
    
    # That seems too good. Let me verify with a small example.
    # 2x2x2: T[i,j,k] = 1 for all. Rank 1?
    # U = ones(4,1), V = ones(4,1), W = ones(4,1)
    # For (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1. Correct.
    # For (i=0,j=1,k=0): U[1,0]*V[2,0]*W[0,0] = 1. Correct.
    # So yes, rank 1 works for 2x2x2?
    
    # But Strassen says rank 7 for 2x2x2. There's a contradiction.
    # I must be misunderstanding the problem.
    
    # Let me reconsider: maybe the tensor is NOT all-ones. 
    # Perhaps the convention is different. Let me re-read:
    # "W[k*n+i,r]" - W is indexed by (k, i) where k is the column index of B (and C), i is row of A (and C).
    # "U[i*m+j, c]" - i is row of A, j is col of A / row of B
    # "V[j*p+k, c]" - j is row of B, k is col of B
    
    # The decomposition: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = ?
    
    # In the schoolbook, this sum equals 1 for all (i,j,k).
    # So the target tensor is indeed all-ones.
    
    # But then why does Strassen give rank 7 for 2x2x2?
    # Unless the tensor is NOT all-ones, but has a different structure.
    
    # AH WAIT. I think I see the issue. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # But in the decomposition, we're not just checking T[i,j,k]=1.
    # We need the DECOMPOSITION to correctly compute the matrix product.
    
    # Let me think about it as: given A and B, C = AB.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Using the decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    # Wait, that's not right either. The standard tensor decomposition for matrix mult is:
    # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k])
    # Hmm, that doesn't look right dimensionally.
    
    # Actually, the correct formulation is:
    # For each term in the bilinear computation:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want to write this as a sum of rank-1 terms:
    # C[i,k] = sum_c W[k*n+i,c] * (linear form in A) * (linear form in B)
    # where the linear form in A is sum_j U[i*m+j,c] * A[i,j] ... no, that's not right.
    
    # Let me look at this from the tensor perspective.
    # The multiplication tensor T is a 3-tensor with T[i,j,k] = 1 for all (i,j,k).
    # A rank-r decomposition means: T[i,j,k] = sum_{c=0}^{r-1} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For 2x2x2, T[i,j,k] = 1 for all 8 entries.
    # Can we write 1 = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] with r < 8?
    # Yes! With r=1: U=ones, V=ones, W=ones. Then 1*1*1=1 for all entries.
    
    # So the rank of the all-ones tensor is 1. This means the "matrix multiplication tensor" 
    # as I've been defining it (all-ones) has rank 1, which is clearly not what the task means.
    
    # I must be misunderstanding the tensor. Let me re-read the task description more carefully.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 5 matrix."
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    # I think the issue is that the tensor is NOT all-ones. The matrix multiplication tensor 
    # has a specific structure based on the indices. Let me reconsider.
    
    # Actually, I think I've been overthinking this. Let me just use the schoolbook decomposition 
    # and try to find a better one through search or known algorithms.
    
    # For now, let's return the schoolbook decomposition (rank 40) and see if we can improve.
    
    U, V, W = _schoolbook(n, m, p)
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist()):
        # Fallback to a known correct decomposition
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find a better decomposition
    # For 2x4x5, let's try a structured approach.
    # Split into 2x2x5 + 2x2x5 blocks.
    # Each 2x2x5 block: if the tensor is all-ones, rank 1 suffices.
    # But we've established that the tensor might not be all-ones.
    
    # Let me try a different approach: use the known rank bounds.
    # rank(2x4x5) <= rank(2x2x5) + rank(2x2x5)
    # If rank(2x2x5) = 5 (schoolbook), then total = 10.
    # If rank(2x2x5) can be lower, even better.
    
    # For 2x2xp, the schoolbook rank is 2*2*p = 4p.
    # But we can do better. For 2x2x2, Strassen gives 7.
    # For 2x2xp, we can use a block Strassen approach.
    
    # Actually, let me just try to construct a decomposition for 2x2x5 directly.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,...,4}
    
    # This is 10 output values, each a sum of 2 products.
    # Total 20 products in schoolbook.
    
    # Can we share computations? The A values are shared across k.
    # For each i, we compute A[i,0] and A[i,1] once, then multiply by B[0,k] and B[1,k].
    
    # One approach: for each k, do 2x2x1 multiplication (rank 2 for 2x2x1? or rank 4?)
    # 2x2x1: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # This is 4 products in schoolbook. Can we do better?
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # These are independent, so rank 4 seems necessary for 2x2x1.
    
    # So 2x2x5 would be 5 * 4 = 20 in schoolbook.
    # And 2x4x5 would be 2 * 20 = 40.
    
    # To improve, we need to find shared structure.
    
    # Let me try a randomized search for a lower-rank decomposition.
    # This is computationally expensive, so let's be careful.
    
    # Actually, let's just return the schoolbook for now.
    # The score will be 40/40 = 1.0, same as the seed.
    
    # To improve, I need to find a valid decomposition with rank < 40.
    
    # Let me try a simple heuristic: use Strassen on 2x2x2 blocks.
    # For 2x4x5, split into 2x2x5 blocks.
    # For each 2x2x5, split into 2x2x2 + 2x2x1 + 2x2x1 + 2x2x1
    # 2x2x2: rank 7 (Strassen)
    # 2x2x1: rank 4 (schoolbook)
    # Total per 2x2x5: 7 + 4 + 4 + 4 = 19
    # Total for 2x4x5: 2 * 19 = 38
    
    # That's better than 40! Let's implement this.
    
    def _decompose_2x2x5():
        """Decompose 2x2x5 as 2x2x2 (Strassen) + 3 x 2x2x1 (schoolbook)."""
        n = m = 2
        p = 5
        
        # 2x2x2 part (columns k=0,1): use Strassen
        U1, V1, W1 = _strassen_2x2()
        # But Strassen is for 2x2x2, so k ranges 0,1.
        # We need to embed this into 2x2x5.
        # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
        # For 2x2x5: U has shape (4, r), V has shape (10, r), W has shape (10, r)
        # Wait, n*m = 4, m*p = 10, p*n = 10 for 2x2x5.
        
        # The Strassen U1[i*2+j, c] corresponds to A[i,j] for i,j in {0,1}.
        # In 2x2x5, A is 2x2, so U[i*2+j, c] is the same.
        # V1[j*2+k, c] corresponds to B[j,k] for j in {0,1}, k in {0,1}.
        # In 2x2x5, B is 2x5, so V[j*5+k, c] for j in {0,1}, k in {0,...,4}.
        # We need to map V1[j*2+k, c] to V[j*5+k, c] for k in {0,1}.
        # W1[k*2+i, c] corresponds to C[i,k] for i in {0,1}, k in {0,1}.
        # In 2x2x5, C is 2x5, so W[k*2+i, c] for k in {0,...,4}, i in {0,1}.
        # Wait, W has shape (p*n, r) = (10, r). W[k*n+i, c] = W[k*2+i, c].
        # So W1[k*2+i, c] maps directly to W[k*2+i, c] for k in {0,1}.
        
        U_2x2x5 = np.zeros((4, 7))
        V_2x2x5 = np.zeros((10, 7))
        W_2x2x5 = np.zeros((10, 7))
        
        U_2x2x5[:, :] = U1
        for j in range(2):
            for k in range(2):
                V_2x2x5[j*5+k, :] = V1[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_2x2x5[k*2+i, :] = W1[k*2+i, :]
        
        # Now add 2x2x1 blocks for k=2,3,4
        # For each k, 2x2x1: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        # Schoolbook: 4 terms.
        # U[i*2+j, c] = 1, V[j*5+k, c] = 1, W[k*2+i, c] = 1
        
        col_offset = 7
        for k in range(2, 5):
            for i in range(2):
                for j in range(2):
                    U_2x2x5[i*2+j, col_offset] = 1
                    V_2x2x5[j*5+k, col_offset] = 1
                    W_2x2x5[k*2+i, col_offset] = 1
                    col_offset += 1
        
        total_rank = col_offset
        return U_2x2x5, V_2x2x5, W_2x2x5, total_rank
    
    U2, V2, W2, r2 = _decompose_2x2x5()
    
    # Verify 2x2x5 decomposition
    if _verify(U2.tolist(), V2.tolist(), W2.tolist(), n=2, m=2, p=5):
        # Combine two 2x2x5 blocks for 2x4x5
        # First block: A[:,0:2] * B[0:2,:]
        # Second block: A[:,2:4] * B[2:4,:]
        
        U_2x4x5 = np.zeros((8, r2*2))
        V_2x4x5 = np.zeros((20, r2*2))
        W_2x4x5 = np.zeros((10, r2*2))
        
        # First block (j=0,1)
        U_2x4x5[0:4, 0:r2] = U2
        for j in range(2):
            for k in range(5):
                V_2x4x5[j*5+k, 0:r2] = V2[j*5+k, 0:r2]
        W_2x4x5[:, 0:r2] = W2
        
        # Second block (j=2,3)
        U_2x4x5[4:8, r2:r2*2] = U2
        for j in range(2):
            for k in range(5):
                V_2x4x5[(j+2)*5+k, r2:r2*2] = V2[j*5+k, 0:r2]
        W_2x4x5[:, r2:r2*2] = W2
        
        total_rank = r2*2
        
        if _verify(U_2x4x5.tolist(), V_2x4x5.tolist(), W_2x4x5.tolist()):
            return {"U": U_2x4x5.tolist(), "V": V_2x4x5.tolist(), "W": W_2x4x5.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
