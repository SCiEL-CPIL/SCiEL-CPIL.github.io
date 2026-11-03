"""Matrix multiplication 3,5,7 - rank-minimal tensor decomposition."""
import time
import random

DIMENSIONS = [3, 5, 7]

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 145  # seconds, leave 5s margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _time_exceeded():
    return _elapsed() > TIME_LIMIT

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 3x5 matrix
    by a 5x7 matrix.
    
    Strategy: Partition the contraction dimension m=5 into blocks and apply
    known fast matrix multiplication algorithms to each block.
    
    The key insight is that we can partition the middle dimension and use
    Strassen-like decompositions for sub-blocks.
    
    For 3x5x7:
    - Split m=5 into 2+3
    - The 3x2x7 sub-problem and 3x3x7 sub-problem can each be handled
      with optimized decompositions.
    
    However, a simpler and very effective approach: use the fact that
    we can apply Strassen's algorithm ideas to reduce the effective rank.
    
    Schoolbook rank: 3*5*7 = 105
    
    Better approach: Use a block decomposition where we treat the 5x7 
    multiplication as combining smaller Strassen blocks.
    
    Actually, the most practical approach for small fixed dimensions is to
    use known optimal or near-optimal decompositions. Let's use a systematic
    approach:
    
    For multiplying A (3x5) by B (5x7):
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can use the following strategy:
    - Split the 5 contraction indices into groups
    - Apply Strassen (rank 7 for 2x2x2) where applicable
    - Use Laderman (rank 7 for 3x3x3) for 3x3 blocks
    
    Let's think about this more carefully.
    
    Partition: m=5 = 2 + 3
    - Block 1: A[:, 0:2] (3x2) * B[0:2, :] (2x7) -> 3x2x7 sub-tensor
    - Block 2: A[:, 2:5] (3x3) * B[2:5, :] (3x7) -> 3x3x7 sub-tensor
    
    For 3x2x7: We can further decompose. The rank of 3x2x2 is at most 4 (schoolbook)
    but can be 4. Actually for small matrices, let's just use schoolbook for 
    the small pieces but combine smartly.
    
    Alternative better strategy: Use a "Strassen packing" approach.
    
    Actually, let me use a direct approach that's known to work well:
    
    The rank of the 3x5x7 matrix multiplication tensor. Let's use a 
    constructive approach based on partitioning and applying known results.
    
    Known: rank(2x2x2) <= 7 (Strassen), rank(3x3x3) <= 7 (Laderman)
    
    Strategy:
    1. Split the 5x7 part: treat B as [B1; B2] where B1 is 2x7, B2 is 3x7
    2. C = A1*B1 + A2*B2 where A1 is 3x2, A2 is 3x3
    
    For A1*B1 (3x2x7): 
      Further split 7 into 2+2+2+1 or use direct computation
      Rank of 3x2x2 = 4 (can't do better than 2*2=4 for the 2x2 part, times 3 rows)
      Actually rank(3x2x2): we need to compute 3*2*2=12 terms schoolbook.
      Can we do better? For 3x2x2, the rank is exactly 4? No...
      Let me think: C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..2, j in 0..1, k in 0..1
      That's 3*2*2 = 12 scalar products. The tensor rank of 3x2x2...
      Actually it's at most 4 because we can use Strassen on the 2x2 part for each row?
      No, Strassen is for 2x2*2x2. Here it's 3x2 * 2x2.
      
      For 3x2 * 2x2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
      This is just 3*2 = 6 multiplications (each A[i,j] times B[j,k] for fixed j, summed over k).
      Wait no: for each (i,k), we have 2 terms. Total 3*2*2 = 12 products.
      But we can factor: for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
      That's 2 vector multiplications per row = 3*2 = 6 scalar mults per row, 18 total? No.
      Each "A[i,0]*B[0,k]" is one scalar mult. There are 3*2*2 = 12 such.
      
      Can we reduce? The tensor is of format 3x2x2. Its rank... 
      I believe rank(3x2x2) = 4. We can use:
      (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) - A[i,0]*B[0,k] - A[i,1]*B[1,k]... no that's more.
      
      Actually for 2x2x2, Strassen gives rank 7 < 8.
      For 3x2x2, we can think of it as 3 copies of 1x2x2, which has rank 2 each,
      total rank 6? Or we can do better.
      
      Let me just use schoolbook for the small pieces and focus on getting
      a good decomposition overall.
    
    Let me use a different, more systematic approach:
    
    Direct construction with rank reduction via Strassen on 2x2 blocks:
    
    Split all three dimensions:
    n=3 = 2+1, m=5 = 2+2+1, p=7 = 2+2+2+1
    
    This gives 2*3*4 = 24 sub-blocks. For each 2x2x2 sub-block, use Strassen (rank 7).
    For other sub-blocks, use schoolbook.
    
    Sub-block types and their ranks:
    - 2x2x2: rank 7 (Strassen)
    - 2x2x1: rank 4 (schoolbook, can't improve much)  
    - 2x1x2: rank 4
    - 1x2x2: rank 4
    - 2x1x1: rank 2
    - 1x2x1: rank 2
    - 1x1x2: rank 2
    - 1x1x1: rank 1
    
    Let me count:
    n-split: [2, 1]
    m-split: [2, 2, 1]  
    p-split: [2, 2, 2, 1]
    
    Sub-blocks (n_block, m_block, p_block) -> count * rank:
    (2,2,2): 1*1*1 = 1 block, rank 7
    (2,2,2): 1*1*2 = 1 block, rank 7  (p has three 2s)
    (2,2,2): 1*1*3 = 1 block, rank 7
    (2,2,1): 1*1*4 = 1 block, rank 4
    (2,2,1): 1*1*4 = 1 block, rank 4  (second m=2)
    (2,2,1): 1*1*4 = 1 block, rank 4  (third m=1... wait)
    
    Let me be more careful.
    
    n_blocks = [2, 1]  (sizes)
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 2, 1]
    
    For each combination (ni, mj, pk):
    size_n = n_blocks[ni], size_m = m_blocks[mj], size_p = p_blocks[pk]
    
    (2,2,2): appears for ni=0, mj in {0,1}, pk in {0,1,2} -> 1*2*3 = 6 blocks, rank 7 each
    (2,2,1): ni=0, mj in {0,1}, pk=3 -> 1*2*1 = 2 blocks, rank 4 each
    (2,1,2): ni=0, mj=2, pk in {0,1,2} -> 1*1*3 = 3 blocks, rank 4 each
    (2,1,1): ni=0, mj=2, pk=3 -> 1*1*1 = 1 block, rank 2
    (1,2,2): ni=1, mj in {0,1}, pk in {0,1,2} -> 1*2*3 = 6 blocks, rank 4 each
    (1,2,1): ni=1, mj in {0,1}, pk=3 -> 1*2*1 = 2 blocks, rank 2
    (1,1,2): ni=1, mj=2, pk in {0,1,2} -> 1*1*3 = 3 blocks, rank 2
    (1,1,1): ni=1, mj=2, pk=3 -> 1*1*1 = 1 block, rank 1
    
    Total rank = 6*7 + 2*4 + 3*4 + 1*2 + 6*4 + 2*2 + 3*2 + 1*1
              = 42 + 8 + 12 + 2 + 24 + 4 + 6 + 1
              = 99
    
    That's better than 105! Let's verify: 42+8=50, +12=62, +2=64, +24=88, +4=92, +6=98, +1=99.
    
    Yes, rank 99. That's an improvement over schoolbook 105.
    
    Can we do better? Let me try different splits.
    
    Try n=3=2+1, m=5=3+2, p=7=3+2+2:
    
    n_blocks = [2, 1]
    m_blocks = [3, 2]
    p_blocks = [3, 2, 2]
    
    (2,3,3): 1*1*1 = 1, rank(2x3x3) <= ? Use Laderman-like? Rank of 2x3x3...
    Actually for non-square, we need to be careful.
    
    Let me stick with the 2+1, 2+2+1, 2+2+2+1 split which gives 99.
    
    Actually, let me try to improve by using better decompositions for the 
    rectangular cases.
    
    For 2x2x1 (rank 4 schoolbook): C[i,k] = sum_j A[i,j]*B[j,k] where i in 0..1, j in 0..1, k=0
    That's just C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] for i=0,1.
    4 multiplications. Can't reduce below 4 since it's essentially two dot products of length 2.
    Actually rank is exactly 4 for 2x2x1? The tensor is C[i,k] with only k=0, so it's 
    really a 2x2 matrix in (i,j) times B[j,0]. The tensor T[i,j,k] = delta structure...
    For 2x2x1: T[i,j,0] = 1 if we're computing C[i,0] = sum_j A[i,j] B[j,0].
    The rank of this 2x2x1 tensor: it's the sum over j of e_i * e_j * e_0, which is 
    2 terms (j=0 and j=1), each being a 2x1x1 tensor. So rank = 2? 
    Wait: T[i,j,k] for the multiplication tensor. For 2x2x1:
    T[i,j,k] = 1 if k=0, else 0? No.
    
    The matrix multiplication tensor T[i,j,k] satisfies:
    C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    So T[i,j,k] = 1 for all i,j,k (it's just the identity in the summation).
    
    For 2x2x1: T[i,j,0] = 1 for i in {0,1}, j in {0,1}. That's a 2x2x1 tensor that's all ones.
    Rank of all-ones 2x2x1 tensor: it's e_0*e_0*e_0 + e_0*e_1*e_0 + e_1*e_0*e_0 + e_1*e_1*e_0
    = (e_0+e_1) ⊗ (e_0+e_1) ⊗ e_0... no that's not right.
    
    Actually T[i,j,k] = 1 for all i,j when k=0. This is a 2x2x1 array.
    It can be written as sum over j of (column j of A-identity) * (row j of B-identity) * e_k.
    More precisely: T = sum_j u_j ⊗ v_j ⊗ w_j where u_j[i] = delta_{i,?}... 
    
    Hmm, let me think differently. The multiplication tensor for n x m x p is:
    T = sum_{i,j,k} e_i^{(n)} ⊗ e_j^{(m)} ⊗ e_k^{(p)}
    
    So for 2x2x1: T = e_0⊗e_0⊗e_0 + e_0⊗e_1⊗e_0 + e_1⊗e_0⊗e_0 + e_1⊗e_1⊗e_0
    = (e_0+e_1) ⊗ (e_0+e_1) ⊗ e_0? No, that gives all entries as 1 in a 2x2x1, which is correct!
    Wait: (e_0+e_1)⊗(e_0+e_1)⊗e_0 has components [i,j,k] = (1 if i in {0,1})(1 if j in {0,1})(1 if k=0) = 1 for all i,j when k=0. Yes!
    
    So rank(2x2x1) = 1? That can't be right because then we could compute 2x2x1 multiplication with 1 multiplication.
    
    Oh wait, I'm confusing myself. The tensor T is defined such that:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This means T[i,j,k] = 1 for all valid i,j,k. The tensor is just the all-ones tensor of shape (n,m,p).
    
    But the rank of the all-ones tensor of shape (n,m,p) is min(n,m,p) in general? No...
    
    Actually, the all-ones tensor of shape (n,m,p) has rank 1 if and only if it can be written as a single outer product. The all-ones tensor 1_{n×m×p} = 1_n ⊗ 1_m ⊗ 1_p, which IS a single outer product! So its rank is 1.
    
    But that would mean any matrix multiplication can be done with rank 1, which is absurd.
    
    I'm making an error. Let me reconsider.
    
    The matrix multiplication tensor is NOT the all-ones tensor. Let me re-derive.
    
    We want: C[i,k] = sum_j A[i,j] B[j,k]
    
    In tensor form: C = A * B where (A*B)[i,k] = sum_j A[i,j] B[j,k].
    
    The tensor T such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] has T[i,j,k] = 1 for all i,j,k.
    
    But wait, that's the same as saying the multiplication is just the standard sum. The tensor T IS the all-ones tensor.
    
    But then the rank of matrix multiplication would be 1 for all sizes, which is wrong.
    
    I think I'm confusing the tensor definition. Let me look at this more carefully.
    
    The matrix multiplication tensor T_{n,m,p} is defined by:
    T_{n,m,p} = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i ⊗ e_j ⊗ e_k
    
    where e_i, e_j, e_k are basis vectors. This tensor satisfies:
    <T, A ⊗ B ⊗ C> = sum_{i,j,k} A[i,j] B[j,k] C[k,i]... 
    
    No, that's not right either. Let me use the standard definition.
    
    The matrix multiplication tensor M_{n,m,p} is the tensor in R^{n*m} ⊗ R^{m*p} ⊗ R^{n*p} such that:
    For A in R^{n×m}, B in R^{m×p}, C = A*B in R^{n×p}:
    C = <M_{n,m,p}, vec(A) ⊗ vec(B)>
    
    In components: M_{n,m,p}[(i,j), (j,k), (i,k)] = 1, and 0 otherwise.
    
    So the tensor has indices (a, b, c) where:
    a = (i,j) ranges over n*m values
    b = (j,k) ranges over m*p values  
    c = (i,k) ranges over n*p values
    
    M[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k, else 0.
    
    This is NOT the all-ones tensor. It's a sparse tensor with exactly n*m*p nonzero entries.
    
    The rank of this tensor is what we're trying to minimize.
    
    So for 3x5x7, the tensor has 3*5*7 = 105 nonzero entries, and schoolbook gives rank 105.
    
    The Strassen decomposition for 2x2x2 reduces 8 nonzero entries to rank 7.
    
    OK so my earlier analysis was correct. The block decomposition approach is valid.
    
    Let me implement the 99-rank decomposition and see if I can improve it.
    
    Actually, let me also consider: can we use Strassen more effectively?
    
    For the (2,2,2) blocks, Strassen gives rank 7 instead of 8. That saves 1 per block.
    We had 6 such blocks, saving 6 total. 105 - 6 = 99. Correct.
    
    Can we find more 2x2x2 structure? What if we use a different partition?
    
    Try n=3=2+1, m=5=2+2+1, p=7=2+2+2+1: same as before, 99.
    
    Try n=3=1+1+1, m=5=2+2+1, p=7=2+2+2+1:
    (1,2,2): 3*2*3 = 18 blocks, rank 4 each? No, 1x2x2 has rank... 
    T for 1x2x2: M[(i,j),(j,k),(i,k)] with i fixed. So it's sum over j,k of e_(i,j) ⊗ e_(j,k) ⊗ e_(i,k).
    That's 2*2 = 4 terms, each a rank-1 tensor. Rank <= 4.
    Can it be less? For 1x2x2, the tensor is essentially a 2x2 matrix (in j,k) for fixed i.
    The rank of a 2x2 matrix as a tensor... it's just the matrix rank, which is 2.
    Wait no, the tensor M_{1,2,2} has shape (1*2, 2*2, 1*2) = (2, 4, 2).
    M[(i,j),(j,k),(i,k)] = 1. With i=0: M[(0,j),(j,k),(0,k)] = 1.
    So it's a 2x4x2 tensor. In indices (j, jk, k) where jk = j*2+k:
    M[j, j*2+k, k] = 1.
    
    This is the "matrix multiplication" for 1x2 * 2x2. The rank of this tensor...
    I believe it's 4 (schoolbook) and cannot be reduced.
    
    So 18*4 = 72, plus other blocks. This doesn't seem better.
    
    Let me try another approach: what about using the fact that 3 = 2+1 and applying
    a "Strassen-like" decomposition at a higher level?
    
    Actually, let me just go with the 99-rank solution and implement it carefully.
    
    The plan:
    1. Partition n=3 into [2,1], m=5 into [2,2,1], p=7 into [2,2,2,1]
    2. For each sub-block (ni, mj, pk):
       - If (2,2,2): use Strassen (7 terms)
       - Otherwise: use schoolbook
    3. Assemble the full U, V, W matrices
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Partition
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2, 1]
    
    n_offsets = []
    off = 0
    for s in n_parts:
        n_offsets.append(off)
        off += s
    m_offsets = []
    off = 0
    for s in m_parts:
        m_offsets.append(off)
        off += s
    p_offsets = []
    off = 0
    for s in p_parts:
        p_offsets.append(off)
        off += s
    
    # Strassen decomposition for 2x2x2
    # Standard Strassen: 7 products
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # In terms of U, V, W for the 2x2x2 sub-tensor:
    # U has 4 rows (i*2+j), V has 4 rows (j*2+k), W has 4 rows (k*2+i)
    # For each of the 7 Strassen products, we add to U, V, W
    
    def strassen_2x2x2(u_base, v_base, w_base, u, v, w, col_start):
        """Add Strassen decomposition for a 2x2x2 block.
        u_base: (n_off, m_off) for U indexing
        v_base: (m_off, p_off) for V indexing  
        w_base: (p_off, n_off) for W indexing
        Returns number of columns used."""
        # u index: (i+m_off)*(2) + (j+m_off) ... no wait
        # U has shape (n*m, rank). Row index for U: i*m + j where i is row of A, j is col of A
        # V has shape (m*p, rank). Row index for V: j*p + k
        # W has shape (p*n, rank). Row index for W: k*n + i
        
        # For the 2x2x2 block starting at (n_off, m_off, p_off):
        # U rows: (n_off+i)*m + (m_off+j) for i,j in 0..1
        # V rows: (m_off+j)*p + (p_off+k) for j,k in 0..1
        # W rows: (p_off+k)*n + (n_off+i) for k,i in 0..1
        
        col = col_start
        # M1 = (A00+A11)*(B00+B11)
        u[(n_off+0)*m + (m_off+0)][col] = 1
        u[(n_off+1)*m + (m_off+1)][col] = 1
        v[(m_off+0)*p + (p_off+0)][col] = 1
        v[(m_off+0)*p + (p_off+1)][col] = 1
        v[(m_off+1)*p + (p_off+0)][col] = 1
        v[(m_off+1)*p + (p_off+1)][col] = 1
        w[(p_off+0)*n + (n_off+0)][col] = 1
        w[(p_off+0)*n + (n_off+1)][col] = 1
        w[(p_off+1)*n + (n_off+0)][col] = -1
        w[(p_off+1)*n + (n_off+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)*B00
        u[(n_off+1)*m + (m_off+0)][col] = 1
        u[(n_off+1)*m + (m_off+1)][col] = 1
        v[(m_off+0)*p + (p_off+0)][col] = 1
        w[(p_off+0)*n + (n_off+1)][col] = 1
        col += 1
        
        # M3 = A00*(B01-B11)
        u[(n_off+0)*m + (m_off+0)][col] = 1
        v[(m_off+0)*p + (p_off+1)][col] = 1
        v[(m_off+1)*p + (p_off+1)][col] = -1
        w[(p_off+1)*n + (n_off+0)][col] = 1
        col += 1
        
        # M4 = A11*(B10-B00)
        u[(n_off+1)*m + (m_off+1)][col] = 1
        v[(m_off+1)*p + (p_off+0)][col] = 1
        v[(m_off+0)*p + (p_off+0)][col] = -1
        w[(p_off+0)*n + (n_off+1)][col] = 1
        col += 1
        
        # M5 = (A00+A01)*B11
        u[(n_off+0)*m + (m_off+0)][col] = 1
        u[(n_off+0)*m + (m_off+1)][col] = 1
        v[(m_off+1)*p + (p_off+1)][col] = 1
        w[(p_off+1)*n + (n_off+0)][col] = -1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        u[(n_off+1)*m + (m_off+0)][col] = 1
        u[(n_off+0)*m + (m_off+0)][col] = -1
        v[(m_off+0)*p + (p_off+0)][col] = 1
        v[(m_off+0)*p + (p_off+1)][col] = 1
        w[(p_off+1)*n + (n_off+1)][col] = 1
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        u[(n_off+0)*m + (m_off+1)][col] = 1
        u[(n_off+1)*m + (m_off+1)][col] = -1
        v[(m_off+1)*p + (p_off+0)][col] = 1
        v[(m_off+1)*p + (p_off+1)][col] = 1
        w[(p_off+1)*n + (n_off+0)][col] = 1
        col += 1
        
        return col
    
    def schoolbook_block(n_off, m_off, p_off, sn, sm, sp, u, v, w, col_start):
        """Add schoolbook decomposition for an sn x sm x sp block."""
        col = col_start
        for i in range(sn):
            for j in range(sm):
                for k in range(sp):
                    u[(n_off+i)*m + (m_off+j)][col] = 1
                    v[(m_off+j)*p + (p_off+k)][col] = 1
                    w[(p_off+k)*n + (n_off+i)][col] = 1
                    col += 1
        return col
    
    # Build the decomposition
    # First pass: count total rank
    total_rank = 0
    for ni, sn in enumerate(n_parts):
        for mj, sm in enumerate(m_parts):
            for pk, sp in enumerate(p_parts):
                if sn == 2 and sm == 2 and sp == 2:
                    total_rank += 7
                else:
                    total_rank += sn * sm * sp
    
    # Allocate matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ni, sn in enumerate(n_parts):
        n_off = n_offsets[ni]
        for mj, sm in enumerate(m_parts):
            m_off = m_offsets[mj]
            for pk, sp in enumerate(p_parts):
                p_off = p_offsets[pk]
                if sn == 2 and sm == 2 and sp == 2:
                    col = strassen_2x2x2(n_off, m_off, p_off, U, V, W, col)
                else:
                    col = schoolbook_block(n_off, m_off, p_off, sn, sm, sp, U, V, W, col)
    
    return {"U": U, "V": V, "W": W}
