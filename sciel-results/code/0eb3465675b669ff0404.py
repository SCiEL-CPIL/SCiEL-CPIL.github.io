"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank = 125 - 25 + 5 = 105 (BCLR construction)
    # For multiplying two n x n matrices, the BCLR algorithm achieves rank n^3 - n^2 + n
    # by expressing the tensor as a sum of rank-1 terms using a specific pattern.
    
    rank = n * n * n - n * n + n
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # The BCLR construction works by iterating over all (i, j, k) triples
    # and for each triple, it adds a term that covers the (i,j,k) entry
    # but cancels out some overlapping terms to reduce the total rank.
    #
    # The key insight: for each (i, j, k), we have a term:
    #   U[i*m + j] += 1
    #   V[j*p + k] += 1  
    #   W[k*n + i] += 1
    # But we need to subtract terms to avoid overcounting.
    #
    # A simpler exact construction for rank n^3 - n^2 + n:
    # We use the fact that the matrix multiplication tensor can be decomposed
    # using the "diagonal" structure.
    
    # Actually, let me implement the known BCLR algorithm properly.
    # The BCLR algorithm for rank n^3 - n^2 + n uses:
    # For each (i, j, k) with i != j or j != k or k != i (i.e., not all equal),
    # plus some diagonal terms.
    
    # Let me use a verified construction. The standard approach:
    # Rank = n^3 - n^2 + n means we save n^2 - n terms compared to naive.
    
    # Here's a concrete construction that achieves this rank:
    # We iterate over all (i, j, k) in [0,n)^3.
    # For each triple, we create a rank-1 term.
    # But we merge terms where possible.
    
    # A known valid construction for rank n^3 - n^2 + n:
    # Use the identity that involves the "difference" structure.
    
    # Let me just implement the straightforward BCLR:
    # The tensor T = sum_{i,j,k} e_i^A e_j^B e_k^C (with appropriate indices)
    # can be written as sum of (n^3 - n^2 + n) rank-1 tensors.
    
    # Concrete implementation:
    # For each pair (i, j) with i != j, and each k: one term
    # For each pair (j, k) with j != k, and each i: one term  
    # For diagonal terms: handle separately
    # This doesn't quite work out cleanly.
    
    # Let me use a different known construction.
    # The BCLR rank n^3 - n^2 + n is achieved by:
    # 
    # Terms of type 1: For all (i,j,k) where i<j, one term
    # Terms of type 2: For all (i,j,k) where i>j, one term
    # Terms of type 3: Diagonal terms
    #
    # Actually, the simplest correct approach: I'll use the fact that
    # rank(n^3 - n^2 + n) is achievable and construct it explicitly.
    
    # Here's a working construction:
    # We use the "split" method. For the n x n x n multiplication tensor:
    # 
    # Step 1: Handle all off-diagonal blocks
    # Step 2: Handle diagonal blocks with reduced rank
    
    # Let me just verify with a small case and scale up.
    # For n=2: rank = 8 - 4 + 2 = 6. Strassen gives 7, so 6 would be better.
    # But actually for n=2, the known rank is 7 (Strassen), and 6 is not known.
    # So BCLR for n=2 gives 6? That can't be right since Strassen is 7 and lower bound is 7.
    
    # I think I'm confusing myself. Let me just use the naive rank 125 which is definitely correct,
    # but try to improve it.
    
    # Actually, the simplest improvement: use Strassen's algorithm for 2x2 blocks.
    # 5 = 2 + 2 + 1
    # We can decompose 5x5x5 into blocks:
    # - 2x2x2 blocks: use Strassen (rank 7 instead of 8)
    # - 2x2x1, 2x1x2, 1x2x2 blocks: rank 4 each (can't improve)
    # - 1x1x1 blocks: rank 1
    
    # Block decomposition of 5x5 matrix multiplication:
    # A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
    # where A11, A22 are 2x2, others are 2x1, 1x2, or 1x1
    
    # This is complex. Let me just return the naive rank 125 to be safe.
    # But that's the seed score. I need to improve.
    
    # Let me implement Strassen-based block decomposition for 5x5.
    # Split 5 = 2 + 2 + 1.
    
    # The multiplication C = AB where A,B are 5x5.
    # Split into 3x3 blocks of sizes (2,2,1).
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # Block (i,j) of C has size (s_i, s_j) where s = (2,2,1).
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # where A[i,k] is (s_i x s_k), B[k,j] is (s_k x s_j)
    
    # For blocks where both dimensions are 2, we can use Strassen.
    # The 2x2x2 sub-problem has rank 7 instead of 8.
    
    # Let me count: how many 2x2x2 sub-problems do we have?
    # Blocks (i,j,k) where s_i=2, s_j=2, s_k=2:
    # i in {0,1} (size 2), j in {0,1} (size 2), k in {0,1} (size 2)
    # So 2*2*2 = 8 such block triples, but they share the same k.
    
    # Actually, for each output block (i,j) with s_i=2, s_j=2,
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The sum is over k. For each k, A[i,k] is (2 x s_k), B[k,j] is (s_k x 2).
    # If s_k=2, then A[i,k]*B[k,j] is a 2x2x2 multiplication (rank 7 with Strassen).
    # If s_k=1, then it's a 2x1x2 multiplication (rank 4, no improvement).
    
    # For output block (i,j) with s_i=2, s_j=2:
    #   k=0: s_k=2, rank 7
    #   k=1: s_k=2, rank 7  
    #   k=2: s_k=1, rank 4
    #   Total: 7+7+4 = 18 instead of 8+8+4 = 20. Save 2.
    
    # There are 2*2 = 4 such output blocks (i in {0,1}, j in {0,1}).
    # Total savings: 4 * 2 = 8.
    # New rank: 125 - 8 = 117.
    
    # But wait, the Strassen terms for different output blocks might overlap?
    # No, because each output block (i,j) uses its own set of intermediate terms.
    # The U, V, W matrices for different output blocks have disjoint support
    # in the output dimension W.
    
    # So the total rank is indeed 117.
    
    # Let me implement this.
    
    sizes = [2, 2, 1]
    # Block offsets
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    
    rank = 0
    U = []
    V = []
    W = []
    
    for i in range(3):
        for j in range(3):
            si = sizes[i]
            sj = sizes[j]
            oi = offsets[i]
            oj = offsets[j]
            
            for k in range(3):
                sk = sizes[k]
                ok = offsets[k]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Use Strassen: 7 terms
                    # A[i,k] is 2x2, B[k,j] is 2x2
                    # C[i,j] += A[i,k] * B[k,j]
                    # Strassen for 2x2:
                    # M1 = (A00+A11)*(B00+B11)
                    # M2 = (A10+A11)*B00
                    # M3 = A00*(B10-B11)
                    # M4 = A11*(B01-B10)
                    # M5 = (A00+A01)*B11
                    # M6 = (A10-A00)*(B00+B10)
                    # M7 = (A01-A11)*(B01+B11)
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # A[i,k] entries:
                    # A[oi, ok], A[oi, ok+1], A[oi+1, ok], A[oi+1, ok+1]
                    # B[k,j] entries:
                    # B[ok, oj], B[ok, oj+1], B[ok+1, oj], B[ok+1, oj+1]
                    # C[i,j] entries:
                    # C[oj, oi], C[oj, oi+1], C[oj+1, oi], C[oj+1, oi+1]
                    # Note: W index is k*n + i, so for output block (i,j),
                    # the W indices are (oj)*n + oi, etc.
                    
                    a00 = oi * m + ok
                    a01 = oi * m + ok + 1
                    a10 = (oi + 1) * m + ok
                    a11 = (oi + 1) * m + ok + 1
                    
                    b00 = ok * p + oj
                    b01 = ok * p + oj + 1
                    b10 = (ok + 1) * p + oj
                    b11 = (ok + 1) * p + oj + 1
                    
                    c00 = oj * n + oi
                    c01 = oj * n + oi + 1
                    c10 = (oj + 1) * m + oi  # Wait, W index is k*n+i, so row is oj, col is oi
                    # Actually W has shape (p*n, rank), so W[k*n + i, r]
                    # For output block (i,j), the W indices are:
                    # W[oj*n + oi], W[oj*n + oi+1], W[(oj+1)*n + oi], W[(oj+1)*n + oi+1]
                    c10 = (oj + 1) * n + oi
                    c11 = (oj + 1) * n + oi + 1
                    
                    # M1 = (A00+A11)*(B00+B11)
                    U.append([a00, 1, a11, 1])
                    V.append([b00, 1, b11, 1])
                    W.append([c00, 1, c11, -1, c01, 1, c10, 1])
                    rank += 1
                    
                    # M2 = (A10+A11)*B00
                    U.append([a10, 1, a11, 1])
                    V.append([b00, 1])
                    W.append([c10, 1, c11, -1])
                    rank += 1
                    
                    # M3 = A00*(B10-B11)
                    U.append([a00, 1])
                    V.append([b10, 1, b11, -1])
                    W.append([c01, 1, c11, 1])
                    rank += 1
                    
                    # M4 = A11*(B01-B10)
                    U.append([a11, 1])
                    V.append([b01, 1, b10, -1])
                    W.append([c00, 1, c10, 1])
                    rank += 1
                    
                    # M5 = (A00+A01)*B11
                    U.append([a00, 1, a01, 1])
                    V.append([b11, 1])
                    W.append([c00, -1, c01, 1])
                    rank += 1
                    
                    # M6 = (A10-A00)*(B00+B10)
                    U.append([a10, 1, a00, -1])
                    V.append([b00, 1, b10, 1])
                    W.append([c11, 1])
                    rank += 1
                    
                    # M7 = (A01-A11)*(B01+B11)
                    U.append([a01, 1, a11, -1])
                    V.append([b01, 1, b11, 1])
                    W.append([c00, 1, c11, -1])
                    rank += 1
                    
                else:
                    # Naive: si * sj * sk terms
                    for ii in range(si):
                        for jj in range(sj):
                            for kk in range(sk):
                                u_idx = (oi + ii) * m + (ok + kk)
                                v_idx = (ok + kk) * p + (oj + jj)
                                w_idx = (oj + jj) * n + (oi + ii)
                                U.append([u_idx, 1])
                                V.append([v_idx, 1])
                                W.append([w_idx, 1])
                                rank += 1
    
    # Convert sparse lists to dense matrices
    U_dense = [[0] * rank for _ in range(n * m)]
    V_dense = [[0] * rank for _ in range(m * p)]
    W_dense = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for idx, val in U[r]:
            U_dense[idx][r] = val
        for idx, val in V[r]:
            V_dense[idx][r] = val
        for idx, val in W[r]:
            W_dense[idx][r] = val
    
    return {"U": U_dense, "V": V_dense, "W": W_dense}
# EVOLVE-BLOCK-END
