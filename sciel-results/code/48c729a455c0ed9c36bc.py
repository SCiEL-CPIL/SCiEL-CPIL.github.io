"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 120s/150s wall-clock limits

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 36, 36, 36

    # Build the schoolbook decomposition as a fallback
    schoolbook_rank = n * m * p  # 216
    
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Try Block-Strassen: partition 6x6 into 3x3 blocks of 2x2 submatrices
    # Each 2x2 block multiplication uses Strassen's rank-7 algorithm
    # Total rank = 9 block positions * 7 Strassen terms = 63 per output block
    # But we need to be careful about the structure
    
    # Actually, let's think more carefully:
    # A is 6x6, B is 6x6, C = A*B is 6x6
    # Partition into 3x3 grid of 2x2 blocks
    # Each block C[bi,bj] = sum over bk of A[bi,bk] * B[bk,bj]
    # Each 2x2 matrix multiply has rank 7 (Strassen)
    # So for each output block (bi, bj), we need sum over bk of (rank-7 decompositions)
    # That's 3 * 7 = 21 terms per output block
    # 9 output blocks * 21 = 189 total rank
    
    strassen_rank = 9 * 3 * 7  # 189
    
    if strassen_rank < best_rank:
        success = _build_block_strassen(n, m, p, deadline)
        if success is not None:
            U_found, V_found, W_found = success
            best_rank = strassen_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found

    # Convert to lists for the output format
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}


def _build_block_strassen(n, m, p, deadline):
    """Build a block-Strassen decomposition for 6x6x6 matrix multiplication.
    
    Partition into 3x3 blocks of 2x2 submatrices.
    Each 2x2 multiply uses Strassen's rank-7 algorithm.
    For each output block (bi, bj): C[bi,bj] = sum_bk A[bi,bk] * B[bk,bj]
    Each product is rank 7, so each output block needs 3*7=21 terms.
    Total: 9 * 21 = 189 terms.
    """
    nm, mp, pn = n * m, m * p, p * n
    block_size = 2
    num_blocks = 3  # 6 / 2 = 3 blocks per dimension
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen term, we have products of sums/differences of 2x2 blocks
    # We need to express these as rank-1 tensors in the global space
    
    U = np.zeros((nm, 189))
    V = np.zeros((mp, 189))
    W = np.zeros((pn, 189))
    
    col = 0
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                # For block product A[bi,bk] * B[bk,bj] -> contributes to C[bi,bj]
                # Use Strassen's 7 terms
                for s in range(7):
                    # Each Strassen term is a product of two 2x2 matrices
                    # expressed as sums/differences of individual elements
                    # We need to decompose the 2x2 product into rank-1 terms
                    
                    # For Strassen term s, we have:
                    # M_s = (linear combo of A blocks) * (linear combo of B blocks)
                    # Each "linear combo of A blocks" is a 2x2 matrix
                    # The product of two 2x2 matrices has rank 4 (schoolbook)
                    # But Strassen combines them cleverly...
                    
                    # Actually, let me reconsider. The standard Strassen algorithm
                    # computes 7 scalar products where each "product" is of two 
                    # (sums of) 2x2 blocks. But each such product of two 2x2 blocks
                    # is itself a 2x2 matrix product, which has rank 4.
                    
                    # Wait, no. In Strassen's algorithm for 2x2 matrices:
                    # The 7 "products" M1...M7 are each products of two 2x2 matrices
                    # (that are sums/differences of the original 2x2 blocks).
                    # Each 2x2 * 2x2 product is itself a full matrix product of rank 4.
                    
                    # So actually Strassen reduces the number of scalar multiplications
                    # from 8 to 7, but here we're dealing with matrix blocks.
                    
                    # Let me reconsider the problem. For 6x6 matrices partitioned
                    # into 2x2 blocks:
                    # A has 9 blocks A[bi,bj], each 2x2
                    # B has 9 blocks B[bj,bk], each 2x2  
                    # C has 9 blocks C[bi,bk], each 2x2
                    # C[bi,bk] = sum_j A[bi,j] * B[j,k]
                    
                    # Each A[bi,j] * B[j,k] is a 2x2 * 2x2 matrix product
                    # A 2x2 * 2x2 product has rank 4 (schoolbook) or 7 (Strassen)
                    
                    # If we use schoolbook for each 2x2 product: rank 4
                    # Total: 9 blocks * 3 sums * 4 = 108... that's less than 216!
                    
                    # Wait, that doesn't seem right. Let me recalculate.
                    # For each output block C[bi,bk], we sum over j: A[bi,j] * B[j,k]
                    # That's 3 products of 2x2 matrices
                    # Each 2x2 * 2x2 product using schoolbook has rank 4
                    # So each C[bi,bk] needs 3 * 4 = 12 terms
                    # Total: 9 * 12 = 108 terms
                    
                    # That's a huge improvement! But wait, is this correct?
                    
                    # Let me verify: a 2x2 matrix product C = A*B where
                    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
                    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
                    # This is 4 scalar multiplications: ae, bg, af, bh, ce, dg, cf, dh
                    # Wait, that's 8 scalar multiplications, giving rank 8 for 2x2.
                    # No wait: C11 = ae + bg (2 mults), C12 = af + bh (2 mults),
                    # C21 = ce + dg (2 mults), C22 = cf + dh (2 mults)
                    # Total 8 scalar multiplications, so rank 8 for 2x2 schoolbook.
                    
                    # Strassen reduces this to 7.
                    
                    # So for 6x6 with 2x2 blocks:
                    # Schoolbook: 9 output blocks * 3 inner products * 8 = 216 (matches!)
                    # Strassen per block: 9 * 3 * 7 = 189
                    
                    # OK so my original calculation was right. Let me implement Strassen properly.
                    
                    # For Strassen, the 7 "products" are each of two 2x2 matrices.
                    # But each such product is a 2x2 * 2x2 matrix multiply, which
                    # itself needs to be decomposed.
                    
                    # Actually, I think I'm overcomplicating this. Let me think of it
                    # differently.
                    
                    # The tensor for 6x6 matrix multiplication has entries indexed by
                    # (i, j, k) where i, j, k in {0,...,5} and the entry is 1 if
                    # the triple corresponds to a valid term in the product.
                    
                    # For block-Strassen, we can use the following approach:
                    # The 2x2 Strassen decomposition has rank 7. We can embed this
                    # into the 6x6 problem by treating each 2x2 block as a "super element".
                    
                    # Let me just directly construct the 189-term decomposition.
                    
                    # Strassen's 7 formulas for 2x2 matrices A, B:
                    # Let A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)B00
                    # M3 = A00(B10-B11)
                    # M4 = A11(B01-B00)
                    # M5 = (A00+A01)B11
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # Each M_s is a product of two 2x2 matrices (sums/diffs of elements).
                    # But in our tensor decomposition, we need rank-1 terms.
                    
                    # The key insight: each "product" in Strassen is actually a product
                    # of two linear combinations of the original matrix elements.
                    # When we expand, each M_s contributes to multiple entries of C.
                    
                    # For the tensor decomposition, we need to express:
                    # sum over all (i,j,k) of A[i,j] * B[j,k] = C[i,k]
                    # as sum_r U[i,j]_r * V[j,k]_r * W[i,k]_r
                    
                    # Let me use a different approach: directly construct the rank-1
                    # terms for the block-Strassen decomposition.
                    
                    # For each block position (bi, bk, bj) and Strassen term s,
                    # we need to figure out which global indices are involved.
                    
                    # Actually, let me just use the schoolbook for now and try to
                    # find a better decomposition through optimization.
                    
                    pass  # Placeholder - we'll use schoolbook for now
    
    # Since the block-Strassen construction is complex, let's use a simpler approach:
    # Just return the schoolbook decomposition
    return None

def _verify_decomposition(n, m, p, U, V, W, tol=1e-8):
    """Verify that U, V, W form a valid decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    if U.shape != (nm, rank) or V.shape != (mp, rank) or W.shape != (pn, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    np.random.seed(123)
    for _ in range(500):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c = np.random.randint(0, pn)
        
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k1 = b % p
        k2 = c // n
        i2 = c % n
        
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue
        
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    
    return True
# EVOLVE-BLOCK-END
