import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7
    
    # We'll build the decomposition by applying Strassen recursively
    # and tracking which (i, j, k) indices each rank-1 term uses
    
    terms = []  # Each term is a dict: {'U': {idx: coeff}, 'V': {idx: coeff}, 'W': {idx: coeff}}
    
    def add_term(U_entries, V_entries, W_entries):
        """Add a rank-1 term with the given sparse entries."""
        terms.append({'U': dict(U_entries), 'V': dict(V_entries), 'W': dict(W_entries)})
    
    def strassen(n_rows, m_mids, p_cols, row_off, mid_off, col_off, row_bound, mid_bound, col_bound):
        """
        Decompose multiplication of (n_rows x m_mids) by (m_mids x p_cols).
        Only indices within original bounds (row_bound, mid_bound, col_bound) are kept.
        """
        # Base case: 1x1x1 or very small - use schoolbook
        if n_rows == 1 and m_mids == 1 and p_cols == 1:
            i, j, k = row_off, mid_off, col_off
            if i < row_bound and j < mid_bound and k < col_bound:
                add_term({i * m_mids + j: 1}, {j * p_cols + k: 1}, {k * n_rows + i: 1})
            return
        
        # If dimensions are small, use schoolbook
        if n_rows * m_mids * p_cols <= 8:
            for i in range(n_rows):
                for j in range(m_mids):
                    for k in range(p_cols):
                        ai = row_off + i
                        aj = mid_off + j
                        ak = col_off + k
                        if ai < row_bound and aj < mid_bound and ak < col_bound:
                            add_term({ai * m + aj: 1}, {aj * p + ak: 1}, {ak * n + ai: 1})
            return
        
        # Split dimensions into two halves
        n1 = n_rows // 2
        n2 = n_rows - n1
        m1 = m_mids // 2
        m2 = m_mids - m1
        p1 = p_cols // 2
        p2 = p_cols - p1
        
        # Strassen's 7 multiplications:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each multiplication is a bilinear form. We need to decompose each
        # into rank-1 terms.
        
        # For M1 = (A11+A22)(B11+B22):
        # This is a product of (n1+n2) x (m1+m2) and (m1+m2) x (p1+p2)
        # But we only need the parts that contribute to valid output indices.
        
        # Actually, let me think about this differently. Each Strassen "multiplication"
        # M_r is itself a matrix multiplication problem. We recursively decompose it.
        
        # M1 involves:
        # A part: rows [0..n1-1] with mids [0..m1-1] (A11) PLUS rows [n1..n1+n2-1] with mids [m1..m1+m2-1] (A22)
        # B part: mids [0..m1-1] with cols [0..p1-1] (B11) PLUS mids [m1..m1+m2-1] with cols [p1..p1+p2-1] (B22)
        # Result: rows [0..n1-1] with cols [0..p1-1] (C11 contribution)
        
        # This is getting complex. Let me use a different formulation.
        
        # Alternative: Represent each Strassen multiplication as a sum of rank-1 tensors.
        # For a 2x2x2 Strassen, we know the exact coefficients.
        
        # Let me just implement the full recursive Strassen with proper index tracking.
        
        # For each of the 7 multiplications, we have:
        # A combination of sub-blocks of A and B that get multiplied
        # The result contributes to specific sub-blocks of C
        
        # M1 = (A11 + A22) * (B11 + B22)
        #   A factors: A11 (rows 0..n1-1, mids 0..m1-1) + A22 (rows n1..n1+n2-1, mids m1..m1+m2-1)
        #   B factors: B11 (mids 0..m1-1, cols 0..p1-1) + B22 (mids m1..m1+m2-1, cols p1..p1+p2-1)
        #   C contribution: C11 (rows 0..n1-1, cols 0..p1-1)
        
        # To decompose M1 into rank-1 terms, we need to decompose the product
        # (A11 + A22) * (B11 + B22) into rank-1 terms.
        
        # This is itself a matrix multiplication problem! The "A" matrix has entries from
        # two sub-blocks, and the "B" matrix has entries from two sub-blocks.
        
        # Let me define helper to decompose a "sum of blocks" multiplication.
        
        def decomp_sum_blocks(A_blocks, B_blocks, C_rows, C_cols, r_off, c_off):
            """
            Decompose (sum of A_blocks) * (sum of B_blocks) contributing to C sub-block.
            A_blocks: list of (row_start, row_end, mid_start, mid_end)
            B_blocks: list of (mid_start, mid_end, col_start, col_end)
            C_rows: (start, end) for output rows
            C_cols: (start, end) for output cols
            r_off, c_off: offsets for C
            """
            # This is a bilinear form. We can expand:
            # (sum_a A_a) * (sum_b B_b) = sum_a sum_b A_a * B_b
            # Each A_a * B_b is a matrix multiplication, which we can decompose recursively.
            
            for (ar, ac, am, ae) in A_blocks:
                for (bm, be, bc, ce) in B_blocks:
                    # A_a is rows [ar..ac-1], mids [am..ae-1]
                    # B_b is mids [bm..be-1], cols [bc..ce-1]
                    # For the product to be non-zero, we need am..ae-1 overlaps with bm..be-1
                    # The effective mid range is [max(am,bm)..min(ae,be)-1]
                    
                    mid_lo = max(am, bm)
                    mid_hi = min(ae, be)
                    if mid_lo >= mid_hi:
                        continue
                    
                    # A_a * B_b: (ac-ar) x (mid_hi-mid_lo) times (mid_hi-mid_lo) x (ce-bc)
                    # This contributes to C rows [r_off..r_off+(ac-ar)-1], cols [c_off..c_off+(ce-bc)-1]
                    
                    # Recursively decompose this sub-multiplication
                    sub_n = ac - ar
                    sub_m = mid_hi - mid_lo
                    sub_p = ce - bc
                    
                    # The A part has rows [ar..ac-1] (absolute), mids [mid_lo..mid_hi-1] (absolute)
                    # The B part has mids [mid_lo..mid_hi-1] (absolute), cols [bc..ce-1] (absolute)
                    # C part has rows [r_off..r_off+sub_n-1] (absolute), cols [c_off..c_off+sub_p-1] (absolute)
                    
                    strassen(sub_n, sub_m, sub_p, ar, mid_lo, bc, row_bound, mid_bound, col_bound)
        
        # Now apply the 7 Strassen multiplications:
        
        # M1 = (A11 + A22) * (B11 + B22) -> C11
        decomp_sum_blocks(
            [(0, n1, 0, m1), (n1, n1+n2, m1, m1+m2)],  # A11, A22
            [(0, m1, 0, p1), (m1, m1+m2, p1, p1+p2)],  # B11, B22
            (0, n1), (0, p1), 0, 0
        )
        
        # M2 = (A21 + A22) * B11 -> C21
        decomp_sum_blocks(
            [(n1, n1+n2, 0, m1), (n1, n1+n2, m1, m1+m2)],  # A21, A22
            [(0, m1, 0, p1)],  # B11
            (n1, n1+n2), (0, p1), n1, 0
        )
        
        # M3 = A11 * (B21 - B22) -> C12
        decomp_sum_blocks(
            [(0, n1, 0, m1)],  # A11
            [(m1, m1+m2, 0, p1), (m1, m1+m2, p1, p1+p2)],  # B21, B22 (with sign)
            (0, n1), (p1, p1+p2), 0, p1
        )
        
        # M4 = A22 * (B11 - B21) -> C11 and C21
        decomp_sum_blocks(
            [(n1, n1+n2, m1, m1+m2)],  # A22
            [(0, m1, 0, p1), (m1, m1+m2, 0, p1)],  # B11, B21 (with sign)
            (0, n1), (0, p1), 0, 0  # This needs to go to both C11 and C21... 
        )
        
        # This approach is getting too complex and error-prone. Let me use a simpler
        # but still effective method: direct recursive Strassen with proper tracking.
        
        # Actually, let me step back and use a known result. For 3x5x7 matrix multiplication,
        # we can use the following approach:
        # 1. Pad to 4x6x8
        # 2. Apply one level of Strassen: 7 sub-problems of size 2x3x4
        # 3. For each 2x3x4, pad to 2x4x4, apply Strassen: 7 sub-problems of size 2x2x2
        # 4. Each 2x2x2 uses Strassen: 7 rank-1 terms
        # Total: 7 * 7 * 7 = 343? No, that's too many.
        
        # Wait, I'm overcomplicating this. Let me just use the known optimal or near-optimal
        # decomposition for small matrix multiplication.
        
        # Actually, the simplest improvement is to just use Strassen at the top level
        # and schoolbook for the base cases. Let me implement it cleanly.
        
        return
    
    # The recursive approach is too complex. Let me use a direct construction.
    
    # For 3x5x7, I'll use the following strategy:
    # Pad A to 4x5, B to 5x8, middle to 6
    # Apply Strassen: 7 multiplications of 2x3x4 sub-problems
    # For each 2x3x4, use schoolbook (rank 2*3*4=24) or further Strassen
    
    # Actually, let me just implement a clean recursive Strassen.
    
    terms = []
    
    def schoolbook(n_r, m_m, p_c, r_off, m_off, c_off):
        """Add schoolbook rank-1 terms for this sub-problem."""
        for i in range(n_r):
            for j in range(m_m):
                for k in range(p_c):
                    ai = r_off + i
                    aj = m_off + j
                    ak = c_off + k
                    if ai < n and aj < m and ak < p:
                        terms.append({
                            'U': {ai * m + aj: 1},
                            'V': {aj * p + ak: 1},
                            'W': {ak * n + ai: 1}
                        })
    
    def strassen_rec(n_r, m_m, p_c, r_off, m_off, c_off):
        """Recursively apply Strassen."""
        if n_r <= 1 and m_m <= 1 and p_c <= 1:
            schoolbook(n_r, m_m, p_c, r_off, m_off, c_off)
            return
        
        # Use schoolbook for small problems
        if n_r * m_m * p_c <= 16:
            schoolbook(n_r, m_m, p_c, r_off, m_off, c_off)
            return
        
        # Split
        n1 = n_r // 2
        n2 = n_r - n1
        m1 = m_m // 2
        m2 = m_m - m1
        p1 = p_c // 2
        p2 = p_c - p1
        
        # For each of the 7 Strassen multiplications, we need to decompose
        # a product of sums of sub-blocks.
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11: rows [r_off, r_off+n1), mids [m_off, m_off+m1)
        # A22: rows [r_off+n1, r_off+n1+n2), mids [m_off+m1, m_off+m1+m2)
        # B11: mids [m_off, m_off+m1), cols [c_off, c_off+p1)
        # B22: mids [m_off+m1, m_off+m1+m2), cols [c_off+p1, c_off+p1+p2)
        
        # The product (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # But A11*B22 and A22*B11 have no overlap in mid indices, so they're zero!
        # Wait, no. A11 has mids [m_off, m_off+m1), B22 has mids [m_off+m1, m_off+m1+m2).
        # These don't overlap, so A11*B22 = 0 and A22*B11 = 0.
        # So M1 = A11*B11 + A22*B22
        
        # Similarly for the others. Let me just expand each:
        
        # M1 = A11*B11 + A22*B22
        strassen_rec(n1, m1, p1, r_off, m_off, c_off)  # A11*B11
        strassen_rec(n2, m2, p2, r_off+n1, m_off+m1, c_off+p1)  # A22*B22
        
        # M2 = (A21+A22)*B11 = A21*B11 + A22*B11
        # A21: rows [r_off+n1, r_off+n1+n2), mids [m_off, m_off+m1)
        # A22: rows [r_off+n1, r_off+n1+n2), mids [m_off+m1, m_off+m1+m2)
        # B11: mids [m_off, m_off+m1), cols [c_off, c_off+p1)
        # A22*B11: mids don't overlap (A22 has m1..m1+m2, B11 has 0..m1), so A22*B11=0
        # M2 = A21*B11
        strassen_rec(n2, m1, p1, r_off+n1, m_off, c_off)  # A21*B11
        
        # M3 = A11*(B21-B22) = A11*B21 - A11*B22
        # A11: rows [r_off, r_off+n1), mids [m_off, m_off+m1)
        # B21: mids [m_off+m1, m_off+m1+m2), cols [c_off, c_off+p1)
        # B22: mids [m_off+m1, m_off+m1+m2), cols [c_off+p1, c_off+p1+p2)
        # A11*B21: mids don't overlap, so = 0
        # A11*B22: mids don't overlap, so = 0
        # M3 = 0? That can't be right.
        
        # I think I'm confusing myself. In Strassen's algorithm, the sub-blocks are defined
        # such that the multiplications make sense. Let me look at this more carefully.
        
        # Standard Strassen for 2x2 matrices:
        # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # For rectangular blocks, the same formulas apply but the "multiplications"
        # are between rectangular matrices.
        
        # The key point: each M_r is a single matrix multiplication of (sub_A) * (sub_B)
        # where sub_A and sub_B are sums/differences of sub-blocks.
        
        # To decompose M_r into rank-1 terms, I need to decompose the product
        # (sum of A sub-blocks) * (sum of B sub-blocks).
        
        # This is a bilinear form. The cleanest way is to note that:
        # (sum_a A_a) * (sum_b B_b) = sum_{a,b} A_a * B_b
        # where only terms with overlapping mid indices contribute.
        
        # So for M1 = (A11+A22)(B11+B22):
        # = A11*B11 + A11*B22 + A22*B11 + A22*B22
        # A11*B11: A11 mids [0,m1), B11 mids [0,m1) -> overlap [0,m1) -> contributes
        # A11*B22: A11 mids [0,m1), B22 mids [m1,m1+m2) -> no overlap -> 0
        # A22*B11: A22 mids [m1,m1+m2), B11 mids [0,m1) -> no overlap -> 0
        # A22*B22: A22 mids [m1,m1+m2), B22 mids [m1,m1+m2) -> overlap [m1,m1+m2) -> contributes
        # So M1 = A11*B11 + A22*B22
        
        # This means M1 is just the sum of two independent multiplications!
        # So we can decompose each recursively.
        
        # Similarly:
        # M2 = (A21+A22)*B11 = A21*B11 + A22*B11
        # A21*B11: overlap -> contributes
        # A22*B11: no overlap -> 0
        # M2 = A21*B11
        
        # M3 = A11*(B21-B22) = A11*B21 - A11*B22
        # A11*B21: no overlap -> 0
        # A11*B22: no overlap -> 0
        # M3 = 0 ??? 
        
        # This doesn't make sense. I must be misunderstanding the block structure.
        
        # Let me reconsider. In Strassen, for the full 2x2 case:
        # A is 2x2, B is 2x2, C is 2x2
        # A11, A12, A21, A22 are all 1x1
        # B11, B12, B21, B22 are all 1x1
        # M1 = (A11+A22)(B11+B22) is a 1x1 * 1x1 = 1x1 multiplication
        # This is just a scalar multiplication, rank 1.
        
        # For rectangular blocks, say A is (2a x 2b), B is (2b x 2c):
        # A11 is (a x b), A12 is (a x b), A21 is (a x b), A22 is (a x b)
        # B11 is (b x c), B12 is (b x c), B21 is (b x c), B22 is (b x c)
        # M1 = (A11+A22)(B11+B22) is (a x b) * (b x c) = (a x c)
        # This is a standard matrix multiplication of size a x b x c.
        
        # So each M_r is itself a matrix multiplication problem!
        # We can recursively apply Strassen to each M_r.
        
        # The total rank is 7 * (rank of each sub-multiplication)
        # But the sub-multiplications have different sizes!
        
        # For our case 3x5x7, pad to 4x6x8:
        # n_r=4, m_m=6, p_c=8
        # n1=2, n2=2, m1=3, m2=3, p1=4, p2=4
        # M1: (2x3) * (3x4), rank = ?
        # M2: (2x3) * (3x4), rank = ?
        # etc.
        
        # Each is a 2x3x4 multiplication. We can further decompose these.
        
        # For 2x3x4, pad to 2x4x4:
        # n_r=2, m_m=4, p_c=4
        # n1=1, n2=1, m1=2, m2=2, p1=2, p2=2
        # M1: (1x2)*(2x2), M2: (1x2)*(2x2), etc.
        # Each is 1x2x2, which can be done with rank 2 (schoolbook) or...
        
        # Actually, 1x2x2 matrix multiplication has rank 2 (it's just a dot product).
        # So 2x4x4 has rank 7*2 = 14? No, that's not right either.
        
        # I think the issue is that when we say "M1 is a matrix multiplication", 
        # the rank of M1 as a tensor is the tensor rank of that sub-problem.
        
        # Let me just implement it recursively and count the terms.
        
        return
    
    # This is getting too complicated analytically. Let me just implement a clean
    # recursive Strassen and count the resulting rank-1 terms.
    
    terms = []
    
    def strassen(n_r, m_m, p_c, r_off, m_off, c_off):
        """
        Decompose (n_r x m_m) * (m_m x p_c) multiplication.
        Uses Strassen if beneficial, otherwise schoolbook.
        Only adds terms for indices within original bounds.
        """
        # Check time
        if time.time() - START_TIME > TIME_LIMIT:
            return
        
        # Base case: use schoolbook for small problems
        if n_r * m_m * p_c <= 24:
            for i in range(n_r):
                for j in range(m_m):
                    for k in range(p_c):
                        ai = r_off + i
                        aj = m_off + j
                        ak = c_off + k
                        if ai < n and aj < m and ak < p:
                            terms.append({
                                'U': {ai * m + aj: 1},
                                'V': {aj * p + ak: 1},
                                'W': {ak * n + ai: 1}
                            })
            return
        
        # Split into halves
        n1 = n_r // 2
        n2 = n_r - n1
        m1 = m_m // 2
        m2 = m_m - m1
        p1 = p_c // 2
        p2 = p_c - p1
        
        # Define the 7 Strassen multiplications.
        # Each is a product of (sum of A sub-blocks) * (sum of B sub-blocks).
        # We decompose each into its constituent A_a * B_b products and recurse.
        
        # M1 = (A11 + A22)(B11 + B22)
        # A blocks: [(0, n1, 0, m1), (n1, n1+n2, m1, m1+m2)]
        # B blocks: [(0, m1, 0, p1), (m1, m1+m2, p1, p1+p2)]
        # C: rows [0, n1), cols [0, p1)
        
        # M2 = (A21 + A22)(B11)
        # A blocks: [(n1, n1+n2, 0, m1), (n1, n1+n2, m1, m1+m2)]
        # B blocks: [(0, m1, 0, p1)]
        # C: rows [n1, n1+n2), cols [0, p1)
        
        # M3 = (A11)(B21 - B22)
        # A blocks: [(0, n1, 0, m1)]
        # B blocks: [(m1, m1+m2, 0, p1), (m1, m1+m2, p1, p1+p2)]
        # C: rows [0, n1), cols [p1, p1+p2)
        
        # M4 = (A22)(B11 - B21)
        # A blocks: [(n1, n1+n2, m1, m1+m2)]
        # B blocks: [(0, m1, 0, p1), (m1, m1+m2, 0, p1)]
        # C: rows [0, n1), cols [0, p1) AND rows [n1, n1+n2), cols [0, p1)
        # Wait, M4 contributes to both C11 and C21? Let me check.
        
        # From Strassen: C11 = M1 + M4 - M5 + M7, C21 = M2 + M4
        # So M4 contributes to both C11 and C21. That means M4 is a single multiplication
        # whose result is added to both output blocks.
        
        # This means M4 is a multiplication that produces a (n1 x p1) result for C11
        # and the same (n2 x p1) result for C21? No, that doesn't make sense dimensionally.
        
        # I think I need to be more careful. Let me look at the standard Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M4 appears in both C11 and C21. This means M4 must be a multiplication
        # that produces results for both. But M4 = A22*(B11-B21), where A22 is (n2 x m2)
        # and (B11-B21) is (m2 x p1). So M4 is (n2 x p1).
        
        # But C11 is (n1 x p1) and C21 is (n2 x p1). How can a (n2 x p1) matrix
        # contribute to both?
        
        # I think the issue is that in the standard Strassen, all sub-blocks are square.
        # For rectangular, the formulas still work but the "contributions" need to be
        # interpreted carefully.
        
        # Actually, I think the standard Strassen formulas assume square blocks.
        # For rectangular, we need to adapt.
        
        # Let me just use a different approach: since the dimensions are small,
        # let me use a known low-rank decomposition or just improve on schoolbook
        # by using Strassen at one level.
        
        # For 3x5x7, pad to 4x6x8.
        # One level of Strassen gives 7 sub-problems of size 2x3x4.
        # Each 2x3x4 can be done with schoolbook rank 24.
        # Total: 7 * 24 = 168. That's worse than 105!
        
        # So one level of Strassen is not helpful here. We need to go deeper.
        
        # Let me try: pad to 4x8x8 (pad middle to 8).
        # One level: 7 sub-problems of 2x4x4.
        # Each 2x4x4: pad to 2x4x4, one more level of Strassen: 7 sub-problems of 1x2x2.
        # Each 1x2x2: schoolbook rank 2.
        # Total: 7 * 7 * 2 = 98. That's better than 105!
        
        # Let me implement this.
        
        return
    
    # Let me implement the specific decomposition:
    # 3x5x7 -> pad to 4x8x8
    # Level 1: 7 sub-problems of 2x4x4
    # Level 2: each 2x4x4 -> 7 sub-problems of 1x2x2
    # Level 3: each 1x2x2 -> 2 schoolbook terms
    # Total: 7 * 7 * 2 = 98
    
    terms = []
    
    def add_schoolbook(n_r, m_m, p_c, r_off, m_off, c_off):
        for i in range(n_r):
            for j in range(m_m):
                for k in range(p_c):
                    ai = r_off + i
                    aj = m_off + j
                    ak = c_off + k
                    if ai < n and aj < m and ak < p:
                        terms.append({
                            'U': {ai * m + aj: 1},
                            'V': {aj * p + ak: 1},
                            'W': {ak * n + ai: 1}
                        })
    
    def strassen_level(n_r, m_m, p_c, r_off, m_off, c_off, depth=0):
        if time.time() - START_TIME > TIME_LIMIT:
            return
        
        if depth >= 2:
            add_schoolbook(n_r, m_m, p_c, r_off, m_off, c_off)
            return
        
        n1 = n_r // 2
        n2 = n_r - n1
        m1 = m_m // 2
        m2 = m_m - m1
        p1 = p_c // 2
        p2 = p_c - p1
        
        # M1 = (A11+A22)(B11+B22)
        # For rectangular, this expands to:
        # A11*B11 + A22*B22 (cross terms have no mid overlap)
        strassen_level(n1, m1, p1, r_off, m_off, c_off, depth+1)
        strassen_level(n2, m2, p2, r_off+n1, m_off+m1, c_off+p1, depth+1)
        
        # M2 = (A21+A22)*B11 = A21*B11 (A22*B11 has no overlap)
        strassen_level(n2, m1, p1, r_off+n1, m_off, c_off, depth+1)
        
        # M3 = A11*(B21-B22)
        # A11 has mids [m_off, m_off+m1), B21 has mids [m_off+m1, m_off+m1+m2)
        # No overlap! So M3 = 0?
        
        # I'm stuck on the rectangular Strassen. Let me just use a known result.
        
        # After more thought, I believe the correct approach for rectangular Strassen
        # is to note that each M_r is a valid matrix multiplication, and we recurse.
        # The cross terms that don't overlap in mid indices are indeed zero.
        
        # Let me just implement it and verify.
        
        return
    
    # I'll use a simpler approach: just use the schoolbook but with a better
    # organization. Actually, let me try to use the Laderman algorithm or
    # just accept that for these dimensions, a good decomposition might be
    # around 90-100.
    
    # For now, let me just return the schoolbook decomposition to ensure
    # correctness, and add a TODO for improvement.
    
    rank = n * m * p
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = 1.0
                V[j*p + k][idx] = 1.0
                W[k*n + i][idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
