"""Matrix multiplication 5x5x5: return a rank-117 tensor decomposition using 2+2+1 Strassen partition."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    Each term is a tuple (u, v, w) where u, v are length-2 lists (A, B factors)
    and w is a length-4 list (C factor, row-major: [C00, C01, C10, C11]).
    """
    # Standard Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # A indices: 00, 01, 10, 11 -> flattened [A00, A01, A10, A11]
    # B indices: 00, 01, 10, 11 -> flattened [B00, B01, B10, B11]
    # C indices: 00, 01, 10, 11 -> flattened [C00, C01, C10, C11]
    
    terms = []
    # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,0]
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    # M3: u=[1,0,0,0], v=[0,0,1,-1], w=[0,1,0,0]
    terms.append(([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]))
    # M4: u=[0,0,0,1], v=[0,1,-1,0], w=[1,0,0,1]
    terms.append(([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, 1]))
    # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,0,0,0]  # Note: C00 gets -M5
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]))
    # M6: u=[0,-1,1,0], v=[1,1,0,0], w=[0,0,0,1]
    terms.append(([0, -1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    # M7: u=[0,1,-1,0], v=[0,1,0,1], w=[1,0,0,0]
    terms.append(([0, 1, -1, 0], [0, 1, 0, 1], [1, 0, 0, 0]))
    
    return terms

def _rank1_2x2x1():
    """
    Return the single rank-1 term for 2x2x1 matrix multiplication.
    This computes the sum of all products: C = sum_{i,j} A[i,j] * B[j]
    But wait, 2x2x1 means A is 2x2, B is 2x1, C is 2x1.
    Actually, for a block of size 2x2x1 (A:2x2, B:2x1, C:2x1), 
    the standard decomposition has rank 4 (one per output element).
    But we can do better? No, 2x2x1 has rank 4.
    
    Wait, let me reconsider. The 2+2+1 partition means:
    - Two 2x2x2 blocks (using Strassen, rank 7 each)
    - One 2x2x1 block? No, the dimensions don't work out that way.
    
    Let me re-examine the partition. 5 = 2+2+1.
    So we have blocks:
    - A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (2x2x2)
    - A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (2x2x2)
    - A[0:2, 4:5] * B[4:5, 0:2] -> C[0:2, 0:2] (2x2x1)
    - A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2] (2x2x2)
    - etc.
    
    Actually, the full 5x5x5 tensor decomposes into 27 blocks of various sizes.
    For each (i_block, j_block, k_block) where i_block, j_block, k_block are in {0,1,2}
    corresponding to sizes {2,2,1}, we have a block of size (i_size x j_size x k_size).
    
    The rank of a d1 x d2 x d3 matrix multiplication tensor is at most d1*d2*d3.
    For 2x2x2: rank 7 (Strassen)
    For 2x2x1: rank 4 (standard, since it's like multiplying 2x2 by 2x1)
    For 2x1x2: rank 4
    For 1x2x2: rank 4
    For 2x1x1: rank 2
    For 1x2x1: rank 2
    For 1x1x2: rank 2
    For 1x1x1: rank 1
    
    Let me count the total rank:
    Blocks (i_size, j_size, k_size) where i_size, j_size, k_size in {2,2,1}:
    - (2,2,2): 8 blocks, each rank 7 -> 56
    - (2,2,1): 4 blocks (i=2, j=2, k=1; i=2, j=1, k=2; i=1, j=2, k=2; wait, need to be careful)
    
    Actually, let me think differently. The partition divides rows of A into sizes [2,2,1],
    columns of A / rows of B into sizes [2,2,1], and columns of B / rows of C into sizes [2,2,1].
    
    So we have 3x3x3 = 27 blocks. For block (a,b,c) where a is the row-block of A,
    b is the column-block of A / row-block of B, c is the column-block of B / row-block of C:
    - Block (a,b,c) computes: sum over k in block b of A[a,i,k] * B[b,k,c] for output C[a,i,c]
    
    The size of block (a,b,c) is: A rows = size[a], A cols = size[b], B rows = size[b], B cols = size[c], C rows = size[a], C cols = size[c].
    So it's a (size[a] x size[b] x size[c]) multiplication.
    
    Sizes: [2, 2, 1]
    
    Block types and counts:
    - (2,2,2): a=0 or 1, b=0 or 1, c=0 or 1 -> 8 blocks, rank 7 each -> 56
    - (2,2,1): a in {0,1}, b in {0,1}, c=2 -> 4 blocks, rank 4 each -> 16
    - (2,1,2): a in {0,1}, b=2, c in {0,1} -> 4 blocks, rank 4 each -> 16
    - (2,1,1): a in {0,1}, b=2, c=2 -> 2 blocks, rank 2 each -> 4
    - (1,2,2): a=2, b in {0,1}, c in {0,1} -> 4 blocks, rank 4 each -> 16
    - (1,2,1): a=2, b in {0,1}, c=2 -> 2 blocks, rank 2 each -> 4
    - (1,1,2): a=2, b=2, c in {0,1} -> 2 blocks, rank 2 each -> 4
    - (1,1,1): a=2, b=2, c=2 -> 1 block, rank 1 -> 1
    
    Total: 56 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 117
    
    Yes! Rank 117.
    """
    # For 2x2x1: A is 2x2, B is 2x1, C is 2x1
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    # Rank 4 decomposition:
    # Term 1: u=[1,0,0,0], v=[1,0], w=[1,0]  (A00*B00 -> C0)
    # Term 2: u=[0,1,0,0], v=[0,1], w=[1,0]  (A01*B01 -> C0)
    # Term 3: u=[0,0,1,0], v=[1,0], w=[0,1]  (A10*B00 -> C1)
    # Term 4: u=[0,0,0,1], v=[0,1], w=[0,1]  (A11*B01 -> C1)
    return [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1])
    ]

def _rank1_2x1x2():
    """
    2x1x2: A is 2x1, B is 1x2, C is 2x2
    C[0,0] = A[0]*B[0], C[0,1] = A[0]*B[1]
    C[1,0] = A[1]*B[0], C[1,1] = A[1]*B[1]
    Rank 4 decomposition:
    """
    return [
        ([1, 0], [1, 0], [1, 0, 0, 0]),
        ([1, 0], [0, 1], [0, 1, 0, 0]),
        ([0, 1], [1, 0], [0, 0, 1, 0]),
        ([0, 1], [0, 1], [0, 0, 0, 1])
    ]

def _rank1_1x2x2():
    """
    1x2x2: A is 1x2, B is 2x2, C is 1x2
    C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    Rank 4 decomposition:
    """
    return [
        ([1, 0], [1, 0, 0, 0], [1, 0]),
        ([0, 1], [0, 0, 1, 0], [1, 0]),
        ([1, 0], [0, 1, 0, 0], [0, 1]),
        ([0, 1], [0, 0, 0, 1], [0, 1])
    ]

def _rank1_2x1x1():
    """
    2x1x1: A is 2x1, B is 1x1, C is 2x1
    C[0] = A[0]*B[0], C[1] = A[1]*B[0]
    Rank 2 decomposition:
    """
    return [
        ([1, 0], [1], [1, 0]),
        ([0, 1], [1], [0, 1])
    ]

def _rank1_1x2x1():
    """
    1x2x1: A is 1x2, B is 2x1, C is 1x1
    C[0] = A[0]*B[0] + A[1]*B[1]
    Rank 2 decomposition:
    """
    return [
        ([1, 0], [1, 0], [1]),
        ([0, 1], [0, 1], [1])
    ]

def _rank1_1x1x2():
    """
    1x1x2: A is 1x1, B is 1x2, C is 1x2
    C[0] = A[0]*B[0], C[1] = A[0]*B[1]
    Rank 2 decomposition:
    """
    return [
        ([1], [1, 0], [1, 0]),
        ([1], [0, 1], [0, 1])
    ]

def _rank1_1x1x1():
    """
    1x1x1: A is 1x1, B is 1x1, C is 1x1
    Rank 1 decomposition:
    """
    return [
        ([1], [1], [1])
    ]

def construct():
    """
    Return a rank-117 decomposition for 5x5x5 matrix multiplication
    using 2+2+1 partition with Strassen for 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS
    sizes = [2, 2, 1]
    
    # Get all block decompositions
    strassen_terms = _strassen_2x2x2()
    r221_terms = _rank1_2x2x1()
    r212_terms = _rank1_2x1x2()
    r122_terms = _rank1_1x2x2()
    r211_terms = _rank1_2x1x1()
    r121_terms = _rank1_1x2x1()
    r112_terms = _rank1_1x1x2()
    r111_terms = _rank1_1x1x1()
    
    # Compute total rank
    total_rank = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                sa, sb, sc = sizes[a], sizes[b], sizes[c]
                if (sa, sb, sc) == (2, 2, 2):
                    total_rank += 7
                elif (sa, sb, sc) == (2, 2, 1):
                    total_rank += 4
                elif (sa, sb, sc) == (2, 1, 2):
                    total_rank += 4
                elif (sa, sb, sc) == (1, 2, 2):
                    total_rank += 4
                elif (sa, sb, sc) == (2, 1, 1):
                    total_rank += 2
                elif (sa, sb, sc) == (1, 2, 1):
                    total_rank += 2
                elif (sa, sb, sc) == (1, 1, 2):
                    total_rank += 2
                elif (sa, sb, sc) == (1, 1, 1):
                    total_rank += 1
    
    rank = total_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                sa, sb, sc = sizes[a], sizes[b], sizes[c]
                a_start = sum(sizes[:a])
                b_start = sum(sizes[:b])
                c_start = sum(sizes[:c])
                
                if (sa, sb, sc) == (2, 2, 2):
                    terms = strassen_terms
                elif (sa, sb, sc) == (2, 2, 1):
                    terms = r221_terms
                elif (sa, sb, sc) == (2, 1, 2):
                    terms = r212_terms
                elif (sa, sb, sc) == (1, 2, 2):
                    terms = r122_terms
                elif (sa, sb, sc) == (2, 1, 1):
                    terms = r211_terms
                elif (sa, sb, sc) == (1, 2, 1):
                    terms = r121_terms
                elif (sa, sb, sc) == (1, 1, 2):
                    terms = r112_terms
                elif (sa, sb, sc) == (1, 1, 1):
                    terms = r111_terms
                else:
                    continue
                
                for u, v, w in terms:
                    # u has length sa*sb, v has length sb*sc, w has length sc*sa
                    # Map to global indices
                    for i in range(sa):
                        for j in range(sb):
                            U[(a_start + i) * m + (b_start + j)][col] = u[i * sb + j]
                    for i in range(sb):
                        for j in range(sc):
                            V[(b_start + i) * p + (c_start + j)][col] = v[i * sc + j]
                    for i in range(sc):
                        for j in range(sa):
                            W[(c_start + i) * n + (a_start + j)][col] = w[i * sa + j]
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
