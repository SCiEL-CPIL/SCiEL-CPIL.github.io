"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return the 7 terms of the 2x2x2 Strassen decomposition.
    Each term is (u, v, w) where u, v, w are lists of length 4 (indices into A, B, C).
    A is 2x2 (indices 0,1,2,3 row-major), B is 2x2 (indices 0,1,2,3 row-major), C is 2x2 (indices 0,1,2,3 row-major, trace convention: C[k*n+i]).
    Actually for 2x2x2: n=m=p=2.
    A indices: i*m+j = i*2+j -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    B indices: j*p+k = j*2+k -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    C indices: k*n+i = k*2+i -> (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    """
    # We need to express each output C element as sum of products.
    # Each term contributes a scalar multiple to each of the 7 multiplications.
    # The decomposition has 7 terms.
    
    # Term 1: (A00+A11)(B00+B11) -> contributes to C00(+1), C11(+1)
    # u: A00=0, A11=3 -> u=[1,0,0,1]
    # v: B00=0, B11=3 -> v=[1,0,0,1]
    # w: C00=0, C11=3 -> w=[1,0,0,1]
    
    # Term 2: (A10+A11)B00 -> contributes to C10(+1), C11(-1)
    # u: A10=2, A11=3 -> u=[0,0,1,1]
    # v: B00=0 -> v=[1,0,0,0]
    # w: C10=2, C11=3 -> w=[0,0,1,1] with signs: C10 gets +1, C11 gets -1
    # So w = [0, 0, 1, -1]
    
    # Term 3: A00(B10-B11) -> contributes to C01(+1), C11(+1)
    # u: A00=0 -> u=[1,0,0,0]
    # v: B10=2, B11=3 -> v=[0,0,1,-1]
    # w: C01=1, C11=3 -> w=[0,1,0,1]
    
    # Term 4: A11(B01-B00) -> contributes to C00(+1), C10(+1)
    # u: A11=3 -> u=[0,0,0,1]
    # v: B01=1, B00=0 -> v=[-1,1,0,0]
    # w: C00=0, C10=2 -> w=[1,0,1,0]
    
    # Term 5: (A00+A01)B11 -> contributes to C00(-1), C01(+1)
    # u: A00=0, A01=1 -> u=[1,1,0,0]
    # v: B11=3 -> v=[0,0,0,1]
    # w: C00=0(-1), C01=1(+1) -> w=[-1,1,0,0]
    
    # Term 6: (A10-A00)(B00+B10) -> contributes to C11(+1)
    # u: A10=2, A00=0 -> u=[-1,0,1,0]
    # v: B00=0, B10=2 -> v=[1,0,1,0]
    # w: C11=3 -> w=[0,0,0,1]
    
    # Term 7: (A01-A11)(B01+B11) -> contributes to C00(+1)
    # u: A01=1, A11=3 -> u=[0,1,0,-1]
    # v: B01=1, B11=3 -> v=[0,1,0,1]
    # w: C00=0 -> w=[1,0,0,0]
    
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,1,0,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,0,1,0], [0,0,0,1]),
        ([0,1,0,-1], [0,1,0,1], [1,0,0,0]),
    ]
    return terms

def _schoolbook(n, m, p):
    """Return schoolbook terms for n x m x p."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n*m)
                v = [0] * (m*p)
                w = [0] * (p*n)
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                terms.append((u, v, w))
    return terms

def _combine_terms(terms, n, m, p):
    """Convert a list of (u, v, w) terms (each a list of length n*m, m*p, p*n)
    into the factor matrices U, V, W."""
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for col, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            if u[idx] != 0:
                U[idx][col] = u[idx]
        for idx in range(m * p):
            if v[idx] != 0:
                V[idx][col] = v[idx]
        for idx in range(p * n):
            if w[idx] != 0:
                W[idx][col] = w[idx]
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # safety margin
    
    # For 2x5x6, we can split m=5 into 2+3.
    # 2x2x6: split m=2, can we use Strassen? Strassen is for 2x2x2.
    # Actually, let's think about splitting differently.
    # We can view 2x5x6 as sum of 2x2x6 and 2x3x6 (splitting m).
    # Or 2x5x2 + 2x5x4 (splitting p).
    # Or 2x5x6 = 1x5x6 + 1x5x6 (splitting n).
    
    # Let's try to find a good decomposition.
    # Known: rank(2,5,6) is at most 2*5*6 = 60.
    # With Strassen on 2x2x2 blocks:
    # If we split m=5 into 2+3: rank(2,2,6) + rank(2,3,6).
    # rank(2,2,6): split p=6 into 2+4. rank(2,2,2)+rank(2,2,4) = 7 + rank(2,2,4).
    # rank(2,2,4): split p=4 into 2+2. rank(2,2,2)+rank(2,2,2) = 7+7 = 14.
    # So rank(2,2,6) <= 7 + 14 = 21.
    # rank(2,3,6): split p=6 into 2+4. rank(2,3,2)+rank(2,3,4).
    # rank(2,3,2): no obvious Strassen. Schoolbook = 12. Can we do better?
    # Actually for 2x3x2, maybe we can use a different approach.
    # Let's just use schoolbook for small cases and Strassen for 2x2x2.
    
    # A simpler approach: just use Strassen wherever possible by splitting into 2x2x2 blocks.
    # 2x5x6: split m=5 as 2+3, split p=6 as 2+2+2.
    # This gives 2x2x2 (3 times) + 2x3x2 (3 times).
    # 3*7 + 3*12 = 21 + 36 = 57. Not great.
    
    # Alternative: split m=5 as 2+3, and for the 2x3x6 part, split m=3 as 1+2.
    # 2x1x6: schoolbook = 12. Or split p=6 as 2+2+2: 3 * rank(2,1,2) = 3*4 = 12.
    # 2x2x6: as above, 21.
    # Total: 12 + 21 = 33 for 2x5x6? Wait, 2x1x6 + 2x4x6? No, m=5=1+4.
    # 2x1x6 = 12. 2x4x6: split m=4 as 2+2. 2*rank(2,2,6) = 2*21 = 42. Total 54.
    
    # Let's try: m=5 = 2+3.
    # 2x2x6: split p=6 as 2+2+2. 3 * rank(2,2,2) = 3*7 = 21.
    # 2x3x6: split m=3 as 1+2. 
    #   2x1x6: schoolbook 12.
    #   2x2x6: 21.
    #   Total: 33.
    # Grand total: 21 + 33 = 54.
    
    # Even better: for 2x3x6, split p=6 as 2+2+2.
    # 2x3x2: this is the tricky one. Let's use schoolbook = 12.
    # 3 * 12 = 36.
    # 2x2x2 parts: 3 * 7 = 21.
    # Total: 57.
    
    # So the 54 approach is better. Let's implement it.
    # Actually, let me reconsider. The problem is small enough that we can try multiple splits
    # and pick the best one. But for simplicity, let's implement the 54-rank decomposition.
    
    # 2x5x6 = 2x2x6 + 2x3x6
    # 2x2x6 = 2x2x2 + 2x2x2 + 2x2x2 (3 Strassen blocks)
    # 2x3x6 = 2x1x6 + 2x2x6
    #   2x1x6 = schoolbook (12 terms)
    #   2x2x6 = 3 Strassen blocks (21 terms)
    # Total: 21 + 12 + 21 = 54 terms.
    
    terms = []
    
    # Helper to add Strassen terms with offset
    def add_strassen(terms, i_off, j_off, k_off):
        """Add Strassen 2x2x2 terms with row offset i_off, col offset j_off, out-col offset k_off."""
        for (u, v, w) in _strassen_222():
            new_u = [0] * (n*m)
            new_v = [0] * (m*p)
            new_w = [0] * (p*n)
            for a in range(2):
                for b in range(2):
                    new_u[(i_off+a)*m + (j_off+b)] = u[a*2+b]
            for a in range(2):
                for b in range(2):
                    new_v[(j_off+a)*p + (k_off+b)] = v[a*2+b]
            for a in range(2):
                for b in range(2):
                    new_w[(k_off+a)*n + (i_off+b)] = w[a*2+b]
            terms.append((new_u, new_v, new_w))
    
    # Helper to add schoolbook terms with offset
    def add_schoolbook(terms, i_off, j_off, k_off, ni, mj, pk):
        """Add schoolbook terms for ni x mj x pk with offsets."""
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    new_u = [0] * (n*m)
                    new_v = [0] * (m*p)
                    new_w = [0] * (p*n)
                    new_u[(i_off+i)*m + (j_off+j)] = 1
                    new_v[(j_off+j)*p + (k_off+k)] = 1
                    new_w[(k_off+k)*n + (i_off+i)] = 1
                    terms.append((new_u, new_v, new_w))
    
    # Part 1: 2x2x6 (rows 0-1, cols 0-1, all 6 output cols)
    # Split into 3 Strassen blocks: (0,0,0), (0,0,2), (0,0,4)
    for k_off in [0, 2, 4]:
        add_strassen(terms, 0, 0, k_off)
    
    # Part 2: 2x3x6 (rows 0-1, cols 2-4, all 6 output cols)
    # Split into 2x1x6 + 2x2x6
    # 2x1x6: rows 0-1, col 2, all 6 output cols -> schoolbook
    add_schoolbook(terms, 0, 2, 0, 2, 1, 6)
    
    # 2x2x6: rows 0-1, cols 3-4, all 6 output cols -> 3 Strassen blocks
    for k_off in [0, 2, 4]:
        add_strassen(terms, 0, 3, k_off)
    
    result = _combine_terms(terms, n, m, p)
    return result
# EVOLVE-BLOCK-END
