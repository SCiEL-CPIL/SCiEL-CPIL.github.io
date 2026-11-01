"""Rank-minimal tensor decomposition for 2x3 x 3x7 matrix multiplication."""
import time
import itertools
from fractions import Fraction

_START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin below 150s kill

def _check_validity(U, V, W, n, m, p, max_sample=3000):
    """Verify sum_c U[i,j,c]*V[j,k,c]*W[k*n+i,c] == sum_l A[i,l]*B[l,k]."""
    from fractions import Fraction
    r = len(U[0]) if U else 0
    if len(U) != n*m: return False
    if len(V) != m*p: return False
    if len(W) != p*n: return False
    for row in U + V + W:
        if len(row) != r: return False
    # Test with specific random-ish values (deterministic)
    for trial in range(min(max_sample, 500)):
        import random
        rng = random.Random(trial * 12345 + 7)
        A = [[Fraction(rng.randint(0, 3)) for _ in range(m)] for _ in range(n)]
        B = [[Fraction(rng.randint(0, 3)) for _ in range(p)] for _ in range(m)]
        # True product
        C_true = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = Fraction(0)
                for l in range(m):
                    s += A[i][l] * B[l][k]
                C_true[i][k] = s
        # Decoded product
        C_dec = [[Fraction(0) for _ in range(p)] for _ in range(n)]
        for c in range(r):
            # u_{i,j,c} v_{j,k,c}
            for i in range(n):
                for j in range(m):
                    uj = U[i*m + j][c]
                    if uj == 0: continue
                    for k in range(p):
                        vk = V[j*p + k][c]
                        if vk == 0: continue
                        wval = W[k*n + i][c]
                        if wval != 0:
                            C_dec[i][k] += uj * vk * wval
        for i in range(n):
            for k in range(p):
                if C_dec[i][k] != C_true[i][k]:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _rank_via_column_splitting(n, m, p, splits, use_strassen_for_222=False):
    """
    Try to reduce rank by grouping columns of B (the middle index l).
    splits: list of tuples (l_start, l_end) covering [0, m), plus a flag per group.
    For each group, if it's "block-diag" (size 1): schoolbook for that group.
    if it's "2x2" size with size 2, use Strassen for 2x2x? subproblem.
    Actually, simpler approach: split the middle dimension l into groups of size s.
    For each group, if s=1: R(2,1,7) = 14 (must).
    If s=2: R(2,2,7) = min over splits of p of sum of R(2,2,p_chunk).
    """
    # We'll enumerate ways to partition the middle dimension into groups.
    # For 2x3: split into (1,2) or (2,1) or (1,1,1).
    # The (1,2) and (2,1) cases both have one group of size 1 (rank 14) and one of size 2.
    # R(2,2,7): further split p=7 into chunks.
    # R(2,2,2)=7 (Strassen), R(2,2,1)=4, R(2,2,3)=? 
    # Known: R(2,2,3) = 11 (Shitov). Actually R(2,2,3): n=2,m=2,p=3.
    # Bound: max(4,6,6)=6 <= R <= 12. Known R(2,2,3) <= 11.
    # Actually R(2,2,3) = 11 per Shitov. Let me use upper bound 11.
    # But I need to actually CONSTRUCT it, not just bound.
    # For safety, use schoolbook for chunks unless I have an explicit construction.
    # R(2,2,2) = 7: I'll use Strassen.
    # R(2,2,1) = 4: schoolbook.
    # R(2,2,3): schoolbook = 12, or if I can do better...
    # For p-chunk of size q: R(2,2,q) <= 7 * ceil(q/2) if I pair off... no.
    # Simplest: R(2,2,q) <= q * R(2,2,1) = 4q via schoolbook, OR
    # split into ceil(q/2) blocks of 2 (with remainder 1): 
    # R(2,2,q) <= 7*floor(q/2) + 4*(q mod 2)
    # For q=7: 7*3 + 4*1 = 25. Hmm, that's the same as before.
    # Actually R(2,2,7): split 7 = 2+2+2+1: 7+7+7+4 = 25.
    # Can we do 2+2+3? R(2,2,3) <= 12 schoolbook, so 7+7+12 = 26. Worse.
    # So best for R(2,2,7) is 25 via four chunks (2,2,2,1).
    # Total: 14 + 25 = 39.
    
    # But wait, I'm not doing cross-group sharing. Let me just construct rank 39
    # by splitting l into {0} and {1,2}, and for the {1,2} group split k into (0,1), (2,3), (4,5), (6).
    n, m, p = 2, 3, 7
    # Group l=0: R(2,1,7) = 14
    # Group l in {1,2}: R(2,2,7), split k into pairs and single
    # This gives 14 + 25 = 39.
    
    # Actually, let me just be safe and use schoolbook rank 42.
    # The time budget makes complex constructions risky.
    return None

def _construct_strassen_222():
    """Return (U,V,W) for 2x2x2 with rank 7 (Strassen)."""
    # Strassen's 7 multiplications for 2x2:
    # M1 = (A11)(B11+B22)
    # M2 = (A11+A12)(B22)
    # M3 = (A21+A22)(B11)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A22)(B11+B22)  -- no, let me use the standard:
    # Actually standard Strassen:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B12 - B22)
    # P4 = A22(B21 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # In our format: A is 2x2 (n=2,m=2), B is 2x2 (m=2,p=2).
    # U has shape (n*m, 7) = (4, 7). U[i*m+j, c] = u_c^{ij}
    # V has shape (m*p, 7) = (4, 7). V[j*p+k, c] = v_c^{jk}
    # W has shape (p*n, 7) = (4, 7). W[k*n+i, c] = w_c^{ki}
    
    # Strassen 7 mults:
    # P1 = (A11+A22)(B11+B22): u = {00:1, 11:1}, v = {00:1, 11:1}
    # P2 = (A21+A22)B11: u = {10:1, 11:1}, v = {10:1}
    # P3 = A11(B12-B22): u = {00:1}, v = {01:1, 11:-1}
    # P4 = A22(B21-B11): u = {11:1}, v = {11:1, 10:-1}
    # P5 = (A11+A12)B22: u = {00:1, 01:1}, v = {11:1}
    # P6 = (A21-A11)(B11+B12): u = {10:1, 00:-1}, v = {10:1, 01:1}
    # P7 = (A12-A22)(B21+B22): u = {01:1, 11:-1}, v = {11:1, 10:1}
    
    # W entries: W[k*n + i, c] where c is the P index
    # C11 = P1 + P4 - P5 + P7: W[0*2+0, c] = W[0, c]
    #   c=0(P1): +1, c=3(P4): +1, c=4(P5): -1, c=6(P7): +1
    # C12 = P3 + P5: W[1*2+0, c] = W[2, c]
    #   c=2(P3): +1, c=4(P5): +1
    # C21 = P2 + P4: W[0*2+1, c] = W[1, c]
    #   c=1(P2): +1, c=3(P4): +1
    # C22 = P1 - P2 + P3 + P6: W[1*2+1, c] = W[3, c]
    #   c=0(P1): +1, c=1(P2): -1, c=2(P3): +1, c=5(P6): +1
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00: P1, P3, P5, P6
        [0, 0, 0, 0, 1, 0, 1],   # A01: P5, P7
        [0, 1, 0, 0, 0, 1, 0],   # A10: P2, P6
        [1, 1, 0, 1, 0, 0, -1],  # A11: P1, P2, P4, P7
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0],   # B00: P1
        [0, 0, 1, 0, 0, 1, 1],   # B01: P3, P6, P7
        [1, 1, 0, -1, 0, 1, 1],  # B10: P1, P2, P4, P6, P7
        [1, 0, -1, 1, 1, 0, 1],  # B11: P1, P3, P4, P5, P7
    ]
    W = [
        [1, 0, 0, 0, -1, 0, 1],  # C00 (k=0,i=0): P1, P4? No...
        # W[k*n+i, c]: k=0,i=0 -> index 0
        # W[k*n+i, c]: k=0,i=1 -> index 1
        # W[k*n+i, c]: k=1,i=0 -> index 2
        # W[k*n+i, c]: k=1,i=1 -> index 3
    ]
    W = [
        [1, 0, 0, 0, -1, 0, 1],  # W[0, c] for C_{0,0}: P1 - P5 + P7 + P4
        [0, 1, 0, 1, 0, 0, 0],   # W[1, c] for C_{1,0}: P2 + P4
        [0, 0, 1, 0, 1, 0, 0],   # W[2, c] for C_{0,1}: P3 + P5
        [1, -1, 1, 0, 0, 1, 0],  # W[3, c] for C_{1,1}: P1 - P2 + P3 + P6
    ]
    return {"U": U, "V": V, "W": W}

def _construct_212_schoolbook_for_217():
    """R(2,1,7) = 14, trivial."""
    return _schoolbook(2, 1, 7)

def _attempt_rank_reduction(base_U, base_V, base_W, n, m, p, max_time=120.0):
    """
    Attempt to reduce rank via structured operations.
    We start from schoolbook and try to:
    1. Identify linear dependencies among rows of U, V, W
    2. Use symmetry/structure specific to (2,3,7)
    
    Strategy: Since rank 42 = 6*7 = 14*3 = 21*2, and we know R >= 36,
    the best we can reliably construct is 42 (schoolbook) unless we implement
    explicit Strassen-based block decomposition.
    
    Let's try: split middle dimension into l=0 and l in {1,2}.
    For l=0: schoolbook gives 14 terms (2 * 1 * 7).
    For l in {1,2}: R(2,2,7). Split k into (0,1),(2,3),(4,5),(6).
    R(2,2,2) = 7 each (Strassen), R(2,2,1) = 4 (schoolbook).
    Total for l in {1,2}: 7+7+7+4 = 25.
    Grand total: 14 + 25 = 39.
    """
    # Construct rank 39 via block decomposition
    # Part 1: l=0, all k: R(2,1,7) = 14
    # Part 2: l in {1,2}, k split into (0,1), (2,3), (4,5), (6)
    #   Each 2x2x2 block: Strassen rank 7
    #   One 2x2x1 block: schoolbook rank 4
    # Total: 14 + 7 + 7 + 7 + 4 = 39
    
    n, m, p = 2, 3, 7
    rank = 39
    U = [[0]*rank for _ in range(n*m)]  # 6 x 39
    V = [[0]*rank for _ in range(m*p)]  # 21 x 39
    W = [[0]*rank for _ in range(p*n)]  # 14 x 39
    
    col = 0
    
    # Part 1: l=0 (middle index 0), k = 0..6
    # U[i*3+0, c] = 1 for each (i,k)
    # V[0*7+k, c] = 1
    # W[k*2+i, c] = 1
    for i in range(2):
        for k in range(7):
            U[i*3 + 0][col] = 1
            V[0*7 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
    
    # Part 2: l in {1,2}, k pairs and single
    # k_pairs = [(0,1), (2,3), (4,5)], k_single = (6,)
    # For each k-chunk, use the appropriate decomposition for 2x2 x |chunk|
    
    def strassen_222_block(col_offset, k_vals, l_base):
        """Add Strassen block for 2x2 (l in {l_base, l_base+1}) x 2 (k in k_vals)."""
        # l indices: l_base, l_base+1 (these are 1,2 in our case)
        # k indices: k_vals[0], k_vals[1]
        s = _construct_strassen_222()
        sU, sV, sW = s["U"], s["V"], s["W"]
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    val = sU[i*2 + j][c]
                    if val != 0:
                        U[i*3 + (l_base + j)][col_offset + c] = val
            for j in range(2):
                for k in range(2):
                    val = sV[j*2 + k][c]
                    if val != 0:
                        V[(l_base + j)*7 + (k_vals[0] + k)][col_offset + c] = val
            for k in range(2):
                for i in range(2):
                    val = sW[k*2 + i][c]
                    if val != 0:
                        W[(k_vals[0] + k)*2 + i][col_offset + c] = val
        return col_offset + 7
    
    def schoolbook_221_block(col_offset, k_val, l_base):
        """Add schoolbook block for 2x2 (l) x 1 (k)."""
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U[i*3 + (l_base + j)][col_offset] = 1
                    V[(l_base + j)*7 + (k_val)][col_offset] = 1
                    W[k_val*2 + i][col_offset] = 1
                    col_offset += 1
        return col_offset
    
    # k-pairs with l in {1,2}
    col = strassen_222_block(col, (0, 1), 1)
    col = strassen_222_block(col, (2, 3), 1)
    col = strassen_222_block(col, (4, 5), 1)
    col = schoolbook_221_block(col, 6, 1)
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n, m, p = 2, 3, 7
    elapsed = time.time() - _START_TIME
    
    # Try rank 39 construction first
    try:
        result39 = _attempt_rank_reduction(None, None, None, n, m, p, max_time=60.0)
        if result39 is not None and _check_validity(result39["U"], result39["V"], result39["W"], n, m, p):
            elapsed = time.time() - _START_TIME
            if elapsed < TIME_LIMIT:
                return result39
    except Exception:
        pass
    
    # Fall back to schoolbook rank 42
    elapsed = time.time() - _START_TIME
    if elapsed < TIME_LIMIT:
        result42 = _schoolbook(n, m, p)
        return result42
    
    # Last resort (shouldn't reach here)
    return _schoolbook(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
