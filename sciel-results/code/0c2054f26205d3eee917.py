import numpy as np
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N

    # GF(256) arithmetic with polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
    def gf_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= 0x11D
        return res & 0xFF

    def gf_pow(a, n):
        res = 1
        base = a
        while n:
            if n & 1:
                res = gf_mul(res, base)
            base = gf_mul(base, base)
            n >>= 1
        return res

    # Find primitive element
    alpha = None
    for cand in range(2, 256):
        if gf_pow(cand, 255) == 1:
            # Check if it's not a divisor of 255
            is_prim = True
            for factor in [3, 5, 17]:
                if gf_pow(cand, 255 // factor) == 1:
                    is_prim = False
                    break
            if is_prim:
                alpha = cand
                break
    
    if alpha is None:
        # Fallback to random
        random.seed(42)
        code = {0}
        for _ in range(10000):
            if time.time() > deadline - 5.0:
                break
            cand = random.randint(0, TOTAL - 1)
            if cand in code:
                continue
            valid = True
            for w in code:
                if (cand ^ w).bit_count() < MIN_DIST:
                    valid = False
                    break
            if valid:
                code.add(cand)
        return list(code)

    # Compute minimal polynomial of alpha
    # m(x) = x^8 + c7 x^7 + ... + c0
    # alpha^8 = c7 alpha^7 + ... + c0
    # We want to find c's such that alpha is a root.
    # This is equivalent to finding the characteristic polynomial of the multiplication-by-alpha matrix.
    # Or simpler: alpha satisfies P(alpha)=0 where P is the primitive polynomial.
    # The primitive polynomial is x^8 + x^4 + x^3 + x^2 + 1.
    # So m_1(x) = x^8 + x^4 + x^3 + x^2 + 1.
    # Let's verify: 0x11D is 100011101.
    # x^8 + x^4 + x^3 + x^2 + 1.
    # Coeffs: [1, 0, 0, 0, 1, 1, 1, 0, 1] (from x^8 to x^0)
    
    g_coeffs = [1, 0, 0, 0, 1, 1, 1, 0, 1] # g(x) = x^8 + x^4 + x^3 + x^2 + 1
    
    # Generate [17, 9, 5] code
    # Generator matrix G (9x17)
    # Row i is x^i * g(x) mod (x^17 - 1)
    G = []
    for i in range(9):
        row = [0] * 17
        for j in range(9):
            if g_coeffs[j] == 1:
                idx = (i + j) % 17
                row[idx] = 1
        G.append(row)
    
    # G is list of lists. Convert to numpy for speed? Or just use lists.
    # Find dual code basis (null space of G^T)
    # G is 9x17. G^T is 17x9.
    # We want vectors v (length 17) such that G^T v = 0.
    # This is the null space of G^T.
    # Perform Gaussian elimination on G^T.
    
    # Matrix for elimination: 17 rows, 9 columns.
    # We want to find 8 linearly independent vectors in the null space.
    # Actually, the dual code has dimension 8.
    # We can find the basis by identifying free variables.
    
    # Let's use a simple null space finder.
    # A = G^T (17x9)
    # We want x such that A x = 0.
    # Rank is 9. Nullity is 17-9=8.
    
    # Pivot columns
    pivots = []
    A = [list(row) for row in zip(*G)] # 17x9
    # Elimination
    for col in range(9):
        pivot_row = -1
        for r in range(len(pivots), 17):
            if A[r][col] == 1:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        # Swap
        A[len(pivots)], A[pivot_row] = A[pivot_row], A[len(pivots)]
        pivots.append(col)
        # Eliminate
        for r in range(17):
            if r != len(pivots) - 1 and A[r][col] == 1:
                for c in range(9):
                    A[r][c] ^= A[len(pivots)-1][c]
                    
    # Now A is in RREF.
    # Pivots are in columns pivots.
    # Free columns are the others.
    # We have 8 free variables.
    # Basis vectors for null space:
    # For each free column f, set x_f = 1, other free x = 0.
    # Then x_pivot = sum(A[r, f] for r in pivot_rows) ?
    # Equation: sum(A[r, c] x_c) = 0 for each row r.
    # x_p = sum_{f != p} A[r, f] x_f.
    
    free_cols = [c for c in range(9) if c not in pivots]
    dual_basis = []
    for f in free_cols:
        vec = [0] * 17
        vec[f] = 1
        for r, p in enumerate(pivots):
            # x_p = sum(A[r, f] * x_f) = A[r, f]
            # Wait, the equation is x_p + sum_{f} A[r, f] x_f = 0
            # So x_p = sum_{f} A[r, f] x_f
            vec[p] = A[r][f]
        dual_basis.append(vec)
        
    # dual_basis is 8 vectors of length 17.
    # These generate the [17, 8, 6] code.
    
    # Generate all 256 codewords
    current_code = set()
    for u in range(256):
        c = 0
        for k in range(8):
            if (u >> k) & 1:
                for j in range(17):
                    if dual_basis[k][j]:
                        c ^= 1 << j
        current_code.add(c)
        
    code_list = list(current_code)
    
    # Local Search
    random.seed(123)
    
    # Phase 1: Add random words
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 200000:
        cand = random.randint(0, TOTAL - 1)
        if cand in current_code:
            attempts += 1
            continue
        valid = True
        for w in code_list:
            if (cand ^ w).bit_count() < MIN_DIST:
                valid = False
                break
        if valid:
            current_code.add(cand)
            code_list.append(cand)
        attempts += 1
        
    # Phase 2: Swap search (if time permits)
    if time.time() < deadline - 5.0:
        swap_attempts = 0
        while time.time() < deadline - 5.0 and swap_attempts < 50000:
            if not code_list:
                break
            idx = random.randint(0, len(code_list) - 1)
            old_word = code_list[idx]
            current_code.discard(old_word)
            code_list.pop(idx)
            
            new_word = None
            for _ in range(50):
                if time.time() > deadline - 5.0:
                    break
                cand = random.randint(0, TOTAL - 1)
                if cand in current_code:
                    continue
                valid = True
                for w in code_list:
                    if (cand ^ w).bit_count() < MIN_DIST:
                        valid = False
                        break
                if valid:
                    new_word = cand
                    break
                    
            if new_word is not None:
                current_code.add(new_word)
                code_list.append(new_word)
            else:
                current_code.add(old_word)
                code_list.append(old_word)
            swap_attempts += 1
            
    return list(current_code)
