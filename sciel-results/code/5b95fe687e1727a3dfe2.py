import random
import time

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 45.0
    
    # Precompute weights for all 17-bit integers
    # 2^17 = 131072, this is fast
    weights = [i.bit_count() for i in range(1 << N)]
    
    def dist(a, b):
        return weights[a ^ b]
        
    def can_add(w, code):
        for c in code:
            if dist(w, c) < MIN_DIST:
                return False
        return True

    # 1. Generate the [17,8,6] QR code subcode (256 words)
    # The QR code of length 17 is a cyclic code with generator polynomial
    # corresponding to the cyclotomic coset C1 = {1, 2, 4, 8, 16, 15, 13, 9}
    # We need to construct the generator matrix.
    # Alternatively, we can just generate the code using the primitive element.
    
    # Let's use a simpler method: 
    # The code is linear. We can find a basis.
    # But finding the basis requires GF(2^8) arithmetic.
    # Let's implement a minimal GF(2^8) for this specific purpose.
    
    # Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11d)
    # We need an element alpha of order 17.
    # 2^8 - 1 = 255 = 15 * 17.
    # So any non-zero element beta has beta^255 = 1.
    # alpha = beta^15 will have order 17.
    
    # Let's pick a primitive polynomial and a primitive element.
    # Poly: 0x11d
    # We need to find a primitive element in GF(2^8) with this poly.
    # Usually x is primitive if the poly is primitive.
    # Is 0x11d primitive? Yes, it is a common one.
    # So alpha = x (which is 0x02 in polynomial representation? No, x is the element).
    # In GF(2^8) with poly 0x11d, the element 'x' is represented by 0x02.
    # Let's verify order of 0x02.
    # If 0x02 is primitive, its order is 255.
    # Then alpha = 0x02^15 mod poly.
    
    def gf_mul(a, b, poly=0x11d):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= poly
        return res

    def gf_pow(a, n, poly=0x11d):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a, poly)
            a = gf_mul(a, a, poly)
            n >>= 1
        return res

    # Check if 0x02 is primitive.
    # If not, try another.
    # For now, assume 0x02 is primitive.
    # alpha = 0x02^15
    alpha = gf_pow(0x02, 15)
    
    # The generator polynomial g(x) has roots alpha^1, ..., alpha^8
    # We can construct the code by finding the null space of the parity check matrix?
    # Or generate the generator matrix.
    # The generator matrix G of a cyclic code with generator g(x) = sum g_i x^i
    # has rows:
    # [g_0, g_1, ..., g_8, 0, ..., 0]
    # [0, g_0, ..., g_8, 0, ..., 0]
    # ...
    # [0, ..., 0, g_0, ..., g_8]
    
    # How to find coefficients of g(x)?
    # g(x) is the minimal polynomial of alpha.
    # It is the monic polynomial of smallest degree such that g(alpha) = 0.
    # Degree is 8.
    # g(x) = x^8 + c_7 x^7 + ... + c_0
    # We can find c_i by solving the linear system.
    # Or, we can just generate the code using the definition:
    # C = { u(x) g(x) mod (x^17 - 1) | deg(u) < 9 }
    
    # Let's compute g(x) coefficients.
    # We know g(x) divides x^17 - 1.
    # And g(alpha) = 0.
    # Let's construct the basis of GF(2^8) and find the relation.
    # Powers of alpha: 1, alpha, ..., alpha^8
    # We want c_0 + c_1 alpha + ... + c_8 alpha^8 = 0 with c_8 = 1.
    # So c_0 + ... + c_7 alpha^7 = alpha^8.
    # We can compute alpha^k in GF(2^8) and express them in terms of the basis {1, x, ..., x^7}.
    # But wait, the code is over GF(2). The coefficients c_i are in GF(2).
    # So we are looking for a linear dependency over GF(2) among the vectors representing alpha^0, ..., alpha^8 in GF(2^8).
    
    # Let's represent elements of GF(2^8) as integers 0..255.
    # The vectors are the bit representations.
    # v_k = representation of alpha^k.
    # We want sum c_k v_k = 0.
    # This is a linear algebra problem over GF(2) on 8-dimensional vectors.
    # There are 9 vectors, so there is a non-trivial dependency.
    
    vecs = []
    for k in range(9):
        val = gf_pow(alpha, k)
        # val is an integer 0..255
        # The vector is the 8 bits of val
        vecs.append(val)
        
    # Find c_0..c_8 such that XOR of c_k * vecs[k] = 0
    # We can do Gaussian elimination.
    # Since c_8 must be 1 (monic), we can express vecs[8] as XOR of some subset of vecs[0..7].
    # Actually, the minimal polynomial is unique.
    
    # Let's just perform elimination to find the relation.
    # Augmented matrix? No, just find the kernel.
    # Since we know c_8=1, we can set up the system:
    # c_0 v_0 + ... + c_7 v_7 = v_8
    # Solve for c_0..c_7.
    
    # Gaussian elimination on the columns of v_0..v_7 to solve for c.
    # Actually, it's easier to just try all 2^8 subsets? No, 256 is small.
    # But we can do it properly.
    
    # Let's use a simple solver.
    # We want c such that M c = v_8, where M is 8x8 matrix with columns v_0..v_7.
    # If M is invertible, there is a unique solution.
    
    # Construct M (8x8)
    # M[i][j] is the i-th bit of v_j
    M = [[(v_j >> i) & 1 for j in range(8)] for i in range(8)]
    b = [(v_8 >> i) & 1 for i in range(8)]
    
    # Solve M c = b over GF(2)
    # Gaussian Elimination
    n = 8
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    
    for col in range(n):
        # Find pivot
        pivot = -1
        for row in range(col, n):
            if A[row][col] == 1:
                pivot = row
                break
        if pivot == -1:
            # Should not happen if v_0..v_7 are independent
            continue
        A[col], A[pivot] = A[pivot], A[col]
        for row in range(n):
            if row != col and A[row][col] == 1:
                for k in range(n + 1):
                    A[row][k] ^= A[col][k]
    
    # Solution
    c = [0] * 8
    for i in range(n):
        if A[i][i] == 1:
            c[i] = A[i][n]
        else:
            # Free variable, but we expect unique solution
            c[i] = 0
            
    # c_8 is 1
    coeffs = c + [1]
    # coeffs[i] is coefficient of x^i
    # g(x) = sum coeffs[i] x^i
    
    # Now generate the code
    # g(x) coefficients: coeffs[0]..coeffs[8]
    # Generator matrix rows are shifts of g(x)
    # But we need to be careful with the cyclic shift mod x^17-1.
    # For a cyclic code, the generator matrix can be:
    # Row i: coefficients of x^i g(x) mod (x^17-1)
    
    code_words = set()
    # The code is the span of the rows of G.
    # Since it's linear, we can just generate all 2^9 combinations of the basis vectors.
    # Basis vectors are the rows of G.
    
    basis = []
    for i in range(9):
        # Compute x^i * g(x) mod (x^17 - 1)
        # g(x) is represented by coeffs
        # x^i * g(x) shifts coeffs to the right by i
        # If index >= 17, wrap around to index - 17 (add to low end)
        # Wait, x^17 = 1, so x^17 = 1, x^18 = x, etc.
        # So x^i * x^j = x^(i+j) = x^((i+j) % 17)
        
        row = [0] * 17
        for j in range(9):
            if coeffs[j] == 1:
                idx = (i + j) % 17
                row[idx] = 1
        basis.append(row)
        
    # Generate all codewords
    # There are 2^9 = 512 codewords
    # We can iterate through all masks 0..511
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                # XOR with basis[i]
                for j in range(17):
                    if basis[i][j]:
                        word ^= (1 << j)
        code_words.add(word)
        
    # Filter for even weight to get the [17,8,6] subcode
    words = [w for w in code_words if weights[w] % 2 == 0]
    
    # Now we have 256 words.
    # Convert to list for search
    words = list(words)
    
    rng = random.Random(42)
    best_size = len(words)
    best_words = list(words)
    
    def try_add_random(code, rng, attempts=300):
        for _ in range(attempts):
            if time.time() - start_time > time_limit:
                return False
            w = rng.getrandbits(N)
            if can_add(w, code):
                code.append(w)
                return True
        return False
        
    def try_flip(code, rng):
        idx = rng.randrange(len(code))
        bit = rng.randrange(N)
        new_w = code[idx] ^ (1 << bit)
        for j, c in enumerate(code):
            if j != idx and dist(new_w, c) < MIN_DIST:
                return False
        code[idx] = new_w
        return True
        
    def try_swap(code, rng):
        idx = rng.randrange(len(code))
        old_w = code[idx]
        code.pop(idx)
        for _ in range(200):
            w = rng.getrandbits(N)
            if can_add(w, code):
                code.append(w)
                return True
        code.insert(idx, old_w)
        return False
        
    def kick(code, rng, k=3):
        if time.time() - start_time > time_limit:
            return
        removed = []
        for _ in range(min(k, len(code))):
            idx = rng.randrange(len(code))
            removed.append(code.pop(idx))
        for w in removed:
            if can_add(w, code):
                code.append(w)
        try_add_random(code, rng)
        
    # Phase 1: Local search to improve
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit:
            break
        move = rng.random()
        if move < 0.2:
            try_add_random(words, rng, attempts=100)
        elif move < 0.5:
            try_flip(words, rng)
        elif move < 0.8:
            try_swap(words, rng)
        else:
            kick(words, rng, k=rng.randint(2, 5))
            
        if len(words) > best_size:
            best_size = len(words)
            best_words = list(words)
            
    if len(best_words) >= len(words):
        return best_words
    return words
