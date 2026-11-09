"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Build a q-ary code of length n with covering radius `radius`.
    Strategy:
      1. If a perfect code is known for these parameters, return it directly.
      2. Otherwise, use a vectorized greedy set cover over the whole space.
      3. Follow with a greedy deletion pass to remove redundant codewords.
      4. Enforce a wall-clock deadline; return the best valid cover found.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115 s to be safe within 120 s

    # ---------- 1. Perfect / known small cases ----------
    # Ternary Golay / tetracode: q=3, n=4, radius=1 -> 9 codewords (perfect)
    if q == 3 and n == 4 and radius == 1:
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    # General perfect Hamming codes: q-ary Hamming code exists when (q^n - 1) % (n + 1) == 0
    # and q is a prime power; the code has (q^n - 1) / (n + 1) codewords, radius 1.
    # We can construct it via the parity-check matrix.
    if radius == 1:
        hn = (q ** n - 1) // (n + 1)
        if (q ** n - 1) % (n + 1) == 0:
            # Try to build a q-ary Hamming code (only feasible for small n)
            if q ** n <= 200000 and n <= 10:
                try:
                    # Build all nonzero vectors in GF(q)^n / scalar -> parity check columns
                    # Standard construction: H has n rows, columns are all non-equivalent vectors
                    # For small cases this is doable.
                    cols = []
                    for vec in itertools.product(range(q), repeat=n):
                        if any(v != 0 for v in vec):
                            # Normalize: make first nonzero entry 1 (mod q)
                            for i in range(n):
                                if vec[i] != 0:
                                    if vec[i] != 1:
                                        inv = _modinv(vec[i], q)
                                        vec = tuple((v * inv) % q for v in vec)
                                    break
                            cols.append(vec)
                    # Deduplicate
                    cols = list(set(cols))
                    if len(cols) == (q ** n - 1) // (n + 1):
                        # Solve for codewords: all vectors x such that H @ x == 0 mod q
                        # This is linear algebra over Z_q; only works if q is prime
                        if _is_prime(q):
                            H = np.array(cols, dtype=int).T  # n x hn
                            # Null space over Z_q
                            codewords = _nullspace_zq(H, q)
                            if codewords is not None:
                                return [list(c) for c in codewords]
                except Exception:
                    pass

    # ---------- 2. Vectorized greedy set cover ----------
    total = q ** n
    if total > 2000000:
        # Too large for full-space approach; fall back to direct product construction
        return _fallback_product(q, n, radius)

    # Represent each word as an integer in [0, total)
    # Precompute: for each codeword candidate, the set of words it covers
    # Ball of radius r around word w: all words within Hamming distance r

    # Encode/decode
    def encode(word):
        """word: list of n ints -> int"""
        val = 0
        for i, v in enumerate(reversed(word)):
            val = val * q + v
        return val

    def decode(val):
        """int -> list of n ints"""
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word[::-1]

    # Precompute all words
    all_words = np.array([list(w) for w in itertools.product(range(q), repeat=n)], dtype=np.int8)
    all_word_ints = np.array([encode(list(w)) for w in all_words], dtype=np.int32)

    # Precompute coverage: for each candidate codeword index, which word indices it covers
    # This is expensive; instead, we'll use a different approach:
    # For each word in the space, precompute which codeword positions can cover it
    # Actually, let's just do greedy: at each step pick the codeword that covers the most uncovered words.

    # Precompute ball membership: balls[cw_idx, word_idx] = 1 if word is in ball of cw
    # But total can be up to ~1M, so a full boolean matrix is 1M * 1M = too big.
    # Instead: for each word, compute the set of codewords that cover it, or vice versa.

    # Alternative: represent coverage via integer arrays.
    # For each codeword position, generate the list of word indices it covers.
    # Number of words in a ball of radius r in q-ary n-space:
    ball_size = sum((q - 1) ** d * _comb(n, d) for d in range(radius + 1))

    if total > 500000:
        # Too large for precomputing all balls; use direct product fallback
        return _fallback_product(q, n, radius)

    # Precompute for each codeword index, the list of word indices in its ball
    # Use the word integer representation for hashing
    # balls[cw] = sorted list of word indices covered by codeword cw
    balls = []
    for cw in range(total):
        cw_word = decode(cw)
        covered = set()
        covered.add(cw)
        # Enumerate all words at distance 1..radius
        for d in range(1, radius + 1):
            for positions in itertools.combinations(range(n), d):
                for replacements in itertools.product(range(1, q), repeat=d):
                    new_word = list(cw_word)
                    for pos, rep in zip(positions, replacements):
                        new_word[pos] = rep
                    val = encode(new_word)
                    covered.add(val)
        balls.append(np.array(sorted(covered), dtype=np.int32))

    # Greedy set cover
    covered_mask = np.zeros(total, dtype=np.bool_)
    codewords = []

    while not covered_mask.all():
        if time.time() > deadline - 5.0:
            break  # Not enough time for another full pass
        best_cw = -1
        best_count = 0
        # Find codeword covering most uncovered words
        # This is O(total * avg_ball_size) per step which is too slow.
        # Instead, for each uncovered word, count how many candidate codewords cover it,
        # then pick the codeword with max count. But we need the reverse mapping.
        
        # Build reverse: for each word, which codewords cover it
        # This is also expensive. Let's just iterate.
        for cw in range(total):
            if covered_mask.all():
                break
            count = 0
            ball = balls[cw]
            # Count uncovered in this ball
            for idx in ball:
                if not covered_mask[idx]:
                    count += 1
            if count > best_count:
                best_count = count
                best_cw = cw
            if time.time() > deadline - 5.0:
                break
        
        if best_cw == -1 or best_count == 0:
            # Can't improve; add any uncovered word's position
            uncovered_indices = np.where(~covered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            best_cw = int(uncovered_indices[0])
        
        codewords.append(best_cw)
        # Mark covered
        for idx in balls[best_cw]:
            covered_mask[idx] = True

    # Convert to list of words
    result = [decode(cw) for cw in codewords]

    # ---------- 3. Greedy deletion ----------
    if time.time() < deadline - 10.0:
        # Try to remove redundant codewords
        # Rebuild coverage counts
        # For efficiency, work with the current codeword list
        n_cw = len(result)
        if n_cw > 1 and n_cw <= 5000:
            # Build word->codeword index mapping for current codewords
            # For each word, count how many codewords cover it
            cover_count = np.zeros(total, dtype=np.int32)
            cw_balls = []
            for i, cw_word in enumerate(result):
                cw_int = encode(cw_word)
                covered_set = set()
                covered_set.add(cw_int)
                for d in range(1, radius + 1):
                    for positions in itertools.combinations(range(n), d):
                        for replacements in itertools.product(range(1, q), repeat=d):
                            new_word = list(cw_word)
                            for pos, rep in zip(positions, replacements):
                                new_word[pos] = rep
                            val = encode(new_word)
                            covered_set.add(val)
                cw_balls.append(covered_set)
                for w in covered_set:
                    cover_count[w] += 1

            # Try removing each codeword
            changed = True
            while changed and time.time() < deadline - 5.0:
                changed = False
                for i in range(len(result)):
                    if cover_count is None:
                        break
                    # Check if any word would become uncovered
                    redundant = True
                    for w in cw_balls[i]:
                        if cover_count[w] <= 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove codeword i
                        for w in cw_balls[i]:
                            cover_count[w] -= 1
                        del cw_balls[i]
                        del result[i]
                        changed = True
                        break  # Restart scan after removal

    return result


def _fallback_product(q, n, radius):
    """Direct product construction as fallback."""
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # General fallback: brute force all words (only for very small spaces)
    if q ** n <= 10000:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    # Last resort: single codeword (invalid for radius < n, but valid for radius >= n)
    return [[0] * n]


def _comb(n, k):
    """Small combination."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _modinv(a, m):
    """Modular inverse via extended Euclidean."""
    if m == 1:
        return 0
    g, x, _ = _extended_gcd(a, m)
    if g != 1:
        raise ValueError("No modular inverse")
    return x % m


def _extended_gcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x, y = _extended_gcd(b, a % b)
    return (g, y, x - (a // b) * y)


def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def _nullspace_zq(H, q):
    """Find null space of H over Z_q. Only for small H."""
    # Gaussian elimination over Z_q
    H = H.copy()
    m, k = H.shape
    rows = []
    for col in range(k):
        # Find pivot
        pivot_row = None
        for r in range(len(rows), m):
            if H[r, col] % q != 0:
                pivot_row = r
                break
        if pivot_row is None:
            continue
        # Swap
        H[[len(rows), pivot_row]] = H[[pivot_row, len(rows)]]
        # Normalize pivot
        inv = _modinv(H[len(rows), col], q)
        H[len(rows)] = (H[len(rows)] * inv) % q
        # Eliminate
        for r in range(m):
            if r != len(rows) and H[r, col] % q != 0:
                factor = H[r, col]
                H[r] = (H[r] - factor * H[len(rows)]) % q
        rows.append(len(rows))
    
    # Now construct basis vectors for null space
    # Free variables correspond to columns without pivots
    pivot_cols = []
    for r in range(len(rows)):
        for c in range(k):
            if H[r, c] % q == 1 and all(H[rr, c] % q == 0 for rr in range(len(rows)) if rr != r):
                pivot_cols.append(c)
                break
    
    free_cols = [c for c in range(k) if c not in pivot_cols]
    basis = []
    for fc in free_cols:
        vec = np.zeros(k, dtype=int)
        vec[fc] = 1
        # Solve for pivot variables
        for r in range(len(rows)):
            pc = None
            for c in range(k):
                if H[r, c] % q == 1:
                    pc = c
                    break
            if pc is not None:
                # vec[pc] = -sum(H[r, j] * vec[j] for j != pc)
                val = 0
                for j in range(k):
                    if j != pc:
                        val = (val + H[r, j] * vec[j]) % q
                vec[pc] = (-val) % q
        basis.append(vec)
    
    if not basis:
        return [np.zeros(k, dtype=int)]
    return basis

# EVOLVE-BLOCK-END
