import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    if q != 3 or n != 6 or radius != 1:
        # Generic greedy fallback
        total = q ** n
        def w2i(w):
            idx = 0
            for v in w:
                idx = idx * q + v
            return idx
        def i2w(idx):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w

        masks = [0] * total
        for i in range(total):
            wi = i2w(i)
            m = 1 << i
            for pos in range(n):
                for val in range(q):
                    if val == wi[pos]:
                        continue
                    wn = wi[:]
                    wn[pos] = val
                    m |= 1 << w2i(wn)
            masks[i] = m

        uncovered = (1 << total) - 1
        code = []
        while uncovered:
            if time.time() > deadline:
                break
            best_i, best_c = -1, -1
            for i in range(total):
                c = (uncovered & masks[i]).bit_count()
                if c > best_c:
                    best_c = c
                    best_i = i
            if best_i < 0:
                break
            code.append(best_i)
            uncovered &= ~masks[best_i]
        return [i2w(i) for i in code]

    # Optimized path for q=3, n=6, radius=1
    total = 729

    def w2i(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    def i2w(idx):
        w = [0] * 6
        for i in range(5, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute neighbor masks
    masks = [0] * total
    for i in range(total):
        wi = i2w(i)
        m = 1 << i
        for pos in range(6):
            for val in range(3):
                if val == wi[pos]:
                    continue
                wn = wi[:]
                wn[pos] = val
                m |= 1 << w2i(wn)
        masks[i] = m

    def greedy_random(seed):
        """Run greedy set cover with random tie-breaking."""
        rng = random.Random(seed)
        uncovered = (1 << total) - 1
        code = []
        while uncovered:
            if time.time() > deadline:
                return None
            best_c = -1
            candidates = []
            for i in range(total):
                c = (uncovered & masks[i]).bit_count()
                if c > best_c:
                    best_c = c
                    candidates = [i]
                elif c == best_c and c > 0:
                    candidates.append(i)
            if not candidates:
                return None
            code.append(rng.choice(candidates))
            uncovered &= ~masks[code[-1]]
        return code

    def try_remove(code, rng):
        """Try to remove one codeword from the code while maintaining coverage."""
        if not code:
            return code
        code_set = set(code)
        # Build coverage map: for each word, which codewords cover it
        # We'll track for each word in the space, how many codewords cover it
        # and which ones
        cover_count = [0] * total
        for c in code:
            m = masks[c]
            # Count bits in m
            cover_count = [a + ((m >> i) & 1) for i, a in enumerate(cover_count)]
        # Actually that's slow. Let's use a different approach.
        # For each codeword c in code, check if removing it leaves all its neighbors still covered.
        # A codeword c can be removed if for every word w in mask(c), there exists another c' in code (c' != c) such that w in mask(c').
        # Equivalent: for every w in mask(c), (union of masks of other codewords) covers w.
        # Let's try random order.
        indices = list(range(len(code)))
        rng.shuffle(indices)
        for idx in indices:
            c = code[idx]
            # Check if all words in mask(c) are covered by other codewords
            other_mask = 0
            for j, cj in enumerate(code):
                if j != idx:
                    other_mask |= masks[cj]
            if (masks[c] & ~other_mask) == 0:
                # Can remove
                new_code = [c for j, c in enumerate(code) if j != idx]
                return new_code
        return code

    def local_search(code, rng, max_iter=50):
        """Local search: repeatedly try to remove a codeword."""
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            new_code = try_remove(code, rng)
            if len(new_code) < len(code):
                code = new_code
            else:
                break
        return code

    best_code = None
    best_size = float('inf')

    # Run multiple greedy restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        code = greedy_random(r * 12345 + 67890)
        if code is None:
            continue
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            # Try local search to reduce further
            improved = local_search(best_code, random.Random(r * 99999 + 1), max_iter=20)
            if len(improved) < best_size:
                best_size = len(improved)
                best_code = improved[:]
            if best_size <= 73:
                break

    if best_code is None:
        # Fallback: deterministic greedy
        code = greedy_random(0)
        if code is None:
            # Should not happen, but just in case
            return [i2w(i) for i in range(total)]
        best_code = code

    return [i2w(i) for i in best_code]
