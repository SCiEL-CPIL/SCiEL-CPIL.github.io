# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _ball_indices(q, n, radius, center):
    """Return flat indices of all words within Hamming distance <= radius of center."""
    # Precompute single-step offsets per position
    # For each position, list of (pos, delta) pairs for each symbol change
    offsets_per_pos = []
    for pos in range(n):
        base = q ** (n - 1 - pos)
        for d in range(-1, 2):
            if d == 0:
                continue
            offsets_per_pos.append((pos, d, base))
    # Generate combinations of up to `radius` offsets (distinct positions)
    # We'll enumerate subsets of positions and symbol changes
    result = {0}
    # For radius 1, this is just center + single changes
    if radius == 1:
        idx = 0
        for pos in range(n):
            base = q ** (n - 1 - pos)
            val = center[pos]
            for new_val in range(q):
                if new_val == val:
                    continue
                idx = 0
                for p in range(n):
                    v = center[p] if p != pos else new_val
                    idx = idx * q + v
                result.add(idx)
        return sorted(result)
    # General case: enumerate all words within distance radius
    # For small n and radius, brute force over combinations
    # Use recursive generation
    def gen(pos, remaining, current_idx, current_dist):
        if remaining < 0:
            return
        if pos == n:
            result.add(current_idx)
            return
        # Keep same symbol
        gen(pos + 1, remaining, current_idx * q + center[pos], current_dist)
        # Change symbol
        base = q ** (n - 1 - pos)
        for new_val in range(q):
            if new_val == center[pos]:
                continue
            # new index: replace pos
            # current_idx so far is for positions 0..pos-1
            gen(pos + 1, remaining - 1, current_idx * q + new_val, current_dist + 1)

    gen(0, radius, 0, 0)
    return sorted(result)


def _build_cover(q, n, radius):
    """Build an initial cover using tetracode direct sum."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    # For the specific case q=3, n=7, radius=1, use the tetracode construction
    if q == 3 and n == 7 and radius == 1:
        code = _build_cover(3, 7, 1)
    else:
        # Fallback: greedy construction
        code = []
        total = q ** n
        # coverage array
        cov = np.zeros(total, dtype=np.int32)
        # Precompute balls lazily
        ball_cache = {}
        while True:
            if time.time() > deadline - 5:
                break
            # Find uncovered word
            uncovered = np.where(cov == 0)[0]
            if len(uncovered) == 0:
                break
            # Greedy: pick codeword covering most uncovered words
            # For efficiency, sample or use first uncovered
            # Better: for each candidate, count uncovered in ball
            # This is expensive; use heuristic: pick uncovered word as center
            center = int(uncovered[0])
            # Convert to list
            center_list = []
            tmp = center
            for pos in range(n - 1, -1, -1):
                center_list.insert(0, tmp % q)
                tmp //= q
            if center not in ball_cache:
                ball_cache[center] = np.array(_ball_indices(q, n, radius, center_list), dtype=np.int64)
            ball = ball_cache[center]
            cov[ball] += 1
            code.append(center_list)

    # Now try to reduce the code size
    total = q ** n
    # Coverage count array
    cov = np.zeros(total, dtype=np.int32)
    # Precompute balls for all codewords
    balls = []
    for cw in code:
        balls.append(np.array(_ball_indices(q, n, radius, cw), dtype=np.int64))
    for b in balls:
        cov[b] += 1

    # Verify coverage
    assert np.all(cov >= 1), "Initial cover is incomplete"

    # Try to remove redundant codewords
    # A codeword is redundant if all words in its ball have cov >= 2
    # (i.e., removing it still leaves all words covered)
    removed = [False] * len(code)
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(code)):
            if removed[i]:
                continue
            # Check if all words in ball[i] have cov >= 2
            if np.all(cov[balls[i]] >= 2):
                # Remove it
                cov[balls[i]] -= 1
                removed[i] = True
                changed = True

    # Build reduced code
    code = [code[i] for i in range(len(code)) if not removed[i]]
    balls = [balls[i] for i in range(len(code)) if not removed[i]]

    # Rebuild coverage
    cov = np.zeros(total, dtype=np.int32)
    for b in balls:
        cov[b] += 1

    # Try local swaps: replace a codeword with a better one
    # For each codeword, try to find a replacement that covers at least as many
    # unique words and allows removal of another codeword
    # Simplified: try to find a codeword that can be removed after adding a new one
    # This is complex; do a simple greedy improvement:
    # Try to remove codewords one by one, and if coverage breaks, add back the best replacement

    # For now, return the reduced code
    return code
# EVOLVE-BLOCK-END
