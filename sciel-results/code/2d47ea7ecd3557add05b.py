import numpy as np
import time

def construct():
    N = 18
    MIN_DIST = 6
    size = 1 << N
    deadline = time.time() + 55  # safety margin before 60s limit

    # Precompute all masks of weight < MIN_DIST (Hamming ball of radius 5)
    # sum_{i=0}^{5} C(18,i) = 1+18+153+816+3060+8568 = 12616
    masks = np.array([i for i in range(size) if i.bit_count() < MIN_DIST], dtype=np.uint32)

    def mark_forbidden(forbidden, c):
        forbidden[(c ^ masks) % size] = True

    def greedy_fill(forbidden, order, max_add=None):
        """Add codewords from 'order' that aren't forbidden. Returns list added."""
        added = []
        if max_add is None:
            max_add = len(order)
        for c in order:
            if len(added) >= max_add:
                break
            if not forbidden[c]:
                added.append(c)
                mark_forbidden(forbidden, c)
        return added

    # Strategy 1: Extended QR Code Construction (deterministic, instant)
    try:
        # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
        # The QR code of length 17 uses the indicator of Q = {1,2,4,8,9,13,15,16}
        # Extended QR code: append parity bit to get [18, 9, 6] code with 512 words
        Q = [1, 2, 4, 8, 9, 13, 15, 16]
        v = np.zeros(17, dtype=np.uint8)
        v[Q] = 1

        # Generator matrix: 9 rows x 17 cols from cyclic shifts of v
        # Row i is v rolled by i positions (i=0..8)
        G = np.zeros((9, 17), dtype=np.uint8)
        for i in range(9):
            G[i] = np.roll(v, i)

        # Build codewords: for each 9-bit message, compute sum of selected rows mod 2
        # Then extend with overall parity
        row_ints = []
        for i in range(9):
            val = 0
            for k in range(17):
                if G[i, k]:
                    val |= (1 << k)
            row_ints.append(val)

        codes = []
        for m in range(512):
            c = 0
            for j in range(9):
                if (m >> j) & 1:
                    c ^= row_ints[j]
            # Extend with parity bit at position 17
            parity = c.bit_count() % 2
            c_ext = c | (parity << 17)
            codes.append(c_ext)

        # Verify: check minimum distance
        min_d = N + 1
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                d = (codes[i] ^ codes[j]).bit_count()
                if d < min_d:
                    min_d = d
                if min_d < MIN_DIST:
                    break
            if min_d < MIN_DIST:
                break

        if min_d >= MIN_DIST:
            return codes
    except Exception:
        pass

    # Strategy 2: Fast greedy with weight-stratified scan
    try:
        forbidden = np.zeros(size, dtype=bool)
        codes = [0]
        mark_forbidden(forbidden, 0)

        # Weight-stratified scan: process words by increasing weight
        # This systematically fills the space without random collision issues
        # Group words by weight for deterministic, structured exploration
        words_by_weight = [[] for _ in range(N + 1)]
        for w in range(1, size):
            wt = w.bit_count()
            words_by_weight[wt].append(w)

        # Process in order of increasing weight, then within same weight by value
        for wt in range(1, N + 1):
            if time.time() > deadline - 5:
                break
            for w in words_by_weight[wt]:
                if time.time() > deadline - 5:
                    break
                if not forbidden[w]:
                    codes.append(w)
                    mark_forbidden(forbidden, w)
    except Exception:
        pass

    # Strategy 3: Local improvement - try removing one codeword and repacking
    try:
        if time.time() < deadline - 10 and len(codes) > 0:
            current_len = len(codes)
            improved = False

            # Try each codeword removal (limited tries for time)
            max_removal_tries = 200
            for trial in range(min(max_removal_tries, len(codes))):
                if time.time() > deadline - 10:
                    break

                idx = trial % len(codes)
                removed = codes[idx]
                remaining = codes[:idx] + codes[idx + 1:]

                # Rebuild forbidden from remaining
                new_forbidden = np.zeros(size, dtype=bool)
                for c in remaining:
                    mark_forbidden(new_forbidden, c)

                # Try to add as many new codewords as possible
                # Use weight-stratified scan for the fill
                added = 0
                for wt in range(0, N + 1):
                    if time.time() > deadline - 10:
                        break
                    for w in words_by_weight[wt]:
                        if time.time() > deadline - 10:
                            break
                        if not new_forbidden[w]:
                            remaining.append(w)
                            mark_forbidden(new_forbidden, w)
                            added += 1

                if len(remaining) > current_len:
                    codes = remaining
                    current_len = len(codes)
                    improved = True
                    # Rebuild forbidden for next iteration
                    forbidden = np.zeros(size, dtype=bool)
                    for c in codes:
                        mark_forbidden(forbidden, c)

                if not improved and trial > 50:
                    break
    except Exception:
        pass

    # Strategy 4: Additional random-order greedy passes with remaining time
    try:
        if time.time() < deadline - 3 and len(codes) > 0:
            import random
            rng = random.Random(12345)
            current_len = len(codes)

            # Rebuild forbidden
            forbidden = np.zeros(size, dtype=bool)
            for c in codes:
                mark_forbidden(forbidden, c)

            # Try a few random-order fills
            for attempt in range(5):
                if time.time() > deadline - 3:
                    break
                order = list(range(size))
                rng.shuffle(order)
                added = greedy_fill(forbidden, order, max_add=1000)
                if added:
                    codes.extend(added)
                    current_len = len(codes)
    except Exception:
        pass

    # Final validation: spot-check
    if len(codes) > 1:
        import random
        rng = random.Random(999)
        sample_size = min(100, len(codes))
        sample = rng.sample(codes, sample_size)
        valid = True
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                if (sample[i] ^ sample[j]).bit_count() < MIN_DIST:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            return [0]

    return codes if codes else [0]
