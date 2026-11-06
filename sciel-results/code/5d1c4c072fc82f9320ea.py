import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 17
    MIN_DIST = 6
    FULL = (1 << N) - 1

    # ---------- 1. Build the 256-word shortened [18,9,6] QR code ----------
    qres = set((x * x) % 19 for x in range(1, 19))

    G = np.zeros((9, 18), dtype=np.uint8)
    for i in range(9):
        for j in range(17):
            if ((j - i) % 19) in qres:
                G[i, j] = 1
        G[i, 17] = G[i, :17].sum() % 2

    # Shorten: keep only rows of G whose 18th (last) bit is 0 -> 8 info rows
    info_rows = [i for i in range(9) if G[i, 17] == 0]
    G17 = G[info_rows, :17]  # 8 x 17

    # Build 256 codewords
    code_set = set()
    for mask in range(256):
        v = np.zeros(17, dtype=np.uint8)
        for i in range(8):
            if (mask >> i) & 1:
                v ^= G17[i]
        val = 0
        for j in range(17):
            val = (val << 1) | int(v[j])
        code_set.add(val)

    code_list = list(code_set)
    code_arr = np.array(code_list, dtype=np.int64)

    # Verify min distance
    min_d = 999
    for i in range(len(code_list)):
        for j in range(i + 1, len(code_list)):
            d = (code_list[i] ^ code_list[j]).bit_count()
            if d < min_d:
                min_d = d

    # ---------- 2. Precompute popcount for 16-bit chunks ----------
    popcount_table = np.array([i.bit_count() for i in range(65536)], dtype=np.int8)

    def dist_to_code(word, code_arr):
        xors = code_arr ^ word
        low = (xors & 0xFFFF).astype(np.intp)
        high = (xors >> 16).astype(np.intp)
        return popcount_table[low].astype(np.intp) + high

    # ---------- 3. Precompute distance buckets for fast conflict checking ----------
    # For each existing codeword, precompute all words at distance <= 5
    # We use a "conflict set" approach: for each codeword, we know which
    # words conflict. But storing all is too large. Instead, we use the
    # distance array approach but optimized.

    # Better approach: maintain the code as a set, and for conflict checking
    # use the vectorized distance computation but with early exit.

    # Precompute all words at distance <= 5 from each codeword would be
    # 256 * sum(C(17,k) for k=0..5) = 256 * (1+17+136+680+2380+6188) = 256*9402 ≈ 2.4M
    # That's feasible! Let's precompute the "forbidden" set.

    forbidden = set()
    for cw in code_list:
        # Add all words at distance <= 5 from cw
        # Distance 0
        forbidden.add(cw)
        # Distance 1: 17 words
        for bit in range(N):
            forbidden.add(cw ^ (1 << bit))
        # Distance 2: C(17,2) = 136 words
        for b1 in range(N):
            for b2 in range(b1 + 1, N):
                forbidden.add(cw ^ (1 << b1) ^ (1 << b2))
        # Distance 3: C(17,3) = 680 words
        for b1 in range(N):
            for b2 in range(b1 + 1, N):
                for b3 in range(b2 + 1, N):
                    forbidden.add(cw ^ (1 << b1) ^ (1 << b2) ^ (1 << b3))
        # Distance 4: C(17,4) = 2380 words
        for b1 in range(N):
            for b2 in range(b1 + 1, N):
                for b3 in range(b2 + 1, N):
                    for b4 in range(b3 + 1, N):
                        forbidden.add(cw ^ (1 << b1) ^ (1 << b2) ^ (1 << b3) ^ (1 << b4))
        # Distance 5: C(17,5) = 6188 words
        for b1 in range(N):
            for b2 in range(b1 + 1, N):
                for b3 in range(b2 + 1, N):
                    for b4 in range(b3 + 1, N):
                        for b5 in range(b4 + 1, N):
                            forbidden.add(cw ^ (1 << b1) ^ (1 << b2) ^ (1 << b3) ^ (1 << b4) ^ (1 << b5))

    # Now forbidden contains all words that conflict with at least one codeword
    # A word can be added iff it's NOT in forbidden
    # But when we add a word, we need to update forbidden

    # This is too large to update incrementally. Let's use a different approach.
    # Instead, maintain the code as a list and use vectorized distance checks
    # but with numpy for speed.

    # Actually, let's use a hybrid: keep the forbidden set for fast rejection,
    # and when we add/remove, update it.

    # The forbidden set has ~2.4M entries. Adding a word adds ~9402 new forbidden
    # entries (some may already be there). Removing requires checking if other
    # codewords still forbid it, which is expensive.

    # Alternative: use the distance array approach but make it faster.
    # Let's benchmark: 256 codewords, vectorized distance check on 256 elements
    # takes ~10-20 microseconds. That's fast enough for SA.

    # Let's go with the vectorized approach but optimize the inner loop.

    def can_add(word, code_arr, code_set):
        if word in code_set:
            return False
        if len(code_arr) == 0:
            return True
        xors = code_arr ^ word
        low = (xors & 0xFFFF).astype(np.intp)
        high = (xors >> 16).astype(np.intp)
        dists = popcount_table[low].astype(np.intp) + high
        return np.all(dists >= MIN_DIST)

    def find_conflicts(word, code_arr):
        if len(code_arr) == 0:
            return np.array([], dtype=np.intp)
        xors = code_arr ^ word
        low = (xors & 0xFFFF).astype(np.intp)
        high = (xors >> 16).astype(np.intp)
        dists = popcount_table[low].astype(np.intp) + high
        return np.where(dists < MIN_DIST)[0]

    # ---------- 4. Simulated Annealing with fast rejection ----------
    rng = random.Random(42)
    all_words = np.arange(FULL + 1, dtype=np.int64)

    current_code = list(code_list)
    current_set = set(code_list)
    current_arr = np.array(current_code, dtype=np.int64)

    best_code = list(current_code)
    best_size = len(current_code)

    T0 = 5.0
    T_min = 0.001
    cooling = 0.9998

    T = T0
    no_improve = 0
    step = 0

    # Precompute single-bit and double-bit flip masks for speed
    single_flips = [1 << b for b in range(N)]
    double_flips = [(1 << b1) ^ (1 << b2) for b1 in range(N) for b2 in range(b1 + 1, N)]

    while time.time() < deadline:
        step += 1

        # Choose a move type
        # 70%: try to add a random word
        # 20%: try to swap (remove one, add another)
        # 10%: try to flip bits of an existing codeword

        move_type = rng.random()

        if move_type < 0.70:
            # Try to add a random word
            new_word = int(rng.choice(all_words))
            if new_word in current_set:
                no_improve += 1
                continue

            conf_idx = find_conflicts(new_word, current_arr)
            if len(conf_idx) == 0:
                # Can add directly
                current_code.append(new_word)
                current_set.add(new_word)
                current_arr = np.append(current_arr, new_word)
                no_improve = 0
                if len(current_code) > best_size:
                    best_code = list(current_code)
                    best_size = len(current_code)
            else:
                # Try to repair: move one conflicting codeword
                # Pick a random conflicting codeword and try to relocate it
                ci = int(conf_idx[rng.randint(0, len(conf_idx) - 1)])
                old_cw = current_code[ci]

                # Try moving old_cw to a new position
                # Generate candidates: flip 1 or 2 bits of old_cw
                candidates = []
                for flip in single_flips:
                    candidates.append(old_cw ^ flip)
                for flip in double_flips:
                    candidates.append(old_cw ^ flip)
                rng.shuffle(candidates)

                found = False
                for new_pos in candidates:
                    if new_pos == new_word:
                        continue
                    if new_pos in current_set:
                        continue
                    # Check if new_pos is valid for the rest of the code
                    # (excluding the old position)
                    # Build array without old_cw
                    temp_arr = np.delete(current_arr, ci)
                    if len(temp_arr) > 0:
                        xors = temp_arr ^ new_pos
                        low = (xors & 0xFFFF).astype(np.intp)
                        high = (xors >> 16).astype(np.intp)
                        dists = popcount_table[low].astype(np.intp) + high
                        if not np.all(dists >= MIN_DIST):
                            continue
                    # Check distance to new_word
                    if (new_pos ^ new_word).bit_count() < MIN_DIST:
                        continue
                    # Accept the move: replace old_cw with new_pos, then add new_word
                    current_code[ci] = new_pos
                    current_set.discard(old_cw)
                    current_set.add(new_pos)
                    current_code.append(new_word)
                    current_set.add(new_word)
                    current_arr = np.append(np.delete(current_arr, ci), new_word)
                    no_improve = 0
                    if len(current_code) > best_size:
                        best_code = list(current_code)
                        best_size = len(current_code)
                    found = True
                    break

                if not found:
                    no_improve += 1

        elif move_type < 0.90:
            # Swap: remove a random codeword, try to add a different one
            if len(current_code) == 0:
                continue
            idx = rng.randint(0, len(current_code) - 1)
            removed = current_code[idx]
            current_code.pop(idx)
            current_set.discard(removed)
            current_arr = np.delete(current_arr, idx)

            # Try to add a random word
            new_word = int(rng.choice(all_words))
            if new_word in current_set:
                no_improve += 1
                # Restore
                current_code.append(removed)
                current_set.add(removed)
                current_arr = np.append(current_arr, removed)
                continue

            if len(current_arr) > 0:
                xors = current_arr ^ new_word
                low = (xors & 0xFFFF).astype(np.intp)
                high = (xors >> 16).astype(np.intp)
                dists = popcount_table[low].astype(np.intp) + high
                if not np.all(dists >= MIN_DIST):
                    no_improve += 1
                    # Restore
                    current_code.append(removed)
                    current_set.add(removed)
                    current_arr = np.append(current_arr, removed)
                    continue
            else:
                pass

            current_code.append(new_word)
            current_set.add(new_word)
            current_arr = np.append(current_arr, new_word)
            no_improve = 0
            if len(current_code) > best_size:
                best_code = list(current_code)
                best_size = len(current_code)

        else:
            # Flip bits of an existing codeword
            if len(current_code) == 0:
                continue
            idx = rng.randint(0, len(current_code) - 1)
            old_cw = current_code[idx]
            n_flips = 1 if rng.random() < 0.5 else 2

            if n_flips == 1:
                flip = single_flips[rng.randint(0, N - 1)]
            else:
                flip = double_flips[rng.randint(0, len(double_flips) - 1)]

            new_cw = old_cw ^ flip
            if new_cw in current_set:
                no_improve += 1
                continue

            # Check validity against rest of code
            temp_arr = np.delete(current_arr, idx)
            if len(temp_arr) > 0:
                xors = temp_arr ^ new_cw
                low = (xors & 0xFFFF).astype(np.intp)
                high = (xors >> 16).astype(np.intp)
                dists = popcount_table[low].astype(np.intp) + high
                if not np.all(dists >= MIN_DIST):
                    no_improve += 1
                    continue

            current_code[idx] = new_cw
            current_set.discard(old_cw)
            current_set.add(new_cw)
            current_arr[idx] = new_cw
            no_improve = 0
            if len(current_code) > best_size:
                best_code = list(current_code)
                best_size = len(current_code)

        # Cooling
        T = max(T * cooling, T_min)

        # Restart if stuck
        if no_improve > 3000:
            current_code = list(best_code)
            current_set = set(best_code)
            current_arr = np.array(current_code, dtype=np.int64)
            T = T0
            no_improve = 0

    # ---------- 5. Final greedy polish ----------
    result = list(best_code)
    result_set = set(result)
    result_arr = np.array(result, dtype=np.int64)

    # Try to add words greedily in random order
    polish_rng = random.Random(123)
    polish_words = list(range(FULL + 1))
    polish_rng.shuffle(polish_words)

    for w in polish_words:
        if time.time() > deadline - 3:
            break
        if w in result_set:
            continue
        if len(result_arr) == 0:
            result.append(w)
            result_set.add(w)
            result_arr = np.append(result_arr, w)
            continue
        xors = result_arr ^ w
        low = (xors & 0xFFFF).astype(np.intp)
        high = (xors >> 16).astype(np.intp)
        dists = popcount_table[low].astype(np.intp) + high
        if np.all(dists >= MIN_DIST):
            result.append(w)
            result_set.add(w)
            result_arr = np.append(result_arr, w)

    # ---------- 6. Verify ----------
    assert len(set(result)) == len(result)
    assert all(0 <= w < (1 << N) for w in result)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            d = (result[i] ^ result[j]).bit_count()
            assert d >= MIN_DIST, f"Distance {d} < {MIN_DIST}"

    return result
