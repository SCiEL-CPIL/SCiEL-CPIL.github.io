import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s kill

    best_seq = []
    best_len = 0

    def update_forbidden(prefix_sums, forbidden_sums, x):
        """Add x to sequence; return new prefix_sums and forbidden_sums."""
        new_prefix = prefix_sums[-1] + x
        new_prefix_sums = prefix_sums + [new_prefix]
        new_forbidden = forbidden_sums.copy()
        for p in prefix_sums:
            new_forbidden.add(new_prefix - p)
        return new_prefix_sums, new_forbidden

    def is_valid_candidate(x, prefix_sums, forbidden_sums):
        """O(n) check: can x be appended to sequence with given prefix_sums?"""
        new_prefix = prefix_sums[-1] + x
        for p in prefix_sums:
            d = new_prefix - p
            if d in forbidden_sums:
                return False
        return True

    def compute_all_sums(seq):
        """Compute all consecutive block sums for a sequence."""
        n = len(seq)
        sums = set()
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + seq[i]
        for i in range(n):
            for j in range(i + 1, n + 1):
                sums.add(prefix[j] - prefix[i])
        return sums

    def greedy_construct(seed=None, prob_smallest=0.8):
        if seed is not None:
            random.seed(seed)

        seq = []
        prefix_sums = [0]
        forbidden_sums = set()
        last = 0

        for _ in range(256):
            valid_candidates = []
            for x in range(last + 1, 257):
                if is_valid_candidate(x, prefix_sums, forbidden_sums):
                    valid_candidates.append(x)

            if not valid_candidates:
                break

            if random.random() < prob_smallest:
                x = valid_candidates[0]
            else:
                x = random.choice(valid_candidates)

            seq.append(x)
            last = x
            prefix_sums, forbidden_sums = update_forbidden(prefix_sums, forbidden_sums, x)

        return seq

    def local_search_insertion(seq, seed=None, max_trials=200):
        """
        Try to insert unused elements into the sequence to increase length.
        Uses efficient prefix-sum based validity checking.
        """
        if seed is not None:
            random.seed(seed)

        current = seq[:]
        current_sums = compute_all_sums(current)
        used = set(current)
        unused = [x for x in range(1, 257) if x not in used]
        random.shuffle(unused)

        improved = True
        trial = 0
        while improved and trial < max_trials:
            improved = False
            trial += 1

            # Try each unused element at each position
            for x in unused:
                n = len(current)
                for pos in range(n + 1):
                    # Check if inserting x at pos keeps all block sums distinct
                    # New blocks: those that include x
                    # A block including x starts at some i <= pos and ends at some j >= pos
                    # (in the new sequence, x is at index pos)
                    # In the old sequence, prefix before pos is current[0..pos-1], after is current[pos..n-1]

                    valid = True
                    # We need to check all new block sums involving x
                    # Block from i to pos (in new seq): current[i..pos-1] + x  (if i < pos)
                    # Block from pos to j (in new seq): x + current[pos..j-1]  (if j > pos)
                    # Block from i to j (in new seq) where i < pos < j: current[i..pos-1] + x + current[pos..j-1]

                    # Precompute prefix sums of current
                    prefix = [0] * (n + 1)
                    for i in range(n):
                        prefix[i + 1] = prefix[i] + current[i]

                    # Check blocks ending at x (start at some i in [0, pos-1], or just x)
                    # Block [i..pos-1] + x: sum = (prefix[pos] - prefix[i]) + x
                    for i in range(pos):
                        s = (prefix[pos] - prefix[i]) + x
                        if s in current_sums:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Block [x] alone
                    if x in current_sums:
                        valid = False
                        continue

                    # Check blocks starting at x (end at some j in [pos, n-1], or just x)
                    # Block x + [pos..j-1]: sum = x + (prefix[j] - prefix[pos])
                    for j in range(pos, n):
                        s = x + (prefix[j] - prefix[pos])
                        if s in current_sums:
                            valid = False
                            break
                    if not valid:
                        continue

                    # Check blocks spanning x: start at i < pos, end at j >= pos
                    # Block [i..pos-1] + x + [pos..j-1]: sum = (prefix[pos] - prefix[i]) + x + (prefix[j] - prefix[pos])
                    for i in range(pos):
                        base = (prefix[pos] - prefix[i]) + x
                        for j in range(pos, n):
                            s = base + (prefix[j] - prefix[pos])
                            if s in current_sums:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue

                    # Also need to check that new block sums don't conflict with each other
                    # Collect all new sums
                    new_sums = set()
                    for i in range(pos):
                        new_sums.add((prefix[pos] - prefix[i]) + x)
                    new_sums.add(x)
                    for j in range(pos, n):
                        new_sums.add(x + (prefix[j] - prefix[pos]))
                    for i in range(pos):
                        base = (prefix[pos] - prefix[i]) + x
                        for j in range(pos, n):
                            new_sums.add(base + (prefix[j] - prefix[pos]))

                    # Check for duplicates among new sums
                    if len(new_sums) != sum(1 for _ in new_sums):  # This is always true, need different check
                        pass
                    # Actually, let's just check: number of new blocks vs unique sums
                    count_new = 0
                    for i in range(pos):
                        count_new += 1
                    count_new += 1  # x alone
                    for j in range(pos, n):
                        count_new += 1
                    for i in range(pos):
                        for j in range(pos, n):
                            count_new += 1

                    if len(new_sums) != count_new:
                        valid = False
                        continue

                    # All valid! Insert x
                    current = current[:pos] + [x] + current[pos:]
                    current_sums = compute_all_sums(current)
                    used.add(x)
                    unused.remove(x)
                    improved = True
                    break
                if improved:
                    break

        return current

    def local_search_swap(seq, seed=None, max_trials=100):
        """
        Try swapping elements to enable further insertions.
        This is a heuristic: swap two elements, check validity, keep if valid.
        """
        if seed is not None:
            random.seed(seed)

        current = seq[:]
        current_sums = compute_all_sums(current)
        n = len(current)

        improved = True
        trial = 0
        while improved and trial < max_trials:
            improved = False
            trial += 1

            for i in range(n):
                for j in range(i + 1, n):
                    # Try swapping current[i] and current[j]
                    # But sequence must remain strictly increasing, so swap only if
                    # the result is still increasing
                    if current[i] > current[j]:
                        continue

                    new_seq = current[:]
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

                    # Check if still strictly increasing
                    if any(new_seq[k] >= new_seq[k + 1] for k in range(n - 1)):
                        continue

                    # Check if all block sums are distinct
                    new_sums = compute_all_sums(new_seq)
                    if len(new_sums) == n * (n + 1) // 2:
                        # Valid swap! Keep it
                        current = new_seq
                        current_sums = new_sums
                        improved = True
                        break
                if improved:
                    break

        return current

    def beam_search(width, seed=None):
        """Beam search with efficient candidate checking."""
        if seed is not None:
            random.seed(seed)

        beams = [([], [0], set(), 0)]

        max_depth = 30
        for depth in range(max_depth):
            if time.time() > time_limit - 10:
                break

            next_beams = []
            for beam in beams:
                seq, prefix_sums, forbidden_sums, last = beam

                valid_candidates = []
                for x in range(last + 1, 257):
                    if is_valid_candidate(x, prefix_sums, forbidden_sums):
                        valid_candidates.append(x)

                if not valid_candidates:
                    continue

                if len(valid_candidates) > width:
                    k_smallest = width // 2
                    k_random = width - k_smallest
                    smallest = valid_candidates[:k_smallest]
                    rest = valid_candidates[k_smallest:]
                    if len(rest) >= k_random:
                        others = random.sample(rest, k_random)
                    else:
                        others = rest
                    candidates_to_try = smallest + others
                else:
                    candidates_to_try = valid_candidates

                for x in candidates_to_try:
                    new_seq = seq + [x]
                    new_prefix_sums, new_forbidden = update_forbidden(prefix_sums, forbidden_sums, x)
                    next_beams.append((new_seq, new_prefix_sums, new_forbidden, x))

            if not next_beams:
                break

            # Prune: prefer smaller last value (more room) and shorter prefix sums
            next_beams.sort(key=lambda b: (b[3], b[1][-1]))
            beams = next_beams[:width]

        if not beams:
            return []
        best_beam = max(beams, key=lambda b: len(b[0]))
        return best_beam[0]

    def polish(seq, seed=None):
        """Final high-precision polish: try all insertions in order."""
        current = seq[:]
        current_sums = compute_all_sums(current)
        used = set(current)
        unused = sorted([x for x in range(1, 257) if x not in used])

        improved = True
        while improved:
            improved = False
            for x in unused[:]:
                n = len(current)
                prefix = [0] * (n + 1)
                for i in range(n):
                    prefix[i + 1] = prefix[i] + current[i]

                for pos in range(n + 1):
                    valid = True
                    new_sums = set()
                    count_new = 0

                    # Blocks ending at x
                    for i in range(pos):
                        s = (prefix[pos] - prefix[i]) + x
                        if s in current_sums:
                            valid = False
                            break
                        new_sums.add(s)
                        count_new += 1
                    if not valid:
                        continue

                    # x alone
                    if x in current_sums:
                        valid = False
                        continue
                    new_sums.add(x)
                    count_new += 1

                    # Blocks starting at x
                    for j in range(pos, n):
                        s = x + (prefix[j] - prefix[pos])
                        if s in current_sums:
                            valid = False
                            break
                        new_sums.add(s)
                        count_new += 1
                    if not valid:
                        continue

                    # Blocks spanning x
                    for i in range(pos):
                        base = (prefix[pos] - prefix[i]) + x
                        for j in range(pos, n):
                            s = base + (prefix[j] - prefix[pos])
                            if s in current_sums:
                                valid = False
                                break
                            new_sums.add(s)
                            count_new += 1
                        if not valid:
                            break
                    if not valid:
                        continue

                    if len(new_sums) != count_new:
                        continue

                    # Valid insertion
                    current = current[:pos] + [x] + current[pos:]
                    current_sums = compute_all_sums(current)
                    used.add(x)
                    unused.remove(x)
                    improved = True
                    break
                if improved:
                    break

        return current

    # Phase 1: Multiple greedy restarts
    attempt_count = 0
    while time.time() - start_time < time_limit - 30:
        p_smallest = random.uniform(0.5, 1.0)
        seq = greedy_construct(seed=attempt_count, prob_smallest=p_smallest)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq

        if attempt_count % 3 == 0:
            seq2 = greedy_construct(seed=attempt_count + 1000, prob_smallest=0.2)
            if len(seq2) > best_len:
                best_len = len(seq2)
                best_seq = seq2

        attempt_count += 1

    # Phase 2: Local search insertion on best sequence
    if time.time() - start_time < time_limit - 20:
        improved_seq = local_search_insertion(best_seq, seed=42)
        if len(improved_seq) > best_len:
            best_len = len(improved_seq)
            best_seq = improved_seq

        # Try again with different seed
        improved_seq2 = local_search_insertion(best_seq, seed=123)
        if len(improved_seq2) > best_len:
            best_len = len(improved_seq2)
            best_seq = improved_seq2

    # Phase 3: Beam search
    for width in [15, 30, 50]:
        if time.time() - start_time > time_limit - 10:
            break
        beam_seq = beam_search(width, seed=attempt_count + width)
        if len(beam_seq) > best_len:
            best_len = len(beam_seq)
            best_seq = beam_seq

    # Phase 4: Polish the best sequence
    if time.time() - start_time < time_limit - 5:
        polished = polish(best_seq)
        if len(polished) > best_len:
            best_len = len(polished)
            best_seq = polished

    return best_seq
