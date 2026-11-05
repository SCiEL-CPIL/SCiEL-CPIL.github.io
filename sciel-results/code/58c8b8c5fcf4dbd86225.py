import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin (90s hard limit)

    def is_valid(seq):
        """Check if all non-empty consecutive block sums are distinct."""
        if not seq:
            return True
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def compute_sums(seq):
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        s = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s.add(P[j] - P[i])
        return s

    def greedy_increasing():
        values = []
        sums = set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    def greedy_decreasing():
        values = []
        sums = set()
        for x in range(256, 0, -1):
            prefixes = [x]
            total = x
            for a in values:
                total += a
                prefixes.append(total)
            if not sums.intersection(prefixes):
                values.append(x)
                values.sort()
                sums.update(prefixes)
        return values

    # 1. Baselines from greedy
    best_seq = greedy_increasing()
    best_len = len(best_seq)

    dec_seq = greedy_decreasing()
    if len(dec_seq) > best_len:
        best_seq = dec_seq
        best_len = len(dec_seq)

    # 2. Randomized greedy with multiple orders (cheap, fast)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        values = []
        sums = set()
        for x in order:
            # x is not necessarily > values[-1]; find insertion position
            # New sums when inserting x at position p (0..len(values)):
            # blocks that include x and extend left/right within values
            # Simpler: just try to append (x must be > last) OR we do full check
            # For speed, only allow append if x > values[-1]
            if values and x <= values[-1]:
                continue
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        if len(values) > best_len:
            best_len = len(values)
            best_seq = list(values)

    # 3. Local search with position-aware insertion
    current_seq = list(best_seq)
    all_candidates = list(range(1, 257))

    # Helper: try inserting x at every valid position, return best (longest valid) new seq
    def try_insert_all_positions(seq, x):
        """Try inserting x at each position; return (new_seq, new_sums) for the first valid one, or None."""
        L = len(seq)
        # Precompute prefix sums of seq
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        existing_sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                existing_sums.add(P[j] - P[i])

        # Positions 0..L (insert before index pos, i.e., between pos-1 and pos)
        for pos in range(L + 1):
            # x must fit strictly: if pos>0 then seq[pos-1] < x; if pos<L then x < seq[pos]
            if pos > 0 and seq[pos - 1] >= x:
                continue
            if pos < L and x >= seq[pos]:
                continue
            # New sums involving x: for left extension l (0..pos) and right extension r (0..L-pos)
            # not both zero
            # sum = (P[pos] - P[pos-l]) + x + (P[pos+r] - P[pos])  where l in 0..pos, r in 0..L-pos
            new_sums = set()
            base_left = P[pos]
            for l in range(0, pos + 1):
                left_part = base_left - P[pos - l]
                for r in range(0, L - pos + 1):
                    if l == 0 and r == 0:
                        continue
                    right_part = P[pos + r] - P[pos]
                    new_sums.add(left_part + x + right_part)
            # Check conflict with existing sums
            if existing_sums.intersection(new_sums):
                continue
            # Also new_sums must be internally distinct (they are, by construction? verify)
            # They are distinct because (l,r) pairs give distinct sums? Not necessarily, but
            # since seq is strictly increasing and x is fixed, different (l,r) give different
            # left_part + right_part. left_part strictly decreases as l increases, right_part
            # strictly increases as r increases, so sums are distinct. OK.
            # Build new sequence
            new_seq = seq[:pos] + [x] + seq[pos:]
            new_sums = existing_sums | new_sums
            return new_seq, new_sums
        return None

    while time.time() < deadline:
        r = random.random()
        if r < 0.55:  # Add with position-aware insertion
            current_set = set(current_seq)
            available = [x for x in all_candidates if x not in current_set]
            if not available:
                continue
            # Bias towards values that are "near" the sequence (more likely to fit)
            # Pick randomly but also try a few candidates
            tried = 0
            max_tries = 5
            while tried < max_tries and available:
                x = random.choice(available)
                available.remove(x)
                tried += 1
                result = try_insert_all_positions(current_seq, x)
                if result is not None:
                    new_seq, new_sums = result
                    current_seq = new_seq
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = list(current_seq)
                    break
        elif r < 0.75:  # Remove
            if current_seq:
                x = random.choice(current_seq)
                current_seq.remove(x)
        else:  # Swap
            if current_seq:
                current_set = set(current_seq)
                available = [x for x in all_candidates if x not in current_set]
                if available:
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    # Remove x_rem
                    temp_seq = [x for x in current_seq if x != x_rem]
                    # Try inserting x_add at all positions
                    result = try_insert_all_positions(temp_seq, x_add)
                    if result is not None:
                        new_seq, new_sums = result
                        current_seq = new_seq
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)

    return sorted(best_seq)
