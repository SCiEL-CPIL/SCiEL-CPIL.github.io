import time
import random
import numpy as np

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
            if values:
                rev = np.array(values[::-1])
                tails = np.concatenate(([x], x + np.cumsum(rev)))
            else:
                tails = np.array([x])
            if not sums.intersection(tails.tolist()):
                values.append(x)
                sums.update(tails.tolist())
        return values

    def greedy_decreasing():
        values = []
        sums = set()
        for x in range(256, 0, -1):
            if values:
                arr = np.array(values)
                prefixes = np.concatenate(([x], x + np.cumsum(arr)))
            else:
                prefixes = np.array([x])
            if not sums.intersection(prefixes.tolist()):
                values.append(x)
                values.sort()
                sums.update(prefixes.tolist())
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
            if values and x <= values[-1]:
                continue
            if values:
                rev = np.array(values[::-1])
                tails = np.concatenate(([x], x + np.cumsum(rev)))
            else:
                tails = np.array([x])
            if not sums.intersection(tails.tolist()):
                values.append(x)
                sums.update(tails.tolist())
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
        if L == 0:
            return [x], {x}
        P = np.array([0] + list(np.cumsum(seq)), dtype=np.int64)
        # Existing sums
        idx_i, idx_j = np.triu_indices(L + 1, k=1)
        existing_sums = set((P[idx_j] - P[idx_i]).tolist())

        # Positions 0..L
        for pos in range(L + 1):
            if pos > 0 and seq[pos - 1] >= x:
                continue
            if pos < L and x >= seq[pos]:
                continue
            # left_parts[l] = sum of seq[pos-l..pos-1] for l=0..pos  => P[pos]-P[pos-l]
            ls = np.arange(0, pos + 1)
            left_parts = P[pos] - P[pos - ls]  # shape (pos+1,)
            # right_parts[r] = sum of seq[pos..pos+r-1] for r=0..L-pos => P[pos+r]-P[pos]
            rs = np.arange(0, L - pos + 1)
            right_parts = P[pos + rs] - P[pos]  # shape (L-pos+1,)
            # new_sums = left_parts[:,None] + x + right_parts[None,:]
            new_sums_arr = left_parts[:, None] + x + right_parts[None, :]
            # Exclude (l=0, r=0) which is just x (but x itself is a valid block sum)
            # Actually (l=0,r=0) gives x which IS a valid consecutive block (just x alone)
            # So we keep it. All (l,r) with l>=0, r>=0, not both zero... wait.
            # l=0, r=0 means the block is just {x} itself, sum = x. That's valid.
            # l>0, r=0 means block extends left from x. Valid.
            # l=0, r>0 means block extends right from x. Valid.
            # l>0, r>0 means block extends both ways. Valid.
            # So all (l,r) pairs are valid consecutive blocks containing x.
            new_sums_list = new_sums_arr.flatten().tolist()
            new_sums_set = set(new_sums_list)
            # Check internal distinctness: since seq is strictly increasing and x fits,
            # different (l,r) give different sums. Verify:
            if len(new_sums_set) != len(new_sums_list):
                continue
            # Check conflict with existing sums
            if existing_sums.intersection(new_sums_set):
                continue
            new_seq = seq[:pos] + [x] + seq[pos:]
            return new_seq, existing_sums | new_sums_set
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

    # 4. Final polish: systematically try every (available_x, position) on best_seq
    if time.time() < deadline - 2:
        best_set = set(best_seq)
        available = [x for x in range(1, 257) if x not in best_set]
        for x in available:
            if time.time() > deadline - 1:
                break
            result = try_insert_all_positions(best_seq, x)
            if result is not None:
                new_seq, _ = result
                if len(new_seq) > best_len:
                    best_len = len(new_seq)
                    best_seq = new_seq
                    best_set = set(best_seq)
                    available = [a for a in range(1, 257) if a not in best_set]

    return sorted(best_seq)
