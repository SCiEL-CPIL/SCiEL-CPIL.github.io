import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Leave 5s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    # 1. Generate 2-3-5 smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
            
    N = len(smooth_numbers)
    
    # 2. Precompute divisibility matrix for antichain checks
    # div_mask[i] is a bitmask where bit j is set if smooth_numbers[i] divides smooth_numbers[j]
    div_mask = [0] * N
    for i in range(N):
        for j in range(N):
            if i != j and smooth_numbers[j] % smooth_numbers[i] == 0:
                div_mask[i] |= (1 << j)
    
    def is_antichain_mask(mask):
        """Check if the set of indices (represented by bitmask) forms an antichain."""
        n = mask.bit_count()
        if n <= 1:
            return True
        # Extract indices
        idxs = []
        m = mask
        while m:
            low = m & (-m)
            idxs.append(low.bit_length() - 1)
            m ^= low
        # Check all pairs
        for i in range(n):
            for j in range(i+1, n):
                a, b = idxs[i], idxs[j]
                if (div_mask[a] & (1 << b)) or (div_mask[b] & (1 << a)):
                    return False
        return True

    # 3. Pre-compute best representation for each target 1..128
    # answers[n] will store the list of terms for target n+1
    answers = [None] * 128
    
    # Level 1: Single term
    for s in smooth_numbers:
        if s <= 128:
            answers[s - 1] = [s]

    # Level 2: Two terms
    for i in range(N):
        a = smooth_numbers[i]
        for j in range(i + 1, N):
            b = smooth_numbers[j]
            s = a + b
            if s > 128:
                break
            if answers[s - 1] is None:
                # Check antichain: neither divides the other
                if b % a != 0 and a % b != 0:
                    answers[s - 1] = [a, b]

    # Check if all found
    if all(ans is not None for ans in answers):
        return answers
    
    if elapsed() > DEADLINE:
        return answers

    # Level 3: Three terms
    for i in range(N):
        a = smooth_numbers[i]
        for j in range(i + 1, N):
            b = smooth_numbers[j]
            if a + b > 128:
                continue
            # Check a,b antichain
            if b % a == 0 or a % b == 0:
                continue
            for k in range(j + 1, N):
                c = smooth_numbers[k]
                s = a + b + c
                if s > 128:
                    break
                if answers[s - 1] is None:
                    # Check c doesn't divide or get divided by a or b
                    if (c % a != 0 and a % c != 0) and (c % b != 0 and b % c != 0):
                        answers[s - 1] = [a, b, c]
    
    # Check if all found
    if all(ans is not None for ans in answers):
        return answers
    
    if elapsed() > DEADLINE:
        return answers

    # Level 4: Four terms - use bitmask approach for efficiency
    # Build mapping from value to index
    val_to_idx = {v: i for i, v in enumerate(smooth_numbers)}
    
    for i in range(N):
        a = smooth_numbers[i]
        for j in range(i + 1, N):
            b = smooth_numbers[j]
            if a + b > 128:
                break
            # Check a,b antichain
            if b % a == 0 or a % b == 0:
                continue
            for k in range(j + 1, N):
                c = smooth_numbers[k]
                s3 = a + b + c
                if s3 > 128:
                    break
                # Check a,b,c antichain
                if (c % a == 0 or a % c == 0) or (c % b == 0 or b % c == 0):
                    continue
                # For the 4th term, we need d such that a+b+c+d <= 128
                # d must be > c (since we iterate in order) and d must not be divisible by or divide a,b,c
                # Instead of iterating all d, compute remaining and look up
                for l in range(k + 1, N):
                    d = smooth_numbers[l]
                    s = s3 + d
                    if s > 128:
                        break
                    if answers[s - 1] is None:
                        # Check d against a,b,c
                        if (d % a == 0 or a % d == 0) or (d % b == 0 or b % d == 0) or (d % c == 0 or c % d == 0):
                            continue
                        answers[s - 1] = [a, b, c, d]
    
    # Check if all found
    if all(ans is not None for ans in answers):
        return answers
    
    if elapsed() > DEADLINE:
        return answers

    # Level 5: Five terms - only for missing targets
    missing = [i for i in range(128) if answers[i] is None]
    if not missing:
        return answers
    
    for i in range(N):
        a = smooth_numbers[i]
        for j in range(i + 1, N):
            b = smooth_numbers[j]
            if a + b > 128:
                break
            if b % a == 0 or a % b == 0:
                continue
            for k in range(j + 1, N):
                c = smooth_numbers[k]
                s3 = a + b + c
                if s3 > 128:
                    break
                if (c % a == 0 or a % c == 0) or (c % b == 0 or b % c == 0):
                    continue
                for l in range(k + 1, N):
                    d = smooth_numbers[l]
                    s4 = s3 + d
                    if s4 > 128:
                        break
                    if (d % a == 0 or a % d == 0) or (d % b == 0 or b % d == 0) or (d % c == 0 or c % d == 0):
                        continue
                    for m in range(l + 1, N):
                        e = smooth_numbers[m]
                        s = s4 + e
                        if s > 128:
                            break
                        if answers[s - 1] is None:
                            if (e % a == 0 or a % e == 0) or (e % b == 0 or b % e == 0) or (e % c == 0 or c % e == 0) or (e % d == 0 or d % e == 0):
                                continue
                            answers[s - 1] = [a, b, c, d, e]
    
    # Check if all found
    if all(ans is not None for ans in answers):
        return answers
    
    if elapsed() > DEADLINE:
        return answers

    # Level 6: Six terms - only for remaining missing targets
    missing = [i for i in range(128) if answers[i] is None]
    if not missing:
        return answers
    
    for i in range(N):
        a = smooth_numbers[i]
        for j in range(i + 1, N):
            b = smooth_numbers[j]
            if a + b > 128:
                break
            if b % a == 0 or a % b == 0:
                continue
            for k in range(j + 1, N):
                c = smooth_numbers[k]
                s3 = a + b + c
                if s3 > 128:
                    break
                if (c % a == 0 or a % c == 0) or (c % b == 0 or b % c == 0):
                    continue
                for l in range(k + 1, N):
                    d = smooth_numbers[l]
                    s4 = s3 + d
                    if s4 > 128:
                        break
                    if (d % a == 0 or a % d == 0) or (d % b == 0 or b % d == 0) or (d % c == 0 or c % d == 0):
                        continue
                    for m in range(l + 1, N):
                        e = smooth_numbers[m]
                        s5 = s4 + e
                        if s5 > 128:
                            break
                        if (e % a == 0 or a % e == 0) or (e % b == 0 or b % e == 0) or (e % c == 0 or c % e == 0) or (e % d == 0 or d % e == 0):
                            continue
                        for n_idx in range(m + 1, N):
                            f = smooth_numbers[n_idx]
                            s = s5 + f
                            if s > 128:
                                break
                            if answers[s - 1] is None:
                                if (f % a == 0 or a % f == 0) or (f % b == 0 or b % f == 0) or (f % c == 0 or c % f == 0) or (f % d == 0 or d % f == 0) or (f % e == 0 or e % f == 0):
                                    continue
                                answers[s - 1] = [a, b, c, d, e, f]
    
    # Check if all found
    if all(ans is not None for ans in answers):
        return answers
    
    if elapsed() > DEADLINE:
        return answers

    # Fallback: try higher levels with itertools for remaining
    missing = [i for i in range(128) if answers[i] is None]
    for target_idx in missing:
        target = target_idx + 1
        for r in range(7, 31):
            if elapsed() > DEADLINE:
                break
            for combo in itertools.combinations(smooth_numbers, r):
                if sum(combo) == target and is_antichain_mask(0):
                    # Need proper mask check
                    idxs = [val_to_idx[v] for v in combo]
                    mask = 0
                    for idx in idxs:
                        mask |= (1 << idx)
                    if is_antichain_mask(mask):
                        answers[target_idx] = list(combo)
                        break
            if answers[target_idx] is not None:
                break

    return answers
