import time

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before the 90s kill

    MAXV = 256
    MAX_SUM = 256 * 256
    MASK = (1 << (MAX_SUM + 1)) - 1

    best_seq = []
    best_len = 0

    # ---------- Greedy baseline ----------
    def greedy():
        seq = []
        sums = set()
        for x in range(1, MAXV + 1):
            tails = [x]
            total = x
            for a in reversed(seq):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                seq.append(x)
                sums.update(tails)
        return seq

    base = greedy()
    best_seq = base[:]
    best_len = len(base)

    # ---------- Precompute new-suffix-sums cache ----------
    # Key: (last_val, tuple(suffix_sums)) -> list of new suffix sums (ascending)
    # suffix_sums[i] = sum of seq[i:] (i from 0..n-1)
    _cache = {}

    def new_suffix_sums(last_val, suffix_sums):
        key = (last_val, suffix_sums)
        v = _cache.get(key)
        if v is None:
            new_sums = [last_val]
            total = last_val
            for s in suffix_sums:
                total += s
                new_sums.append(total)
            v = new_sums
            _cache[key] = v
        return v

    # ---------- DFS with in-place backtracking ----------
    def dfs(seq, sums_set, sums_bits, last_val):
        nonlocal best_seq, best_len

        if time.time() > deadline:
            return

        n = len(seq)
        if n > best_len:
            best_len = n
            best_seq = seq[:]

        # Pruning: max additional elements we could possibly add
        # (at most one per integer > last_val)
        if n + (MAXV - last_val) <= best_len:
            return

        # Maintain suffix sums incrementally: suffix_sums[i] = sum(seq[i:])
        # For the current seq, suffix_sums is derived from seq.
        # To avoid recomputing, we pass it implicitly via seq.
        # But for the cache key we need the tuple of suffix sums.
        # Compute suffix sums from seq (n is small, <= ~12)
        suffix_sums_list = [0] * n
        running = 0
        for i in range(n - 1, -1, -1):
            running += seq[i]
            suffix_sums_list[i] = running
        suffix_sums_tuple = tuple(suffix_sums_list)

        # Iterate candidate x
        for x in range(last_val + 1, MAXV + 1):
            if time.time() > deadline:
                return

            # Get new suffix sums for appending x
            new_sums = new_suffix_sums(x, suffix_sums_tuple)

            # Conflict check via bitset
            # Build bitset of new_sums
            new_bits = 0
            for s in new_sums:
                new_bits |= (1 << s)

            if (sums_bits & new_bits) != 0:
                continue

            # No conflict: add x
            sums_bits |= new_bits
            for s in new_sums:
                sums_set.add(s)
            seq.append(x)

            dfs(seq, sums_set, sums_bits, x)

            # Backtrack
            seq.pop()
            for s in new_sums:
                sums_set.discard(s)
            sums_bits &= ~new_bits

    # Start DFS
    dfs([], set(), 0, 0)

    return best_seq
