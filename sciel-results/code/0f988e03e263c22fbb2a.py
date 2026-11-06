import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0
    MAX_VAL = 256

    def get_consecutive_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def is_valid(seq):
        if not seq:
            return True
        if seq[0] < 1 or seq[-1] > MAX_VAL:
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        sums = get_consecutive_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    # ---- STRUCTURAL IDEA: reverse-order (descending) greedy ----
    # Appending values from high to low yields a strictly-increasing sequence.
    # This explores a fundamentally different search space than ascending greedy.
    def greedy_desc(start=MAX_VAL, step=1):
        vals = []
        sums = set()
        for x in range(start, 0, -step):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums.intersection(new_sums):
                vals.append(x)
                sums.update(new_sums)
        vals.sort()
        return vals

    def greedy_asc(start=1, step=1):
        vals = []
        sums = set()
        for x in range(start, MAX_VAL + 1, step):
            new_sums = set()
            total = x
            new_sums.add(total)
            for a in reversed(vals):
                total += a
                new_sums.add(total)
            if not sums.intersection(new_sums):
                vals.append(x)
                sums.update(new_sums)
        return vals

    # ---- REFINEMENT MACHINERY: insertion-extension ----
    def try_extend(seq, time_limit):
        current = list(seq)
        used = set(current)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for pos in range(len(current) + 1):
                if pos == 0:
                    lo, hi = 1, (current[0] - 1 if current else MAX_VAL)
                elif pos == len(current):
                    lo, hi = current[-1] + 1, MAX_VAL
                else:
                    lo, hi = current[pos - 1] + 1, current[pos] - 1
                if lo > hi:
                    continue
                # Try larger values first (more headroom)
                for x in range(hi, lo - 1, -1):
                    if x in used:
                        continue
                    new_seq = current[:pos] + [x] + current[pos:]
                    new_sums = get_consecutive_sums(new_seq)
                    if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                        current = new_seq
                        used.add(x)
                        improved = True
                        break
                if improved:
                    break
        return current

    # ---- REFINEMENT MACHINERY: local search (remove + reinsert + extend) ----
    def local_search(seq, time_limit, rng):
        current = list(seq)
        current_len = len(current)
        for _ in range(500):
            if time.time() > time_limit:
                break
            if len(current) < 2:
                break
            i = rng.randint(0, len(current) - 1)
            val = current.pop(i)
            inserted = False
            for pos in range(len(current) + 1):
                if pos == i:
                    continue
                if pos == 0:
                    if current and current[0] <= val:
                        continue
                elif pos == len(current):
                    if current and current[-1] >= val:
                        continue
                else:
                    if current[pos - 1] >= val or current[pos] <= val:
                        continue
                new_seq = current[:pos] + [val] + current[pos:]
                new_sums = get_consecutive_sums(new_seq)
                if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                    current = new_seq
                    inserted = True
                    break
            if not inserted:
                current.insert(i, val)
            extended = try_extend(current, time_limit)
            if len(extended) > current_len:
                return extended
        return current

    # ---- Build diverse candidate pool ----
    candidates = []
    candidates.append(greedy_desc())
    candidates.append(greedy_asc())
    for step in (2, 3, 4):
        candidates.append(greedy_desc(step=step))
        candidates.append(greedy_asc(step=step))

    best_seq = max(candidates, key=len)
    best_len = len(best_seq)

    # Refinement pass on the best candidates
    rng = random.Random(42)
    for seq in candidates:
        if time.time() > TIME_LIMIT - 3.0:
            break
        if len(seq) < best_len - 2:
            continue
        ext = try_extend(list(seq), TIME_LIMIT - 3.0)
        if len(ext) > best_len:
            best_seq, best_len = ext, len(ext)
        searched = local_search(list(best_seq), TIME_LIMIT - 3.0, rng)
        if len(searched) > best_len:
            best_seq, best_len = searched, len(searched)

    # Final extension of the global best
    ext = try_extend(list(best_seq), TIME_LIMIT - 1.0)
    if len(ext) > best_len:
        best_seq = ext

    return best_seq
