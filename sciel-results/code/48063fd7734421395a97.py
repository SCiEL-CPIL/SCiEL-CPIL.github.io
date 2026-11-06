import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 40s safety margin before 90s kill

    MAX_SUM = 20000
    used_sums = bytearray(MAX_SUM + 1)

    def is_valid(x, suffix_sums):
        if used_sums[x]:
            return False
        for s in suffix_sums:
            if x + s <= MAX_SUM and used_sums[x + s]:
                return False
        return True

    def add_sums(x, suffix_sums):
        added = [x]
        used_sums[x] = 1
        for s in suffix_sums:
            ns = x + s
            if ns <= MAX_SUM:
                used_sums[ns] = 1
                added.append(ns)
        return added

    def remove_sums(added):
        for v in added:
            used_sums[v] = 0

    # Greedy to get a good starting point
    greedy_seq = []
    suffix_sums = []

    for x in range(1, 257):
        if is_valid(x, suffix_sums):
            greedy_seq.append(x)
            added = add_sums(x, suffix_sums)
            new_suffix = [x] + [x + s for s in suffix_sums if x + s <= MAX_SUM]
            suffix_sums = new_suffix

    best_seq = greedy_seq.copy()
    best_len = len(best_seq)

    # Reset used_sums
    used_sums[:] = b'\x00' * (MAX_SUM + 1)

    def time_up():
        return time.time() > deadline

    def dfs(seq, suffix_sums, last_val, target_len):
        nonlocal best_seq, best_len

        current_len = len(seq)

        if current_len > best_len:
            best_len = current_len
            best_seq = seq.copy()

        if time_up():
            return False

        if current_len == target_len:
            return True

        # Pruning: not enough numbers left
        remaining_slots = target_len - current_len
        available_numbers = 256 - last_val
        if available_numbers < remaining_slots:
            return False

        # Pruning: theoretical max extension
        # Even if we pick every other number, max additional is ~available/2
        # But we can do better: if we need k more elements from [last_val+1, 256],
        # the minimum span is k-1 (consecutive), so if 256 - last_val < remaining_slots - 1, impossible
        # Actually available_numbers = 256 - last_val, we need remaining_slots elements
        # from that range, so we need available_numbers >= remaining_slots
        # Which we already checked above.

        for x in range(last_val + 1, 257):
            if time_up():
                return False

            if is_valid(x, suffix_sums):
                added = add_sums(x, suffix_sums)
                new_suffix = [x] + [x + s for s in suffix_sums if x + s <= MAX_SUM]
                seq.append(x)

                if dfs(seq, new_suffix, x, target_len):
                    # Found a valid sequence of target_len
                    remove_sums(added)
                    seq.pop()
                    return True

                remove_sums(added)
                seq.pop()

        return False

    # Iterative deepening: try to find sequences of increasing length
    for target_len in range(best_len + 1, best_len + 15):
        if time_up():
            break

        # Reset used_sums for each new target
        used_sums[:] = b'\x00' * (MAX_SUM + 1)

        if dfs([], [], 0, target_len):
            # Found a sequence of target_len, continue to next
            continue
        else:
            # If we timed out, stop
            if time_up():
                break
            # Otherwise, no sequence of this length exists (exhausted search)
            # Since valid lengths are contiguous, we can stop
            break

    return best_seq
