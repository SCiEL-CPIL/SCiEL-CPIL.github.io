"""Combined program: bitset-based sum checking + beam search."""
import time

# Max possible sum: if we take all 256 values 1..256, total = 32896.
# But we only take a subset in increasing order; worst case sum is still bounded.
# To be safe, allocate a bitset large enough.
MAX_SUM = 32896 + 256  # generous bound

def _greedy():
    """Greedy: pick smallest available value that doesn't collide."""
    values = []
    used = 0  # bitset of used sums
    for x in range(1, 257):
        # Compute new sums (suffix sums ending at x)
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        # Check collision via bitset
        mask = 0
        for t in tails:
            mask |= (1 << t)
        if used & mask:
            continue
        values.append(x)
        used |= mask
    return values, used

def _check_and_extend(values, used, x):
    """
    Check if x can be appended to values.
    Returns (new_used_bitset, new_sums_list) or None.
    values: list of ints (the current sequence)
    used: bitset of already-used sums
    x: candidate value > values[-1]
    """
    tails = [x]
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    mask = 0
    for t in tails:
        mask |= (1 << t)
    if used & mask:
        return None
    return (used | mask, tails)

def _beam_search(greedy_seq, greedy_used, deadline, beam_width=50):
    """
    Beam search to find a longer sequence than greedy.
    Each beam entry: (sequence_list, used_bitset, last_val)
    We try to extend each entry and keep the top beam_width by:
      - prefer smaller last_val (more room)
      - prefer shorter sequences? No, we want longer.
      Actually, we rank by (length descending, last_val ascending) so we keep
      promising long sequences with room to grow.
    """
    best_seq = greedy_seq[:]
    best_len = len(greedy_seq)

    # Initialize beam with the greedy sequence
    beam = [(greedy_seq, greedy_used, greedy_seq[-1])]

    # We'll do depth-first-like expansion but keep top K at each "level"
    # Since sequences can have different lengths, we process by expanding
    # each beam entry and collecting all valid extensions, then pruning.

    # To avoid infinite loop, we track that we only expand if we can improve.
    # We'll do iterative rounds: in each round, expand all beam entries by one
    # step, collect valid extensions, update best, then prune to beam_width.

    # But sequences have variable length; a cleaner approach:
    # Use a priority queue ordered by (length desc, last_val asc), and do
    # a best-first search, expanding the most promising state each time.
    # Stop when the top state can't beat best_len (i.e., its length + remaining
    # possible values <= best_len) or time runs out.

    import heapq

    # State: (neg_length, last_val, seq, used)
    # neg_length so that longer sequences come first (min-heap on neg_length)
    # Among same length, smaller last_val is better (more room).
    start_state = (-len(greedy_seq), greedy_seq[-1], greedy_seq, greedy_used)
    heap = [start_state]
    heapq.heapify(heap)

    # Track visited states to avoid re-expanding (by (last_val, used) is too
    # expensive; instead we just rely on the search being finite and bounded).
    # We'll limit total expansions.
    max_expansions = 200000
    expansions = 0

    while heap:
        if time.time() > deadline:
            break
        neg_len, last_val, seq, used = heapq.heappop(heap)
        cur_len = -neg_len

        # Prune: even if we add all remaining values, can we beat best_len?
        remaining = 256 - last_val
        if cur_len + remaining <= best_len:
            continue

        expansions += 1
        if expansions > max_expansions:
            break

        # Try to extend by each x > last_val
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                break
            result = _check_and_extend(seq, used, x)
            if result is None:
                continue
            new_used, _ = result
            new_seq = seq + [x]
            new_len = cur_len + 1

            if new_len > best_len:
                best_len = new_len
                best_seq = new_seq[:]

            # Push new state
            heapq.heappush(heap, (-new_len, x, new_seq, new_used))

    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    # Greedy baseline
    greedy_seq, greedy_used = _greedy()
    best_seq = greedy_seq
    best_len = len(greedy_seq)

    # Try to extend greedy by one more element (quick check)
    if time.time() < deadline - 5:
        for x in range(greedy_seq[-1] + 1, 257):
            result = _check_and_extend(greedy_seq, greedy_used, x)
            if result is not None:
                best_seq = greedy_seq + [x]
                best_len = len(best_seq)
                break

    # Beam search to find even longer sequences
    if time.time() < deadline - 3:
        result_seq = _beam_search(greedy_seq, greedy_used, deadline, beam_width=50)
        if len(result_seq) > best_len:
            best_seq = result_seq
            best_len = len(result_seq)

    return best_seq
