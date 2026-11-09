"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import functools

# EVOLVE-BLOCK-START
def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Classic recursive construction
    word = [1]
    perm_orderings = []

    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        perm_orderings.append(perms)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new

    if n <= 4:
        return "".join(map(str, word))

    # For n >= 5, try local search to improve
    try:
        result = _local_search_superperm(n, start_time, deadline, word, perm_orderings)
        if result is not None:
            return result
    except Exception:
        pass

    return "".join(map(str, word))

def _local_search_superperm(n, start_time, deadline, word, perm_orderings):
    """Local search to improve the classic superpermutation construction."""
    # Reconstruct the (n-1) word
    word_nm1 = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word_nm1) - m + 2):
            window = tuple(word_nm1[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word_nm1 = new

    # Get the permutation ordering for the last step
    perms = perm_orderings[-1]
    num_perms = len(perms)

    @functools.lru_cache(maxsize=100000)
    def get_overlap(i, j):
        """Overlap between block_i and block_j for distinct permutations."""
        if i == j:
            return 2 * (n - 1)
        p = perms[i]
        q = perms[j]
        for k in range(n - 1, 0, -1):
            if p[n-1-k:] == q[:k]:
                return k
        return 0

    @functools.lru_cache(maxsize=10000)
    def get_word_overlap(i):
        """Overlap between word_nm1 and block_i."""
        p = perms[i]
        block = list(p) + [n] + list(p)
        max_ov = min(len(word_nm1), len(block))
        for k in range(max_ov, -1, -1):
            if word_nm1[len(word_nm1)-k:] == block[:k]:
                return k
        return 0

    block_len = 2 * (n - 1) + 1

    def total_length(order):
        total = len(word_nm1)
        for i, p_idx in enumerate(order):
            if i == 0:
                ov = get_word_overlap(p_idx)
            else:
                ov = get_overlap(order[i-1], p_idx)
            total += block_len - ov
        return total

    order = list(range(num_perms))
    current_len = total_length(order)
    best_len = current_len
    best_order = list(order)

    # Local search: try adjacent swaps
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(order) - 1):
            if time.time() > deadline - 5:
                break
            a, b = order[i], order[i+1]
            if a == b:
                continue

            old_delta = 0
            if i > 0:
                old_delta += get_overlap(order[i-1], a)
            else:
                old_delta += get_word_overlap(a)
            old_delta += get_overlap(a, b)
            if i + 1 < len(order) - 1:
                old_delta += get_overlap(b, order[i+2])

            new_delta = 0
            if i > 0:
                new_delta += get_overlap(order[i-1], b)
            else:
                new_delta += get_word_overlap(b)
            new_delta += get_overlap(b, a)
            if i + 1 < len(order) - 1:
                new_delta += get_overlap(a, order[i+2])

            if new_delta > old_delta:
                order[i], order[i+1] = b, a
                current_len += old_delta - new_delta
                improved = True
                if current_len < best_len:
                    best_len = current_len
                    best_order = list(order)

    # Reconstruct word from best ordering
    if best_order != list(range(num_perms)):
        new_word = list(word_nm1)
        for i, p_idx in enumerate(best_order):
            p = perms[p_idx]
            block = list(p) + [n] + list(p)
            if i == 0:
                ov = get_word_overlap(p_idx)
            else:
                ov = get_overlap(best_order[i-1], p_idx)
            new_word += block[ov:]
        return "".join(map(str, new_word))

    return "".join(map(str, word))
# EVOLVE-BLOCK-END
