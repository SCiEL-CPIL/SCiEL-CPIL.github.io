import time
from collections import Counter

def construct(n):
    """
    Constructs a superpermutation using classic recursive construction,
    then optimizes block ordering via TSP local search,
    then applies character-level deletion local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # Step 1: Classic recursive construction to get blocks
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_o = min(len(new), len(block))
            for o in range(max_o, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        if time.time() > deadline:
            break

    # Step 2: Extract blocks and compute overlap matrix for TSP reordering
    m = n
    # Re-extract the (n-1)-perms in classic order
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)

    blocks = [list(p) + [m] + list(p) for p in perms]
    n_blocks = len(blocks)

    if n_blocks <= 1:
        return "".join(map(str, word))

    # Compute overlap matrix
    ov = [[0] * n_blocks for _ in range(n_blocks)]
    for i in range(n_blocks):
        bi = blocks[i]
        for j in range(n_blocks):
            if i == j:
                continue
            bj = blocks[j]
            max_o = min(len(bi), len(bj))
            for o in range(max_o, 0, -1):
                if bi[-o:] == bj[:o]:
                    ov[i][j] = o
                    break
    if time.time() > deadline:
        return "".join(map(str, word))

    # Step 3: TSP local search on block ordering
    order = list(range(n_blocks))

    def order_cost(ord_):
        if not ord_:
            return 0
        cost = len(blocks[ord_[0]])
        for k in range(1, len(ord_)):
            i, j = ord_[k - 1], ord_[k]
            cost += len(blocks[j]) - ov[i][j]
        return cost

    best_cost = order_cost(order)

    # 2-opt
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n_blocks - 1):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, n_blocks):
                if time.time() > deadline - 10:
                    break
                new_order = order[:i + 1] + order[i + 1:j + 1][::-1] + order[j + 1:]
                nc = order_cost(new_order)
                if nc < best_cost:
                    best_cost = nc
                    order = new_order
                    improved = True

    # Or-opt
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        improved = False
        for seg_len in [1, 2, 3]:
            if time.time() > deadline - 10:
                break
            for i in range(n_blocks - seg_len):
                if time.time() > deadline - 10:
                    break
                seg = order[i:i + seg_len]
                for j in range(n_blocks):
                    if time.time() > deadline - 10:
                        break
                    if j <= i or j >= i + seg_len:
                        continue
                    new_order = order[:i] + order[i + seg_len:]
                    ins = j if j < i else j - seg_len
                    new_order = new_order[:ins] + seg + new_order[ins:]
                    nc = order_cost(new_order)
                    if nc < best_cost:
                        best_cost = nc
                        order = new_order
                        improved = True
        if not improved:
            break

    # Rebuild word from optimized order
    new_word = []
    for idx in order:
        block = blocks[idx]
        if not new_word:
            new_word.extend(block)
        else:
            o = ov[order[order.index(idx) - 1] if order.index(idx) > 0 else 0][idx] if len(order) > 1 else 0
            # Recompute overlap properly
            if len(new_word) > 0:
                prev_block = blocks[order[order.index(idx) - 1]] if order.index(idx) > 0 else None
                if prev_block is not None:
                    max_o = min(len(new_word), len(block))
                    ov2 = 0
                    for oo in range(max_o, 0, -1):
                        if new_word[-oo:] == block[:oo]:
                            ov2 = oo
                            break
                    new_word.extend(block[ov2:])
                else:
                    new_word.extend(block)
            else:
                new_word.extend(block)
    word = new_word

    if time.time() > deadline:
        return "".join(map(str, word))

    # Step 4: Character-level deletion local search
    if n >= 4:
        def get_counts(lst, nn):
            counts = Counter()
            for i in range(len(lst) - nn + 1):
                w = tuple(lst[i:i + nn])
                counts[w] += 1
            return counts

        counts = get_counts(word, n)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(word)):
                if time.time() > deadline:
                    break

                removed_windows = []
                for k in range(max(0, i - n + 1), min(i + 1, len(word) - n + 1)):
                    w = tuple(word[k:k + n])
                    removed_windows.append(w)

                added_windows = []
                for j in range(max(0, i - n + 1), min(i, len(word) - n)):
                    w_left = word[j:i]
                    w_right = word[i + 1:i + 1 + (n - len(w_left))]
                    w = tuple(w_left + w_right)
                    added_windows.append(w)

                valid = True
                for w in removed_windows:
                    if counts[w] == 1:
                        if w not in added_windows:
                            valid = False
                            break

                if valid:
                    for w in removed_windows:
                        counts[w] -= 1
                    for w in added_windows:
                        counts[w] += 1
                    word.pop(i)
                    improved = True
                    break

    return "".join(map(str, word))
