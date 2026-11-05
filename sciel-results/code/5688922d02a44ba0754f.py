import time
import random
import sys

def construct(n):
    """
    Constructs a superpermutation for n symbols.
    For n=7, uses overlap-matrix local search to find a better ordering
    of the (n-1)-permutation blocks than the standard left-to-right order.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    if n <= 1:
        return str(n) if n == 1 else ""

    # Build superpermutation up to n-1 using classic recursive construction
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_ov = min(len(new), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # Extract all (n-1)-permutations in order of appearance
    perm_list, seen = [], set()
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perm_list.append(window)

    num_perms = len(perm_list)

    if n != 7:
        # For smaller n, just do the standard construction
        new_word = []
        for p in perm_list:
            block = list(p) + [n] + list(p)
            max_ov = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        return "".join(map(str, new_word))

    # For n=7: precompute overlap matrix
    # M[i][j] = longest suffix of perm_list[i] matching prefix of perm_list[j]
    # Max possible overlap is n-1 = 6
    M = [[0] * num_perms for _ in range(num_perms)]
    for i in range(num_perms):
        p1 = perm_list[i]
        for j in range(num_perms):
            p2 = perm_list[j]
            for k in range(n - 1, -1, -1):
                if k == 0 or p1[-k:] == p2[:k]:
                    M[i][j] = k
                    break

    # Initial order: standard left-to-right order
    order = list(range(num_perms))

    # Compute initial total overlap
    def total_overlap(ord):
        s = 0
        for i in range(len(ord) - 1):
            s += M[ord[i]][ord[i + 1]]
        return s

    best_order = order[:]
    best_score = total_overlap(order)

    # Local search with delta-based scoring
    random.seed(42)

    def try_swap(ord, i, j, current_score):
        """Try swapping positions i and j. Returns (new_score, accepted)."""
        if i == j:
            return current_score, False
        if abs(i - j) == 1:
            # Adjacent swap: positions i, i+1
            a, b = ord[i], ord[i + 1]
            old_mid = M[a][b]
            new_mid = M[b][a]
            delta = new_mid - old_mid
            # Check neighbors
            if i > 0:
                left = ord[i - 1]
                delta += M[left][b] - M[left][a]
            if i + 1 < len(ord) - 1:
                right = ord[i + 2]
                delta += M[a][right] - M[b][right]
            new_score = current_score + delta
            if new_score > current_score:
                ord[i], ord[i + 1] = b, a
                return new_score, True
            return current_score, False
        else:
            # Non-adjacent swap
            a, b = ord[i], ord[j]
            old_score = 0
            new_score = 0
            # Affected edges: (i-1,i), (i,i+1), (j-1,j), (j,j+1)
            for pos in [i - 1, i, j - 1, j]:
                if 0 <= pos < len(ord) - 1:
                    old_score += M[ord[pos]][ord[pos + 1]]
            # Apply swap
            ord[i], ord[j] = b, a
            for pos in [i - 1, i, j - 1, j]:
                if 0 <= pos < len(ord) - 1:
                    new_score += M[ord[pos]][ord[pos + 1]]
            delta = new_score - old_score
            if delta > 0:
                return current_score + delta, True
            else:
                # Revert
                ord[i], ord[j] = a, b
                return current_score, False

    def try_move(ord, i, j, current_score):
        """Move element from position i to position j. Returns (new_score, accepted)."""
        if i == j:
            return current_score, False
        item = ord[i]
        # Compute old affected edges
        old_edges = set()
        for pos in [i - 1, i]:
            if 0 <= pos < len(ord) - 1:
                old_edges.add((pos, pos + 1))
        if j > i:
            for pos in range(j - 1, j):
                if 0 <= pos < len(ord) - 1:
                    old_edges.add((pos, pos + 1))
        else:
            for pos in range(j, j + 1):
                if 0 <= pos < len(ord) - 1:
                    old_edges.add((pos, pos + 1))

        old_total = sum(M[ord[p]][ord[p + 1]] for p in range(len(ord) - 1) if (p, p + 1) in old_edges)

        # Perform move
        ord.pop(i)
        if j > i:
            j -= 1
        ord.insert(j, item)

        new_edges = set()
        for pos in [j - 1, j]:
            if 0 <= pos < len(ord) - 1:
                new_edges.add((pos, pos + 1))
        if j > i - 1:
            for pos in range(j, min(j + 1, len(ord) - 1)):
                new_edges.add((pos, pos + 1))
        else:
            for pos in range(i - 1, min(i, len(ord) - 1)):
                new_edges.add((pos, pos + 1))

        new_total = sum(M[ord[p]][ord[p + 1]] for p in range(len(ord) - 1) if (p, p + 1) in new_edges)

        delta = new_total - old_total
        if delta > 0:
            return current_score + delta, True
        else:
            # Revert
            ord.pop(j)
            ord.insert(i, item)
            return current_score, False

    # Main search loop
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 50000

    while time.time() < deadline:
        move_type = random.random()

        if move_type < 0.4:
            # Adjacent swap
            i = random.randint(0, num_perms - 2)
            new_score, accepted = try_swap(order, i, i + 1, current_score)
        elif move_type < 0.7:
            # Random swap
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            new_score, accepted = try_swap(order, i, j, current_score)
        else:
            # Move element
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            new_score, accepted = try_move(order, i, j, current_score)

        if accepted:
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_order = order[:]
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Reset to best and try different seed
                order = best_order[:]
                current_score = best_score
                no_improve_count = 0
                random.seed(random.randint(0, 10**9))

    # Construct final word from best_order
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perm_list[p_idx]
        block = list(p) + [n] + list(p)
        if idx == 0:
            word = block
        else:
            max_ov = min(len(word), len(block))
            ov = 0
            for k in range(max_ov, -1, -1):
                if word[-k:] == block[:k]:
                    ov = k
                    break
            word += block[ov:]

    return "".join(map(str, word))
