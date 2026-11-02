import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit

    if n <= 1:
        return "1" * n

    # Generate all permutations
    from itertools import permutations
    perms = list(permutations(range(1, n + 1)))
    num_perms = len(perms)

    # Build standard superpermutation word recursively
    def build_standard_word(n):
        if n == 1:
            return [1]
        word = build_standard_word(n - 1)
        # Extract order of (n-1)-permutations from current word
        order = []
        seen = set()
        for i in range(len(word) - (n - 1) + 1):
            w = tuple(word[i:i + n - 1])
            if w not in seen:
                seen.add(w)
                order.append(w)
        # Build new word
        new_word = []
        for p in order:
            block = list(p) + [n] + list(p)
            # Find max overlap with end of new_word
            max_ov = min(len(new_word), len(block))
            ov = 0
            for o in range(max_ov, 0, -1):
                if new_word[-o:] == block[:o]:
                    ov = o
                    break
            new_word += block[ov:]
        return new_word

    std_word = build_standard_word(n)
    best_len = len(std_word)
    best_word = std_word[:]

    # Extract permutation order from standard word
    std_order = []
    seen = set()
    for i in range(len(std_word) - n + 1):
        w = tuple(std_word[i:i + n])
        if w not in seen:
            seen.add(w)
            std_order.append(w)

    # Precompute: for each pair (a, b), the max overlap of a's suffix with b's prefix
    # overlap[i][j] = max k such that order[i][-k:] == order[j][:k]
    # This is expensive for n=7 (5040^2), but we can compute it efficiently
    # Actually, let's compute it on the fly with memoization or just use a dict

    # For efficiency, precompute overlap matrix for the standard order
    # But 5040^2 = ~25M entries, too much memory. Let's compute on demand.

    # Function to compute max overlap between suffix of perm a and prefix of perm b
    def max_overlap(a, b):
        # a and b are tuples of length n
        max_k = min(len(a), len(b))
        for k in range(max_k, 0, -1):
            if a[-k:] == b[:k]:
                return k
        return 0

    # Build the word from an order and return (word, length)
    def build_word(order):
        word = list(order[0])
        for i in range(1, len(order)):
            ov = max_overlap(order[i-1], order[i])
            word += list(order[i][ov:])
        return word, len(word)

    # Validate: check all n! permutations appear as windows
    def is_valid_word(word):
        seen = set()
        for i in range(len(word) - n + 1):
            seen.add(tuple(word[i:i+n]))
        return len(seen) == num_perms

    # Initial validation
    if not is_valid_word(best_word):
        # Should not happen, but fallback
        pass

    current_order = std_order[:]
    current_len = best_len

    # 2-opt local search: try reversing segments
    # Also try single swaps and 3-opt moves

    # Precompute overlap for adjacent pairs in current order for fast updates
    # Actually, let's just do full recomputation but with early termination

    # Efficient length calculation
    def calc_len(order):
        total = n  # first permutation
        for i in range(1, len(order)):
            ov = max_overlap(order[i-1], order[i])
            total += n - ov
        return total

    current_len = calc_len(current_order)

    # Local search with multiple move types
    no_improve_count = 0
    max_no_improve = 5000

    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break

        move_type = random.random()
        improved = False

        if move_type < 0.3:
            # Single swap
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j:
                continue
            current_order[i], current_order[j] = current_order[j], current_order[i]
            new_len = calc_len(current_order)
            if new_len < current_len:
                current_len = new_len
                improved = True
                if new_len < best_len:
                    word, _ = build_word(current_order)
                    if is_valid_word(word):
                        best_word = word
                        best_len = new_len
                    else:
                        # Invalid, revert
                        current_order[i], current_order[j] = current_order[j], current_order[i]
                        current_len = calc_len(current_order)
                        improved = False
            else:
                current_order[i], current_order[j] = current_order[j], current_order[i]

        elif move_type < 0.6:
            # Reverse segment (2-opt)
            i = random.randint(0, num_perms - 1)
            j = random.randint(i, num_perms - 1)
            if j - i < 2:
                continue
            # Reverse order[i:j+1]
            segment = current_order[i:j+1]
            segment.reverse()
            current_order[i:j+1] = segment
            new_len = calc_len(current_order)
            if new_len < current_len:
                current_len = new_len
                improved = True
                if new_len < best_len:
                    word, _ = build_word(current_order)
                    if is_valid_word(word):
                        best_word = word
                        best_len = new_len
                    else:
                        # Revert
                        segment.reverse()
                        current_order[i:j+1] = segment
                        current_len = calc_len(current_order)
                        improved = False
            else:
                segment.reverse()
                current_order[i:j+1] = segment

        else:
            # Move one element to another position
            i = random.randint(0, num_perms - 1)
            j = random.randint(0, num_perms - 1)
            if i == j:
                continue
            elem = current_order.pop(i)
            current_order.insert(j, elem)
            new_len = calc_len(current_order)
            if new_len < current_len:
                current_len = new_len
                improved = True
                if new_len < best_len:
                    word, _ = build_word(current_order)
                    if is_valid_word(word):
                        best_word = word
                        best_len = new_len
                    else:
                        # Revert: remove from j and insert back at i
                        current_order.pop(j)
                        current_order.insert(i, elem)
                        current_len = calc_len(current_order)
                        improved = False
            else:
                # Revert
                current_order.pop(j)
                current_order.insert(i, elem)

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

    # Final: return best valid word
    return "".join(map(str, best_word))
