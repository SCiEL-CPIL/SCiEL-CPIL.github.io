import time
import random

def construct():
    deadline = time.time() + 85
    n = 256

    # Precompute conflict mask for all pairs
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    def is_compatible(x, current_mask):
        return (conflict_mask[x] & current_mask) == 0

    def count_bits(mask):
        return bin(mask).count('1')

    # Conflict counts
    conflict_counts = [0] * (n + 1)
    for i in range(1, n + 1):
        conflict_counts[i] = count_bits(conflict_mask[i])

    order_by_conflict = list(range(1, n + 1))
    order_by_conflict.sort(key=lambda x: conflict_counts[x])

    best_mask = 0
    best_size = 0

    # Strategy 1: Greedy from empty, sorted by conflict count
    current_mask = 0
    for x in order_by_conflict:
        if is_compatible(x, current_mask):
            current_mask |= (1 << x)
    sz = count_bits(current_mask)
    if sz > best_size:
        best_size = sz
        best_mask = current_mask

    # Strategy 2: Greedy from primes+1 start
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False

    initial_mask = 0
    for i in range(1, n + 1):
        if is_prime[i] or i == 1:
            initial_mask |= (1 << i)

    current_mask = initial_mask
    for x in order_by_conflict:
        if not (current_mask >> x) & 1:
            if is_compatible(x, current_mask):
                current_mask |= (1 << x)
    sz = count_bits(current_mask)
    if sz > best_size:
        best_size = sz
        best_mask = current_mask

    # Strategy 3: Greedy from odd numbers
    current_mask = 0
    for i in range(1, n + 1, 2):
        current_mask |= (1 << i)
    for x in order_by_conflict:
        if not (current_mask >> x) & 1:
            if is_compatible(x, current_mask):
                current_mask |= (1 << x)
    sz = count_bits(current_mask)
    if sz > best_size:
        best_size = sz
        best_mask = current_mask

    # Strategy 4: Greedy from high numbers
    current_mask = 0
    for x in range(n, 0, -1):
        if is_compatible(x, current_mask):
            current_mask |= (1 << x)
    sz = count_bits(current_mask)
    if sz > best_size:
        best_size = sz
        best_mask = current_mask

    # Multiple randomized greedy restarts
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    num_restarts = max(1, int(remaining_time / 0.3))

    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break

        order = list(range(1, n + 1))
        rng.shuffle(order)
        order.sort(key=lambda x: conflict_counts[x] + rng.random() * 8)

        current_mask = 0
        for x in order:
            if is_compatible(x, current_mask):
                current_mask |= (1 << x)
        sz = count_bits(current_mask)
        if sz > best_size:
            best_size = sz
            best_mask = current_mask

    # Add-only pass
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            for x in range(1, n + 1):
                if not (best_mask >> x) & 1:
                    if is_compatible(x, best_mask):
                        best_mask |= (1 << x)
                        best_size += 1
                        improved = True

    # 1-swap local search: aggressive version
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 4:
            improved = False
            for x in range(1, n + 1):
                if (best_mask >> x) & 1:
                    continue
                conflicts_with_x = conflict_mask[x] & best_mask
                if conflicts_with_x == 0:
                    best_mask |= (1 << x)
                    best_size += 1
                    improved = True
                    break
                # Try removing one conflicting element
                c = conflicts_with_x
                while c:
                    lb = c & (-c)
                    y = lb.bit_length() - 1
                    c ^= lb
                    new_mask = (best_mask & ~(1 << y)) | (1 << x)
                    # Check if this swap enables more additions
                    if conflict_counts[x] < conflict_counts[y]:
                        best_mask = new_mask
                        improved = True
                        break
            if improved:
                break

    # 2-swap local search: remove 2, add 2
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try all pairs (x1, x2) not in set, and (y1, y2) in set
            # Only try if x1 and x2 are compatible with each other
            for x1 in range(1, n + 1):
                if (best_mask >> x1) & 1:
                    continue
                for x2 in range(x1 + 1, n + 1):
                    if (best_mask >> x2) & 1:
                        continue
                    if (conflict_mask[x1] >> x2) & 1:
                        continue
                    # Find conflicts of x1 and x2 with best_mask
                    c1 = conflict_mask[x1] & best_mask
                    c2 = conflict_mask[x2] & best_mask
                    if c1 == 0 and c2 == 0:
                        # Can add both directly
                        best_mask |= (1 << x1) | (1 << x2)
                        best_size += 2
                        improved = True
                        break
                    # Try removing y1 from c1 and y2 from c2
                    # Only if c1 and c2 have exactly one element each, or share an element
                    if count_bits(c1) <= 2 and count_bits(c2) <= 2:
                        # Get conflicting elements
                        c1_list = []
                        c1_tmp = c1
                        while c1_tmp:
                            lb = c1_tmp & (-c1_tmp)
                            c1_list.append(lb.bit_length() - 1)
                            c1_tmp ^= lb
                        c2_list = []
                        c2_tmp = c2
                        while c2_tmp:
                            lb = c2_tmp & (-c2_tmp)
                            c2_list.append(lb.bit_length() - 1)
                            c2_tmp ^= lb

                        # Try all combinations of removing up to 2 elements
                        # Case 1: c1 and c2 are disjoint, remove one from each
                        if len(c1_list) == 1 and len(c2_list) == 1:
                            y1 = c1_list[0]
                            y2 = c2_list[0]
                            if y1 != y2:
                                new_mask = (best_mask & ~(1 << y1) & ~(1 << y2)) | (1 << x1) | (1 << x2)
                                if conflict_counts[x1] + conflict_counts[x2] < conflict_counts[y1] + conflict_counts[y2]:
                                    best_mask = new_mask
                                    improved = True
                                    break
                        # Case 2: c1 and c2 share one element, remove it and one more
                        common = set(c1_list) & set(c2_list)
                        if len(common) == 1:
                            y_common = list(common)[0]
                            # Remove y_common and one other from the union
                            other1 = [y for y in c1_list if y != y_common]
                            other2 = [y for y in c2_list if y != y_common]
                            for y_other in other1 + other2:
                                if y_other != y_common:
                                    new_mask = (best_mask & ~(1 << y_common) & ~(1 << y_other)) | (1 << x1) | (1 << x2)
                                    if conflict_counts[x1] + conflict_counts[x2] < conflict_counts[y_common] + conflict_counts[y_other]:
                                        best_mask = new_mask
                                        improved = True
                                        break
                    if improved:
                        break
                if improved:
                    break

    # Final add-only pass
    if time.time() < deadline - 0.5:
        for x in range(1, n + 1):
            if not (best_mask >> x) & 1:
                if is_compatible(x, best_mask):
                    best_mask |= (1 << x)
                    best_size += 1

    best = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    return best
