import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 82.0

    N = 256
    all_nums = list(range(1, N + 1))

    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N

    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)

    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]

    # Start with all odds
    base_mask = 0
    for i in odds_indices:
        base_mask |= (1 << i)

    best_mask = base_mask
    best_size = bin(base_mask).count('1')

    def greedy_add(mask, candidates):
        """Greedily add elements from candidates that have no conflicts with mask."""
        changed = True
        while changed:
            changed = False
            for e in candidates:
                if (mask >> e) & 1:
                    continue
                if (conflict_mask[e] & mask) == 0:
                    mask |= (1 << e)
                    changed = True
        return mask

    # Initial greedy: add evens with 0 conflicts
    current_mask = greedy_add(base_mask, evens_indices)
    current_size = bin(current_mask).count('1')
    if current_size > best_size:
        best_mask = current_mask
        best_size = current_size

    def local_search_2swap(mask, size, time_limit):
        """Local search with 2-swap moves: remove 2 elements, add up to 2 elements."""
        nonlocal best_mask, best_size
        T = 1.5
        cooling = 0.9997
        no_improve = 0
        max_no_improve = 5000

        while time.time() < time_limit:
            if no_improve > max_no_improve:
                break

            r = random.random()
            if r < 0.6:
                # 2-swap: remove 2 random elements from mask, try to add 2 elements
                bits = [i for i in range(N) if (mask >> i) & 1]
                if len(bits) < 2:
                    continue
                to_remove = random.sample(bits, 2)
                temp_mask = mask
                for rm in to_remove:
                    temp_mask &= ~(1 << rm)

                # Try to add up to 2 elements that don't conflict with temp_mask
                not_in_temp = [idx for idx in range(N) if not ((temp_mask >> idx) & 1)]
                if not not_in_temp:
                    continue

                # Prefer evens
                not_in_evens = [idx for idx in not_in_temp if (idx + 1) % 2 == 0]
                not_in_odds = [idx for idx in not_in_temp if (idx + 1) % 2 == 1]

                added = 0
                # Try to add first element
                pool = not_in_evens if (not_in_evens and random.random() < 0.7) else not_in_temp
                if pool:
                    e1 = random.choice(pool)
                    conflicts1 = conflict_mask[e1] & temp_mask
                    temp_mask2 = (temp_mask | (1 << e1))
                    added += 1
                    # Try to add second element
                    not_in_temp2 = [idx for idx in range(N) if not ((temp_mask2 >> idx) & 1)]
                    if not_in_temp2 and random.random() < 0.5:
                        pool2 = [idx for idx in not_in_temp2 if (idx + 1) % 2 == 0]
                        if not pool2:
                            pool2 = not_in_temp2
                        e2 = random.choice(pool2)
                        temp_mask2 |= (1 << e2)
                        added += 1

                new_size = len(bits) - 2 + added
                delta = new_size - size

                if delta > 0:
                    mask = temp_mask2
                    size = new_size
                    no_improve = 0
                    if size > best_size:
                        best_mask = mask
                        best_size = size
                elif delta == 0:
                    if random.random() < 0.3:
                        mask = temp_mask2
                        no_improve += 1
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        mask = temp_mask2
                        size = new_size
                        no_improve += 1
                    else:
                        no_improve += 1
            elif r < 0.8:
                # Single add: add one element, remove its conflicts
                not_in_mask = [idx for idx in range(N) if not ((mask >> idx) & 1)]
                if not not_in_mask:
                    continue
                not_in_evens = [idx for idx in not_in_mask if (idx + 1) % 2 == 0]
                pool = not_in_evens if (not_in_evens and random.random() < 0.8) else not_in_mask
                e = random.choice(pool)
                conflicts = conflict_mask[e] & mask
                num_conflicts = bin(conflicts).count('1')
                new_size = size - num_conflicts + 1
                delta = new_size - size

                if delta > 0:
                    mask = (mask & ~conflicts) | (1 << e)
                    size = new_size
                    no_improve = 0
                    if size > best_size:
                        best_mask = mask
                        best_size = size
                elif delta == 0:
                    if random.random() < 0.4:
                        mask = (mask & ~conflicts) | (1 << e)
                        no_improve += 1
                    else:
                        no_improve += 1
                else:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        mask = (mask & ~conflicts) | (1 << e)
                        size = new_size
                        no_improve += 1
                    else:
                        no_improve += 1
            elif r < 0.9:
                # Single remove: remove one random element
                bits = [i for i in range(N) if (mask >> i) & 1]
                if bits:
                    rm = random.choice(bits)
                    mask &= ~(1 << rm)
                    size -= 1
                    no_improve += 1
            else:
                # Random restart towards odds
                if random.random() < 0.5:
                    mask = base_mask
                    size = best_size
                    no_improve = 0

            T *= cooling
            if T < 0.01:
                T = 0.01

        return mask, size

    # Multiple restarts with local search
    restart_count = 0
    while time.time() < DEADLINE - 10:
        restart_count += 1
        if restart_count == 1:
            # First restart: use the greedy result
            mask = current_mask
            size = current_size
        else:
            # Subsequent restarts: start from base odds, maybe with some random evens
            mask = base_mask
            size = bin(mask).count('1')
            if random.random() < 0.5:
                # Add a random subset of evens
                num_random = random.randint(0, 5)
                for _ in range(num_random):
                    e = random.choice(evens_indices)
                    if not ((mask >> e) & 1):
                        mask |= (1 << e)
                        size += 1
            # Greedy add
            mask = greedy_add(mask, evens_indices)
            size = bin(mask).count('1')

        # Run local search for a chunk of time
        chunk_end = min(time.time() + 15, DEADLINE - 10)
        mask, size = local_search_2swap(mask, size, chunk_end)

    # Final polish: try to add any remaining elements with 0 conflicts
    for e in evens_indices:
        if not ((best_mask >> e) & 1):
            if (conflict_mask[e] & best_mask) == 0:
                best_mask |= (1 << e)

    # Convert back to numbers
    result = sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])

    return result
