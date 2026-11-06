"""Sum-free subset finder using Bitmask-based Local Search with Swaps and Restarts."""
import time
import random
import math
from collections import defaultdict


def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(elems):
        """Verify a list is sum-free."""
        s = set(elems)
        for a in elems:
            for b in elems:
                if a + b in s:
                    return False
        return True

    def find_sum_free(elements, time_budget=45.0):
        """Find a large sum-free subset using Bitmask Local Search."""
        if not elements:
            return []

        n = len(elements)
        elem_set = set(elements)
        elem_to_idx = {x: i for i, x in enumerate(elements)}
        my_deadline = time.time() + time_budget

        # Precompute conflict structures
        sum_pairs = [[] for _ in range(n)]
        add_conflicts = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    sum_pairs[k].append((i, j))
                    add_conflicts[i].append((k, j))
                    if i != j:
                        add_conflicts[j].append((k, i))

        # Re-do add_conflicts correctly
        add_conflicts = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    add_conflicts[i].append((k, j))

        def can_add(mask, k):
            """Check if element k can be added to the set represented by mask."""
            for (i, j) in sum_pairs[k]:
                if (mask & (1 << i)) and (mask & (1 << j)):
                    return False
            for (d, j) in add_conflicts[k]:
                if (mask & (1 << d)) and (mask & (1 << j)):
                    return False
            return True

        def count_conflicts(mask):
            """Count number of violated sum-free constraints in the set represented by mask."""
            count = 0
            m = mask
            while m:
                lsb = m & -m
                k = lsb.bit_length() - 1
                m ^= lsb
                for (i, j) in sum_pairs[k]:
                    if (mask & (1 << i)) and (mask & (1 << j)):
                        count += 1
            return count

        def greedy_completion(mask):
            """Greedy add any element that fits, until no more can be added."""
            changed = True
            while changed:
                changed = False
                for k in range(n):
                    if not (mask & (1 << k)) and can_add(mask, k):
                        mask |= (1 << k)
                        changed = True
            return mask

        def local_search_1flip(mask, max_iters=2000):
            """1-flip local search (add/remove)."""
            current_conflicts = count_conflicts(mask)
            best_mask = mask
            best_size = bin(mask).count('1')
            rng = random.Random(99999)

            for _ in range(max_iters):
                if time.time() > my_deadline:
                    break

                if current_conflicts == 0:
                    added = False
                    for k in range(n):
                        if not (mask & (1 << k)) and can_add(mask, k):
                            mask |= (1 << k)
                            added = True
                            break
                    if not added:
                        if bin(mask).count('1') > 2:
                            active_indices = [i for i in range(n) if mask & (1 << i)]
                            to_remove = rng.choice(active_indices)
                            mask &= ~(1 << to_remove)
                            mask = greedy_completion(mask)
                            current_conflicts = count_conflicts(mask)
                    else:
                        current_conflicts = count_conflicts(mask)
                        size = bin(mask).count('1')
                        if size > best_size:
                            best_size = size
                            best_mask = mask
                    continue

                conflict_found = None
                m = mask
                while m:
                    lsb = m & -m
                    k = lsb.bit_length() - 1
                    m ^= lsb
                    for (i, j) in sum_pairs[k]:
                        if (mask & (1 << i)) and (mask & (1 << j)):
                            conflict_found = (i, j, k)
                            break
                    if conflict_found:
                        break

                if not conflict_found:
                    break

                i, j, k = conflict_found
                candidates = [i, j, k]
                rng.shuffle(candidates)

                best_new_conflicts = current_conflicts
                best_remove = None
                for c in candidates:
                    new_mask = mask & ~(1 << c)
                    new_conflicts = count_conflicts(new_mask)
                    if new_conflicts < best_new_conflicts:
                        best_new_conflicts = new_conflicts
                        best_remove = c

                if best_remove is not None:
                    mask &= ~(1 << best_remove)
                    current_conflicts = best_new_conflicts
                    mask = greedy_completion(mask)
                    current_conflicts = count_conflicts(mask)
                    size = bin(mask).count('1')
                    if size > best_size and current_conflicts == 0:
                        best_size = size
                        best_mask = mask
                else:
                    if bin(mask).count('1') > 1:
                        active_indices = [i for i in range(n) if mask & (1 << i)]
                        to_remove = rng.choice(active_indices)
                        mask &= ~(1 << to_remove)
                        mask = greedy_completion(mask)
                        current_conflicts = count_conflicts(mask)

            return best_mask

        def local_search_2flip(mask, max_iters=1000):
            """2-flip local search (swap)."""
            current_conflicts = count_conflicts(mask)
            best_mask = mask
            best_size = bin(mask).count('1')
            rng = random.Random(12345)

            for _ in range(max_iters):
                if time.time() > my_deadline:
                    break

                if current_conflicts == 0:
                    added = False
                    for k in range(n):
                        if not (mask & (1 << k)) and can_add(mask, k):
                            mask |= (1 << k)
                            added = True
                            break
                    if not added:
                        active_indices = [i for i in range(n) if mask & (1 << i)]
                        inactive_indices = [i for i in range(n) if not (mask & (1 << i))]
                        if active_indices and inactive_indices:
                            rem = rng.choice(active_indices)
                            base_mask = mask & ~(1 << rem)
                            added = False
                            for add in inactive_indices:
                                if add == rem:
                                    continue
                                if can_add(base_mask, add):
                                    mask = base_mask | (1 << add)
                                    added = True
                                    break
                            if not added:
                                mask = base_mask
                        else:
                            break
                        current_conflicts = count_conflicts(mask)
                        size = bin(mask).count('1')
                        if size > best_size:
                            best_size = size
                            best_mask = mask
                    else:
                        current_conflicts = count_conflicts(mask)
                        size = bin(mask).count('1')
                        if size > best_size:
                            best_size = size
                            best_mask = mask
                    continue

                conflict_found = None
                m = mask
                while m:
                    lsb = m & -m
                    k = lsb.bit_length() - 1
                    m ^= lsb
                    for (i, j) in sum_pairs[k]:
                        if (mask & (1 << i)) and (mask & (1 << j)):
                            conflict_found = (i, j, k)
                            break
                    if conflict_found:
                        break

                if not conflict_found:
                    break

                i, j, k = conflict_found
                candidates = [i, j, k]
                rng.shuffle(candidates)

                improved = False
                for c in candidates:
                    base_mask = mask & ~(1 << c)
                    for add in range(n):
                        if (mask & (1 << add)) or add == c:
                            continue
                        if can_add(base_mask, add):
                            new_mask = base_mask | (1 << add)
                            new_conflicts = count_conflicts(new_mask)
                            if new_conflicts < current_conflicts:
                                mask = new_mask
                                current_conflicts = new_conflicts
                                improved = True
                                break
                    if improved:
                        break

                if not improved:
                    mask &= ~(1 << rng.choice(candidates))
                    current_conflicts = count_conflicts(mask)

                mask = greedy_completion(mask)
                current_conflicts = count_conflicts(mask)
                size = bin(mask).count('1')
                if size > best_size and current_conflicts == 0:
                    best_size = size
                    best_mask = mask

            return best_mask

        # ========== RUN SEARCH ==========
        best_mask = 0

        # 1. Sign Partition
        pos_mask = 0
        neg_mask = 0
        for i in range(n):
            if elements[i] > 0:
                pos_mask |= (1 << i)
            else:
                neg_mask |= (1 << i)

        for m in [pos_mask, neg_mask]:
            gm = greedy_completion(m)
            if bin(gm).count('1') > bin(best_mask).count('1'):
                best_mask = gm

        # 2. Residue Classes
        for mod in [3, 5]:
            for r in range(mod):
                m = 0
                for i in range(n):
                    if elements[i] % mod == r:
                        m |= (1 << i)
                gm = greedy_completion(m)
                if bin(gm).count('1') > bin(best_mask).count('1'):
                    best_mask = gm

        # 3. Randomized Greedy
        for t in range(5):
            rng = random.Random(1000 + t)
            order = list(range(n))
            rng.shuffle(order)
            m = 0
            for idx in order:
                if can_add(m, idx):
                    m |= (1 << idx)
            if bin(m).count('1') > bin(best_mask).count('1'):
                best_mask = m

        # 4. Local Search 1-flip
        if bin(best_mask).count('1') > 0:
            ls1 = local_search_1flip(best_mask)
            if bin(ls1).count('1') > bin(best_mask).count('1'):
                best_mask = ls1

        # 5. Local Search 2-flip
        if bin(best_mask).count('1') > 0:
            ls2 = local_search_2flip(best_mask)
            if bin(ls2).count('1') > bin(best_mask).count('1'):
                best_mask = ls2

        # 6. Final Polish
        best_mask = greedy_completion(best_mask)

        # Convert to values
        best_result = [elements[i] for i in range(n) if best_mask & (1 << i)]

        # Final Verification
        if not is_sum_free(best_result):
            result_set = set(best_result)
            changed = True
            while changed:
                changed = False
                for a in list(result_set):
                    for b in list(result_set):
                        if a + b in result_set:
                            result_set.discard(a + b)
                            changed = True
            best_result = list(result_set)
            if not is_sum_free(best_result):
                best_result = [x for x in sorted(best_result, key=lambda x: -abs(x))[:len(best_result)//2]]

        return best_result

    # Solve each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = deadline - time.time()
        if remaining < 5:
            elems = instance(size, seed)
            results.append([x for x in elems if abs(x) % 2 == 1])
            continue

        time_budget = min(remaining - 3, 50.0)
        elems = instance(size, seed)
        result = find_sum_free(elems, time_budget=time_budget)
        results.append(result)

    return results
