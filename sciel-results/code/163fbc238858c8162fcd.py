import time
import random
import math
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit

    n = 12
    m = 40
    N = 1 << n

    def is_subset(a, b):
        return (a & b) == a and a != b

    def count_comparable(family):
        count = 0
        len_fam = len(family)
        for i in range(len_fam):
            ai = family[i]
            for j in range(i + 1, len_fam):
                aj = family[j]
                if is_subset(ai, aj) or is_subset(aj, ai):
                    count += 1
        return count

    def greedy_start():
        """Greedy construction: add set that maximizes comparability with current family."""
        family = []
        used = set()
        for _ in range(m):
            best_mask = -1
            best_gain = -1
            for mask in range(N):
                if mask in used:
                    continue
                gain = 0
                for f in family:
                    if (mask & f) == mask or (f & mask) == f:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            if best_mask == -1:
                while True:
                    mask = random.randint(0, N - 1)
                    if mask not in used:
                        best_mask = mask
                        break
            family.append(best_mask)
            used.add(best_mask)
        return family

    def random_start():
        fam = []
        used = set()
        while len(fam) < m:
            mask = random.randint(0, N - 1)
            if mask not in used:
                used.add(mask)
                fam.append(mask)
        return fam

    def local_search(initial_family, deadline):
        family = initial_family[:]
        family_set = set(family)
        current_score = count_comparable(family)
        best_family = family[:]
        best_score = current_score
        temp = 20.0
        no_improve_count = 0

        iter_count = 0
        while time.time() < deadline:
            iter_count += 1

            if iter_count % 2000 == 0 and temp > 0.5:
                temp *= 0.99
                if no_improve_count > 8000:
                    no_improve_count = 0
                    temp = 10.0
                    continue

            idx = random.randint(0, m - 1)
            old_mask = family[idx]

            r = random.random()
            if r < 0.35:
                new_mask = random.randint(0, N - 1)
            elif r < 0.55:
                # Flip one bit
                new_mask = old_mask ^ (1 << random.randint(0, n - 1))
            elif r < 0.75:
                # Flip two bits
                b1 = random.randint(0, n - 1)
                b2 = random.randint(0, n - 1)
                while b2 == b1:
                    b2 = random.randint(0, n - 1)
                new_mask = old_mask ^ (1 << b1) ^ (1 << b2)
            elif r < 0.85:
                # Random subset: clear 1-2 random set bits
                new_mask = old_mask
                for _ in range(random.randint(1, 2)):
                    bits = [b for b in range(n) if (new_mask >> b) & 1]
                    if not bits:
                        break
                    b = random.choice(bits)
                    new_mask &= ~(1 << b)
            else:
                # Random superset: add 1-2 random unset bits
                new_mask = old_mask
                for _ in range(random.randint(1, 2)):
                    bits = [b for b in range(n) if not (new_mask >> b) & 1]
                    if not bits:
                        break
                    b = random.choice(bits)
                    new_mask |= (1 << b)

            if new_mask == old_mask:
                continue
            if new_mask in family_set:
                continue

            lost = 0
            gained = 0
            for j in range(m):
                if j == idx:
                    continue
                other = family[j]
                if is_subset(old_mask, other) or is_subset(other, old_mask):
                    lost += 1
                if is_subset(new_mask, other) or is_subset(other, new_mask):
                    gained += 1

            delta = gained - lost

            accept = False
            if delta > 0:
                accept = True
            elif temp > 0.5:
                if random.random() < math.exp(delta / temp):
                    accept = True

            if accept:
                family[idx] = new_mask
                family_set.remove(old_mask)
                family_set.add(new_mask)
                current_score += delta
                no_improve_count = 0

                if current_score > best_score:
                    best_score = current_score
                    best_family = family[:]
            else:
                no_improve_count += 1

        return best_family, best_score


    # The problem: maximize the number of comparable pairs among 40 sets in B_12.
    # Use local search with multiple initializations and improved move sets.

    def chain_based_start():
        """Create a family based on multiple chains with varied element orders."""
        family = []
        used = set()

        # Create chains by varying the order of adding elements
        orders = [
            list(range(n)),
            [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [3, 1, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11],
            [0, 2, 1, 3, 5, 4, 6, 7, 8, 9, 10, 11],
            [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ]

        for order in orders:
            mask = 0
            for i in range(n + 1):
                if i > 0:
                    mask |= (1 << order[i - 1])
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
                if len(family) >= m:
                    break
            if len(family) >= m:
                break

        # Add more sets to reach 40
        while len(family) < m:
            mask = random.randint(0, N - 1)
            if mask not in used:
                family.append(mask)
                used.add(mask)

        return family[:m]

    best_overall_family = None
    best_overall_score = 0

    # Try multiple initializations with improved local search
    initializations = [
        greedy_start(),
        chain_based_start(),
        random_start(),
        chain_based_start(),
        greedy_start(),
    ]

    for init in initializations:
        if time.time() > deadline - 8:
            break
        fam, score = local_search(init, deadline)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_family = fam

    if best_overall_family is None:
        best_overall_family = list(range(m))

    result = []
    for mask in best_overall_family:
        result.append([b for b in range(n) if (mask >> b) & 1])

    return result
