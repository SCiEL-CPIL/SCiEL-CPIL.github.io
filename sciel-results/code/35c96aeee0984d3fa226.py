"""Combined approach: in-place incremental eval + smooth-number-aware SA + structured refinement."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 83.0  # Safety margin before 90s kill
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    def fast_eval(A):
        """Evaluate max(|A+A|, |A*A|) for sorted list A."""
        a = sorted(A)
        sums = set()
        prods = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
                prods.add(a[i] * a[j])
        return max(len(sums), len(prods))

    def eval_with_components(A):
        """Return (score, sums_set, prods_set)."""
        a = sorted(A)
        sums = set()
        prods = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
                prods.add(a[i] * a[j])
        return max(len(sums), len(prods)), sums, prods

    def in_place_replace(current, sums, prods, idx, new_val):
        """Replace current[idx] with new_val, updating sums/prods in place. Returns old_val."""
        old_val = current[idx]
        for j in range(n):
            if j == idx:
                continue
            s = old_val + current[j]
            p = old_val * current[j]
            sums.discard(s)
            prods.discard(p)
        sums.discard(old_val + old_val)
        prods.discard(old_val * old_val)

        current[idx] = new_val
        for j in range(n):
            if j == idx:
                continue
            s = new_val + current[j]
            p = new_val * current[j]
            sums.add(s)
            prods.add(p)
        sums.add(new_val + new_val)
        prods.add(new_val * new_val)
        return old_val

    def in_place_restore(current, sums, prods, idx, old_val):
        """Revert a replacement at idx."""
        new_val = current[idx]
        for j in range(n):
            if j == idx:
                continue
            s = new_val + current[j]
            p = new_val * current[j]
            sums.discard(s)
            prods.discard(p)
        sums.discard(new_val + new_val)
        prods.discard(new_val * new_val)

        current[idx] = old_val
        for j in range(n):
            if j == idx:
                continue
            s = old_val + current[j]
            p = old_val * current[j]
            sums.add(s)
            prods.add(p)
        sums.add(old_val + old_val)
        prods.add(old_val * old_val)

    # --- Generate diverse initial candidates ---
    candidates = []

    # Sequential
    candidates.append(list(range(1, 21)))

    # Geometric progressions
    for ratio in [2, 3, 4, 5, 6, 7, 8, 10]:
        A = [ratio**i for i in range(20)]
        if max(A) <= max_val:
            candidates.append(A)

    # Arithmetic progressions
    for a in range(1, 15):
        for d in range(1, 10):
            A_ap = [a + i * d for i in range(n)]
            if max(A_ap) <= max_val:
                candidates.append(A_ap)

    # Divisor sets of highly composite numbers
    def get_divisors(m):
        divs = set()
        i = 1
        while i * i <= m:
            if m % i == 0:
                divs.add(i)
                divs.add(m // i)
            i += 1
        return sorted(divs)

    highly_composite = [720, 2520, 5040, 10080, 15120, 20160, 27720, 45360, 50400,
                        55440, 83160, 110880, 166320, 221760, 277200, 332640, 498960,
                        554400, 665280, 720720, 1081080, 1441440, 2162160, 2882880,
                        3603600, 4324320, 6486480, 7207200, 10810800, 14414400]

    for N in highly_composite:
        if time.time() > deadline - 10:
            break
        divs = [d for d in get_divisors(N) if d <= max_val]
        if len(divs) < n:
            continue
        candidates.append(divs[:n])
        if len(divs) >= 2 * n:
            candidates.append(divs[-n:])
        if len(divs) >= 40:
            mid = len(divs) // 2
            candidates.append(divs[mid - n // 2:mid + n // 2])
            candidates.append(divs[:10] + divs[-10:])

    # Smooth numbers (key for product-set compression)
    def gen_smooth_numbers(primes, max_exp, limit):
        vals = set([1])
        for p in primes:
            new_vals = set()
            for v in vals:
                cur = v
                for e in range(max_exp):
                    cur *= p
                    if cur > limit:
                        break
                    new_vals.add(cur)
            vals |= new_vals
        return sorted([v for v in vals if v <= limit])

    smooth_23 = gen_smooth_numbers([2, 3], 20, max_val)
    if len(smooth_23) >= n:
        candidates.append(smooth_23[:n])
        if len(smooth_23) >= 40:
            step = len(smooth_23) // n
            candidates.append(smooth_23[::step][:n])

    smooth_235 = gen_smooth_numbers([2, 3, 5], 15, max_val)
    if len(smooth_235) >= n:
        candidates.append(smooth_235[:n])
        if len(smooth_235) >= 40:
            step = len(smooth_235) // n
            candidates.append(smooth_235[::step][:n])

    smooth_2357 = gen_smooth_numbers([2, 3, 5, 7], 12, max_val)
    if len(smooth_2357) >= n:
        candidates.append(smooth_2357[:n])

    # Numbers with many divisors
    div_counts = [0] * (max_val + 1)
    for i in range(1, int(max_val**0.5) + 1):
        for j in range(i, max_val + 1, i):
            div_counts[j] += 1
            if i != j // i:
                div_counts[j] += 1

    top_div = sorted(range(1, max_val + 1), key=lambda x: div_counts[x], reverse=True)
    candidates.append(top_div[:n])
    if len(top_div) >= 40:
        candidates.append(top_div[:10] + top_div[10:20])

    # Powers
    powers = []
    for base in range(2, 26):
        v = 1
        while v <= max_val:
            powers.append(v)
            v *= base
    powers = sorted(set(powers))
    if len(powers) >= n:
        candidates.append(powers[:n])
        if len(powers) >= 40:
            step = len(powers) // n
            candidates.append(powers[::step][:n])

    # Evaluate all candidates
    valid_candidates = []
    for A in candidates:
        if time.time() > deadline - 10:
            break
        if len(set(A)) != n:
            continue
        if max(A) > max_val or min(A) < 1:
            continue
        A_sorted = sorted(A)
        score = fast_eval(A_sorted)
        if score < best_score:
            best_score = score
            best_set = A_sorted[:]
        if score < 350:
            valid_candidates.append((A_sorted, score))

    # --- Phase 1: Hill climbing with in-place eval from multiple starts ---
    random.seed(42)

    starts = [(best_set, best_score)]
    for A_s, sc in valid_candidates:
        if A_s != best_set:
            starts.append((A_s, sc))
    starts = starts[:12]

    for start_set, start_score in starts:
        if time.time() > deadline - 5:
            break

        current = start_set[:]
        current_score = start_score
        _, current_sums, current_prods = eval_with_components(current)
        current_set = set(current)

        no_improve = 0
        max_no_improve = 2000

        while no_improve < max_no_improve and time.time() < deadline - 5:
            move_type = random.random()
            i = random.randint(0, n - 1)

            if move_type < 0.30:
                new_val = random.randint(1, max_val)
                if new_val in current_set:
                    no_improve += 1
                    continue
            elif move_type < 0.50:
                base = current[i]
                delta = random.randint(-15, 15)
                new_val = base + delta
                if new_val < 1 or new_val > max_val or new_val in current_set:
                    no_improve += 1
                    continue
            elif move_type < 0.65:
                factor = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15])
                new_val = current[i] * factor
                if new_val > max_val or new_val in current_set:
                    no_improve += 1
                    continue
            elif move_type < 0.80:
                val = current[i]
                divs = []
                for d in range(1, int(val**0.5) + 1):
                    if val % d == 0:
                        divs.append(d)
                        if d != val // d:
                            divs.append(val // d)
                if not divs:
                    no_improve += 1
                    continue
                new_val = random.choice(divs)
                if new_val in current_set:
                    no_improve += 1
                    continue
            else:
                j = random.randint(0, n - 1)
                if i == j:
                    no_improve += 1
                    continue
                new_val = current[i] * current[j]
                if new_val > max_val or new_val in current_set:
                    no_improve += 1
                    continue

            old_val = in_place_replace(current, current_sums, current_prods, i, new_val)
            new_score = max(len(current_sums), len(current_prods))

            if new_score < current_score:
                current_set.discard(old_val)
                current_set.add(new_val)
                current.sort()
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                in_place_restore(current, current_sums, current_prods, i, old_val)
                no_improve += 1

    # --- Phase 2: Smooth-number-aware Simulated Annealing ---
    # Key insight: moves that stay within the smooth-number lattice preserve
    # product-set structure, enabling escape from local minima.
    if time.time() < deadline - 8 and best_set is not None:
        current = best_set[:]
        current_score = best_score
        _, current_sums, current_prods = eval_with_components(current)
        current_set = set(current)

        # Precompute a pool of smooth numbers for structured moves
        smooth_pool = gen_smooth_numbers([2, 3, 5, 7], 12, max_val)
        smooth_pool_set = set(smooth_pool)

        temp = 8.0
        cooling = 0.9997
        max_iter = 60000

        for iteration in range(max_iter):
            if time.time() > deadline - 4:
                break
            temp *= cooling

            i = random.randint(0, n - 1)
            move_choice = random.random()

            if move_choice < 0.40:
                # Random replacement
                new_val = random.randint(1, max_val)
            elif move_choice < 0.70:
                # Smooth-number replacement (stays in multiplicative lattice)
                if smooth_pool:
                    new_val = random.choice(smooth_pool)
                else:
                    new_val = random.randint(1, max_val)
            elif move_choice < 0.85:
                # Multiplicative perturbation of current element
                factor = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20])
                new_val = current[i] * factor
                if new_val > max_val:
                    # Try division instead
                    for d in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        if current[i] % d == 0:
                            new_val = current[i] // d
                            break
                    else:
                        new_val = random.randint(1, max_val)
            else:
                # Neighbor in smooth lattice: multiply/divide by small prime
                p = random.choice([2, 3, 5, 7])
                if random.random() < 0.5:
                    new_val = current[i] * p
                    if new_val > max_val:
                        new_val = current[i] // p if current[i] % p == 0 else random.randint(1, max_val)
                else:
                    if current[i] % p == 0:
                        new_val = current[i] // p
                    else:
                        new_val = current[i] * p
                        if new_val > max_val:
                            new_val = random.randint(1, max_val)

            if new_val < 1 or new_val > max_val or new_val in current_set:
                continue

            old_val = in_place_replace(current, current_sums, current_prods, i, new_val)
            new_score = max(len(current_sums), len(current_prods))
            diff = new_score - current_score

            if diff < 0 or random.random() < np.exp(-diff / max(temp, 0.01)):
                current_set.discard(old_val)
                current_set.add(new_val)
                current.sort()
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                in_place_restore(current, current_sums, current_prods, i, old_val)

    # --- Phase 3: 2-swap local search ---
    if time.time() < deadline - 6 and best_set is not None:
        current = best_set[:]
        current_score = best_score
        _, current_sums, current_prods = eval_with_components(current)
        current_set = set(current)

        no_improve = 0
        max_no_improve = 4000

        while no_improve < max_no_improve and time.time() < deadline - 4:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                no_improve += 1
                continue

            new_val_i = random.randint(1, max_val)
            new_val_j = random.randint(1, max_val)
            if (new_val_i in current_set and new_val_i != current[i]) or \
               (new_val_j in current_set and new_val_j != current[j]) or \
               (new_val_i == new_val_j) or \
               (new_val_i == current[j] and new_val_j == current[i]):
                no_improve += 1
                continue

            old_i = current[i]
            old_j = current[j]

            # Remove old_i contributions
            for k in range(n):
                if k == i:
                    continue
                current_sums.discard(old_i + current[k])
                current_prods.discard(old_i * current[k])
            current_sums.discard(old_i + old_i)
            current_prods.discard(old_i * old_i)

            # Remove old_j contributions
            for k in range(n):
                if k == j or k == i:
                    continue
                current_sums.discard(old_j + current[k])
                current_prods.discard(old_j * current[k])
            current_sums.discard(old_i + old_j)
            current_prods.discard(old_i * old_j)
            current_sums.discard(old_j + old_j)
            current_prods.discard(old_j * old_j)

            current[i] = new_val_i
            current[j] = new_val_j

            # Add new_val_i contributions
            for k in range(n):
                if k == i:
                    continue
                current_sums.add(new_val_i + current[k])
                current_prods.add(new_val_i * current[k])
            current_sums.add(new_val_i + new_val_i)
            current_prods.add(new_val_i + new_val_i)

            # Add new_val_j contributions
            for k in range(n):
                if k == j:
                    continue
                current_sums.add(new_val_j + current[k])
                current_prods.add(new_val_j * current[k])
            current_sums.add(new_val_j + new_val_j)
            current_prods.add(new_val_j * new_val_j)

            new_score = max(len(current_sums), len(current_prods))

            if new_score < current_score:
                current_set.discard(old_i)
                current_set.discard(old_j)
                current_set.add(new_val_i)
                current_set.add(new_val_j)
                current.sort()
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            else:
                # Restore
                for k in range(n):
                    if k == i:
                        continue
                    current_sums.discard(new_val_i + current[k])
                    current_prods.discard(new_val_i * current[k])
                current_sums.discard(new_val_i + new_val_i)
                current_prods.discard(new_val_i + new_val_i)

                for k in range(n):
                    if k == j:
                        continue
                    current_sums.discard(new_val_j + current[k])
                    current_prods.discard(new_val_j * current[k])
                current_sums.discard(new_val_j + new_val_j)
                current_prods.discard(new_val_j * new_val_j)

                current[i] = old_i
                current[j] = old_j

                for k in range(n):
                    if k == i:
                        continue
                        current_sums.add(old_i + current[k])
                        current_prods.add(old_i * current[k])
                current_sums.add(old_i + old_i)
                current_prods.add(old_i * old_i)

                for k in range(n):
                    if k == j:
                        continue
                    current_sums.add(old_j + current[k])
                    current_prods.add(old_j * current[k])
                current_sums.add(old_j + old_j)
                current_prods.add(old_j * old_j)

                no_improve += 1

    # --- Phase 4: Final structured refinement ---
    if time.time() < deadline - 3 and best_set is not None:
        current = best_set[:]
        current_score = best_score
        _, current_sums, current_prods = eval_with_components(current)
        current_set = set(current)

        # Build structured candidate values
        structured_vals = set()
        for x in range(1, 120):
            structured_vals.add(x)
        for x in range(1, 120):
            for m in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
                if x * m <= max_val:
                    structured_vals.add(x * m)
        for v in range(1, max_val + 1, 50):
            structured_vals.add(v)
        structured_vals = sorted(structured_vals)

        for i in range(n):
            if time.time() > deadline - 1:
                break
            old_val = current[i]
            for new_val in structured_vals:
                if new_val == old_val or new_val in current_set:
                    continue
                in_place_replace(current, current_sums, current_prods, i, new_val)
                new_score = max(len(current_sums), len(current_prods))
                if new_score < current_score:
                    current_set.discard(old_val)
                    current_set.add(new_val)
                    current.sort()
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current[:]
                    break
                else:
                    in_place_restore(current, current_sums, current_prods, i, old_val)

    return best_set
# EVOLVE-BLOCK-END
