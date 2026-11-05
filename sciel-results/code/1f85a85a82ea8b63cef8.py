import time
import random
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    base = 51975

    # Efficient divisor enumeration
    def get_divisors(n):
        divs = []
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                other = n // i
                if other != i:
                    divs.append(other)
        return sorted(divs)

    divisors = get_divisors(base)
    divisors_sorted = divisors  # already sorted

    def compute_coverage(classes):
        """Compute coverage array from list of classes."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return covered

    def score_from_covered(covered):
        return int(base - covered.sum())

    def greedy_adaptive(remaining_moduli, covered, time_budget):
        """
        Adaptive greedy: at each step, select the modulus with highest expected
        marginal gain, using probabilistic sampling for large moduli.
        """
        classes = []
        used = set()
        while remaining_moduli and time.time() < time_budget:
            # For each candidate, estimate marginal gain
            best_m = None
            best_r = None
            best_gain = -1

            # Sample a subset if too many candidates
            candidates = remaining_moduli
            if len(candidates) > 30:
                # Prioritize: mix of large (more coverage per class) and small
                large = [m for m in candidates if m > base // 10]
                small = [m for m in candidates if m <= base // 10]
                # Take top 15 from each by size
                large_sample = sorted(large, reverse=True)[:15]
                small_sample = sorted(small)[:15]
                candidates = large_sample + small_sample
                random.shuffle(candidates)

            for m in candidates:
                if m in used:
                    continue
                # Estimate gain: sample residues
                if m <= 50:
                    # Try all residues
                    for r in range(m):
                        gain = int(np.sum(covered[r::m] == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_m = m
                            best_r = r
                else:
                    # Sample residues
                    num_samples = min(20, m)
                    for _ in range(num_samples):
                        r = random.randint(0, m - 1)
                        gain = int(np.sum(covered[r::m] == 0))
                        if gain > best_gain:
                            best_gain = gain
                            best_m = m
                            best_r = r
                    # Also check r=0
                    gain0 = int(np.sum(covered[0::m] == 0))
                    if gain0 > best_gain:
                        best_gain = gain0
                        best_m = m
                        best_r = 0

            if best_m is None:
                break

            classes.append([best_m, best_r])
            used.add(best_m)
            covered[best_r::best_m] = 1
            remaining_moduli = [m for m in remaining_moduli if m != best_m]

        return classes, covered

    def local_search(classes, covered, max_iterations=200, time_budget=None):
        """
        Local search with incremental updates.
        Tries: (1) change residue of existing class, (2) add unused modulus, (3) remove weak class.
        """
        if not classes:
            return classes, covered

        if time_budget is None:
            time_budget = deadline - 10

        sc = score_from_covered(covered)
        used_moduli = set(m for m, _ in classes)
        improved = True
        iteration = 0

        while improved and iteration < max_iterations and time.time() < time_budget:
            improved = False
            iteration += 1

            # --- Move 1: Change residue of existing classes ---
            for i in range(len(classes)):
                if time.time() > time_budget:
                    return classes, covered
                m, r = classes[i]

                # Generate candidate residues
                if m <= 60:
                    candidates = [tr for tr in range(m) if tr != r]
                else:
                    # Sample + current + 0
                    candidates = [0]
                    if r != 0:
                        candidates.append(r)
                    candidates += [random.randint(0, m - 1) for _ in range(30)]
                    candidates = list(set(candidates))

                best_r = r
                best_sc = sc

                for tr in candidates:
                    if tr == r:
                        continue

                    # Quick estimate
                    old_zeros = int(np.sum(covered[r::m] == 0))
                    new_zeros = int(np.sum(covered[tr::m] == 0))

                    # The actual change in uncovered count:
                    # We lose coverage on positions in r::m that were ONLY covered by this class
                    # We gain coverage on positions in tr::m that were uncovered
                    # For a quick filter, if new_zeros < old_zeros, it's likely better
                    # But we need to be careful: positions in r::m might be covered by others
                    # and positions in tr::m might already be covered by others

                    # Do a more careful estimate:
                    # Positions in tr::m that are currently uncovered
                    gain_positions = (covered[tr::m] == 0)
                    gain_count = int(np.sum(gain_positions))

                    # Positions in r::m that are currently covered ONLY by this class
                    # This requires checking if any other class covers them
                    # Expensive, so use a heuristic: if gain_count > 0, try it
                    if gain_count <= 0:
                        continue

                    # Try the move
                    old_r = classes[i][1]
                    classes[i][1] = tr

                    # Recompute coverage efficiently
                    new_covered = covered.copy()
                    # Remove old contribution: set positions in r::m back, but only if not covered by others
                    # This is tricky. Let's just recompute from scratch for correctness
                    # but only when we think it might improve
                    if gain_count > 2:  # threshold to avoid too many recomputes
                        new_covered_arr = np.zeros(base, dtype=np.uint8)
                        for j, (mj, rj) in enumerate(classes):
                            new_covered_arr[rj::mj] = 1
                        new_sc = score_from_covered(new_covered_arr)
                        if new_sc < best_sc:
                            best_sc = new_sc
                            best_r = tr
                            covered = new_covered_arr
                        else:
                            classes[i][1] = old_r

                if best_r != r:
                    classes[i][1] = best_r
                    sc = best_sc
                    improved = True

            # --- Move 2: Add an unused modulus ---
            if time.time() < time_budget - 5:
                unused = [m for m in divisors if m not in used_moduli]
                if unused:
                    # Sort by potential: larger moduli cover more
                    unused_sorted = sorted(unused, reverse=True)
                    # Take top 10
                    for m in unused_sorted[:10]:
                        if time.time() > time_budget:
                            break
                        best_r = 0
                        best_gain = -1

                        if m <= 50:
                            for r in range(m):
                                gain = int(np.sum(covered[r::m] == 0))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_r = r
                        else:
                            for _ in range(15):
                                r = random.randint(0, m - 1)
                                gain = int(np.sum(covered[r::m] == 0))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_r = r
                            gain0 = int(np.sum(covered[0::m] == 0))
                            if gain0 > best_gain:
                                best_gain = gain0
                                best_r = 0

                        if best_gain > 0:
                            classes.append([m, best_r])
                            covered[best_r::m] = 1
                            used_moduli.add(m)
                            sc = score_from_covered(covered)
                            improved = True
                            break  # Only add one at a time

            # --- Move 3: Remove a class that contributes little ---
            if len(classes) > 8 and time.time() < time_budget - 5:
                for i in range(len(classes)):
                    if time.time() > time_budget:
                        break
                    m, r = classes[i]

                    # Compute how many positions are ONLY covered by this class
                    # Create a coverage count array
                    # This is expensive, so do it sparingly
                    if i % 3 == 0:  # Only check every 3rd class
                        count_cover = np.zeros(base, dtype=np.uint8)
                        for j, (mj, rj) in enumerate(classes):
                            if j != i:
                                count_cover[rj::mj] += 1

                        # Positions in r::m that have count_cover == 0 are only covered by this class
                        only_this = (count_cover[r::m] == 0)
                        only_count = int(np.sum(only_this))

                        if only_count == 0:
                            # This class doesn't uniquely cover anything, try removing
                            classes.pop(i)
                            used_moduli.discard(m)
                            new_covered = np.zeros(base, dtype=np.uint8)
                            for mj, rj in classes:
                                new_covered[rj::mj] = 1
                            new_sc = score_from_covered(new_covered)
                            if new_sc <= sc:
                                covered = new_covered
                                sc = new_sc
                                improved = True
                                break
                            else:
                                classes.insert(i, [m, r])
                                used_moduli.add(m)

        return classes, covered

    best_classes = []
    best_score = base + 1

    # Phase 1: Adaptive greedy with multiple random seeds
    num_restarts = 25
    for attempt in range(num_restarts):
        if time.time() > deadline - 40:
            break

        if attempt == 0:
            # Deterministic: sorted by modulus ascending
            order = divisors_sorted[:]
        elif attempt == 1:
            # Deterministic: sorted by modulus descending
            order = sorted(divisors, reverse=True)
        elif attempt == 2:
            # Mix: interleave small and large
            small = divisors_sorted[:len(divisors_sorted)//2]
            large = sorted(divisors_sorted[len(divisors_sorted)//2:], reverse=True)
            order = []
            for i in range(max(len(small), len(large))):
                if i < len(small):
                    order.append(small[i])
                if i < len(large):
                    order.append(large[i])
        else:
            random.seed(attempt * 7919 + 13)
            order = divisors_sorted[:]
            random.shuffle(order)

        covered = np.zeros(base, dtype=np.uint8)
        classes, covered = greedy_adaptive(order, covered, deadline - 30)
        sc = score_from_covered(covered)

        if sc < best_score:
            best_score = sc
            best_classes = classes[:]

        # Local search refinement
        if time.time() < deadline - 20 and classes:
            classes_ls, covered_ls = local_search(classes, covered, max_iterations=100, time_budget=deadline - 15)
            sc_ls = score_from_covered(covered_ls)
            if sc_ls < best_score:
                best_score = sc_ls
                best_classes = classes_ls[:]

    # Phase 2: Aggressive local search on best solution
    if best_classes and time.time() < deadline - 15:
        covered_best = compute_coverage(best_classes)
        best_classes, covered_best = local_search(best_classes, covered_best, max_iterations=300, time_budget=deadline - 5)
        final_sc = score_from_covered(covered_best)
        if final_sc < best_score:
            best_score = final_sc

    # Phase 3: Final random restarts
    if time.time() < deadline - 20:
        for attempt in range(15):
            if time.time() > deadline - 10:
                break
            random.seed(attempt * 31337 + 999 + 10000)
            order = divisors_sorted[:]
            random.shuffle(order)

            covered = np.zeros(base, dtype=np.uint8)
            classes, covered = greedy_adaptive(order, covered, deadline - 15)
            sc = score_from_covered(covered)

            if sc < best_score:
                best_score = sc
                best_classes = classes[:]

            if time.time() < deadline - 10 and classes:
                classes_ls, covered_ls = local_search(classes, covered, max_iterations=60, time_budget=deadline - 5)
                sc_ls = score_from_covered(covered_ls)
                if sc_ls < best_score:
                    best_score = sc_ls
                    best_classes = classes_ls[:]

    if not best_classes:
        best_classes = [[3, 0]]

    return best_classes
