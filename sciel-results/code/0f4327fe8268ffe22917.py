import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    base = 51975
    # All divisors > 1 of base (odd divisors since base is odd)
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors_sorted = sorted(divisors)

    def score_of(classes):
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return int(base - covered.sum())

    def greedy_construct(order, covered=None):
        """Greedy: process moduli in given order, pick best residue for each."""
        if covered is None:
            covered = np.zeros(base, dtype=np.uint8)
        else:
            covered = covered.copy()
        classes = []
        for m in order:
            if time.time() > deadline - 20:
                break
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = int(np.sum(covered[r::m] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes, covered

    def local_search(classes, covered, max_iterations=200):
        """Iteratively try to improve by changing residues or adding/removing classes."""
        if not classes:
            return classes, covered
        sc = int(base - covered.sum())
        used_moduli = set(m for m, _ in classes)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 10:
            improved = False
            iteration += 1

            # Try changing residue of existing classes
            for i in range(len(classes)):
                if time.time() > deadline - 10:
                    return classes, covered
                m, r = classes[i]
                best_r = r
                best_sc = sc
                # Try all residues if small, else sample
                if m <= 80:
                    candidates = list(range(m))
                else:
                    candidates = [r] + [random.randint(0, m - 1) for _ in range(40)]
                for tr in candidates:
                    if tr == r:
                        continue
                    # Compute gain/loss efficiently
                    old_slice = covered[r::m]
                    new_slice = covered[tr::m]
                    # gain = number of zeros in new_slice that are zero in old context
                    # Actually we need to check: positions in tr::m that are currently 0
                    # and positions in r::m that are currently 1 but would become 0
                    # But since other classes may cover those, we need full recompute for accuracy
                    # However, for speed, let's do a quick estimate first
                    gain_estimate = int(np.sum(new_slice == 0))
                    loss_estimate = int(np.sum(old_slice == 1))
                    # Only do full recompute if promising
                    if gain_estimate > loss_estimate - 5:
                        classes[i][1] = tr
                        new_covered_arr = np.zeros(base, dtype=np.uint8)
                        for j, (mj, rj) in enumerate(classes):
                            new_covered_arr[rj::mj] = 1
                        new_sc = int(base - new_covered_arr.sum())
                        if new_sc < best_sc:
                            best_sc = new_sc
                            best_r = tr
                            covered = new_covered_arr
                        else:
                            classes[i][1] = r
                if best_r != r:
                    classes[i][1] = best_r
                    sc = best_sc
                    improved = True

            # Try adding an unused modulus
            if time.time() < deadline - 15:
                unused = [m for m in divisors if m not in used_moduli]
                if unused:
                    random.shuffle(unused)
                    for m in unused[:8]:
                        if time.time() > deadline - 10:
                            break
                        best_r = 0
                        best_gain = -1
                        # Sample residues
                        for _ in range(min(30, m)):
                            r = random.randint(0, m - 1)
                            gain = int(np.sum(covered[r::m] == 0))
                            if gain > best_gain:
                                best_gain = gain
                                best_r = r
                        # Also check r=0 and r=1 explicitly
                        for r in [0, 1]:
                            if r < m:
                                gain = int(np.sum(covered[r::m] == 0))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_r = r
                        if best_gain > 0:
                            classes.append([m, best_r])
                            covered[best_r::m] = 1
                            used_moduli.add(m)
                            sc = int(base - covered.sum())
                            improved = True

            # Try removing a class that contributes little
            if len(classes) > 5 and time.time() < deadline - 12:
                for i in range(len(classes)):
                    if time.time() > deadline - 10:
                        break
                    m, r = classes[i]
                    # Count how many residues are ONLY covered by this class
                    # This is expensive, so skip for now
                    pass

        return classes, covered

    best_classes = []
    best_score = base + 1

    # Phase 1: Multiple greedy constructions with different orderings
    num_greedy = 20
    for attempt in range(num_greedy):
        if time.time() > deadline - 30:
            break

        if attempt == 0:
            order = divisors_sorted
        elif attempt == 1:
            order = sorted(divisors, reverse=True)
        elif attempt == 2:
            # Adaptive look-ahead: prioritize moduli with high potential coverage
            # Use a probabilistic selection based on m (larger m covers more per class)
            order = sorted(divisors, key=lambda m: -m)
        else:
            random.seed(attempt * 7919 + 13)
            order = divisors_sorted[:]
            random.shuffle(order)

        classes, covered = greedy_construct(order)
        sc = int(base - covered.sum())
        if sc < best_score:
            best_score = sc
            best_classes = classes[:]

        # Phase 2: Local search on this solution
        if time.time() < deadline - 20:
            classes_ls, covered_ls = local_search(classes, covered, max_iterations=80)
            sc_ls = int(base - covered_ls.sum())
            if sc_ls < best_score:
                best_score = sc_ls
                best_classes = classes_ls[:]

    # Phase 3: More aggressive local search on the best solution
    if best_classes and time.time() < deadline - 15:
        covered_best = np.zeros(base, dtype=np.uint8)
        for m, r in best_classes:
            covered_best[r::m] = 1
        best_classes, covered_best = local_search(best_classes, covered_best, max_iterations=300)
        final_sc = int(base - covered_best.sum())
        if final_sc < best_score:
            best_score = final_sc

    # Phase 4: Final polish - try a few more random restarts with local search
    if time.time() < deadline - 25:
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            random.seed(attempt * 31337 + 999)
            order = divisors_sorted[:]
            random.shuffle(order)
            classes, covered = greedy_construct(order)
            sc = int(base - covered.sum())
            if sc < best_score:
                best_score = sc
                best_classes = classes[:]
            if time.time() < deadline - 15:
                classes_ls, covered_ls = local_search(classes, covered, max_iterations=50)
                sc_ls = int(base - covered_ls.sum())
                if sc_ls < best_score:
                    best_score = sc_ls
                    best_classes = classes_ls[:]

    if not best_classes:
        best_classes = [[3, 0]]

    return best_classes
