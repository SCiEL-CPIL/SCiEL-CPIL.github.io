import numpy as np
import time
import random

def _divisors_gt1(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

def _score(covered):
    return int(np.count_nonzero(covered == 0))

def construct():
    base = 51975
    start_time = time.time()
    time_limit = 110.0

    moduli_all = _divisors_gt1(base)
    n_mod = len(moduli_all)

    def elapsed():
        return time.time() - start_time

    def greedy_pass(order):
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            best_r = 0
            best_gain = -1
            for r in range(m):
                seg = covered[r::m]
                gain = np.count_nonzero(seg == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes, covered

    def polish(classes, covered, max_rounds=40):
        cur_uncovered = _score(covered)
        for _ in range(max_rounds):
            if elapsed() > time_limit:
                break
            improved = False
            for i in range(len(classes)):
                m, r = classes[i]
                best_r = r
                best_uncovered = cur_uncovered
                for cand in range(m):
                    if cand == r:
                        continue
                    old_seg = covered[r::m]
                    new_seg = covered[cand::m]
                    newly_unc = np.count_nonzero(old_seg == 0)
                    newly_cov = np.count_nonzero(new_seg == 0)
                    cand_uncovered = cur_uncovered + newly_unc - newly_cov
                    if cand_uncovered < best_uncovered:
                        best_uncovered = cand_uncovered
                        best_r = cand
                if best_r != r:
                    covered[r::m] = 0
                    covered[best_r::m] = 1
                    classes[i] = [m, best_r]
                    cur_uncovered = best_uncovered
                    improved = True
            if not improved:
                break
        return classes, covered, cur_uncovered

    desc_order = sorted(moduli_all, reverse=True)

    best_classes = None
    best_uncovered = base

    classes, covered = greedy_pass(desc_order)
    classes, covered, unc = polish(classes, covered)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = [c[:] for c in classes]

    if best_uncovered == 0 or elapsed() > time_limit:
        return best_classes

    rng = random.Random(12345)
    while elapsed() < time_limit - 2.0:
        order = list(desc_order)
        for _ in range(4):
            i = rng.randrange(n_mod)
            j = rng.randrange(n_mod)
            order[i], order[j] = order[j], order[i]
        classes, covered = greedy_pass(order)
        classes, covered, unc = polish(classes, covered)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = [c[:] for c in classes]
        if best_uncovered == 0:
            break

    return best_classes
