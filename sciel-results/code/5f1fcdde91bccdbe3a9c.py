import time
import random
import numpy as np

_BASE = 51975
_DIVISORS = [d for d in range(3, _BASE + 1, 2) if _BASE % d == 0]
_DIVISORS_SET = frozenset(_DIVISORS)


def _build_covered(classes, base):
    """Build covered array from list of [m, r] classes."""
    covered = np.zeros(base, dtype=np.uint8)
    for m, r in classes:
        covered[r::m] = 1
    return covered


def _score(covered, base):
    return base - int(np.sum(covered))


def _best_residue(covered, m, base):
    """Find residue r for modulus m that covers most uncovered residues.
    Returns (best_r, best_count)."""
    uncovered = 1 - covered
    best_r = 0
    best_cnt = -1
    for r in range(m):
        cnt = int(np.sum(uncovered[r::m]))
        if cnt > best_cnt:
            best_cnt = cnt
            best_r = r
    return best_r, best_cnt


def _greedy_add_best(covered, used, classes, base, deadline, best, order=None):
    """Repeatedly add the best (m, r) pair that covers most uncovered."""
    if order is None:
        order = list(_DIVISORS)

    while time.time() < deadline:
        remaining = [m for m in order if m not in used]
        if not remaining:
            break

        best_m, best_r, best_cnt = None, None, -1
        for m in remaining:
            r, cnt = _best_residue(covered, m, base)
            if cnt > best_cnt:
                best_cnt = cnt
                best_m = m
                best_r = r

        if best_m is None or best_cnt <= 0:
            break

        classes.append([best_m, best_r])
        used.add(best_m)
        covered[best_r::best_m] = 1

        s = int(np.sum(covered))
        if s > best[0]:
            best[0] = s
            best[1] = list(classes)

        if time.time() > deadline:
            break


def _local_search(covered, used, classes, base, deadline, best):
    """Try to improve by re-optimizing residues, swapping, or removing classes."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            m_old, r_old = classes[i]
            # Try re-optimizing this class's residue
            best_r_new, best_cnt_new = _best_residue(covered, m_old, base)
            if best_r_new != r_old:
                old_classes = list(classes)
                classes[i] = [m_old, best_r_new]
                new_covered = _build_covered(classes, base)
                old_score = _score(covered, base)
                new_score = _score(new_covered, base)
                if new_score < old_score:
                    covered = new_covered
                    improved = True
                    s = int(np.sum(covered))
                    if s > best[0]:
                        best[0] = s
                        best[1] = list(classes)
                else:
                    classes[i] = [m_old, r_old]

            # Try removing this class if it doesn't help
            if len(classes) > 1:
                trial_classes = [c for j, c in enumerate(classes) if j != i]
                trial_covered = _build_covered(trial_classes, base)
                trial_score = _score(trial_covered, base)
                cur_score = _score(covered, base)
                if trial_score <= cur_score:
                    classes = trial_classes
                    covered = trial_covered
                    used.discard(m_old)
                    improved = True
                    s = int(np.sum(covered))
                    if s > best[0]:
                        best[0] = s
                        best[1] = list(classes)
                    break  # restart loop since classes changed

        # Try adding a new class from unused moduli
        if time.time() < deadline:
            remaining = [m for m in _DIVISORS if m not in used]
            if remaining:
                best_m, best_r, best_cnt = None, None, -1
                for m in remaining:
                    r, cnt = _best_residue(covered, m, base)
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_m = m
                        best_r = r
                if best_m is not None and best_cnt > 0:
                    classes.append([best_m, best_r])
                    used.add(best_m)
                    covered = _build_covered(classes, base)
                    improved = True
                    s = int(np.sum(covered))
                    if s > best[0]:
                        best[0] = s
                        best[1] = list(classes)


def _pairwise_swap(covered, used, classes, base, deadline, best):
    """Try swapping residues of pairs of classes to improve coverage."""
    n = len(classes)
    for i in range(n):
        if time.time() > deadline:
            return
        for j in range(i + 1, n):
            if time.time() > deadline:
                return
            m1, r1 = classes[i]
            m2, r2 = classes[j]
            # Try swapping residues
            old_classes = list(classes)
            classes[i] = [m1, r2 % m1] if m1 != m2 else [m1, r1]
            classes[j] = [m2, r1 % m2] if m2 != m1 else [m2, r2]
            new_covered = _build_covered(classes, base)
            new_score = _score(new_covered, base)
            old_score = _score(covered, base)
            if new_score < old_score:
                covered = new_covered
                s = int(np.sum(covered))
                if s > best[0]:
                    best[0] = s
                    best[1] = list(classes)
            else:
                classes[i] = old_classes[i]
                classes[j] = old_classes[j]


def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin before 120s
    rng = random.Random(1234567)
    base = _BASE
    best = [0, []]  # [covered_count, classes]

    # Initial greedy: process divisors in various orders
    orders = [
        sorted(_DIVISORS),
        sorted(_DIVISORS, reverse=True),
        sorted(_DIVISORS, key=lambda m: base // m),
        sorted(_DIVISORS, key=lambda m: -base // m),
        list(_DIVISORS),
    ]
    for o in orders:
        rng.shuffle(o)

    iters = 0
    while time.time() < deadline - 10.0:
        iters += 1
        order = orders[iters % len(orders)]

        if iters % 3 == 0:
            # Fresh start with random order
            fresh_order = list(_DIVISORS)
            rng.shuffle(fresh_order)
            covered = np.zeros(base, dtype=np.uint8)
            used = set()
            classes = []
            _greedy_add_best(covered, used, classes, base, min(deadline - 5.0, start + 115.0), best, fresh_order)
        else:
            # Perturb best: drop some classes and re-greedy
            covered = _build_covered(best[1], base)
            used = set(m for m, r in best[1])
            classes = list(best[1])

            # Randomly drop 1-3 classes
            if classes and rng.random() < 0.6:
                drop = rng.randrange(1, max(2, len(classes) // 3 + 1))
                for _ in range(drop):
                    if not classes:
                        break
                    idx = rng.randrange(len(classes))
                    m = classes[idx][0]
                    classes.pop(idx)
                    used.discard(m)
                covered = _build_covered(classes, base)

            _greedy_add_best(covered, used, classes, base, min(deadline - 5.0, start + 115.0), best, order)

        # Local search on best
        if time.time() < deadline - 5.0 and best[1]:
            covered_ls = _build_covered(best[1], base)
            used_ls = set(m for m, r in best[1])
            classes_ls = list(best[1])
            _local_search(covered_ls, used_ls, classes_ls, base, min(deadline - 3.0, start + 115.0), best)

            # Pairwise swap refinement
            if time.time() < deadline - 2.0 and len(best[1]) > 1:
                covered_ps = _build_covered(best[1], base)
                used_ps = set(m for m, r in best[1])
                classes_ps = list(best[1])
                _pairwise_swap(covered_ps, used_ps, classes_ps, base, min(deadline - 1.0, start + 114.0), best)

    return best[1] if best[1] else [[3, 0]]


if __name__ == "__main__":
    res = construct()
    print(len(res))
