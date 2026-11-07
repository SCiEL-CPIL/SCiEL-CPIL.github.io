import time
import random
import itertools

def construct():
    deadline = time.time() + 85.0

    # Build initial solution: Boolean lattice on {0..4} + chain of 8 extensions
    base = []
    for r in range(6):
        for combo in itertools.combinations(range(5), r):
            m = 0
            for x in combo:
                m |= (1 << x)
            base.append(m)
    ext = []
    for k in range(5, 13):
        m = (1 << k) - 1
        ext.append(m)
    current = base + ext

    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b

    def score(masks):
        c = 0
        n = len(masks)
        for i in range(n):
            for j in range(i + 1, n):
                if is_comparable(masks[i], masks[j]):
                    c += 1
        return c

    best_score = score(current)
    best = current[:]

    # Generate all possible masks
    all_masks = []
    for r in range(13):
        for combo in itertools.combinations(range(12), r):
            m = 0
            for x in combo:
                m |= (1 << x)
            all_masks.append(m)

    def delta_score(masks, idx, new_mask):
        old = masks[idx]
        if old == new_mask:
            return 0
        delta = 0
        for j in range(len(masks)):
            if j == idx:
                continue
            other = masks[j]
            was_comp = is_comparable(old, other)
            is_comp = is_comparable(new_mask, other)
            if is_comp and not was_comp:
                delta += 1
            elif was_comp and not is_comp:
                delta -= 1
        return delta

    no_improve = 0
    while time.time() < deadline - 5 and no_improve < 300:
        improved = False
        for i in range(40):
            if time.time() > deadline - 5:
                break
            old = current[i]
            for _ in range(30):
                new = random.choice(all_masks)
                if new == old or new in current:
                    continue
                d = delta_score(current, i, new)
                if d > 0:
                    current[i] = new
                    best_score += d
                    best = current[:]
                    improved = True
                    no_improve = 0
                    break
            if time.time() > deadline - 5:
                break
        if not improved:
            no_improve += 1
            if no_improve % 10 == 0:
                for _ in range(5):
                    idx = random.randint(0, 39)
                    nm = random.choice(all_masks)
                    if nm not in current:
                        current[idx] = nm
                        best_score = score(current)
                        best = current[:]

    result = []
    for m in best:
        result.append([x for x in range(12) if (m >> x) & 1])
    return result
