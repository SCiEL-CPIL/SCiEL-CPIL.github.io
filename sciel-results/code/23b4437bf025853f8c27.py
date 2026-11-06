import math
import random
import time

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _compatible(existing, m_new, r_new):
    for m, r in existing:
        g = _gcd(m, m_new)
        if g > 1 and (r_new % g) == (r % g):
            return False
    return True

def _score(pairs):
    return sum(1.0 / m for m, r in pairs)

def _greedy_init():
    moduli = list(range(20, 201))
    moduli.sort(key=lambda m: -1.0 / m)
    pairs = []
    for m in moduli:
        for r in range(m):
            if _compatible(pairs, m, r):
                pairs.append((m, r))
                break
    return pairs

def _try_add(pairs, m, r):
    if _compatible(pairs, m, r):
        return pairs + [(m, r)]
    return None

def _try_remove(pairs, idx):
    return pairs[:idx] + pairs[idx + 1:]

def _try_change_residue(pairs, idx, m, r_new):
    without = pairs[:idx] + pairs[idx + 1:]
    if _compatible(without, m, r_new):
        return without + [(m, r_new)]
    return None

def construct():
    deadline = time.time() + 110.0
    random.seed(42)

    best = _greedy_init()
    best_score = _score(best)

    T = 0.5
    T_min = 1e-6
    cooling = 0.9995

    current = list(best)
    current_score = best_score

    all_moduli = set(range(20, 201))

    while time.time() < deadline and T > T_min:
        move_type = random.random()

        if move_type < 0.35:
            candidates = [m for m in all_moduli if m not in {p[0] for p in current}]
            if not candidates:
                T *= cooling
                continue
            m_new = random.choice(candidates)
            new_pairs = None
            for _ in range(30):
                r_try = random.randint(0, m_new - 1)
                new_pairs = _try_add(current, m_new, r_try)
                if new_pairs is not None:
                    break
            if new_pairs is None:
                T *= cooling
                continue
            new_score = _score(new_pairs)
        elif move_type < 0.65:
            if len(current) <= 1:
                T *= cooling
                continue
            idx = random.randint(0, len(current) - 1)
            new_pairs = _try_remove(current, idx)
            new_score = _score(new_pairs)
        elif move_type < 0.85:
            if not current:
                T *= cooling
                continue
            idx = random.randint(0, len(current) - 1)
            m = current[idx][0]
            new_pairs = None
            for _ in range(30):
                r_try = random.randint(0, m - 1)
                new_pairs = _try_change_residue(current, idx, m, r_try)
                if new_pairs is not None:
                    break
            if new_pairs is None:
                T *= cooling
                continue
            new_score = _score(new_pairs)
        else:
            if not current:
                T *= cooling
                continue
            idx = random.randint(0, len(current) - 1)
            removed_m = current[idx][0]
            base = _try_remove(current, idx)
            candidates = [m for m in all_moduli if m != removed_m and m not in {p[0] for p in base}]
            if not candidates:
                T *= cooling
                continue
            m_new = random.choice(candidates)
            new_pairs = None
            for _ in range(40):
                r_try = random.randint(0, m_new - 1)
                new_pairs = _try_add(base, m_new, r_try)
                if new_pairs is not None:
                    break
            if new_pairs is None:
                T *= cooling
                continue
            new_score = _score(new_pairs)

        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-12)):
            current = new_pairs
            current_score = new_score
            if current_score > best_score:
                best = list(current)
                best_score = current_score

        T *= cooling

    return [[m, r] for m, r in best]
