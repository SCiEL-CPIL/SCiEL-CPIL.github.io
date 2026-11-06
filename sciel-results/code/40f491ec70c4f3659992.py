import math
import random
import time

MAX_MOD = 128
DEADLINE_MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _valid(pairs):
    n = len(pairs)
    for i in range(n):
        for j in range(i + 1, n):
            if not _disjoint(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]):
                return False
    return True

def _greedy_build(moduli_order, time_limit, rng, randomize_residues=False):
    """Given an ordered list of moduli, greedily assign residues."""
    pairs = []
    used_mods = set()
    for m in moduli_order:
        if m in used_mods:
            continue
        if time.time() > time_limit:
            break
        residues = list(range(m))
        if randomize_residues:
            rng.shuffle(residues)
        best_r = None
        for r in residues:
            ok = True
            for (r2, m2) in pairs:
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                best_r = r
                break
        if best_r is not None:
            pairs.append((best_r, m))
            used_mods.add(m)
    return pairs

def _coprime_core(all_mods, rng):
    """Seed with a maximal set of pairwise co-prime moduli (always compatible)."""
    core = []
    core_mods = set()
    for m in all_mods:
        if m in core_mods:
            continue
        ok = True
        for _, m2 in core:
            if math.gcd(m, m2) != 1:
                ok = False
                break
        if ok:
            core.append((0, m))  # residue 0 is fine; co-prime => disjoint
            core_mods.add(m)
    return core

def _local_search(best, all_mods, deadline, rng):
    """Perturbation + greedy repair to escape local optima."""
    current_best = [[r, m] for r, m in best]
    for _ in range(5000):
        if time.time() > deadline:
            break
        perturbed = current_best[:]
        if perturbed:
            k = min(len(perturbed), rng.randint(1, 3))
            indices = rng.sample(range(len(perturbed)), k)
            indices.sort(reverse=True)
            for idx in indices:
                perturbed.pop(idx)
        used_mods = set(m for _, m in perturbed)
        remaining_mods = [m for m in all_mods if m not in used_mods]
        remaining_mods.sort()
        new_pairs = perturbed[:]
        for m in remaining_mods:
            if time.time() > deadline:
                break
            residues = list(range(m))
            rng.shuffle(residues)
            for r in residues:
                ok = True
                for (r2, m2) in new_pairs:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    new_pairs.append([r, m])
                    break
        if len(new_pairs) > len(current_best):
            current_best = new_pairs
    return current_best

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN

    all_mods = list(range(2, MAX_MOD + 1))

    # Precompute GCD scores for ordering heuristics
    scores = {}
    for m in all_mods:
        s = 0
        for k in all_mods:
            if m != k:
                s += math.gcd(m, k)
        scores[m] = s

    order_score = sorted(all_mods, key=lambda m: scores[m], reverse=True)
    order_score_asc = sorted(all_mods, key=lambda m: scores[m])
    order_asc = sorted(all_mods)
    order_desc = sorted(all_mods, reverse=True)

    best_pairs = []
    best_len = 0

    def consider(pairs):
        nonlocal best_pairs, best_len
        if len(pairs) > best_len:
            best_len = len(pairs)
            best_pairs = pairs

    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng = random.Random(num_restarts)

        # Choose an ordering strategy
        r = rng.random()
        if r < 0.5:
            order = all_mods[:]
            rng.shuffle(order)
        elif r < 0.75:
            order = order_score[:]
        elif r < 0.875:
            order = order_asc[:]
        else:
            order = order_desc[:]

        # Seed with a co-prime core (guaranteed compatible), then greedily fill
        core = _coprime_core(order, rng)
        core_mods = set(m for _, m in core)
        remaining = [m for m in order if m not in core_mods]
        pairs = core + _greedy_build(remaining, deadline, rng, randomize_residues=True)
        consider(pairs)

    # Local search to improve the best found
    if time.time() < deadline - 3.0 and best_pairs:
        rng = random.Random(999999)
        improved = _local_search(best_pairs, all_mods, deadline, rng)
        consider(improved)

    return [[r, m] for r, m in best_pairs]
