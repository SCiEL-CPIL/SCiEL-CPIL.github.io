"""Combined: greedy coprime set + local search + targeted small-modulus inclusion."""
import time
import random
import math
from fractions import Fraction
from collections import defaultdict

def _disjoint(m1, r1, m2, r2):
    """Check if residue classes are pairwise disjoint: r1 != r2 mod gcd(m1, m2)."""
    g = math.gcd(m1, m2)
    return (r1 % g) != (r2 % g)

def _is_valid(classes):
    """Check pairwise disjointness for a list of [m, r] pairs with distinct moduli."""
    if not classes:
        return True
    moduli = [c[0] for c in classes]
    if len(moduli) != len(set(moduli)):
        return False
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if not _disjoint(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                return False
    return True

def _score(classes):
    """Exact rational score, returned as float."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return float(total)

def _find_valid_residue(m, selected):
    """Find a residue r in [0, m) such that (m, r) is disjoint from all in selected.
    Returns r or None."""
    for r in range(m):
        ok = True
        for (m2, r2) in selected:
            if not _disjoint(m, r, m2, r2):
                ok = False
                break
        if ok:
            return r
    return None

def _find_valid_residue_random(m, selected, rng):
    """Find a residue r in [0, m) using random order. Returns r or None."""
    residues = list(range(m))
    rng.shuffle(residues)
    for r in residues:
        ok = True
        for (m2, r2) in selected:
            if not _disjoint(m, r, m2, r2):
                ok = False
                break
        if ok:
            return r
    return None

def _count_valid_residues(m, selected):
    """Count how many residues are valid for modulus m given selected classes."""
    count = 0
    for r in range(m):
        ok = True
        for (m2, r2) in selected:
            if not _disjoint(m, r, m2, r2):
                ok = False
                break
        if ok:
            count += 1
    return count

def _local_search_improve(classes, deadline, rng):
    """Try to improve classes by adding more moduli or swapping residues."""
    classes = classes[:]
    used_moduli = set(c[0] for c in classes)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding unselected small moduli
        candidates = [m for m in range(20, 201) if m not in used_moduli]
        candidates.sort()  # Prefer small moduli (high value)
        
        for m in candidates[:]:
            if time.time() > deadline:
                break
            r = _find_valid_residue(m, classes)
            if r is not None:
                classes.append([m, r])
                used_moduli.add(m)
                improved = True
        
        # Try local search: for each selected class, try different residues
        # that might allow adding more moduli
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            m_i, r_i = classes[i]
            # Try random alternative residues
            for _ in range(3):
                if time.time() > deadline:
                    break
                alt_r = rng.randint(0, m_i - 1)
                if alt_r == r_i:
                    continue
                # Check if alt_r is valid with all other classes
                ok = True
                for j in range(len(classes)):
                    if i == j:
                        continue
                    m_j, r_j = classes[j]
                    if not _disjoint(m_i, alt_r, m_j, r_j):
                        ok = False
                        break
                if ok:
                    classes[i][1] = alt_r
                    improved = True
    
    return classes

def _backtrack_with_forward_checking(target_moduli, deadline, rng):
    """Backtracking search with forward checking for residue assignment."""
    n = len(target_moduli)
    assignment = [-1] * n
    # For each modulus, track which residues are still available given prior assignments
    # We'll use a simple approach: for each new modulus, check against all prior
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        m = target_moduli[idx]
        # Get valid residues in random order
        residues = list(range(m))
        rng.shuffle(residues)
        
        for r in residues:
            ok = True
            for i in range(idx):
                if not _disjoint(m, r, target_moduli[i], assignment[i]):
                    ok = False
                    break
            if ok:
                assignment[idx] = r
                if backtrack(idx + 1):
                    return True
                assignment[idx] = -1
        return False
    
    if backtrack(0):
        return [[target_moduli[i], assignment[i]] for i in range(n)]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 150s kill

    best_classes = None
    best_score = 0.0
    rng = random.Random(12345)

    # Strategy A: Greedy pairwise coprime set (baseline from current)
    def greedy_coprime_set(moduli_list):
        picked = []
        for m in moduli_list:
            ok = True
            for p in picked:
                if math.gcd(m, p) != 1:
                    ok = False
                    break
            if ok:
                picked.append(m)
        return picked

    moduli_range = list(range(20, 201))

    # Pure greedy coprime
    picked = greedy_coprime_set(moduli_range)
    classes = [[m, 0] for m in picked]
    if _is_valid(classes):
        sc = _score(classes)
        if sc > best_score:
            best_score = sc
            best_classes = classes[:]

    # Strategy B: Greedy inclusion with residue search (include non-coprime small moduli)
    # Process moduli in ascending order; for each, try to find a valid residue.
    # This is the key improvement: we don't restrict to coprime sets.

    for attempt in range(30):
        if time.time() > deadline - 15:
            break

        rng.seed(42 + attempt)
        moduli_sorted = list(range(20, 201))

        # For later attempts, shuffle the larger moduli to explore different orderings
        if attempt > 0:
            # Keep small moduli (20-40) in order, shuffle the rest
            small = moduli_sorted[:21]  # 20 through 40
            rest = moduli_sorted[21:]
            rng.shuffle(rest)
            moduli_sorted = small + rest

        selected = []  # list of [m, r]

        for m in moduli_sorted:
            if time.time() > deadline - 10:
                break
            r = _find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])

        if selected:
            if _is_valid(selected):
                sc = _score(selected)
                if sc > best_score:
                    best_score = sc
                    best_classes = selected[:]

    # Strategy C: Local search / improvement on the best found so far
    if best_classes is not None:
        best_classes = _local_search_improve(best_classes, deadline - 10, rng)
        if _is_valid(best_classes):
            sc = _score(best_classes)
            if sc > best_score:
                best_score = sc

    # Strategy D: Randomized greedy with different heuristics
    for attempt in range(40):
        if time.time() > deadline - 5:
            break

        rng.seed(1000 + attempt)
        selected = []
        used_moduli = set()

        # Weighted random selection biased toward small moduli
        for _ in range(80):
            if time.time() > deadline - 5:
                break
            available = [m for m in range(20, 201) if m not in used_moduli]
            if not available:
                break

            # Weight by 1/m^alpha where alpha varies per attempt
            alpha = 0.5 + (attempt % 5) * 0.5
            weights = [(1.0 / (m ** alpha)) for m in available]
            tw = sum(weights)
            if tw <= 0:
                break
            r = rng.random() * tw
            cumsum = 0
            chosen_m = available[-1]
            for m, w in zip(available, weights):
                cumsum += w
                if r <= cumsum:
                    chosen_m = m
                    break

            m = chosen_m
            res = _find_valid_residue(m, selected)
            if res is not None:
                selected.append([m, res])
                used_moduli.add(m)

        if selected:
            if _is_valid(selected):
                # Apply local search improvement
                improved = _local_search_improve(selected, deadline - 5, rng)
                if _is_valid(improved):
                    sc = _score(improved)
                    if sc > best_score:
                        best_score = sc
                        best_classes = improved[:]

    # Strategy E: Backtracking with forward checking for small moduli sets
    # Try to pack the smallest N moduli by finding valid residue assignments
    for N in [20, 30, 40, 50, 60, 80, 100, 120, 150]:
        if time.time() > deadline - 3:
            break

        target_moduli = list(range(20, 20 + N))
        if 20 + N > 201:
            target_moduli = list(range(20, 201))

        # Use backtracking with forward checking
        result = _backtrack_with_forward_checking(target_moduli, deadline - 3, rng)
        if result is not None:
            if _is_valid(result):
                sc = _score(result)
                if sc > best_score:
                    best_score = sc
                    best_classes = result[:]
                # Also try local search on this result
                improved = _local_search_improve(result, deadline - 3, rng)
                if _is_valid(improved):
                    sc2 = _score(improved)
                    if sc2 > best_score:
                        best_score = sc2
                        best_classes = improved[:]

    # Strategy F: Targeted small-modulus packing with base structure
    # Use a common base to pack many small moduli
    for base in [10, 15, 20, 25, 30]:
        if time.time() > deadline - 2:
            break
        
        # Try to pack moduli that are multiples or related to the base
        candidates = []
        for m in range(20, 201):
            # Prioritize moduli that share structure with base
            if m % base == 0 or math.gcd(m, base) > 1:
                candidates.append(m)
        
        # Sort by value (1/m) descending, then by m ascending
        candidates.sort(key=lambda m: (-1.0/m, m))
        
        selected = []
        for m in candidates:
            if time.time() > deadline - 2:
                break
            r = _find_valid_residue(m, selected)
            if r is not None:
                selected.append([m, r])
        
        if selected:
            if _is_valid(selected):
                improved = _local_search_improve(selected, deadline - 2, rng)
                if _is_valid(improved):
                    sc = _score(improved)
                    if sc > best_score:
                        best_score = sc
                        best_classes = improved[:]

    # Strategy G: Seed fallback
    seed_classes = [[20 * j, j - 1] for j in range(1, 11)]
    seed_score = _score(seed_classes)
    if seed_score > best_score:
        best_score = seed_score
        best_classes = seed_classes[:]

    if best_classes is None:
        best_classes = seed_classes

    return best_classes
