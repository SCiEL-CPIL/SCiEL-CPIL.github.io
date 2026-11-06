"""Model-and-optimize: parameterize modulus selection and fit directly against
the exact gcd/CRT disjointness criterion, maximizing the number of classes."""
# EVOLVE-BLOCK-START
import time
import math
import random

_START = None


def _prime_factors(m):
    """Distinct prime factors of m."""
    facs = set()
    d = 2
    while d * d <= m:
        if m % d == 0:
            facs.add(d)
            while m % d == 0:
                m //= d
        d += 1
    if m > 1:
        facs.add(m)
    return facs


def construct():
    global _START
    if _START is None:
        _START = time.time()

    deadline = _START + 85.0
    MAX_MOD = 128
    moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcds
    n_mods = len(moduli)
    gcds = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    # Calculate constraint degree: number of other moduli with gcd > 1
    degree = [0] * n_mods
    for i in range(n_mods):
        for j in range(n_mods):
            if i != j and gcds[i][j] > 1:
                degree[i] += 1

    # Order moduli by descending degree (most constrained first)
    order = sorted(range(n_mods), key=lambda i: -degree[i])
    
    best_count = 0
    best_result = []

    # Try multiple random restarts with different tie-breaking or residue choices
    # Since the problem is small, we can try to solve for the full set first.
    # If we fail, we can try subsets.
    
    # Strategy: Greedy assignment with backtracking on the full set.
    # If the full set is not assignable, we need to drop some.
    # A heuristic: The moduli that are "hardest" to assign should be prioritized.
    # If we process in order of descending degree, we are prioritizing the hardest.
    
    def solve_greedy(order, randomize_residues=False):
        assigned = [None] * n_mods
        # We process moduli in the given order
        for idx in order:
            m = moduli[idx]
            found = False
            # Try residues in a specific order
            if randomize_residues:
                residues = list(range(m))
                random.shuffle(residues)
            else:
                residues = list(range(m))
                
            for r in residues:
                ok = True
                for j in range(n_mods):
                    if assigned[j] is None:
                        continue
                    g = gcds[idx][j]
                    if g > 1: # Only check if they share a factor
                        if (r - assigned[j]) % g == 0:
                            ok = False
                            break
                if ok:
                    assigned[idx] = r
                    found = True
                    break
            if not found:
                # If we can't assign this one, we might need to backtrack or drop it.
                # For a greedy "maximize count" approach, we can just drop it and continue.
                # However, dropping a high-degree node might make others easier.
                # Let's just mark it as unassigned and continue.
                assigned[idx] = -1 
                
        result = []
        for i in range(n_mods):
            if assigned[i] is not None and assigned[i] != -1:
                result.append([assigned[i], moduli[i]])
        return result

    # First, try deterministic greedy
    res = solve_greedy(order, randomize_residues=False)
    best_count = len(res)
    best_result = res

    # Try randomized greedy multiple times to see if we can get a better count
    # (e.g., if a bad early choice blocks many later ones, a random shuffle might help)
    # Note: The order is fixed by degree. We can shuffle the residue choices.
    for _ in range(500):
        if time.time() > deadline:
            break
        res = solve_greedy(order, randomize_residues=True)
        if len(res) > best_count:
            best_count = len(res)
            best_result = res
            # If we found a perfect assignment (all 127), we can stop
            if best_count == n_mods:
                break

    # If we didn't get a high count, try a different ordering:
    # Maybe ascending modulus? Or random order?
    # Let's try a few random orderings of the moduli themselves.
    for _ in range(100):
        if time.time() > deadline:
            break
        rand_order = list(range(n_mods))
        random.shuffle(rand_order)
        res = solve_greedy(rand_order, randomize_residues=True)
        if len(res) > best_count:
            best_count = len(res)
            best_result = res
            if best_count == n_mods:
                break

    # Enforce canonical residues and distinct moduli
    seen = set()
    out = []
    for r, m in best_result:
        if m in seen:
            continue
        if not (0 <= r < m):
            continue
        seen.add(m)
        out.append([r, m])

    if not out:
        out = [[1, 2]]

    return out
# EVOLVE-BLOCK-END
