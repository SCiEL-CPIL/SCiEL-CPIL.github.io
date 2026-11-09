"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START

def _try_add_class(classes, new_class):
    """Check if a new class can be added to the existing set while maintaining disjointness."""
    m_new, r_new = new_class
    for m, r in classes:
        g = gcd(m_new, m)
        if g == 1:
            return False  # coprime moduli always intersect
        if r_new % g == r % g:
            return False
    return True

def _score(classes):
    """Compute the exact rational score (sum of 1/m)."""
    total = Fraction(0, 1)
    for m, r in classes:
        total += Fraction(1, m)
    return total

def _score_float(classes):
    """Compute float score for fast comparison."""
    return sum(1.0 / m for m, r in classes)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    all_moduli = list(range(20, 201))
    
    def greedy_construct(moduli_list):
        """Greedy: add moduli in given order, pick first valid residue."""
        classes = []
        for m in moduli_list:
            found = False
            for r in range(m):
                if _try_add_class(classes, [m, r]):
                    classes.append([m, r])
                    found = True
                    break
        return classes
    
    # Strategy 1: Greedy with smallest moduli first
    best_classes = greedy_construct(all_moduli)
    best_score = _score(best_classes)
    best_score_f = _score_float(best_classes)
    
    # Local search improvement
    def try_improve(classes, max_iters=1000):
        current = [c[:] for c in classes]
        current_score_f = _score_float(current)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5:
                break
            
            # Try adding a new modulus
            for m in all_moduli:
                if any(c[0] == m for c in current):
                    continue
                added = False
                for r in range(m):
                    if _try_add_class(current, [m, r]):
                        new_classes = current + [[m, r]]
                        new_score_f = current_score_f + 1.0/m
                        if new_score_f > current_score_f:
                            current = new_classes
                            current_score_f = new_score_f
                            improved = True
                            added = True
                            break
                if added:
                    break
            
            # Try replacing a class with a smaller modulus
            for i in range(len(current)):
                m_old = current[i][0]
                replaced = False
                for m_new in all_moduli:
                    if m_new >= m_old:
                        continue
                    if any(c[0] == m_new for c in current if c[0] != m_old):
                        continue
                    others = [c for c in current if c[0] != m_old]
                    for r_new in range(m_new):
                        if _try_add_class(others, [m_new, r_new]):
                            new_classes = others + [[m_new, r_new]]
                            new_score_f = _score_float(new_classes)
                            if new_score_f > current_score_f:
                                current = new_classes
                                current_score_f = new_score_f
                                improved = True
                                replaced = True
                                break
                    if replaced:
                        break
                if replaced:
                    break
                    
        return current, _score(current)
    
    best_classes, best_score = try_improve(best_classes, max_iters=500)
    best_score_f = _score_float(best_classes)
    
    # Randomized restarts
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        rng = random.Random(42)
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            shuffled = all_moduli[:]
            rng.shuffle(shuffled)
            candidate = greedy_construct(shuffled)
            cand_score_f = _score_float(candidate)
            if cand_score_f > best_score_f * 0.98:
                improved_c, improved_s = try_improve(candidate, max_iters=200)
                improved_f = _score_float(improved_c)
                if improved_f > best_score_f:
                    best_classes = improved_c
                    best_score = improved_s
                    best_score_f = improved_f
    
    return [[m, r] for m, r in best_classes]

# EVOLVE-BLOCK-END
