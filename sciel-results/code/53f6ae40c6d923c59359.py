"""Optimized program: local search over residue assignments for odd covering systems."""
import time
import random
import math
from collections import defaultdict

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11
# All odd divisors > 1
_DIVISORS = [d for d in range(3, BASE + 1, 2) if BASE % d == 0]
_DIVISORS_SORTED_DESC = sorted(_DIVISORS, reverse=True)  # larger moduli first
_DIVISORS_SORTED_ASC = sorted(_DIVISORS)  # smaller moduli first


def _uncovered_count(classes):
    """Count residues mod BASE covered by no class. classes = list of (m, r)."""
    covered = bytearray(BASE)
    for m, r in classes:
        for x in range(r, BASE, m):
            covered[x] = 1
    return BASE - sum(covered)


def _build_initial_greedy_desc():
    """Greedy construction: for each modulus (largest first), pick residue that maximizes new coverage."""
    covered = bytearray(BASE)
    classes = []
    for m in _DIVISORS_SORTED_DESC:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = 0
            for x in range(r, BASE, m):
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_gain > 0:
            classes.append((m, best_r))
            for x in range(best_r, BASE, m):
                covered[x] = 1
    return classes


def _build_initial_greedy_asc():
    """Greedy construction: for each modulus (smallest first), pick residue that maximizes new coverage."""
    covered = bytearray(BASE)
    classes = []
    for m in _DIVISORS_SORTED_ASC:
        best_r = 0
        best_gain = -1
        for r in range(m):
            gain = 0
            for x in range(r, BASE, m):
                if not covered[x]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        if best_gain > 0:
            classes.append((m, best_r))
            for x in range(best_r, BASE, m):
                covered[x] = 1
    return classes


def _fast_uncovered(classes):
    """Fast uncovered count using bytearray."""
    covered = bytearray(BASE)
    for m, r in classes:
        for x in range(r, BASE, m):
            covered[x] = 1
    return BASE - sum(covered)


def _local_search(classes, deadline):
    """Local search: try changing residues, adding, removing classes."""
    curr_classes = [list(c) for c in classes]
    curr_score = _fast_uncovered(curr_classes)
    improved = True
    max_iterations = 100
    iter_count = 0
    
    while improved and iter_count < max_iterations and time.time() < deadline:
        improved = False
        iter_count += 1
        
        # 1. Try changing residue of existing classes
        for i in range(len(curr_classes)):
            m, r = curr_classes[i]
            best_new_r = r
            best_new_score = curr_score
            
            # Try neighbors and some random residues
            candidates = set()
            candidates.add((r - 1) % m)
            candidates.add((r + 1) % m)
            if m > 10:
                for _ in range(min(5, m)):
                    candidates.add(random.randint(0, m - 1))
            
            for new_r in candidates:
                if new_r == r:
                    continue
                old_r = curr_classes[i][1]
                curr_classes[i][1] = new_r
                new_score = _fast_uncovered(curr_classes)
                curr_classes[i][1] = old_r
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_r = new_r
            
            if best_new_r != r and best_new_score < curr_score:
                curr_classes[i][1] = best_new_r
                curr_score = best_new_score
                improved = True
        
        # 2. Try adding a new class
        used_mods = set(m for m, _ in curr_classes)
        available = [m for m in _DIVISORS if m not in used_mods]
        
        if available:
            # Sample a few
            samples = random.sample(available, min(5, len(available)))
            for m in samples:
                best_r = 0
                best_gain = -1
                for r in range(m):
                    gain = 0
                    # Count how many new residues would be covered
                    # We need to check if adding this class covers any currently uncovered residues
                    # This is expensive, so let's approximate or do a quick check
                    # For now, just pick the residue that covers the most total (ignoring overlap)
                    # Actually, we need to check against current coverage
                    # Let's compute the gain properly but efficiently
                    # For each x in range(r, BASE, m), check if not covered by other classes
                    # This is O(BASE) per r, which is too slow for large m
                    # Let's just try a few residues
                    pass
                
                # Simplified: just pick a random residue and see if it helps
                test_r = random.randint(0, m - 1)
                test_classes = curr_classes + [[m, test_r]]
                test_score = _fast_uncovered(test_classes)
                if test_score < curr_score:
                    curr_classes.append([m, test_r])
                    curr_score = test_score
                    improved = True
        
        # 3. Try removing a class
        for i in range(len(curr_classes)):
            test_classes = curr_classes[:i] + curr_classes[i+1:]
            test_score = _fast_uncovered(test_classes)
            if test_score < curr_score:
                del curr_classes[i]
                curr_score = test_score
                improved = True
                break
    
    return curr_classes, curr_score


def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 150s limit
    
    # Build initial solutions using both greedy orders
    classes_desc = _build_initial_greedy_desc()
    classes_asc = _build_initial_greedy_asc()
    
    score_desc = _fast_uncovered(classes_desc)
    score_asc = _fast_uncovered(classes_asc)
    
    # Pick the better one
    if score_asc < score_desc:
        best_classes = classes_asc
        best_score = score_asc
    else:
        best_classes = classes_desc
        best_score = score_desc
    
    # Run local search
    if time.time() < DEADLINE:
        improved_classes, improved_score = _local_search(best_classes, DEADLINE)
        if improved_score < best_score:
            best_classes = improved_classes
            best_score = improved_score
    
    # Try a few more restarts if time permits
    restart_count = 0
    while time.time() < DEADLINE - 5.0 and restart_count < 5:
        restart_count += 1
        # Randomize the initial solution
        random_classes = []
        random_mods = random.sample(_DIVISORS, min(len(_DIVISORS), 20))
        for m in random_mods:
            r = random.randint(0, m - 1)
            random_classes.append((m, r))
        
        rand_score = _fast_uncovered(random_classes)
        if rand_score < best_score:
            best_classes = random_classes
            best_score = rand_score
        
        if time.time() < DEADLINE - 2.0:
            improved_classes, improved_score = _local_search(best_classes, DEADLINE - 2.0)
            if improved_score < best_score:
                best_classes = improved_classes
                best_score = improved_score
        
        if best_score == 0:
            break
    
    # Convert to list of lists for return
    return [[m, r] for m, r in best_classes]
