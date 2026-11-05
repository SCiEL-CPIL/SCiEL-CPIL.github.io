"""Optimized odd covering system search for moduli dividing 51975."""
import time
import random
import numpy as np

BASE = 51975
DEADLINE = 140  # seconds, safety margin before 150s kill

def _divisors_of(n):
    """Return all divisors of n greater than 1, sorted."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            if i > 1:
                divs.append(i)
            j = n // i
            if j != i and j > 1:
                divs.append(j)
        i += 1
    return sorted(divs)

# Precompute divisors
DIVISORS = _divisors_of(BASE)

def _uncovered_count(covered, m, r):
    """Count uncovered residues in class (m, r)."""
    return covered[r::m].sum()  # sum of 0s = count of uncovered

def _build_covered(classes):
    """Build covered bytearray from list of [m, r] classes."""
    covered = bytearray(BASE)
    for m, r in classes:
        covered[r::m] = b'\x01' * len(range(r, BASE, m))
    return covered

def _score(classes):
    """Compute number of uncovered residues."""
    covered = _build_covered(classes)
    return BASE - sum(covered)

def _greedy_init(divisors_order=None, rng=None):
    """Greedy initialization: add moduli in given order, each picks best residue."""
    if divisors_order is None:
        divisors_order = sorted(DIVISORS)
    if rng is None:
        rng = random.Random(42)
    
    covered = bytearray(BASE)
    classes = []
    
    for m in divisors_order:
        best_r = 0
        best_gain = -1
        # Sample residues if m is large to save time
        if m > 500:
            candidates = rng.sample(range(m), min(50, m))
        else:
            candidates = range(m)
        
        for r in candidates:
            gain = covered[r::m].count(0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        classes.append([m, best_r])
        covered[best_r::m] = b'\x01' * len(range(best_r, BASE, m))
    
    return classes

def _local_search(classes, rng, time_limit):
    """Local search: try swapping residues or removing/adding moduli."""
    start = time.time()
    current_classes = [c[:] for c in classes]
    current_score = _score(current_classes)
    best_classes = [c[:] for c in current_classes]
    best_score = current_score
    
    used_moduli = set(m for m, r in current_classes)
    available_moduli = [m for m in DIVISORS if m not in used_moduli]
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try changing residue of an existing class
        for i in range(len(current_classes)):
            m, r = current_classes[i]
            # Try a few random alternative residues
            for _ in range(3):
                new_r = rng.randrange(m)
                if new_r == r:
                    continue
                new_classes = [c[:] for c in current_classes]
                new_classes[i][1] = new_r
                new_score = _score(new_classes)
                if new_score < current_score:
                    current_classes = new_classes
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                    improved = True
                    break
        
        # Try removing a class
        if len(current_classes) > 2:
            for i in range(len(current_classes)):
                new_classes = [c for j, c in enumerate(current_classes) if j != i]
                new_score = _score(new_classes)
                if new_score < current_score:
                    current_classes = new_classes
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
                    improved = True
                    break
        
        # Try adding a new class
        if available_moduli:
            m = rng.choice(available_moduli)
            # Find best residue for this modulus
            best_r = 0
            best_gain = -1
            covered = _build_covered(current_classes)
            for r in range(m):
                gain = covered[r::m].count(0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            new_classes = current_classes + [[m, best_r]]
            new_score = _score(new_classes)
            if new_score < current_score:
                current_classes = new_classes
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in current_classes]
                improved = True
        
        if not improved:
            # Random restart within local search: perturb
            if len(current_classes) > 2:
                i = rng.randrange(len(current_classes))
                m = current_classes[i][0]
                current_classes[i][1] = rng.randrange(m)
                current_score = _score(current_classes)
            else:
                break
    
    return best_classes, best_score

def construct():
    start_time = time.time()
    rng = random.Random(12345)
    
    # Multiple random restarts with different ordering
    best_classes = None
    best_score = BASE  # worst case: nothing covered
    
    # Start with greedy in natural order
    classes = _greedy_init(sorted(DIVISORS), rng)
    score = _score(classes)
    if score < best_score:
        best_score = score
        best_classes = [c[:] for c in classes]
    
    # Try a few different orderings
    for attempt in range(5):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Shuffle moduli for different greedy order
        shuffled = DIVISORS[:]
        rng.shuffle(shuffled)
        classes = _greedy_init(shuffled, rng)
        score = _score(classes)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
    
    # Local search on best solution
    remaining = DEADLINE - 10 - (time.time() - start_time)
    if remaining > 5 and best_classes is not None:
        best_classes, best_score = _local_search(best_classes, rng, remaining)
    
    return best_classes if best_classes is not None else _greedy_init(sorted(DIVISORS), rng)
