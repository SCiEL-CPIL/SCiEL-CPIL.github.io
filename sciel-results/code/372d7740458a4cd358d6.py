import time
import random
import numpy as np

_BASE = 51975
_DEADLINE = 115.0
_START = time.time()

# Precompute all divisors > 1 of 51975
def _get_divisors(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _get_divisors(_BASE)
# Sort by size descending - larger moduli first for sparse baseline
_DIVS_DESC = sorted(_DIVS, reverse=True)
# Sort by size ascending - smaller moduli cover more residues
_DIVS_ASC = sorted(_DIVS)

def _count_uncovered(covered):
    """Count how many positions are 0 (uncovered) in the bytearray."""
    return covered.count(0)

def _build_coverage(classes):
    """Build a coverage count array: coverage[pos] = number of classes covering pos."""
    coverage = np.zeros(_BASE, dtype=np.int32)
    for m, r in classes:
        coverage[r::_BASE//m+1] += 1
    return coverage

def _greedy_build(order):
    """Greedy: add moduli in given order, each picks residue covering most uncovered."""
    covered = bytearray(_BASE)  # 0 = uncovered
    classes = []
    
    for m in order:
        # Find residue r that covers the most currently uncovered positions
        best_r = 0
        best_count = -1
        for r in range(m):
            # Count uncovered in positions r, r+m, r+2m, ...
            count = 0
            for pos in range(r, _BASE, m):
                if covered[pos] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
            if best_count == 0:
                break
        
        classes.append([m, best_r])
        # Mark as covered
        for pos in range(best_r, _BASE, m):
            covered[pos] = 1
    
    return classes, _count_uncovered(covered)

def _local_refine_np(classes, coverage, uncovered_count):
    """Fast local refinement using NumPy vectorized operations."""
    classes = [c[:] for c in classes]
    improved = True
    iterations = 0
    
    while improved and iterations < 300:
        improved = False
        iterations += 1
        
        for i in range(len(classes)):
            m, r = classes[i]
            # Try all other residues for this modulus
            for r2 in range(m):
                if r2 == r:
                    continue
                
                # Gain: positions in r2-class that are currently uncovered (coverage==0)
                gain = np.sum(coverage[r2::_BASE//m+1] == 0)
                
                # Loss: positions in r-class that are only covered by this class (coverage==1)
                loss = np.sum(coverage[r::_BASE//m+1] == 1)
                
                if gain > loss:
                    # Apply change: update coverage array
                    coverage[r::_BASE//m+1] -= 1
                    coverage[r2::_BASE//m+1] += 1
                    classes[i] = [m, r2]
                    uncovered_count += loss - gain
                    improved = True
                    break
    
    return classes, uncovered_count

def _random_restart_build(rng, order):
    """Build with random ordering of moduli."""
    shuffled_order = order[:]
    rng.shuffle(shuffled_order)
    
    covered = bytearray(_BASE)
    classes = []
    
    for m in shuffled_order:
        best_r = 0
        best_count = -1
        for r in range(m):
            count = 0
            for pos in range(r, _BASE, m):
                if covered[pos] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_r = r
            if best_count == 0:
                break
        
        classes.append([m, best_r])
        for pos in range(best_r, _BASE, m):
            covered[pos] = 1
    
    return classes, _count_uncovered(covered)

def construct():
    global _START
    
    # Strategy: Start with descending order (large moduli first) for sparse baseline
    classes, uncovered = _greedy_build(_DIVS_DESC)
    best_classes = [c[:] for c in classes]
    best_uncovered = uncovered
    
    # Time check
    if time.time() - _START > _DEADLINE - 10:
        return best_classes
    
    # Local refinement on the descending greedy result
    coverage = _build_coverage(best_classes)
    refined_classes, refined_uncovered = _local_refine_np(best_classes, coverage, best_uncovered)
    if refined_uncovered < best_uncovered:
        best_classes = [c[:] for c in refined_classes]
        best_uncovered = refined_uncovered
    
    # Time check
    if time.time() - _START > _DEADLINE - 5:
        return best_classes
    
    # Also try ascending order greedy
    classes_asc, uncovered_asc = _greedy_build(_DIVS_ASC)
    if uncovered_asc < best_uncovered:
        best_classes = [c[:] for c in classes_asc]
        best_uncovered = uncovered_asc
        coverage = _build_coverage(best_classes)
        refined_classes, refined_uncovered = _local_refine_np(best_classes, coverage, best_uncovered)
        if refined_uncovered < best_uncovered:
            best_classes = [c[:] for c in refined_classes]
            best_uncovered = refined_uncovered
    
    # Time check
    if time.time() - _START > _DEADLINE - 5:
        return best_classes
    
    # Try random restarts with both orders
    rng = random.Random(42)
    restarts = 0
    while time.time() - _START < _DEADLINE - 5:
        restarts += 1
        # Alternate between descending and ascending random orders
        order = _DIVS_DESC if restarts % 2 == 1 else _DIVS_ASC
        rc, ru = _random_restart_build(rng, order)
        if ru < best_uncovered:
            best_classes = [c[:] for c in rc]
            best_uncovered = ru
            # Local refine the new best
            coverage = _build_coverage(best_classes)
            refined_classes, refined_uncovered = _local_refine_np(best_classes, coverage, best_uncovered)
            if refined_uncovered < best_uncovered:
                best_classes = [c[:] for c in refined_classes]
                best_uncovered = refined_uncovered
        
        if time.time() - _START > _DEADLINE - 5:
            break
    
    return best_classes
