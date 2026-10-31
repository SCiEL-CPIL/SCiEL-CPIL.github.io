"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 150.0
_TIME_MARGIN = 20.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _TIME_MARGIN - _elapsed()

def _score_of(U):
    """Compute 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = np.add.outer(U_arr, U_arr)
    diffs = np.subtract.outer(U_arr, U_arr)
    n_sum = len(set(sums.flatten()))
    n_diff = len(set(diffs.flatten()))
    if n_sum == 0:
        return 0.0
    ratio = n_diff / n_sum
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_max = math.log(2 * max_u + 1)
    if log_max == 0:
        return 0.0
    return 1.0 + log_ratio / log_max

def _is_golomb(U):
    """Check if U is a Golomb ruler (all pairwise differences distinct)."""
    if len(U) < 2:
        return True
    diffs = set()
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _count_diff_collisions(U):
    """Count how many difference pairs collide (non-Golomb measure)."""
    if len(U) < 2:
        return 0
    diffs = {}
    collisions = 0
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            d = U[j] - U[i]
            if d in diffs:
                collisions += 1
            else:
                diffs[d] = 1
    return collisions

def _construct_golomb_ruler(n, max_val=1000000):
    """
    Construct a Golomb ruler with n marks using greedy + local search.
    Returns the ruler as a sorted list starting with 0.
    """
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    # Greedy construction: place each new mark as far as possible
    ruler = [0]
    for i in range(1, n):
        best_pos = None
        best_score = -1
        # Try positions from high to low
        for pos in range(max_val, -1, -1):
            # Check if pos creates a Golomb ruler
            new_ruler = ruler + [pos]
            new_ruler.sort()
            # Quick check: all differences must be distinct
            diffs = set()
            valid = True
            for a in range(len(new_ruler)):
                for b in range(a + 1, len(new_ruler)):
                    d = new_ruler[b] - new_ruler[a]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
            if valid:
                best_pos = pos
                break
        if best_pos is None:
            # Fallback: try smaller positions
            for pos in range(1, max_val):
                new_ruler = ruler + [pos]
                new_ruler.sort()
                diffs = set()
                valid = True
                for a in range(len(new_ruler)):
                    for b in range(a + 1, len(new_ruler)):
                        d = new_ruler[b] - new_ruler[a]
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
                if valid:
                    best_pos = pos
                    break
        if best_pos is None:
            # Can't extend, return what we have
            break
        ruler.append(best_pos)
    
    ruler.sort()
    return ruler

def _local_search_golomb(U, time_budget):
    """
    Local search that tries to maintain/improve Golomb property while maximizing score.
    Strategy: try to add elements that preserve Golomb property, or remove elements
    that cause many collisions.
    """
    U = sorted(set(U))
    if 0 not in U:
        U.insert(0, 0)
    
    best_U = list(U)
    best_score = _score_of(U)
    
    start = time.time()
    max_u = max(U) if U else 0
    
    while time.time() - start < time_budget and _time_left() > _TIME_MARGIN:
        improved = False
        max_u = max(U) if U else 0
        
        # Strategy 1: Try adding an element that preserves Golomb property
        # Search in a range around current max
        search_lo = max(1, max_u // 4)
        search_hi = max_u * 3 + 1000
        
        # Try several random positions
        for _ in range(200):
            if time.time() - start > time_budget or _time_left() <= _TIME_MARGIN:
                break
            pos = random.randint(search_lo, search_hi)
            if pos in U:
                continue
            new_U = U + [pos]
            new_U.sort()
            # Check Golomb property
            if _is_golomb(new_U):
                s = _score_of(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    max_u = max(U)
                    improved = True
                    break
        
        if improved:
            continue
        
        # Strategy 2: Try removing an element to reduce collisions
        if len(U) > 3:
            collisions = _count_diff_collisions(U)
            if collisions > 0:
                # Try removing each element and see if collisions decrease
                for idx in range(1, len(U)):  # don't remove 0
                    new_U = U[:idx] + U[idx+1:]
                    new_collisions = _count_diff_collisions(new_U)
                    if new_collisions < collisions:
                        s = _score_of(new_U)
                        if s > best_score:
                            best_score = s
                            best_U = new_U
                            U = new_U
                            improved = True
                            break
        
        # Strategy 3: Try small perturbations of existing elements
        if not improved:
            for _ in range(50):
                if time.time() - start > time_budget or _time_left() <= _TIME_MARGIN:
                    break
                idx = random.randint(1, len(U) - 1)  # don't move 0
                old_val = U[idx]
                delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5, 10, 20])
                new_val = old_val + delta
                if new_val < 0 or new_val in U:
                    continue
                new_U = U[:idx] + [new_val] + U[idx+1:]
                new_U.sort()
                s = _score_of(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    improved = True
                    break
    
    return best_U, best_score

def _polish(U, time_budget):
    """Final high-precision polish: try all single-element additions/removals in a tighter range."""
    U = sorted(set(U))
    best_U = list(U)
    best_score = _score_of(U)
    
    start = time.time()
    max_u = max(U) if U else 0
    
    # Try adding elements in a focused range
    for pos in range(max(1, max_u // 2), max_u * 2 + 500):
        if time.time() - start > time_budget or _time_left() <= _TIME_MARGIN:
            break
        if pos in U:
            continue
        new_U = U + [pos]
        new_U.sort()
        s = _score_of(new_U)
        if s > best_score:
            best_score = s
            best_U = new_U
            U = new_U
            max_u = max(U)
    
    # Try removing elements
    for idx in range(len(U) - 1, 0, -1):
        if time.time() - start > time_budget or _time_left() <= _TIME_MARGIN:
            break
        new_U = U[:idx] + U[idx+1:]
        s = _score_of(new_U)
        if s > best_score:
            best_score = s
            best_U = new_U
            U = new_U
    
    return best_U, best_score

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the sumset/difference-set exponent lower bound.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_U = [0, 1, 3]
    best_score = _score_of(best_U)
    
    # Phase 1: Try known good Golomb rulers of various sizes
    for n in range(3, 25):
        if _time_left() < _TIME_MARGIN + 5:
            break
        try:
            ruler = _construct_golomb_ruler(n, max_val=n * n * 100)
            if len(ruler) >= 2:
                s = _score_of(ruler)
                if s > best_score:
                    best_score = s
                    best_U = list(ruler)
        except Exception:
            continue
    
    # Phase 2: Local search from best Golomb ruler
    if _time_left() > _TIME_MARGIN + 10:
        budget = min(_time_left() * 0.4, 60.0)
        U, s = _local_search_golomb(best_U, budget)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Phase 3: Try a few random restarts with Golomb-inspired starts
    if _time_left() > _TIME_MARGIN + 10:
        for restart in range(3):
            if _time_left() < _TIME_MARGIN + 5:
                break
            # Create a random sparse set and try to Golomb-ify it
            size = random.randint(5, 15)
            max_val = random.randint(200, 2000)
            U = [0] + sorted(random.sample(range(1, max_val), min(size - 1, max_val - 1)))
            s = _score_of(U)
            if s > best_score:
                best_score = s
                best_U = list(U)
            
            if _time_left() > _TIME_MARGIN + 5:
                budget = min(_time_left() * 0.2, 30.0)
                U2, s2 = _local_search_golomb(best_U, budget)
                if s2 > best_score:
                    best_score = s2
                    best_U = U2
    
    # Phase 4: Final polish
    if _time_left() > _TIME_MARGIN + 5:
        budget = min(_time_left() * 0.3, 40.0)
        U, s = _polish(best_U, budget)
        if s > best_score:
            best_score = s
            best_U = U
    
    return best_U
# EVOLVE-BLOCK-END
