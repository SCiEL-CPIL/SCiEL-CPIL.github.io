"""Rewritten program: narrow admissible tuple via branch-and-bound on residue selection."""
import time
import math
import random
import numpy as np

N = 5000
DEADLINE = 315.0  # safety margin before 330s kill

# ---- primes ----
def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

_PRIMES_FULL = _primes(N)

# For optimization, we only need primes up to a cap.
# Primes p > N/2 remove at most 1 element (floor(5000/p)+1 <= 2).
# We'll include primes up to ~2000 to be safe for the branch-and-bound.
K_CAP = 2000
_PRIMES_OPT = [p for p in _PRIMES_FULL if p <= K_CAP]

# Precompute class membership as numpy boolean arrays for fast union
# For each prime p, class_data[p] is a list of p boolean arrays of length N+1
def _build_class_data():
    data = {}
    for p in _PRIMES_OPT:
        classes = []
        for r in range(p):
            arr = np.zeros(N + 1, dtype=bool)
            arr[r::p] = True
            classes.append(arr)
        data[p] = classes
    return data

_CLASS_DATA = _build_class_data()

# Precompute counts for each class
_CLASS_COUNTS = {}
for p in _PRIMES_OPT:
    _CLASS_COUNTS[p] = [int(np.sum(arr)) for arr in _CLASS_DATA[p]]

def _admissible(points):
    """Check if a set of points is admissible."""
    k = len(points)
    if k == 0:
        return True
    for p in _PRIMES_FULL:
        if p > k:
            break
        seen = set()
        for v in points:
            seen.add(v % p)
        if len(seen) == p:
            return False
    return True

def _construct_from_selection(selection):
    """Given a selection of removed residue classes, construct the kept set and ensure admissibility."""
    removed = np.zeros(N + 1, dtype=bool)
    for p, r in selection.items():
        removed |= _CLASS_DATA[p][r]
    points = [v for v in range(N + 1) if not removed[v]]
    
    # Ensure admissibility: iteratively remove the least-populated residue class for violated primes
    changed = True
    while changed:
        changed = False
        k = len(points)
        for p in _PRIMES_FULL:
            if p > k:
                break
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                points = [v for v in points if v not in smallest]
                changed = True
                break
    return points

def _kept_count(selection):
    """Return number of elements not in any removed class."""
    removed = np.zeros(N + 1, dtype=bool)
    for p, r in selection.items():
        removed |= _CLASS_DATA[p][r]
    return int(np.sum(~removed))

def _greedy_init():
    """Greedy initialization: for each prime, pick the residue class that removes the most currently-unremoved elements."""
    selection = {}
    removed = np.zeros(N + 1, dtype=bool)
    # Process primes in order of increasing size of their smallest class (small primes first)
    primes_sorted = sorted(_PRIMES_OPT, key=lambda p: min(_CLASS_COUNTS[p]))
    for p in primes_sorted:
        counts = _CLASS_COUNTS[p]
        classes = _CLASS_DATA[p]
        best_r = -1
        best_gain = float('inf')
        for r in range(p):
            # gain = number of NEW elements removed by this class
            gain = int(np.sum(classes[r] & ~removed))
            if gain < best_gain:
                best_gain = gain
                best_r = r
        selection[p] = best_r
        removed |= classes[best_r]
    return selection

def _polish(points):
    """Deterministic local search: repeatedly remove least-populated class for violated primes."""
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        k = len(points)
        for p in _PRIMES_FULL:
            if p > k:
                break
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                points = [v for v in points if v not in smallest]
                improved = True
    return points

def construct():
    start = time.time()
    random.seed(12345)
    
    # ---- Greedy initialization ----
    best_sel = _greedy_init()
    best_points = _construct_from_selection(best_sel)
    best_points = _polish(best_points)
    best_score = len(best_points)
    
    # ---- Simulated annealing over residue choices ----
    def evaluate(sel):
        return _kept_count(sel)
    
    cur_sel = dict(best_sel)
    cur_kept = evaluate(cur_sel)
    
    T0 = 5.0
    T_end = 0.01
    while time.time() < DEADLINE - 30:
        # Pick a random prime to reassign
        p = random.choice(_PRIMES_OPT)
        old_r = cur_sel[p]
        new_r = random.randrange(p)
        if new_r == old_r:
            continue
        cur_sel[p] = new_r
        new_kept = evaluate(cur_sel)
        delta = new_kept - cur_kept
        if delta >= 0 or random.random() < math.exp(delta / max(T0, 1e-9)):
            cur_kept = new_kept
            if new_kept > best_score:
                best_score = new_kept
                best_sel = dict(cur_sel)
                best_points = _construct_from_selection(best_sel)
                best_points = _polish(best_points)
                best_score = len(best_points)
        else:
            cur_sel[p] = old_r
        T0 = max(T_end, T0 * 0.9999)
    
    # ---- Multi-start with random initial selections ----
    for trial in range(30):
        if time.time() > DEADLINE - 15:
            break
        sel = {}
        for p in _PRIMES_OPT:
            counts = _CLASS_COUNTS[p]
            maxc = max(counts)
            candidates = [r for r in range(p) if counts[r] == maxc]
            sel[p] = random.choice(candidates)
        pts = _construct_from_selection(sel)
        pts = _polish(pts)
        if len(pts) > best_score:
            best_score = len(pts)
            best_points = pts
    
    return best_points

if __name__ == "__main__":
    r = construct()
    print(len(r))
