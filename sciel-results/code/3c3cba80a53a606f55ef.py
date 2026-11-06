import time
import random
from itertools import combinations

def _is_sidon_with_diagonals(subset):
    """Check if all unordered pair sums (including diagonals) are unique."""
    s = sorted(subset)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            total = s[i] + s[j]
            if total in sums:
                return False
            sums.add(total)
    return True

def _max_sidon_size_ambient(ambient):
    """Find the largest Sidon subset of ambient set."""
    n = len(ambient)
    # Try from largest subset size down
    for size in range(n, 0, -1):
        for subset in combinations(ambient, size):
            if _is_sidon_with_diagonals(subset):
                return size
    return 0

def _evaluate_ambient(ambient):
    """Return the max Sidon subset size for this ambient set."""
    return _max_sidon_size_ambient(ambient)

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    
    best_score = 12  # worst case: all 12 form a Sidon set
    best_ambient = list(range(12))
    
    # First, try some structured candidates
    structured_candidates = []
    
    # Symmetric pairs: create many sum collisions
    # If a+b = c+d, then {a,b,c,d} can't all be in a Sidon set
    # Try: pairs summing to same value
    for center in range(20, 120, 5):
        cand = []
        for d in range(1, 12):
            if center - d >= 0 and center + d <= 128:
                cand.append(center - d)
                cand.append(center + d)
        if len(cand) >= 12:
            # Take 12 distinct ones
            cand = sorted(set(cand))[:12]
            if len(cand) == 12:
                structured_candidates.append(cand)
    
    # Try arithmetic progressions with small step (more collisions)
    for start in range(0, 120):
        for step in range(1, 8):
            cand = [start + i * step for i in range(12)]
            if all(0 <= x <= 128 for x in cand):
                structured_candidates.append(cand)
    
    # Try: two interleaved sequences
    for base in range(0, 100):
        for gap in range(2, 15):
            cand = []
            for i in range(12):
                if i % 2 == 0:
                    cand.append(base + (i // 2) * gap)
                else:
                    cand.append(base + gap - 1 + (i // 2) * gap)
            if all(0 <= x <= 128 for x in cand) and len(set(cand)) == 12:
                structured_candidates.append(cand)
    
    # Evaluate structured candidates
    for cand in structured_candidates:
        if time.time() > deadline:
            break
        score = _evaluate_ambient(cand)
        if score < best_score:
            best_score = score
            best_ambient = cand
    
    # Random search with local improvement
    random.seed(42)
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Generate random candidate
        cand = sorted(random.sample(range(0, 129), 12))
        score = _evaluate_ambient(cand)
        
        if score < best_score:
            best_score = score
            best_ambient = cand
        
        # Local improvement: try swapping one element
        if score <= best_score:
            for i in range(12):
                if time.time() > deadline:
                    break
                for new_val in random.sample(range(0, 129), 5):
                    if new_val in cand:
                        continue
                    trial = cand[:i] + [new_val] + cand[i+1:]
                    trial = sorted(trial)
                    trial_score = _evaluate_ambient(trial)
                    if trial_score < best_score:
                        best_score = trial_score
                        best_ambient = trial
                        cand = trial
                        score = trial_score
                        break
    
    return best_ambient
