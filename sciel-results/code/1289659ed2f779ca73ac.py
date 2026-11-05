"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import Counter

def _pair_sums_with_diag(subset):
    """Return list of all unordered pair sums including diagonals (i<=j)."""
    sums = []
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            sums.append(subset[i] + subset[j])
    return sums

def _count_collisions(sums):
    """Count number of duplicate sums (excess over unique count)."""
    c = Counter(sums)
    return sum(v - 1 for v in c.values() if v > 1)

def _max_sidon_size(subset):
    """Compute the largest subset of 'subset' with all pair sums (i<=j) distinct."""
    n = len(subset)
    if n <= 1:
        return n
    elems = sorted(subset)
    best = [0]
    
    def backtrack(idx, current, current_sums):
        if idx >= n:
            best[0] = max(best[0], len(current))
            return
        if len(current) + (n - idx) <= best[0]:
            return
        # Try not taking idx first (to find smaller subsets quickly for pruning)
        backtrack(idx + 1, current, current_sums)
        # Try taking idx
        elem = elems[idx]
        new_sums = set(current_sums)
        valid = True
        for j in range(len(current)):
            val = current[j] + elem
            if val in new_sums:
                valid = False
                break
            new_sums.add(val)
        if valid:
            val = elem + elem
            if val in new_sums:
                valid = False
            else:
                new_sums.add(val)
        if valid:
            backtrack(idx + 1, current + [elem], new_sums)
    
    backtrack(0, [], set())
    return best[0]

def _eval_candidate(cand):
    """Return the max Sidon subset size of the candidate."""
    return _max_sidon_size(cand)

def _valid(cand):
    """Check candidate is 12 distinct integers in [0,128]."""
    if len(cand) != 12:
        return False
    if len(set(cand)) != 12:
        return False
    return all(0 <= x <= 128 for x in cand)

def _collision_penalty(cand):
    """
    Calculate a penalty score for the candidate.
    Higher penalty means more collisions, which is desirable for minimizing max Sidon size.
    We count the number of pairs of pairs (i,j) and (k,l) with i<=j, k<=l, (i,j)!=(k,l)
    such that cand[i]+cand[j] == cand[k]+cand[l].
    """
    n = len(cand)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = cand[i] + cand[j]
            sums[s] = sums.get(s, 0) + 1
    
    penalty = 0
    for count in sums.values():
        if count > 1:
            # Number of pairs of pairs is C(count, 2)
            penalty += count * (count - 1) // 2
    return penalty

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin within 60s limit
    
    best_cand = None
    best_score = 99
    seen = set()
    
    def consider(cand):
        nonlocal best_cand, best_score
        if not _valid(cand):
            return
        key = tuple(sorted(cand))
        if key in seen:
            return
        seen.add(key)
        s = _eval_candidate(cand)
        if s < best_score:
            best_score = s
            best_cand = list(cand)
    
    rng = random.Random(12345)
    
    # Initialize with a random candidate
    current_cand = sorted(rng.sample(range(0, 129), 12))
    current_penalty = _collision_penalty(current_cand)
    
    # Simulated Annealing
    T = 1000.0
    cooling = 0.999
    min_T = 1.0
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        # Generate a neighbor by swapping or modifying one element
        idx = rng.randint(0, 11)
        old_val = current_cand[idx]
        
        # Choose a new value
        # Bias towards values that might create collisions?
        # For now, just random sampling in [0, 128]
        new_val = rng.randint(0, 128)
        
        if new_val == old_val:
            continue
            
        # Ensure distinctness
        if new_val in current_cand:
            continue
            
        # Try the move
        current_cand[idx] = new_val
        new_penalty = _collision_penalty(current_cand)
        
        delta = new_penalty - current_penalty
        
        if delta > 0 or rng.random() < pow(2.718281828, delta / T):
            # Accept the move
            current_penalty = new_penalty
        else:
            # Reject the move
            current_cand[idx] = old_val
            
        # Periodically check the true score
        if rng.random() < 0.05: # 5% chance
            consider(current_cand)
            
        T *= cooling
        
        # If we find a very good candidate (score 4 or 5), we might want to polish it
        if best_score <= 5:
             # We can break early if we think we found a good one, 
             # but let's continue to see if we can get 4
             pass

    # Final polish phase
    if best_cand is None:
        best_cand = list(range(12))
        best_score = _eval_candidate(best_cand)
        
    # Local search around the best candidate using the exact evaluator
    if time.time() < deadline:
        cand = list(best_cand)
        for iteration in range(500):
            if time.time() > deadline:
                break
            cand_copy = list(cand)
            # Perturb 1 element
            idx = rng.randint(0, 11)
            old_val = cand_copy[idx]
            new_val = rng.randint(0, 128)
            if new_val == old_val or new_val in cand_copy:
                continue
            cand_copy[idx] = new_val
            if not _valid(cand_copy):
                continue
            key = tuple(sorted(cand_copy))
            if key in seen:
                continue
            seen.add(key)
            s = _eval_candidate(cand_copy)
            if s < best_score:
                best_score = s
                best_cand = cand_copy
                cand = list(best_cand)
            # If we improve, continue from there. If not, maybe stick to best_cand for next step?
            # For now, just random walk around best_cand
    
    return best_cand
# EVOLVE-BLOCK-END
