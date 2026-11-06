import time
import random
from itertools import combinations

def is_sidon_fast(subset):
    """Check if subset is Sidon using sorted sums with early exit."""
    sums = []
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            sums.append(subset[i] + subset[j])
    sums.sort()
    for k in range(1, len(sums)):
        if sums[k] == sums[k-1]:
            return False
    return True

def max_sidon_size_fast(S):
    """Find largest Sidon subset size of S using backtracking with pruning."""
    S = sorted(S)
    n = len(S)
    
    # Try k from n down to 1
    for k in range(n, 0, -1):
        # Use DFS with pruning to find any Sidon subset of size k
        found = _find_sidon_subset(S, k, 0, [], set())
        if found:
            return k
    return 0

def _find_sidon_subset(S, k, start, current, used_sums):
    """DFS to find a Sidon subset of size k starting from index 'start'."""
    if len(current) == k:
        return True
    
    # Pruning: not enough elements left
    if len(S) - start < k - len(current):
        return False
    
    for i in range(start, len(S)):
        x = S[i]
        # Check all pair sums involving x with existing elements
        valid = True
        new_sums = []
        for j in range(len(current)):
            s = x + current[j]
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if not valid:
            continue
        s = x + x
        if s in used_sums:
            continue
        new_sums.append(s)
        
        # Add x and its new sums
        for s in new_sums:
            used_sums.add(s)
        current.append(x)
        
        if _find_sidon_subset(S, k, i + 1, current, used_sums):
            return True
        
        # Backtrack
        current.pop()
        for s in new_sums:
            used_sums.discard(s)
    
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 78  # Safety margin: evaluator kills at 90s
    
    # Initial candidates: various structured sets
    candidates = []
    
    # Pure AP
    candidates.append(list(range(12)))
    
    # AP with different step sizes
    for step in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        cand = [i * step for i in range(12)]
        if max(cand) <= 128:
            candidates.append(cand)
    
    # Geometric-like sequences
    candidates.append([1, 2, 4, 8, 16, 32, 64, 5, 10, 20, 40, 80])
    candidates.append([1, 2, 4, 8, 16, 32, 64, 3, 6, 12, 24, 48])
    
    # Clustered sets (many close together)
    candidates.append(list(range(6)) + list(range(60, 66)))
    candidates.append(list(range(4)) + list(range(20, 24)) + list(range(40, 44)) + list(range(60, 64)))
    
    # Evaluate initial candidates
    best_set = candidates[0]
    best_score = max_sidon_size_fast(best_set)
    
    for cand in candidates[1:]:
        if time.time() > deadline:
            break
        score = max_sidon_size_fast(cand)
        if score < best_score:
            best_score = score
            best_set = cand
    
    # Random search with local improvement
    random.seed(42)
    for i in range(50000):
        if time.time() > deadline:
            break
        
        # Generate random candidate
        candidate = random.sample(range(129), 12)
        score = max_sidon_size_fast(candidate)
        
        if score < best_score:
            best_score = score
            best_set = candidate
            if best_score <= 3:
                break
        
        # Local search: try small perturbations of the best set
        if i % 10 == 0 and best_score > 3:
            # Try swapping one element
            for _ in range(5):
                perturbed = list(best_set)
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in perturbed:
                    perturbed[idx] = new_val
                    perturbed.sort()
                    p_score = max_sidon_size_fast(perturbed)
                    if p_score < best_score:
                        best_score = p_score
                        best_set = perturbed
                        if best_score <= 3:
                            break
    
    return best_set
