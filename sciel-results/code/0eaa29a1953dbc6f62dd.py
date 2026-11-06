"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _is_valid(s_sorted):
    """Check if sorted list s_sorted satisfies the property."""
    n = len(s_sorted)
    for i in range(n):
        a = s_sorted[i]
        for j in range(i + 1, n):
            b = s_sorted[j]
            target = (-b) % a
            for k in range(j, n):
                c = s_sorted[k]
                if c % a == target:
                    return False
    return True

def _count_conflicts(s_sorted):
    """Count conflicts for each element in sorted set."""
    n = len(s_sorted)
    conflicts = [0] * n
    for i in range(n):
        a = s_sorted[i]
        for j in range(i + 1, n):
            b = s_sorted[j]
            target = (-b) % a
            for k in range(j, n):
                c = s_sorted[k]
                if c % a == target:
                    conflicts[i] += 1
                    conflicts[j] += 1
                    conflicts[k] += 1
    return conflicts

def _greedy_upper_tail(start_val, time_limit=5.0):
    """Start from upper tail and greedily add smaller elements."""
    start_time = time.time()
    s = list(range(start_val, N + 1))
    s.sort()
    s_set = set(s)
    
    # Try to add elements from start_val-1 down to 1
    for candidate in range(start_val - 1, 0, -1):
        if time.time() - start_time > time_limit:
            break
        if candidate in s_set:
            continue
        # Check if adding candidate creates a conflict
        new_s = s + [candidate]
        new_s.sort()
        if _is_valid(new_s):
            s = new_s
            s_set.add(candidate)
    
    return s

def _local_search(s, time_limit=10.0):
    """Local search: try swaps and additions."""
    start_time = time.time()
    best = list(s)
    current = list(s)
    current.sort()
    best_len = len(best)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to add any element
        for candidate in range(1, N + 1):
            if candidate in current:
                continue
            new_s = current + [candidate]
            new_s.sort()
            if _is_valid(new_s):
                current = new_s
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try single swap: remove one, add another
        if len(current) > 0:
            removed_idx = random.randint(0, len(current) - 1)
            removed = current.pop(removed_idx)
            current.sort()
            
            added = False
            for candidate in range(1, N + 1):
                if candidate == removed or candidate in current:
                    continue
                new_s = current + [candidate]
                new_s.sort()
                if _is_valid(new_s) and len(new_s) >= len(current):
                    current = new_s
                    if len(current) > best_len:
                        best = list(current)
                        best_len = len(best)
                    added = True
                    break
            
            if not added:
                # Revert
                current.append(removed)
                current.sort()
    
    return best

def _multi_start_greedy(time_budget=30.0):
    """Multi-start greedy with different upper tail starts."""
    start_time = time.time()
    best = []
    best_len = 0
    
    # Try different starting points for upper tail
    start_vals = [192, 180, 170, 160, 150, 140, 130, 120, 110, 100]
    
    for sv in start_vals:
        if time.time() - start_time > time_budget:
            break
        remaining = time_budget - (time.time() - start_time)
        if remaining < 3:
            break
        s = _greedy_upper_tail(sv, time_limit=min(remaining * 0.5, 5.0))
        if len(s) > best_len:
            best = s
            best_len = len(s)
    
    return best

def _randomized_search(time_budget=20.0):
    """Randomized restart with local search."""
    start_time = time.time()
    best = []
    best_len = 0
    rng = random.Random(42)
    
    while time.time() - start_time < time_budget:
        remaining = time_budget - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Random pool from upper half
        pool_size = rng.randint(60, 140)
        pool = list(range(80, N + 1))
        rng.shuffle(pool)
        candidate = sorted(pool[:pool_size])
        
        # Resolve conflicts by removing high-conflict elements
        max_iters = 50
        for _ in range(max_iters):
            conflicts = _count_conflicts(candidate)
            if not any(c > 0 for c in conflicts):
                break
            # Remove element with highest conflict
            max_idx = max(range(len(candidate)), key=lambda i: conflicts[i])
            candidate.pop(max_idx)
        
        # Expand
        candidate = _greedy_upper_tail(1, time_limit=min(2.0, remaining * 0.3))
        # Actually, let's do a proper expansion from scratch
        s = list(candidate)
        s.sort()
        s_set = set(s)
        for cand in range(1, N + 1):
            if cand in s_set:
                continue
            new_s = s + [cand]
            new_s.sort()
            if _is_valid(new_s):
                s = new_s
                s_set.add(cand)
        
        if len(s) > best_len:
            best = s
            best_len = len(s)
        
        # Local search on best
        if remaining > 8:
            improved = _local_search(best, time_limit=min(remaining * 0.4, 8.0))
            if len(improved) > best_len:
                best = improved
                best_len = len(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = 85.0
    
    random.seed(42)
    
    # Phase 1: Multi-start greedy
    best = _multi_start_greedy(time_budget=25.0)
    best_len = len(best)
    
    if time.time() - start_time > deadline:
        return best
    
    # Phase 2: Local search on best
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        improved = _local_search(best, time_limit=min(remaining * 0.5, 15.0))
        if len(improved) > best_len:
            best = improved
            best_len = len(best)
    
    if time.time() - start_time > deadline:
        return best
    
    # Phase 3: Randomized search
    remaining = deadline - (time.time() - start_time)
    if remaining > 15:
        rand_best = _randomized_search(time_budget=min(remaining * 0.7, 25.0))
        if len(rand_best) > best_len:
            best = rand_best
            best_len = len(best)
    
    # Final: try to add any remaining elements
    remaining = deadline - (time.time() - start_time)
    if remaining > 3:
        s = list(best)
        s.sort()
        s_set = set(s)
        for cand in range(1, N + 1):
            if time.time() - start_time > deadline - 2:
                break
            if cand in s_set:
                continue
            new_s = s + [cand]
            new_s.sort()
            if _is_valid(new_s):
                s = new_s
                s_set.add(cand)
        if len(s) > best_len:
            best = s
    
    return best
# EVOLVE-BLOCK-END
