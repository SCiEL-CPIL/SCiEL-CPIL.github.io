"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _max_sidon_subset_size(arr):
    """Find the maximum size of a Sidon subset of arr (all pair sums a_i+a_j for i<=j unique)."""
    n = len(arr)
    arr_sorted = sorted(arr)
    best = [0]
    
    # Precompute all pair sums for quick lookup
    # For a candidate subset, we need all sums s_i + s_j for i <= j to be distinct
    
    def backtrack(start, current, current_sums):
        if len(current) > best[0]:
            best[0] = len(current)
        
        # Pruning: even if we add all remaining, can we beat best?
        remaining = n - start
        if len(current) + remaining <= best[0]:
            return
        
        for i in range(start, n):
            val = arr_sorted[i]
            # Check if adding val keeps Sidon property
            valid = True
            new_sums = []
            for j in range(len(current)):
                s = val + current[j]
                if s in current_sums:
                    valid = False
                    break
                new_sums.append(s)
            if not valid:
                continue
            # Also check val + val
            if val + val in current_sums:
                valid = False
            else:
                new_sums.append(val + val)
            
            if valid:
                # Check new_sums among themselves
                if len(set(new_sums)) != len(new_sums):
                    continue
                # Also check new_sums against existing sums
                all_sums = current_sums | set(new_sums)
                if len(all_sums) != len(current_sums) + len(new_sums):
                    continue
                
                current.append(val)
                backtrack(i + 1, current, all_sums)
                current.pop()
    
    backtrack(0, [], set())
    return best[0]


def _max_sidon_subset_size_fast(arr):
    """Faster version using bitmasks and greedy + backtracking hybrid."""
    n = len(arr)
    if n <= 1:
        return n
    
    arr_sorted = sorted(arr)
    best = [0]
    
    # Try all subsets in decreasing size order, stop at first Sidon
    # For n=12, we can try sizes from 12 down to 1
    # But that's 2^12 = 4096 subsets max, manageable
    
    # Better: backtracking with good pruning
    def backtrack(start, current, sum_set):
        if len(current) > best[0]:
            best[0] = len(current)
        
        remaining = n - start
        if len(current) + remaining <= best[0]:
            return
        
        for i in range(start, n):
            val = arr_sorted[i]
            # Check all new sums
            new_sums = [val + val]
            for j in range(len(current)):
                new_sums.append(val + current[j])
            
            # Check for duplicates in new_sums
            if len(set(new_sums)) != len(new_sums):
                continue
            
            # Check against existing
            for s in new_sums:
                if s in sum_set:
                    break
            else:
                new_sum_set = sum_set | set(new_sums)
                current.append(val)
                backtrack(i + 1, current, new_sum_set)
                current.pop()
    
    backtrack(0, [], set())
    return best[0]


def _count_sidon_conflicts(arr):
    """Count the number of distinct sum collisions in the full set.
    Higher means the set is 'harder' to extract a large Sidon subset from."""
    n = len(arr)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                # collision
                pass
            else:
                sums[s] = 1
    # Number of sums that appear more than once
    total_pairs = n * (n + 1) // 2
    unique_sums = len(sums)
    return total_pairs - unique_sums


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    best_set = list(range(12))
    best_score = _max_sidon_subset_size_fast(best_set)  # should be 12
    
    # Also track the "hardness" metric (number of sum collisions)
    best_conflicts = _count_sidon_conflicts(best_set)
    
    random.seed(42)
    
    def evaluate(arr):
        """Return (max_sidon_size, conflict_count)."""
        ms = _max_sidon_subset_size_fast(arr)
        conf = _count_sidon_conflicts(arr)
        return ms, conf
    
    # Local search with simulated annealing
    current_set = list(range(12))
    current_ms, current_conf = evaluate(current_set)
    
    # We want to MINIMIZE max_sidon_size
    # Primary objective: minimize max_sidon_size
    # Secondary: maximize conflicts (harder to find Sidon subsets)
    
    temp = 1.0
    cooling = 0.999
    
    iteration = 0
    no_improve = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # Try a swap: replace one element with a random other value
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        new_val = random.randint(0, 128)
        
        if new_val == old_val:
            continue
        
        # Check if new_val is already in the set
        if new_val in current_set:
            continue
        
        # Make the swap
        current_set[idx] = new_val
        
        new_ms, new_conf = evaluate(current_set)
        
        # Accept if better (lower max_sidon_size), or with some probability if worse
        if new_ms < current_ms or (new_ms == current_ms and new_conf > current_conf):
            current_ms = new_ms
            current_conf = new_conf
            no_improve = 0
            if current_ms < best_score or (current_ms == best_score and current_conf > best_conflicts):
                best_score = current_ms
                best_conflicts = current_conf
                best_set = list(current_set)
        else:
            # Reject, revert
            current_set[idx] = old_val
            no_improve += 1
            # Occasionally accept worse to escape local minima
            delta = new_ms - current_ms  # positive means worse
            if delta > 0 and random.random() < (1.0 / (1.0 + delta * 2.0)):
                current_set[idx] = new_val
                current_ms = new_ms
                current_conf = new_conf
        
        temp *= cooling
        
        # Periodic restart if stuck
        if no_improve > 500:
            # Try a more aggressive perturbation
            for _ in range(5):
                idx2 = random.randint(0, 11)
                old2 = current_set[idx2]
                new2 = random.randint(0, 128)
                if new2 not in current_set and new2 != old2:
                    current_set[idx2] = new2
            current_ms, current_conf = evaluate(current_set)
            no_improve = 0
    
    return best_set
# EVOLVE-BLOCK-END
