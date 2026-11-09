import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Greedy baseline: smallest valid extension
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)
    
    # Try to extend the greedy baseline with targeted search
    # State: (last_value, frozenset_of_used_sums) -> max additional length achievable
    # But used_sums can be large, so we use a different approach:
    # DFS with memoization on (current_prefix_tuple, frozenset_of_used_sums)
    
    # First, let's try iterative deepening from the greedy prefix
    # to see if we can find a longer valid sequence
    
    # Try local search / hill climbing to improve the sequence
    improved = _local_search_refine(best_seq, deadline)
    if len(improved) > best_len:
        best_seq = improved
        best_len = len(best_seq)
    
    # Try multiple random restarts with greedy
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        seq = _construct_greedy_random(deadline)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            improved = _local_search_refine(best_seq, deadline)
            if len(improved) > best_len:
                best_seq = improved
                best_len = len(improved)
    
    # Final validation
    if not _check_valid(best_seq):
        best_seq = _construct_powers()
    
    return best_seq

def _construct_greedy_smallest():
    """Build longest sequence by always picking the smallest valid next element."""
    seq = []
    used_sums = set()
    
    for _ in range(50):  # max length guard
        # Find smallest x > last such that all new consecutive block sums are unused
        last = seq[-1] if seq else 0
        found = False
        for x in range(last + 1, 257):
            # Compute all new consecutive block sums ending at x
            new_sums = []
            s = 0
            for j in range(len(seq), -1, -1):
                s += seq[j] if j < len(seq) else x
                # Actually, let me recompute properly
                pass
            
            # Recompute: new blocks are [seq[i:] + [x]] for i in 0..len(seq)-1, and [x]
            new_sums = []
            s = x
            new_sums.append(s)
            for j in range(len(seq) - 1, -1, -1):
                s += seq[j]
                new_sums.append(s)
            
            # Check if any new sum is already used
            if any(sm in used_sums for sm in new_sums):
                continue
            
            # All good, add x
            seq.append(x)
            used_sums.update(new_sums)
            found = True
            break
        
        if not found:
            break
    
    return seq

def _construct_greedy_random(deadline):
    """Build sequence with some randomness to explore different branches."""
    seq = []
    used_sums = set()
    
    for _ in range(50):
        last = seq[-1] if seq else 0
        candidates = []
        for x in range(last + 1, 257):
            # Compute new consecutive block sums
            new_sums = []
            s = x
            new_sums.append(s)
            for j in range(len(seq) - 1, -1, -1):
                s += seq[j]
                new_sums.append(s)
            
            if not any(sm in used_sums for sm in new_sums):
                candidates.append(x)
        
        if not candidates:
            break
        
        # Pick randomly from candidates, weighted toward smaller values
        # but with some exploration
        if random.random() < 0.7:
            # Mostly pick small ones
            x = random.choice(candidates[:min(10, len(candidates))])
        else:
            x = random.choice(candidates)
        
        # Add x and its sums
        new_sums = []
        s = x
        new_sums.append(s)
        for j in range(len(seq) - 1, -1, -1):
            s += seq[j]
            new_sums.append(s)
        
        seq.append(x)
        used_sums.update(new_sums)
        
        if time.time() > deadline:
            break
    
    return seq

def _local_search_refine(seq, deadline):
    """Try to extend or improve the sequence via local search."""
    if not _check_valid(seq):
        return seq
    
    best = list(seq)
    best_len = len(best)
    
    # Try to extend by finding a valid next element
    while True:
        if time.time() > deadline:
            break
        
        last = best[-1] if best else 0
        used_sums = _get_all_sums(best)
        
        found = False
        for x in range(last + 1, 257):
            new_sums = []
            s = x
            new_sums.append(s)
            for j in range(len(best) - 1, -1, -1):
                s += best[j]
                new_sums.append(s)
            
            if not any(sm in used_sums for sm in new_sums):
                best.append(x)
                found = True
                break
        
        if not found:
            break
    
    # Try swapping elements to potentially unlock extensions
    for _ in range(100):
        if time.time() > deadline:
            break
        
        candidate = list(best)
        # Try swapping two adjacent elements (if it maintains increasing order)
        if len(candidate) >= 2:
            i = random.randint(0, len(candidate) - 2)
            # Only swap if it maintains strict increase
            if candidate[i] + 1 <= candidate[i+1] - 1:
                candidate[i], candidate[i+1] = candidate[i+1], candidate[i]
                if _check_valid(candidate):
                    # Try to extend
                    extended = _try_extend(candidate, used_sums=None, deadline=deadline)
                    if len(extended) > len(best):
                        best = extended
    
    return best

def _try_extend(seq, used_sums=None, deadline=None):
    """Try to extend a sequence greedily."""
    if used_sums is None:
        used_sums = _get_all_sums(seq)
    
    result = list(seq)
    while True:
        if deadline and time.time() > deadline:
            break
        
        last = result[-1] if result else 0
        found = False
        for x in range(last + 1, 257):
            new_sums = []
            s = x
            new_sums.append(s)
            for j in range(len(result) - 1, -1, -1):
                s += result[j]
                new_sums.append(s)
            
            if not any(sm in used_sums for sm in new_sums):
                result.append(x)
                used_sums.update(new_sums)
                found = True
                break
        
        if not found:
            break
    
    return result

def _get_all_sums(seq):
    """Get all consecutive block sums for a sequence."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _construct_powers():
    """Powers of 2 fallback."""
    seq = []
    s = 0
    x = 1
    while x <= 256:
        seq.append(x)
        s += x
        x = s + 1
    return seq

def _check_valid(seq):
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True
