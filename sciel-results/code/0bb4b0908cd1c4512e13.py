# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
from itertools import combinations

DEADLINE_SECONDS = 58.0  # safety margin before 60s kill


def _build_dp_state(S):
    """
    Build DP state for set S:
      sums_by_k[k] = set of sums achievable with exactly k elements
    """
    n = len(S)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in S:
        for k in range(n, 0, -1):
            if sums_by_k[k - 1]:
                for s in sums_by_k[k - 1]:
                    sums_by_k[k].add(s + x)
    return sums_by_k


def _check_add_valid_and_growth(sums_by_k, candidate, n):
    """
    Check if candidate can be added and calculate the growth in sum sets.
    Returns (is_valid, growth).
    """
    # Build a map: sum -> set of cardinalities where it appears
    # We can cache this? No, sums_by_k changes every step.
    # But we can build it once per step for all candidates?
    # No, we need it for the check.
    
    sum_to_ks = defaultdict(set)
    for k in range(n + 1):
        for s in sums_by_k[k]:
            sum_to_ks[s].add(k)
            
    is_valid = True
    growth = 0
    
    # We iterate through all existing sums to check collisions and count growth
    # For each k, we generate new sums s+x
    # We need to check if s+x collides with any sum at cardinality j != k+1
    
    for k in range(n + 1):
        src = sums_by_k[k]
        if not src:
            continue
        
        target_set = sums_by_k[k + 1] # The set where these new sums will land
        
        for s in src:
            new_s = s + candidate
            
            # Check validity
            # If new_s is in sum_to_ks, check if it's ONLY at cardinality k+1
            # If it's at any other cardinality, it's a collision.
            if new_s in sum_to_ks:
                if len(sum_to_ks[new_s]) > 1 or (k + 1) not in sum_to_ks[new_s]:
                    # If it's in sum_to_ks but not at k+1, or in multiple places
                    # Wait, if it's in sum_to_ks[k+1], that's fine.
                    # If it's in sum_to_ks[j] where j != k+1, it's bad.
                    
                    # Optimization:
                    # If new_s is in target_set (sums_by_k[k+1]), it's already there.
                    # If it's in some other set, it's a collision.
                    
                    # Let's just check:
                    # Is there any j in sum_to_ks[new_s] such that j != k+1?
                    for j in sum_to_ks[new_s]:
                        if j != k + 1:
                            is_valid = False
                            break
                    if not is_valid:
                        return False, 0
            
            # Calculate growth
            # If new_s is NOT in target_set, it's a new sum added to the set of sums for cardinality k+1
            if new_s not in target_set:
                growth += 1
                
    return is_valid, growth


def _construct_greedy_compact(candidates, deadline):
    """
    Greedy construction using compactness heuristic:
    Pick the candidate that adds the fewest new sums to the DP state.
    """
    current = []
    sums_by_k = [set()]
    sums_by_k[0].add(0)
    
    remaining = list(candidates)
    
    while remaining and time.time() < deadline and len(current) < 60:
        n = len(current)
        
        best_x = None
        min_growth = float('inf')
        
        # Pre-compute sum_to_ks for this step to avoid rebuilding for each candidate?
        # _check_add_valid_and_growth builds it internally. 
        # We can refactor to pass it in.
        
        sum_to_ks = defaultdict(set)
        for k in range(n + 1):
            for s in sums_by_k[k]:
                sum_to_ks[s].add(k)
                
        for x in remaining:
            if time.time() > deadline:
                break
            
            is_valid = True
            growth = 0
            
            for k in range(n + 1):
                src = sums_by_k[k]
                if not src:
                    continue
                
                target_set = sums_by_k[k + 1]
                
                for s in src:
                    new_s = s + x
                    
                    # Check validity
                    if new_s in sum_to_ks:
                        # Check if it appears at any cardinality other than k+1
                        # We know it appears at least at some cardinality.
                        # If it appears at k+1, that's fine.
                        # If it appears at any j != k+1, it's invalid.
                        
                        # Efficient check:
                        # If sum_to_ks[new_s] contains any element other than k+1, it's invalid.
                        # Since sum_to_ks[new_s] is a set, we can check:
                        # if sum_to_ks[new_s] != {k+1} and (k+1) in sum_to_ks[new_s]:
                        #    # It's in k+1 and others -> Invalid? 
                        #    # Wait. If it's in k+1 and j, then the sum is achievable with k+1 and j elements.
                        #    # This violates the condition.
                        #    # So if len(sum_to_ks[new_s]) > 1, it's invalid?
                        #    # No. If it's in {k+1, j}, it's invalid.
                        #    # If it's in {k+1}, it's valid (and not new).
                        #    # If it's in {j} where j != k+1, it's invalid.
                        
                        # So: if any j in sum_to_ks[new_s] is != k+1, it's invalid.
                        # We can check: if sum_to_ks[new_s] - {k+1}:
                        if sum_to_ks[new_s] - {k + 1}:
                            is_valid = False
                            break
                    
                    # Calculate growth
                    if new_s not in target_set:
                        growth += 1
                        
                if not is_valid:
                    break
                    
            if not is_valid:
                continue
                
            if growth < min_growth:
                min_growth = growth
                best_x = x
                
        if best_x is None:
            break
            
        current.append(best_x)
        remaining.remove(best_x)
        sums_by_k = _build_dp_state(current)
        
    return current


def _construct_diverse_sets(deadline):
    """
    Construct multiple diverse sets using different strategies.
    Returns the best set found.
    """
    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with compactness on ascending order
    candidates = list(range(1, 257))
    current = _construct_greedy_compact(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
        
    # Strategy 2: Greedy with compactness on descending order
    candidates = list(range(256, 0, -1))
    current = _construct_greedy_compact(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
        
    # Strategy 3: Random restarts with compactness greedy
    random.seed(42)
    for attempt in range(200): # More restarts since greedy is faster
        if time.time() > deadline:
            break
            
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        current = _construct_greedy_compact(candidates, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            
    return best_set


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    best_set = _construct_diverse_sets(deadline)
    
    # Final: ensure all elements are Python int
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
