"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_DEADLINE = 50.0  # seconds, safety margin before 60s limit

def _check_violation_fast(candidate, a_idx):
    """
    Check if element at a_idx is the mean of any subset of at least 2 OTHER elements.
    Returns True if violation exists.
    
    For element a, others = candidate without a.
    Need: exists subset S of others, |S|>=2, such that sum(S) = |S|*a.
    Equivalent: exists subset S of {y-a for y in others}, |S|>=2, sum(S)=0.
    """
    a = candidate[a_idx]
    others = candidate[:a_idx] + candidate[a_idx+1:]
    n = len(others)
    if n < 2:
        return False
    
    # Transform: z_i = y_i - a
    zs = [y - a for y in others]
    
    # DP: dp[k] = set of achievable sums using exactly k elements from zs
    # But sets can be large. Use a different approach for small n.
    
    # For k=2: check all pairs
    for i in range(n):
        for j in range(i+1, n):
            if zs[i] + zs[j] == 0:
                return True
    
    # For k>=3: use DP with sets, but limit size
    # dp[k] = set of sums
    dp = [set() for _ in range(n+1)]
    dp[0].add(0)
    
    for z in zs:
        # Update backwards
        for k in range(n, 0, -1):
            if dp[k-1]:
                new_sums = set()
                for s in dp[k-1]:
                    ns = s + z
                    new_sums.add(ns)
                dp[k] |= new_sums
        # Early exit: check if 0 is in dp[k] for any k>=3
        # Only check after processing all elements for correctness
        # Actually, we need to check after all elements are processed
    
    # Check all k >= 3
    for k in range(3, n+1):
        if 0 in dp[k]:
            return True
    
    return False


def _full_check(candidate):
    """Check all elements for violations."""
    for i in range(len(candidate)):
        if _check_violation_fast(candidate, i):
            return True
    return False


def construct():
    start_time = time.time()
    
    # Strategy: Greedy construction with smart ordering
    # Try to build a large set by adding elements that are "hard" to be means
    # 
    # Key insight: If we use elements that are spaced out in a non-arithmetic way,
    # it's harder to form means. 
    #
    # Approach: Try all numbers 1-512 in a specific order, add if valid.
    # Order: Try to maximize density. One good order is to alternate or use
    # a pattern that avoids arithmetic progressions.
    #
    # Better: Use a two-phase approach.
    # Phase 1: Greedy add in increasing order with fast checking
    # Phase 2: Try to add more by rearranging or using different order
    
    selected = []
    
    # Try multiple orders and keep the best
    orders = [
        list(range(1, 513)),                          # increasing
        list(range(512, 0, -1)),                      # decreasing
        list(range(256, 513)) + list(range(1, 256)),  # upper then lower
        list(range(1, 513, 2)) + list(range(2, 513, 2)),  # odds then evens
    ]
    
    best_set = []
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        selected = []
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            candidate = selected + [x]
            
            # Quick check: only check if x is mean of others (new element)
            # AND if any existing element becomes mean due to x
            # For efficiency, first check if x is mean of subset of selected
            # Then check if any selected element is mean of subset including x
            
            # Fast path: check x against selected
            if selected:
                # Check if x is mean of any subset of selected with size >= 2
                # Transform: need subset S of selected, |S|>=2, sum(S) = |S|*x
                # i.e., sum(S - x) = 0 where S-x means {s-x for s in S}
                # But this is the same as _check_violation_fast but with x as target
                # Let's just do the full check on candidate but optimize
                
                # Quick reject: if x < min(selected) or x > max(selected),
                # x can still be mean if there are elements on both sides
                
                # Do the check
                if _check_violation_fast(candidate, len(selected)):
                    continue
                
                # Also need to check that existing elements aren't now means
                # This is expensive. For now, do full check if set is small
                if len(selected) <= 15:
                    if _full_check(candidate):
                        continue
                else:
                    # For larger sets, trust that if x wasn't a mean and 
                    # we've been careful, existing elements are likely fine.
                    # But this is risky. Let's do a partial check.
                    # Check a few existing elements
                    for idx in [0, len(selected)//2, len(selected)-1]:
                        if _check_violation_fast(candidate, idx):
                            break
                    else:
                        # No violation found in sampled elements
                        pass
                    # For safety, do full check if we're close to deadline
                    if time.time() - start_time < _DEADLINE - 10:
                        if _full_check(candidate):
                            continue
            
            selected.append(x)
            
            if len(selected) >= 48:
                break
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Final verification of best_set
    if best_set and not _full_check(best_set):
        return best_set
    else:
        # Return a known valid small set as fallback
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
