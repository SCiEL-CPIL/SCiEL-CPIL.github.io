"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

_DEADLINE = 55.0  # seconds, safety margin before 60s limit


def _check_element_violation(candidate, a_idx):
    """
    Check if element at a_idx is the mean of any subset of at least 2 OTHER elements.
    Returns True if violation exists.
    
    For element a, others = candidate without a.
    Need: exists subset S of others, |S|>=2, such that sum(S) = |S|*a.
    Equivalent: exists subset S of {y-a for y in others}, |S|>=2, sum(S)=0.
    
    Uses bitset DP: dp[k] is an integer where bit s is set if sum s is achievable
    with exactly k elements. We offset sums by a constant to handle negatives.
    """
    a = candidate[a_idx]
    others = candidate[:a_idx] + candidate[a_idx+1:]
    n = len(others)
    if n < 2:
        return False
    
    # Transform: z_i = y_i - a
    zs = [y - a for y in others]
    
    # For k=2: check all pairs directly (fast)
    for i in range(n):
        for j in range(i+1, n):
            if zs[i] + zs[j] == 0:
                return True
    
    # For k>=3: use DP with sets (simpler than bitset for variable ranges)
    # dp[k] = set of achievable sums using exactly k elements from zs
    dp = [set() for _ in range(n+1)]
    dp[0].add(0)
    
    for z in zs:
        for k in range(n, 0, -1):
            if dp[k-1]:
                # Add z to all sums in dp[k-1]
                for s in dp[k-1]:
                    dp[k].add(s + z)
        # Early exit: check if 0 is in dp[k] for any k>=3 after each element
        # Actually, we need to check after processing all elements for correctness
        # because we're building up incrementally
    
    # Check all k >= 3
    for k in range(3, n+1):
        if 0 in dp[k]:
            return True
    
    return False


def _full_check(candidate):
    """Check all elements for violations."""
    for i in range(len(candidate)):
        if _check_element_violation(candidate, i):
            return True
    return False


def _quick_pair_check(candidate, a_idx):
    """Quick check: is element at a_idx the mean of any 2-element subset of others?"""
    a = candidate[a_idx]
    others = candidate[:a_idx] + candidate[a_idx+1:]
    oset = set(others)
    for y in others:
        complement = 2 * a - y
        if complement in oset and complement != y:
            return True
    return False


def construct():
    start_time = time.time()
    
    selected = []
    best_set = []
    
    # Try multiple orders to find the largest valid set
    orders = [
        list(range(1, 513)),                          # increasing
        list(range(512, 0, -1)),                      # decreasing
        list(range(256, 513)) + list(range(1, 256)),  # upper then lower
        list(range(1, 513, 2)) + list(range(2, 513, 2)),  # odds then evens
    ]
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        selected = []
        for x in order:
            if time.time() - start_time > _DEADLINE - 10:
                break
            
            candidate = selected + [x]
            new_idx = len(selected)
            
            # Quick reject: check if x is mean of any pair in selected
            if len(selected) >= 2:
                if _quick_pair_check(candidate, new_idx):
                    continue
            
            # For small sets, do full verification of all elements
            if len(candidate) <= 12:
                if _full_check(candidate):
                    continue
            else:
                # For larger sets, check new element fully and spot-check existing
                if _check_element_violation(candidate, new_idx):
                    continue
                
                # Spot check a few existing elements
                check_indices = [0, len(selected)//2, len(selected)-1]
                violation_found = False
                for idx in check_indices:
                    if idx != new_idx and _check_element_violation(candidate, idx):
                        violation_found = True
                        break
                
                if violation_found:
                    continue
                
                # If we have time, do a more thorough check
                if time.time() - start_time < _DEADLINE - 15 and len(candidate) <= 20:
                    if _full_check(candidate):
                        continue
            
            selected.append(x)
            
            if len(selected) >= 48:
                break
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Final verification of best_set
    if best_set:
        if _full_check(best_set):
            # Fallback: return a known valid set
            return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        else:
            return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# EVOLVE-BLOCK-END
