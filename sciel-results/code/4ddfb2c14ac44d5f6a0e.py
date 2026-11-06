"""Program for large reciprocal-subsum-free sets using backtracking."""
import time
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin under the 90s wall-clock limit

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _can_add(k, selected):
    """
    Check if we can add k to the selected set.
    Condition: For every x in selected, 1/x must NOT be equal to the sum of reciprocals
    of any non-empty subset of (selected \ {x}) U {k}.
    
    Wait, the condition is: "no reciprocal of a selected element may equal the sum of 
    reciprocals of a subset of the OTHER selected elements."
    
    So if we have set S, for each x in S, 1/x != sum(1/y for y in T) where T is a 
    non-empty subset of S \ {x}.
    
    When we add k to S to form S' = S U {k}:
    1. We must check if 1/k is a sum of reciprocals of a subset of S. 
       (Since k is new, the "other selected elements" for k are exactly S).
    2. For each existing x in S, we must check if 1/x is a sum of reciprocals of a 
       subset of S' \ {x}.
       S' \ {x} = (S \ {x}) U {k}.
       The subsets of S' \ {x} are either subsets of S \ {x} (which were already safe 
       when x was added, assuming we maintained the invariant) OR subsets that include k.
       So we only need to check if 1/x is a sum of reciprocals of a subset T of S \ {x} 
       such that 1/x = sum(1/y for y in T) + 1/k.
       This is equivalent to: sum(1/y for y in T) = 1/x - 1/k.
       Since T must be a subset of S \ {x}, and we know that 1/x is NOT a sum of any 
       subset of S \ {x}, this new condition is distinct.
       
       Actually, a simpler check: 
       Is 1/k a subset sum of S? (For the new element k)
       For each x in S: Is (1/x - 1/k) a subset sum of S \ {x}?
    """
    target_k = Fraction(1, k)
    
    # Check 1: Is 1/k a subset sum of reciprocals of S?
    if _is_subset_sum(target_k, selected):
        return False
        
    # Check 2: For each x in S, is (1/x - 1/k) a subset sum of reciprocals of S \ {x}?
    for x in selected:
        target_x = Fraction(1, x) - Fraction(1, k)
        if target_x <= 0:
            continue # 1/x - 1/k <= 0 implies 1/k >= 1/x => k <= x. 
                     # But we are adding k in descending order? 
                     # If we build in descending order, k < x is not guaranteed if we pick arbitrary.
                     # Let's assume we pick k from a candidate list. 
                     # If 1/x - 1/k <= 0, the sum of positive reciprocals cannot be <= 0.
                     # So if 1/x <= 1/k, i.e., x >= k, then 1/x - 1/k <= 0. 
                     # A sum of positive fractions is always > 0. So this case is automatically safe.
                     # We only need to check if x < k (i.e., 1/x > 1/k).
        
        # Optimization: If 1/x - 1/k is very small, it might not be formable.
        remaining = [y for y in selected if y != x]
        if _is_subset_sum(target_x, remaining):
            return False
            
    return True

def _is_subset_sum(target, elements):
    """
    Check if target can be formed by a sum of reciprocals of a non-empty subset of elements.
    Elements are integers.
    """
    if not elements:
        return False
    
    # Pruning: If sum of all reciprocals < target, return False
    total = sum(Fraction(1, x) for x in elements)
    if total < target:
        return False
        
    # Pruning: If any single element reciprocal equals target, return True
    for x in elements:
        if Fraction(1, x) == target:
            return True
            
    # DFS with memoization? Or just simple DFS with pruning?
    # Given small sizes (max 24 elements), DFS is feasible if pruned well.
    # Sort elements by value descending (reciprocals ascending) to find large sums quickly?
    # Or ascending? 
    # Let's use a set of visited sums to avoid re-computation? 
    # The number of distinct subset sums can be large, but for N=24, it's bounded.
    
    # A more efficient check for small N:
    # Generate all subset sums? 2^24 is too big.
    # But we can prune heavily.
    
    # Let's use a recursive DFS with memoization on (index, current_sum) is hard with Fractions.
    # Instead, let's use a simple recursive search with pruning.
    
    # To speed up, we can check if target is in the set of all subset sums.
    # For N < 20, 2^N is feasible? 2^20 is 1 million. Maybe.
    # But we call this function many times.
    
    # Better: Check specific properties.
    # Since we are adding k in a specific order, we can optimize.
    
    # Let's implement a DFS that tries to match the target.
    # Sort elements by reciprocal magnitude (descending) to fail fast?
    # Actually, if we sort by value ascending (largest reciprocals first), we can prune 
    # if remaining sum < target.
    
    elems = sorted(elements, key=lambda x: x) # Smallest x first -> Largest 1/x first
    n = len(elems)
    
    def dfs(idx, current_sum):
        if current_sum == target:
            return True
        if idx >= n:
            return False
        
        # Pruning: If current_sum + sum(1/elems[idx:]) < target, return False
        # Precompute suffix sums of reciprocals for pruning
        # This is expensive to compute every time. 
        # Alternative: Just pass remaining_sum.
        
        # Let's use a simpler approach: 
        # Since N is small, we can just iterate subsets if N is small enough, 
        # but N can be up to 23.
        
        # Let's use the standard subset sum DFS.
        # Option 1: Include elems[idx]
        new_sum = current_sum + Fraction(1, elems[idx])
        if new_sum <= target: # Since all terms are positive
            if dfs(idx + 1, new_sum):
                return True
        # Option 2: Exclude elems[idx]
        if dfs(idx + 1, current_sum):
            return True
            
        return False
        
    # Optimization: If target > sum of all, return False (done above)
    # If target < min(1/x), return False (done by loop above? No, loop checks equality)
    # Actually, if target < min(1/x for x in elements), then no subset can sum to it 
    # (since min subset sum is min(1/x)).
    min_recip = min(Fraction(1, x) for x in elements)
    if target < min_recip:
        return False
        
    # Memoization might help if called with similar sets, but sets change.
    # Let's just run the DFS.
    return dfs(0, Fraction(0))

def construct():
    """
    Construct the largest subset of [1, 24] that is reciprocal-subsum-free.
    Strategy: Greedy/Backtracking.
    We want to maximize cardinality.
    Larger numbers are generally "safer" to include because their reciprocals are smaller,
    making it harder for them to be the target of a sum, and harder for them to contribute
    significantly to a sum that equals another reciprocal.
    
    However, small numbers are "dangerous" as targets (1/1, 1/2 are large) but also 
    "dangerous" as contributors? 
    Actually, if we include 1, 1/1 = 1. Can 1 be a sum of reciprocals of others? 
    Yes, 1/2+1/3+1/6 = 1. So 1 is hard to keep if 2,3,6 are present.
    
    Heuristic: Try to include as many large numbers as possible.
    Start with all numbers 24 down to 1.
    Remove numbers that cause violations.
    
    Better: Build the set incrementally.
    To maximize size, we should prefer including numbers that are "least likely" to cause conflicts.
    Large numbers are less likely to be the target of a sum (since sum of reciprocals of smaller numbers is larger? No, sum of reciprocals of ANY subset is bounded by sum of all).
    
    Let's try a greedy approach:
    Start with an empty set.
    Iterate candidates from 24 down to 1.
    If adding candidate c to current set S is valid, add it.
    
    Does this maximize cardinality? Not necessarily. 
    Example: Maybe excluding 24 allows us to include 12, 13, 14... 
    But 24 is very safe. 1/24 is small. It's unlikely to be a sum of others.
    And 1/24 as a contributor is small.
    So including 24 is almost always good.
    
    Let's try the greedy descending order first.
    If time permits, try local search (swap) to improve.
    """
    S = []
    candidates = list(range(24, 0, -1)) # 24 down to 1
    
    for c in candidates:
        if _time_remaining() < 5.0:
            break
        if _can_add(c, S):
            S.append(c)
            
    # Current S is a valid set.
    # Try to improve via local search?
    # If we missed some numbers, can we swap?
    # Since we went descending, we included large numbers.
    # Maybe we can add smaller numbers now?
    # The greedy loop above already considered all numbers.
    # If a number was rejected, it was because it conflicted with existing numbers.
    # Can we remove an existing number to allow a smaller one?
    # That would keep cardinality same or reduce it. We want to INCREASE cardinality.
    # So we can only add numbers that were previously rejected?
    # But they were rejected because they conflicted. 
    # If we don't remove anything, we can't add them.
    
    # So the greedy result is a local maximum under "add only".
    # To increase cardinality, we might need to swap: Remove X, Add Y, Z.
    # This is complex.
    
    # Let's check if the greedy set is likely optimal.
    # The observed score 1.583 corresponds to size 19.
    # Greedy descending might get 20 or more.
    
    # Let's try a different order: Random restarts?
    # Or just return the greedy set.
    
    # One optimization: The condition "1/x - 1/k is a subset sum of S\{x}" 
    # is the bottleneck.
    
    # Let's verify the greedy result size.
    return S

# Quick test in head:
# S = []
# c=24: _can_add(24, []) -> True. S=[24]
# c=23: _can_add(23, [24]) -> 
#   1/23 sum of subset of [24]? 1/24 != 1/23. No.
#   x=24: 1/24 - 1/23 < 0. Safe.
#   S=[24, 23]
# ...
# It will likely include almost all large numbers.
# Where it breaks is when small numbers are involved.
# 1/1 = 1. Sum of reciprocals of others must not be 1.
# 1/2 = 0.5. Sum of others must not be 0.5.
#
# Let's run a mental simulation for small numbers.
# If we have {24, 23, ..., 10}, sum of reciprocals is approx integral 10 to 24 of 1/x dx ~ ln(2.4) ~ 0.87.
# 1/9 ~ 0.11. 1/8 ~ 0.12.
# It's possible to include down to 1 or 2.

# To be safe and fast, the greedy approach is a solid baseline improvement over static range(13,25).
