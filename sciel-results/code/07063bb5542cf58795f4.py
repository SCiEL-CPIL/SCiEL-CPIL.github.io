"""
Strategy: Structural Decomposition via 2-adic Valuation
1. Partition candidates [2, 256] based on v2(n) (exponent of 2 in prime factorization).
2. Identify "High" numbers: v2(n) >= 8 (i.e., divisible by 256). Only n=256.
   Actually, let's look at the constraint: lcm(a,b) > 256.
   If a and b are both odd, lcm(a,b) = a*b. If a,b <= 256, a*b can be <= 256.
   If v2(a) >= 1 and v2(b) >= 1, lcm(a,b) has v2 >= 1.
   
   Key Insight:
   If we pick a number with high v2, it is "safe" with many numbers.
   If we pick a number with low v2 (e.g., odd), it restricts other numbers significantly.
   
   Let's refine the partition:
   Group 1: Numbers with v2(n) >= 8. Only 256.
   Group 2: Numbers with v2(n) == 7. 128.
   Group 3: Numbers with v2(n) == 6. 64, 192.
   ...
   
   A more robust structural approach:
   The condition lcm(a,b) <= 256 is the "forbidden" case.
   lcm(a,b) = a*b / gcd(a,b).
   Forbidden if a*b / gcd(a,b) <= 256.
   
   This problem is equivalent to finding a maximum weight independent set in a graph where edges connect incompatible pairs.
   However, the graph has a specific structure related to divisibility.
   
   Alternative Structural Insight:
   If a number x is in the set, any other number y must satisfy lcm(x,y) > 256.
   This implies y > 256 / gcd(x,y).
   
   Let's try a hybrid approach:
   1. Enumerate subsets of "small" numbers (low value, high constraint).
      Numbers <= 16 are very restrictive.
      2: requires others to be odd and > 128.
      3: requires others to be not divisible by 3 and > 85 (if gcd=1) or > 256/3 (impossible) -> if gcd=3, lcm=y, so y>256 (impossible). So if 3 is in S, no multiple of 3 can be in S. And lcm(3,y)=3y/gcd(3,y). If 3|y, lcm=y<=256 (bad). If 3!|y, lcm=3y. 3y>256 => y>85.
      
   So, including small numbers drastically reduces the candidate pool.
   
   Plan:
   1. Identify "Core" candidates: numbers <= 32. There are 31 of them.
      The choice of which core numbers to include determines the "allowed" set of larger numbers.
      2^31 is too large to enumerate.
      
   Refined Plan:
   Use the fact that if we include a number x, it forbids a specific set of numbers.
   We can use a recursive search that prioritizes "critical" numbers (small ones) but uses a smarter bound.
   
   Actually, let's look at the "High" numbers again.
   If we pick NO numbers from [2, 32], what is the best set?
   We can pick any subset of [33, 256] such that pairwise lcm > 256.
   For a,b in [33, 256], lcm(a,b) <= 256 is rare.
   Example: 33 and 36. gcd=3. lcm=396 > 256. OK.
   34 and 34 (distinct? no).
   33 and 33 (no).
   33 and 99. gcd=33. lcm=99 <= 256. BAD.
   So we can't just pick all large numbers.
   
   However, the number of "bad" pairs among large numbers is much smaller than among small numbers.
   
   Let's try a different structural decomposition:
   Sort by v2(n).
   If v2(a) >= v2(b), then v2(lcm(a,b)) = v2(a).
   lcm(a,b) > 256 => (a/gcd(a,b)) * b > 256.
   
   Let's implement a B&B that sorts by "constraint strength" (inverse of value density of constraints).
   Small numbers are high value but high constraint.
   Large numbers are low value but low constraint.
   
   The previous B&B sorted by x ascending. This is good for value, but bad for pruning because it adds constraints early.
   If we sort by x DESCENDING, we add low-value, low-constraint items first. This builds a large base set quickly, allowing the upper bound to be tightened?
   No, the upper bound is sum of remaining values. If we pick low value first, the remaining sum is still high.
   
   Let's try a "Core Enumeration" approach.
   Let C be the set of numbers <= 16.
   There are 15 such numbers. 2^15 = 32768 subsets.
   For each subset S_C of C:
     1. Check if S_C is internally valid.
     2. Determine the set of "allowed" numbers from [17, 256] that are compatible with all in S_C.
     3. Solve the remaining problem on [17, 256] \ Forbidden.
        The remaining problem is much easier because:
        - All numbers are >= 17.
        - The constraint lcm(a,b) > 256 is easier to satisfy.
        - The number of candidates is ~240.
        - But we can use a simple greedy or a fast B&B on the remainder.
        Actually, for numbers >= 17, the "bad" pairs are those where one divides the other or they share a large factor.
        We can use a maximum weight independent set algorithm (or B&B) on this reduced graph.
        Since the graph is sparse, B&B will be fast.
        
   This is a structural change: Decoupling the "hard" small numbers from the "easy" large numbers via enumeration of the small ones.
"""

import time
from math import gcd
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # Candidates 2..256
    all_candidates = list(range(2, 257))
    
    # Split into Core (small, high constraint) and Rest (large, low constraint)
    # Threshold: 16. Numbers <= 16 are 2..16 (15 numbers).
    core_candidates = [x for x in all_candidates if x <= 16]
    rest_candidates = [x for x in all_candidates if x > 16]
    
    best_global_sum = Fraction(0)
    best_global_set = []
    
    # Precompute compatibility for all pairs? 
    # 255x255 matrix is fine.
    n = 255
    # Map value to index for faster lookup? Or just use values directly.
    # Let's use a set for the current core selection to check compatibility.
    
    def is_compatible(a, b):
        return (a * b) // gcd(a, b) > 256

    # Enumerate all subsets of core_candidates
    # There are 2^15 = 32768 subsets.
    # For each subset, check validity.
    # If valid, determine which rest_candidates are compatible with the core subset.
    # Then solve the Max Weight Independent Set on the induced subgraph of compatible rest_candidates.
    
    # To speed up the "solve rest" part, we can use a greedy heuristic first to get a good lower bound,
    # then B&B if time permits. Or just a very fast B&B.
    
    # Optimization: 
    # If core_subset is invalid, skip.
    # If core_subset is valid, the set of compatible rest candidates R' is:
    # R' = { y in rest_candidates | is_compatible(y, x) for all x in core_subset }
    
    # The problem on R' is: Maximize sum(1/y) subject to pairwise lcm > 256.
    # Note: The constraint among elements in R' is the same as the original problem.
    
    # Let's implement a fast B&B for the rest part.
    # Sort rest_candidates by 1/x descending (x ascending).
    # Actually, for the rest part, x >= 17.
    
    # To make the inner B&B fast, we can pre-compute the adjacency of incompatibility for rest_candidates.
    # But R' changes for each core_subset.
    
    # Alternative: 
    # Instead of enumerating all 2^15 core subsets, we can do a joint B&B but with a better ordering?
    # The previous B&B was too slow. Why? 
    # Probably because the branching factor was high and pruning was weak.
    # With Core Enumeration, the inner problem is on ~240 nodes, but the graph is sparser?
    # Not necessarily sparser, but the "bad" edges are fewer relative to total?
    # Actually, for x,y >= 17, lcm(x,y) <= 256 implies x*y/gcd(x,y) <= 256.
    # Since x,y >= 17, x*y >= 289.
    # So gcd(x,y) must be at least 289/256 ~ 1.13. So gcd >= 2.
    # So x and y must share a factor >= 2.
    # This means the incompatibility graph on [17, 256] only has edges between numbers sharing a factor.
    # This is a MUCH sparser graph than the full [2, 256] graph.
    # This suggests that B&B on the rest part will be very fast.
    
    # Let's proceed with Core Enumeration.
    
    # Precompute suffix sums for rest_candidates for B&B upper bound?
    # No, R' is a subset of rest_candidates. We need to compute the sum of available candidates.
    
    # Let's refine the inner solver.
    def solve_rest(available_rest):
        # available_rest is a list of ints
        # Sort by value (1/x) descending -> x ascending
        # available_rest is already > 16.
        # We want to maximize sum(1/x).
        # B&B on this set.
        
        if not available_rest:
            return Fraction(0), []
            
        # Sort by x ascending
        items = sorted(available_rest)
        
        best_sum = Fraction(0)
        best_set = []
        
        # Precompute suffix sums of 1/x for the sorted items
        n_items = len(items)
        suffix_sums = [Fraction(0)] * (n_items + 1)
        for i in range(n_items - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i+1] + Fraction(1, items[i])
            
        # B&B
        # To speed up, we can use a bitset or simple list for current set.
        # Since n_items <= 240, recursion depth is manageable.
        
        def bnb(idx, current_set, current_sum):
            nonlocal best_sum, best_set
            
            if time.time() > deadline:
                return
                
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(current_set)
                
            if idx == n_items:
                return
                
            # Prune
            if current_sum + suffix_sums[idx] <= best_sum:
                return
                
            x = items[idx]
            
            # Check compatibility with current_set
            can_add = True
            for y in current_set:
                if not is_compatible(x, y):
                    can_add = False
                    break
                    
            if can_add:
                bnb(idx + 1, current_set + [x], current_sum + Fraction(1, x))
                
            # Exclude
            bnb(idx + 1, current_set, current_sum)
            
        bnb(0, [], Fraction(0))
        return best_sum, best_set

    # Enumerate core subsets
    # Use recursion to generate subsets of core_candidates
    def enumerate_core(idx, current_core, current_core_sum):
        nonlocal best_global_sum, best_global_set
        
        if time.time() > deadline:
            return
            
        # Update global best if current core is valid and we assume empty rest?
        # No, we solve rest later.
        
        if idx == len(core_candidates):
            # Check if current_core is internally valid
            valid = True
            for i in range(len(current_core)):
                for j in range(i+1, len(current_core)):
                    if not is_compatible(current_core[i], current_core[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Determine compatible rest candidates
                compatible_rest = []
                for y in rest_candidates:
                    ok = True
                    for x in current_core:
                        if not is_compatible(x, y):
                            ok = False
                            break
                    if ok:
                        compatible_rest.append(y)
                        
                # Solve rest
                rest_sum, rest_set = solve_rest(compatible_rest)
                total_sum = current_core_sum + rest_sum
                total_set = current_core + rest_set
                
                if total_sum > best_global_sum:
                    best_global_sum = total_sum
                    best_global_set = total_set
                    
            return
            
        # Branch 1: Include core_candidates[idx]
        x = core_candidates[idx]
        # Check compatibility with existing core
        can_add = True
        for y in current_core:
            if not is_compatible(x, y):
                can_add = False
                break
                
        if can_add:
            enumerate_core(idx + 1, current_core + [x], current_core_sum + Fraction(1, x))
            
        # Branch 2: Exclude
        enumerate_core(idx + 1, current_core, current_core_sum)

    # Start enumeration
    # To improve performance, we can order core_candidates by "constraint impact" or just value.
    # Small numbers are more restrictive.
    # Let's sort core_candidates by x ascending.
    core_candidates.sort()
    
    enumerate_core(0, [], Fraction(0))
    
    return best_global_set

# Test/Debug
if __name__ == "__main__":
    res = construct()
    print(f"Result: {res}")
    print(f"Sum: {sum(1/x for x in res)}")
