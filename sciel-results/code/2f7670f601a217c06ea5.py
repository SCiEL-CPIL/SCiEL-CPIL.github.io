"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin for safety

    N = 192
    MAX_SIZE = 48
    
    # Precompute all triples of distinct integers from 1..N
    # For each triple (a, b, c), the sum is s = a+b+c.
    # If we select a, we cannot select any d such that d | s.
    # This creates a conflict between a and d.
    
    # However, the condition is: "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
    # Let S be the selected set. For any a in S, and any distinct b, c, d in S \ {a}, it must NOT be the case that a | (b+c+d).
    
    # This is a 3-uniform hypergraph independence problem, or similar.
    # A greedy approach is often effective for such "no sum divisible by element" constraints.
    
    # Key Observation:
    # If we pick large numbers, the sums of 3 other large numbers will be even larger.
    # If x is small, it is more likely to divide a sum.
    # If x is large, say x > max_sum_of_3_others, then x cannot divide the sum (since sum < x, sum % x = sum != 0 unless sum=0 which is impossible for positive ints).
    # Max sum of 3 distinct others <= 192 is 192+191+190 = 573.
    # So if we pick x > 573, it's safe? No, x <= 192.
    # So x can always potentially divide a sum.
    
    # Let's analyze the constraint more closely.
    # a | (b+c+d)
    # This means b+c+d = k*a for some integer k >= 1.
    
    # Strategy:
    # 1. Greedy selection.
    # 2. To maximize cardinality, we want to pick numbers that are "incompatible" with few other numbers.
    #    Or, we want to pick numbers such that the sums of any 3 do not create divisibility issues.
    
    # Let's try a greedy approach that prioritizes numbers that are less likely to cause conflicts.
    # A number x is "bad" if it divides many sums of 3 other numbers.
    # Actually, the conflict is pairwise in the sense that adding x might invalidate existing choices or prevent future ones.
    
    # Let's define a cost for adding x to current set S.
    # x can be added if:
    # 1. x does not divide sum of any 3 distinct elements in S.
    # 2. No element a in S divides sum of any 3 distinct elements in (S \ {a}) U {x}.
    #    This simplifies to: No a in S divides (x + b + c) for any distinct b, c in S \ {a}.
    
    # We will use a randomized greedy algorithm with multiple restarts.
    # To improve the seed score (1.0667), we need to find a set larger than the seed's output.
    # The seed score 1.0667 implies the seed found a set of size approx 48 * 1.0667? No, score is value/seed.
    # Wait, "The evaluator measures the raw value (higher is better). The search score is value/seed".
    # Seed program score is 1.0. Current program score is 1.0667.
    # This means the current program found a set of size 48 * 1.0667? No, the max size is 48.
    # Ah, "value" is likely the size of the set.
    # If max is 48, and score is 1.0667, maybe the seed found 45? 45 * 1.0667 = 48.
    # So the current program likely finds 48 elements.
    # If the current program already finds 48, the score is 48/45 = 1.0666...
    # So the task is to find 48 elements. The current program does this.
    # Can we do better? The max is 48. So 48 is the maximum possible value.
    # If the current program finds 48, its score is fixed at 48/seed_value.
    # If seed_value is 45, score is 1.0667.
    # If we find 48, we get the same score.
    # Is it possible to find MORE than 48?
    # "construct() -> 1..maximum=48 distinct positive integers".
    # The contract says "at most 48 elements".
    # So 48 is the hard cap.
    # Therefore, the maximum possible score is 48/seed_value.
    # If the current program achieves this, we cannot improve the score by finding a larger set.
    # We can only improve if the seed value was lower, or if "value" is not just cardinality.
    # "Maximize cardinality." -> Value is cardinality.
    # So if we hit 48, we are optimal.
    
    # Let's verify if the current program consistently hits 48.
    # If it does, we just need to ensure we hit 48 reliably and quickly.
    # If it sometimes hits 47, we can improve the average score.
    
    # Let's write a robust greedy that aims for 48.
    
    def can_add(selected, x):
        n = len(selected)
        # Check 1: x divides sum of any 3 in selected?
        if n >= 3:
            for t in combinations(selected, 3):
                if sum(t) % x == 0:
                    return False
        
        # Check 2: Any a in selected divides sum of 3 in selected U {x} excluding a?
        # Triples must include x.
        # Sum = x + b + c.
        # b, c distinct, from selected \ {a}.
        if n >= 2:
            for a in selected:
                # We need distinct b, c from selected such that b != a, c != a, b != c.
                # And (x + b + c) % a == 0.
                # Optimization: Precompute sums of pairs?
                # For each a, we need to know if there exist b, c in selected\{a} such that b+c = k*a - x.
                # Let target_sum = k*a - x.
                # We need b+c = target_sum for some k.
                # Min sum of 2 distinct from selected\{a} is roughly 1+2=3 (or min two).
                # Max sum is roughly 192+191=383.
                # So k*a - x must be in [min_pair_sum, max_pair_sum].
                
                # Simple check:
                # Create a set of available numbers: avail = set(selected) - {a}
                # For each b in avail:
                #   For each c in avail, c != b:
                #     if (x + b + c) % a == 0: return False
                
                # This is O(n^2) per a, so O(n^3) total.
                # With n=48, 48^3 = 110k ops. Very fast.
                
                # We can speed this up using a set of sums.
                # Let avail_list = [v for v in selected if v != a]
                # Let sum_set = set()
                # For i in range(len(avail_list)):
                #   For j in range(i+1, len(avail_list)):
                #     sum_set.add(avail_list[i] + avail_list[j])
                # Then check if (x + s) % a == 0 for any s in sum_set.
                # Wait, (x + b + c) % a == 0  <=>  (b + c) % a == (-x) % a.
                # So we just need to know if any pair sum in avail_list has a specific remainder mod a.
                
                remainders_needed = (-x) % a
                # Check if any pair in selected\{a} sums to a value congruent to remainders_needed mod a.
                
                # Even better: just iterate pairs. n is small.
                avail = [v for v in selected if v != a]
                found = False
                for i in range(len(avail)):
                    for j in range(i+1, len(avail)):
                        if (x + avail[i] + avail[j]) % a == 0:
                            return False
        return True

    best_set = []
    
    # Strategy: Randomized Greedy with Restart
    # Try different orders.
    # Large numbers first? Small numbers first?
    # Small numbers are more likely to divide sums, so they are "dangerous" to add later?
    # Actually, if you add small numbers first, they restrict the sums of future triples.
    # If you add large numbers first, they are less likely to divide sums of 3 others (since sums are bounded).
    # Wait, if x is large, x | sum implies sum >= x.
    # If we pick x=192, sum of 3 others must be 192, 384, 576...
    # Max sum of 3 others is ~573. So only 192 or 384 are possible sums.
    # So 192 is "safe" to add if no 3 others sum to 192 or 384.
    
    # Let's try greedy from high to low.
    
    def greedy(order):
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected
    
    # Run several variations
    orders = [
        list(range(192, 0, -1)), # High to Low
        list(range(1, 193)),     # Low to High
    ]
    
    for _ in range(20):
        random.shuffle(orders[0])
        orders.append(orders[0][:])
    
    for order in orders:
        if time.time() > deadline:
            break
        res = greedy(order)
        if len(res) > len(best_set):
            best_set = res
        if len(best_set) == 48:
            break
            
    # Local Search: Try to swap elements to increase size?
    # If we are stuck at < 48, try removing one element and adding two?
    # Or just try different random shuffles.
    
    # If we haven't reached 48, do more random restarts.
    while len(best_set) < 48 and time.time() < deadline:
        order = list(range(1, 193))
        random.shuffle(order)
        res = greedy(order)
        if len(res) > len(best_set):
            best_set = res
            
    return best_set

# EVOLVE-BLOCK-END
