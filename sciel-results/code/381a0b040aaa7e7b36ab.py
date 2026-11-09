"""Rewritten program for large interval sets avoiding a prescribed subset sum."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,100] with no subset summing to 200.
    Maximize cardinality.
    """
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Greedy approach with bitset for reachable sums
    # We want to maximize the number of elements from [1,100] 
    # such that no subset sums to 200.
    
    # Strategy: Use a bitset (Python int as bitmask) to track reachable sums.
    # Process numbers in an order that helps maximize inclusion.
    # Key insight: If we include numbers > 100, they can't be part of a sum to 200 
    # with other positive numbers (since min other is 1, max sum with one >100 
    # would need another number >= 200-x which is < 100, but we're limited to [1,100]).
    # Actually, numbers in [1,100] can combine to 200. Numbers > 100 cannot appear 
    # in [1,100], so all our candidates are in [1,100].
    
    # Let's use a greedy approach: try to include as many numbers as possible.
    # A number x can be included if adding it to any subset of already-included 
    # numbers doesn't create sum 200.
    
    # Use bitset: bit i set means sum i is reachable.
    # Initially, only sum 0 is reachable.
    
    # Heuristic: Process numbers in a specific order. 
    # If we process larger numbers first, they have fewer combinations that sum to 200.
    # Alternatively, process in an order that minimizes conflict.
    
    # Let's try: include all numbers, then remove the minimum set that breaks all 200-sums.
    # But that's a hitting set problem, which is NP-hard.
    
    # Better greedy: iterate through numbers, include if possible.
    # The order matters. Let's try descending order first (larger numbers are "safer" 
    # in the sense that fewer combinations of smaller numbers can complement them to 200).
    
    def greedy(order):
        reachable = 1  # bit 0 set, sum 0 reachable
        selected = []
        for x in order:
            # Check if adding x creates sum 200
            # New sums would be {s + x for s in reachable}
            # We need to check if any s + x == 200, i.e., s == 200 - x
            # So check if bit (200 - x) is set in reachable
            if 200 - x <= 0:
                # x > 200, can't be part of sum 200 with non-negative sums
                # But x is in [1,100], so 200-x is in [100, 199], always positive
                pass
            if (reachable >> (200 - x)) & 1:
                # Adding x would create sum 200, skip
                continue
            # Include x
            selected.append(x)
            # Update reachable: new sums are old sums + x
            # We only care about sums up to 200
            reachable |= (reachable << x)
            # Mask to keep only bits 0..200
            reachable &= (1 << 201) - 1
        return selected
    
    # Try different orderings
    best = []
    
    # Order 1: descending
    order1 = list(range(100, 0, -1))
    res1 = greedy(order1)
    if len(res1) > len(best):
        best = res1
    
    # Order 2: ascending
    order2 = list(range(1, 101))
    res2 = greedy(order2)
    if len(res2) > len(best):
        best = res2
    
    # Order 3: even numbers first, then odd
    order3 = list(range(2, 101, 2)) + list(range(1, 100, 2))
    res3 = greedy(order3)
    if len(res3) > len(best):
        best = res3
    
    # Order 4: numbers > 100/2 first (i.e., 51..100), then 1..50
    order4 = list(range(100, 50, -1)) + list(range(50, 0, -1))
    res4 = greedy(order4)
    if len(res4) > len(best):
        best = res4
    
    # Order 5: try to include numbers that are "hard" to combine first
    # Numbers near 100 are hard to combine to 200 with other [1,100] numbers
    # since they need a complement near 100, which is also in the set
    # Let's try: 100, 99, 98, ..., then 1, 2, 3, ...
    # Actually let's try a few more heuristics
    
    # Order 6: multiples of 2 first
    order6 = list(range(2, 101, 2)) + list(range(1, 100, 2))
    res6 = greedy(order6)
    if len(res6) > len(best):
        best = res6
    
    # Order 7: try to maximize by including numbers that don't conflict
    # Let's try a local search: start with greedy, then try to swap
    
    # Actually, let's think about this more carefully.
    # The problem is: find the largest subset S of [1,100] such that no subset of S sums to 200.
    # 
    # Key observation: If we take all numbers from 51 to 100, that's 50 numbers.
    # Can any subset of {51,...,100} sum to 200? 
    # The minimum sum of two numbers is 51+52=103, max is 99+100=199.
    # So no two numbers sum to 200. 
    # Three numbers: min 51+52+53=156, max 98+99+100=297. 
    # Can three sum to 200? 51+52+97=200, but 97 is in the set. So {51,52,97} sums to 200.
    # So we can't take all of 51..100.
    
    # Let's try: take all numbers > 100? No, we're limited to [1,100].
    
    # Alternative: take numbers in a range that avoids subset sums to 200.
    # If we take {101, ..., 200}, none can sum to 200 with others since all > 100.
    # But we're limited to [1,100].
    
    # What about taking {1, 2, ..., 99}? Can any subset sum to 200? Yes, many ways.
    
    # Let's try a different approach: use the fact that if all elements are > 100, 
    # no subset can sum to 200 (since min element > 100, so any sum of 2+ elements > 200).
    # But we can only use [1,100].
    
    # If we take elements in (100, 200), i.e., 101..199, that's 99 elements, but they're not in [1,100].
    
    # So within [1,100], what's the max?
    # Let's try: take all even numbers from 2 to 100: 50 numbers.
    # Can a subset of even numbers sum to 200? Yes, e.g., 2+4+...+98 = 2470, but we need exactly 200.
    # 2+4+6+...+98 = 2*(1+2+...+49) = 2*1225 = 2450. That's way more than 200.
    # But can we find a subset that sums to exactly 200? 
    # 200 is even, and we have even numbers. 200 = 2+4+6+...+? 
    # Sum of first k even numbers = k(k+1). k(k+1)=200? k≈14, 14*15=210. So no exact match with consecutive.
    # But we can choose any subset. 200 = 2+4+6+8+10+12+14+16+18+20+22+24+26+28 = 2*(1+2+...+14) = 2*105 = 210. No.
    # 200 = 4+6+8+...+? Let's just say it's likely possible.
    
    # Let's just run the greedy with multiple orderings and take the best.
    
    # Additional orderings
    import random
    random.seed(42)
    
    # Try random orderings
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        res = greedy(order)
        if len(res) > len(best):
            best = res
    
    # Try more structured orderings
    # Order: numbers coprime to 200 first? No, that doesn't help directly.
    
    # Let's try: process numbers in order of (200 - x) being "hard" to reach
    # i.e., process x where 200-x is large first, since those sums are harder to have reached
    
    # Actually, the greedy is quite simple. Let's try to improve by using a different 
    # criterion: instead of just checking if 200-x is reachable, we could also consider
    # the "cost" of including x in terms of how many future sums it blocks.
    
    # For now, let's return the best we found.
    
    # Verify: ensure no subset sums to 200
    def verify(subset):
        if not subset:
            return False
        # Check all subsets? That's exponential. Instead, use DP.
        reachable = 1
        for x in subset:
            reachable |= (reachable << x)
            reachable &= (1 << 201) - 1
        return ((reachable >> 200) & 1) == 0
    
    if not best:
        # Fallback: just return [100]
        return [100]
    
    if not verify(best):
        # Shouldn't happen, but fallback
        return [100]
    
    return best

# EVOLVE-BLOCK-END
