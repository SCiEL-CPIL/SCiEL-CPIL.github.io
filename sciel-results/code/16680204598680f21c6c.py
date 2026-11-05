"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # The task: find a nonempty subset S of [1,100] with maximum cardinality
    # such that no subset of S sums to exactly 200.
    #
    # Key insight: If we take all numbers > 100, none can be in a subset summing to 200
    # (since all are positive and each > 100, any single element > 200 is impossible, 
    # and any pair > 200). Wait, numbers are in [1,100], so max element is 100.
    #
    # Let's think more carefully. We want to maximize |S| where S ⊆ {1,...,100}
    # and no subset of S sums to 200.
    #
    # A simple approach: take all numbers from 51 to 100. That's 50 numbers.
    # Can any subset of {51,...,100} sum to 200?
    # - Single element: max is 100 < 200. No.
    # - Two elements: min pair is 51+52=103, max is 100+99=199. So no two elements sum to 200.
    # - Three elements: min is 51+52+53=156, max is 100+99+98=297. Could three sum to 200?
    #   51+52+97=200. Yes! 51, 52, 97 are all in [51,100]. So this doesn't work.
    #
    # Let's try {100, 99, 98, ..., 61} (numbers 61-100, that's 40 numbers).
    # - Two elements: min 61+62=123, max 100+99=199. No pair sums to 200.
    # - Three elements: min 61+62+63=186, max 100+99+98=297.
    #   Can three sum to 200? 61+62+77=200. 77 is in [61,100]. So yes.
    #
    # Let's try numbers > 100... but we're limited to [1,100].
    #
    # What about taking all even numbers? {2,4,6,...,100} - 50 numbers.
    # Subset sum 200: 2+4+...+? Many combinations possible. E.g., 2+4+6+...+100 = 2550.
    # 2+4+6+...+100 can certainly make 200. E.g., 2+4+6+8+...+100, pick subset.
    # 2+4+6+8+10+12+14+16+18+20+22+24+26+28+30 = 200? Sum of first 15 evens = 2*(1+...+15)=2*120=240. Too high.
    # 2+4+...+28 = 2*(1+...+14) = 2*105 = 210. Close.
    # 2+4+...+26 = 2*(1+...+13) = 2*91 = 182. Then +18 = 200. 18 is in the set. So yes.
    #
    # Let me think about this differently. The greedy algorithm in the seed picks from 100 down to 1,
    # adding x if it doesn't create a subset sum of 200. This is a standard greedy for this type of problem.
    #
    # Let me trace through the seed algorithm to see what it produces:
    # Start: reachable = {0}
    # x=100: new = {100}. 200 not in new. Add 100. reachable = {0, 100}
    # x=99: new = {99, 199}. 200 not in new. Add 99. reachable = {0, 99, 100, 199}
    # x=98: new = {98, 197, 198}. 200 not in new. Add 98. reachable = {0, 98, 99, 100, 197, 198, 199}
    # x=97: new = {97, 196, 197, 198, 199, 200}... wait, 100+97=197, 99+97=196, 98+97=195... 
    #   Actually: for each s in reachable, s+97 if <=200. 
    #   0+97=97, 98+97=195, 99+97=196, 100+97=197, 197+97=294>200, etc.
    #   new = {97, 195, 196, 197}. 200 not in new. Add 97.
    # ...
    # This greedy approach will add many large numbers. The question is whether we can do better.
    #
    # Upper bound: If we take all of [1,100], there are many subsets summing to 200.
    # What's the maximum size? 
    #
    # Consider: if S contains both x and 200-x for some x in [1,100], that alone doesn't force a subset sum of 200 (just those two). But combined with other elements, it could.
    #
    # Actually, a cleaner upper bound: partition [1,100] into pairs {x, 200-x} where both are in [1,100]. 
    # x ranges from 100 to 101... wait, 200-x in [1,100] means x in [100,199]. So x=100 gives 200-100=100. That's the same element.
    # For x in [1,99], 200-x is in [101,199], which is NOT in [1,100]. So no pairs within [1,100] sum to 200!
    #
    # Wait, that's a key insight. For any x in [1,100], 200-x is in [100,199]. The only case where 200-x is also in [1,100] is when 200-x <= 100, i.e., x >= 100. So x=100, 200-x=100. That's just the element itself.
    #
    # So no TWO distinct elements in [1,100] sum to 200. The minimum sum of two distinct elements is 1+2=3, max is 100+99=199. So indeed, no pair sums to 200.
    #
    # What about three elements? x+y+z=200 with x,y,z in [1,100], distinct (or not, but in our set they're distinct since it's a set).
    # E.g., 50+60+90=200. All in [1,100].
    #
    # So the constraint is about subsets of size >= 3 (since size 1 max is 100 < 200, size 2 max is 199 < 200).
    #
    # The greedy algorithm picks large numbers first. Let's see how many it can pick.
    #
    # Actually, let me just run the greedy and count. But I can't run it here. Let me think about what the optimal is.
    #
    # A known result: the maximum size of a subset of [1,n] with no subset summing to n is related to the problem. For n=200 and [1,100]:
    #
    # Consider taking all numbers from 67 to 100. That's 34 numbers.
    # Min 3-element sum: 67+68+69=204 > 200. So no 3-element subset sums to 200.
    # Any larger subset has even larger sum. So {67,...,100} has NO subset summing to 200!
    # Cardinality: 100-67+1 = 34.
    #
    # Can we do better? What about {66,...,100}? 35 numbers.
    # 66+67+67... wait, distinct. 66+67+68=201 > 200. Still no 3-element sum to 200.
    # So {66,...,100} works! Cardinality 35.
    #
    # {65,...,100}? 36 numbers. 65+66+67=198 < 200. Can we make 200? 65+66+69=200. Yes! 65,66,69 all in [65,100]. So this fails.
    #
    # So {66,...,100} with 35 elements works. Can we add any element from [1,65]?
    # If we add 65: 65+66+69=200. Fails.
    # If we add 64: 64+66+70=200. 70 is in [66,100]. Fails.
    # If we add 63: 63+66+71=200. Fails.
    # ...
    # In general, if we add x < 66, we need to check if there exist a,b in {66,...,100} with x+a+b=200, i.e., a+b=200-x.
    # For x=65: a+b=135. 66+69=135. Yes.
    # For x=64: a+b=136. 66+70=136. Yes.
    # For x=1: a+b=199. 99+100=199. Yes.
    # For x=50: a+b=150. 66+84=150. Yes (84 in [66,100]).
    # 
    # What about removing some elements from [66,100] to add more from [1,65]?
    # This becomes a complex optimization.
    #
    # The seed greedy starts from 100 and works down. It will add 100, 99, 98, ..., and stop adding when it would create a sum of 200.
    #
    # Let me think about what the greedy produces. It adds x if no subset of current selected + {x} sums to 200.
    # Since no pair sums to 200, the first conflict arises with triples.
    #
    # The greedy will add 100, 99, 98, ..., and at some point, adding x would create a triple (or larger) summing to 200.
    #
    # For the set {100, 99, ..., k}, when does adding k-1 create a sum of 200?
    # We need a+b+(k-1)=200 where a,b in {k,...,100}. So a+b=200-(k-1)=201-k.
    # Min a+b = k+(k+1)=2k+1. Max a+b = 100+99=199.
    # So we need 2k+1 <= 201-k <= 199.
    # 201-k <= 199 => k >= 2.
    # 2k+1 <= 201-k => 3k <= 200 => k <= 66.
    # So for k <= 66, there exist a,b in {k,...,100} with a+b=201-k.
    # 
    # When the greedy has selected {100, 99, ..., k+1} and considers k:
    # It checks if any subset of {k, k+1, ..., 100} sums to 200.
    # Specifically, does k + a + b = 200 for some a,b in {k+1,...,100}?
    # a+b = 200-k. Min a+b = (k+1)+(k+2)=2k+3. 
    # So we need 2k+3 <= 200-k => 3k <= 197 => k <= 65.
    # For k=65: a+b=135. (k+1)+(k+2)=66+67=133 <= 135. And 100+99=199 >= 135. So yes, there exist a,b.
    # Specifically, 66+69=135. Both in {66,...,100}. So adding 65 to {66,...,100} creates 65+66+69=200.
    # 
    # For k=66: The greedy considers adding 66 to {67,...,100}. 
    # Does 66+a+b=200 for a,b in {67,...,100}? a+b=134. Min=67+68=135 > 134. So no.
    # Can 66+a+b+c=200? 66+67+68+69=270 > 200. No.
    # So 66 can be added.
    #
    # So the greedy will add 100, 99, ..., 66. That's 35 elements: {66, 67, ..., 100}.
    # Then for 65: as shown, 65+66+69=200, so 65 is rejected.
    # For 64: 64+66+70=200? 70 is in the set. Yes. Rejected.
    # For 63: 63+66+71=200. Yes. Rejected.
    # ...
    # For x: x+66+y=200 => y=134-x. Need y in [67,100] and y != 66. 
    # 134-x in [67,100] => x in [34,67]. For x in [34,65], y=134-x in [69,100], which is in the set.
    # So all of 34-65 are rejected.
    # For x=33: 33+66+y=200 => y=101. Not in set. 
    #   Try other combos: 33+a+b=200, a,b in {66,...,100}. a+b=167. 66+101=167, 101 not in set. 67+100=167. Yes! 67 and 100 are in the set.
    #   So 33+67+100=200. Rejected.
    # For x=32: 32+a+b=200, a+b=168. 68+100=168. Yes. Rejected.
    # ...
    # For x: x+68+100=200 => x=32. So x=32 rejected.
    # x+69+100=200 => x=31. Rejected.
    # ...
    # x+100+99=200 => x=1. Rejected.
    # 
    # So it seems like ALL of 1-65 are rejected? Let's check x=1: 1+99+100=200. Yes.
    # x=2: 2+98+100=200. Yes.
    # ...
    # x=33: 33+67+100=200. Yes.
    # x=34: 34+66+100=200. Yes.
    # ...
    # x=65: 65+66+69=200. Yes.
    #
    # So the greedy produces exactly {66, 67, ..., 100} with 35 elements.
    #
    # Can we do better than 35? The set {66,...,100} has 35 elements and no subset sums to 200.
    # We showed that adding any single element from [1,65] creates a conflict.
    # But maybe we can remove some elements from [66,100] and add more from [1,65] to get a net gain?
    #
    # For example, if we remove 66 from the set, can we add 65 and some others?
    # {65, 67, 68, ..., 100} - 35 elements. Check: 65+67+68=200. Yes! Conflict.
    # So removing 66 and adding 65 doesn't help.
    #
    # What if we remove two elements and add three?
    # This is getting complex. The greedy solution of 35 is likely optimal or near-optimal.
    #
    # Let me verify: is 35 the maximum? 
    # The set {66,...,100} works. I believe this is optimal based on the analysis.
    #
    # Let's just return {66, 67, ..., 100} directly. It's 35 elements, no subset sums to 200.
    #
    # Actually, let me double-check {66,...,100} more carefully.
    # Smallest 3-element sum: 66+67+68 = 201 > 200. 
    # So no 3-element subset sums to 200. Any subset of size >= 3 has sum >= 201.
    # Subsets of size 1: max 100 < 200.
    # Subsets of size 2: max 100+99 = 199 < 200.
    # So indeed, NO subset of {66,...,100} sums to 200.
    #
    # This is a clean, verifiable solution with 35 elements.
    
    # Return the set {66, 67, ..., 100}
    result = list(range(66, 101))
    
    # Verify no subset sums to 200 (for safety)
    # Since min 3-element sum is 201, this is guaranteed, but let's be safe
    # For a quick check with small subsets:
    from itertools import combinations
    for r in range(1, 4):  # Check subsets of size 1, 2, 3
        for combo in combinations(result, r):
            if sum(combo) == 200:
                # This shouldn't happen, but if it does, fall back to greedy
                break
        else:
            continue
        break
    else:
        return result
    
    # Fallback: use the greedy algorithm
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        if time.time() > deadline:
            break
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected
# EVOLVE-BLOCK-END
