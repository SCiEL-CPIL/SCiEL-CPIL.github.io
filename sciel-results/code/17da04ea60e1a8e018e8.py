import random
import math
from itertools import combinations

def construct():
    def is_compatible(S, x):
        """
        Check if x can be added to set S.
        S is a list of integers.
        Conditions:
        1. x does not divide the sum of any 3 distinct elements in S.
        2. For each a in S, a does not divide the sum of any 3 distinct elements from (S \ {a}) U {x}.
           The sums in (S \ {a}) U {x} that include x are x + y + z where y, z in S \ {a}, y != z.
           The sums not including x are already valid by assumption (S is valid).
           So we only need to check sums involving x for each a in S.
        """
        # Check 1: x divides sum of 3 from S
        # If len(S) < 3, this is trivially true.
        if len(S) >= 3:
            # Generate all 3-combinations of S
            # To speed up, we can break early if found
            for t in combinations(S, 3):
                if sum(t) % x == 0:
                    return False
        
        # Check 2: For each a in S, does a divide x + y + z for any y, z in S \ {a}?
        for i, a in enumerate(S):
            # We need y, z distinct from each other and from a.
            # y, z must be in S.
            # If len(S) < 2, we can't form a pair.
            if len(S) < 2:
                continue
            
            # Get indices of S excluding a
            # Instead of creating a new list, iterate over indices
            # We need pairs (j, k) such that j != k and j, k != i
            # And check (x + S[j] + S[k]) % a == 0
            
            # Optimization: Precompute? No, S is small.
            # Just loop
            for j in range(len(S)):
                if j == i:
                    continue
                for k in range(j + 1, len(S)):
                    if k == i:
                        continue
                    if (x + S[j] + S[k]) % a == 0:
                        return False
        return True

    # Greedy Construction: Largest to Smallest
    S = []
    for x in range(192, 0, -1):
        if is_compatible(S, x):
            S.append(x)
    
    # Local Search to improve
    # Try to add any remaining number?
    # The greedy already tried all.
    # Try to swap?
    # If we remove a small number, can we add two larger ones that were blocked?
    # Or remove one to allow one? (No size gain)
    # Remove one to allow TWO?
    
    # Let's try a simple local search:
    # 1. Try to add any unselected x. (Already done in greedy, but maybe order matters? 
    #    Greedy from large to small is usually good. But maybe a smaller x fits better with the rest?
    #    Actually, greedy picks the largest compatible. If we skip a large one, we might fit two smaller ones?
    #    But we want max cardinality.
    #    Let's try Randomized Greedy with several seeds.
    
    best_S = list(S)
    best_size = len(S)
    
    # Run a few randomized greedy attempts
    for _ in range(50):
        random.shuffle(list(range(1, 193))) # Wait, we need to iterate in a specific order?
        # The order of addition matters for greedy.
        # Let's try shuffling the range and picking the first compatible?
        # Or sorting by some heuristic?
        
        # Actually, let's just do a local search on the current S.
        # Try to remove each element and see if we can add two others.
        
        # For now, let's just return the greedy result if it's fast.
        # The greedy result is likely decent.
        
        # Let's try to improve by swapping.
        # Pick a random element a in S.
        # Remove it.
        # Try to add any unselected x.
        # If we can add one, size is same.
        # If we can add two, size increases.
        
        # This is complex. Let's stick to the fast Greedy.
        # But maybe we can try **Greedy from Small to Large**?
        # Small numbers are more likely to divide sums, so they are "more constrained".
        # Picking large numbers first leaves "room" for small numbers?
        # Or picking small numbers first fills up the "divisibility" slots?
        # Usually, "No element divides sum of others" favors large numbers because they are "harder" to divide.
        # So Large-to-Small is likely better.
        
        # Let's try a local search:
        # 1. Try to add any x not in S.
        #    We already did this in the greedy loop.
        #    But maybe if we remove one element, we can add two?
        
        # Let's implement a "Remove and Fill" strategy.
        # For each a in S:
        #   S_temp = S \ {a}
        #   Try to add as many unselected numbers as possible to S_temp (Greedy).
        #   If len(S_temp_new) > len(S), update S.
        
        # This is O(|S| * 192 * Cost(Compat)).
        # Cost(Compat) is ~40k ops.
        # 40 * 192 * 40k = 300M ops. Too slow.
        
        # We need a faster check or fewer iterations.
        
        # Let's just return the greedy result. It's a strong baseline.
        # To beat 1.0, we need to find a larger set.
        # If the greedy finds 48, we are at max.
        # If it finds less, we need to improve.
        
        # Let's check if we can just use a precomputed solution or a better heuristic.
        # Heuristic: Pick numbers in [96, 192] that are coprime? No.
        
        # Let's try **Simulated Annealing** but with a **much faster** compatibility check.
        # The check `is_compatible` is the bottleneck.
        # Can we precompute sums?
        
        # Let's try a different move: **Swap**.
        # Remove a, add b.
        # Check if b is compatible with S \ {a}.
        # This is just one `is_compatible` call.
        # We can do many swaps.
        
        # Let's do a local search with swaps and adds.
        
        S_curr = list(S)
        unselected = [x for x in range(1, 193) if x not in S_curr]
        
        # Try to add any unselected
        # (Already done in greedy, but let's be sure)
        
        # Try Swaps
        for _ in range(1000): # Limit iterations
            if not unselected or not S_curr:
                break
            a = random.choice(S_curr)
            b = random.choice(unselected)
            
            # Check if b is compatible with S_curr \ {a}
            S_temp = [x for x in S_curr if x != a]
            if is_compatible(S_temp, b):
                # Perform swap
                S_curr.remove(a)
                S_curr.append(b)
                unselected.remove(b)
                unselected.append(a)
        
        if len(S_curr) > best_size:
            best_S = S_curr
            best_size = len(S_curr)

    return best_S
