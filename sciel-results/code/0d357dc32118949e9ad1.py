import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute reciprocals as Fractions for exact arithmetic
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    
    def is_valid(S):
        """
        Returns True if S is reciprocal-subsum-free.
        For every x in S, 1/x is not the sum of reciprocals of any subset of S\{x}.
        """
        S_list = list(S)
        n = len(S_list)
        if n < 2:
            return True
            
        for i, x in enumerate(S_list):
            target = recip[x]
            others = S_list[:i] + S_list[i+1:]
            if not others:
                continue
                
            # Check if target can be formed by subset sum of recip[others]
            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Generate all subset sums for left
            left_sums = [0]
            for y in left:
                ry = recip[y]
                # Optimization: if any sum exceeds target, we might prune? 
                # Since all reciprocals are positive, we can prune sums > target.
                new_sums = []
                for s in left_sums:
                    ns = s + ry
                    if ns <= target:
                        new_sums.append(ns)
                left_sums.extend(new_sums)
                
            # Generate all subset sums for right
            right_sums = [0]
            for y in right:
                ry = recip[y]
                new_sums = []
                for s in right_sums:
                    ns = s + ry
                    if ns <= target:
                        new_sums.append(ns)
                right_sums.extend(new_sums)
                
            # Check if any l + r == target
            right_set = set(right_sums)
            
            found = False
            for l in left_sums:
                if (target - l) in right_set:
                    found = True
                    break
                    
            if found:
                return False
                
        return True

    def greedy_build(order):
        """
        Builds a set by iterating through 'order' and adding elements if valid.
        """
        S = set()
        for x in order:
            if x in S:
                continue
            if is_valid(S | {x}):
                S.add(x)
        return S

    def local_search_improve(S):
        """
        Tries to add any element from [1..24] \ S to S if it remains valid.
        Repeats until no more elements can be added.
        """
        S = set(S)
        improved = True
        while improved:
            improved = False
            for x in range(1, 25):
                if x not in S:
                    if is_valid(S | {x}):
                        S.add(x)
                        improved = True
                        # Restart loop to check if adding x allows adding others?
                        # Actually, adding x only makes it *harder* to add others 
                        # (more summands available for violations of new targets? 
                        # No, existing targets might now be formable by the new summand x).
                        # So adding x might invalidate the set? 
                        # Wait, is_valid(S | {x}) checks if S U {x} is valid.
                        # This includes checking if 1/x is formed by S, AND 
                        # if 1/y is formed by S U {x} for y in S.
                        # So if is_valid returns True, the new set is valid.
                        # So we can just continue.
                        break # Restart inner loop to re-evaluate all candidates? 
                        # Actually, just continuing the for loop is fine, 
                        # but a fresh pass is safer to ensure we don't miss interactions.
                        # However, for efficiency, let's just do one pass and repeat if changed.
        return S

    best_set = set()
    best_size = 0
    
    # Try deterministic orders
    orders = [
        list(range(1, 25)),
        list(range(24, 0, -1)),
        # Alternate small/large?
        [1, 24, 2, 23, 3, 22, 4, 21, 5, 20, 6, 19, 7, 18, 8, 17, 9, 16, 10, 15, 11, 14, 12, 13]
    ]
    
    # Add random orders
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline:
            break
        o = list(range(1, 25))
        random.shuffle(o)
        orders.append(o)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        S = greedy_build(order)
        S = local_search_improve(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Early exit if we found a very large set (e.g., > 15?)
        # Max possible is 24.
        if best_size >= 18: # Heuristic threshold
             break

    # Fallback if no valid set found (should not happen, singleton is always valid)
    if not best_set:
        best_set = {24}
        
    return list(best_set)
