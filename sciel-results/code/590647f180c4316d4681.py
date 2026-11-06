import time
import random
import sys

def _is_valid_set(S):
    """
    Checks if no element in S is the mean of a subset of at least 2 other elements.
    Uses bitset DP for subset sums with exact count.
    """
    n = len(S)
    if n < 3:
        return True
        
    for i, a in enumerate(S):
        others = [x for j, x in enumerate(S) if j != i]
        
        dp = [0] * (len(others) + 1)
        dp[0] = 1
        
        for x in others:
            for j in range(len(others) - 1, -1, -1):
                if dp[j]:
                    dp[j+1] |= (dp[j] << x)
                    
        for k in range(2, len(others) + 1):
            target_sum = k * a
            if (dp[k] >> target_sum) & 1:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin for safety
    
    best_set = []
    
    # 1. Pre-compute 3-term arithmetic progression conflicts
    # A set is "locally valid" if it contains no 3-term AP (a, b, c) with 2a = b+c.
    # This is a necessary condition for the global validity (no subset mean).
    # If a set has no 3-term AP, it is much more likely to be globally valid.
    # We build a conflict graph where edge (u, v) exists if there is a w in [1, 512]
    # such that {u, v, w} forms a 3-term AP.
    
    # Actually, a simpler heuristic: 
    # A number x is "conflicting" with y and z if 2x = y+z.
    # We want to pick a set with no such triples.
    
    # Let's use a greedy approach with backtracking or a simple local search on the 3-AP constraint,
    # then verify globally.
    
    # Strategy:
    # 1. Generate random sets of size ~48 from [1, 512].
    # 2. Check for 3-term APs. If any exist, remove one of the conflicting elements.
    # 3. Repeat until no 3-term APs exist.
    # 4. Check global validity. If valid, return.
    # 5. If not, try local search: swap elements to break higher-order means.
    
    # To make it faster, we pre-compute for each pair (u, v) if they form a 3-AP with some w.
    # 2u = v + w => w = 2u - v. If 1 <= w <= 512 and w != u and w != v, then (u, v, w) is a 3-AP.
    # We can build a graph where nodes are numbers 1..512.
    # Edge (u, v) exists if they are part of ANY 3-AP.
    # Then we want a Maximum Independent Set in this graph?
    # No, an independent set in this graph means no pair (u, v) is in ANY 3-AP.
    # That is too strong. We just want no TRIPLE in the set to form a 3-AP.
    
    # Let's use a greedy construction with "conflict avoidance".
    # Maintain a set S.
    # To add x, check if x creates a 3-AP with any pair in S.
    # i.e., for any a, b in S, is 2x = a+b OR 2a = x+b OR 2b = x+a?
    # This is O(N^2) per insertion. With N=512, this is fast enough for many restarts.
    
    rng = random.Random(42)
    
    for trial in range(1000):
        if time.time() > deadline:
            break
            
        # Randomly initialize a set of size 48? No, greedy build is safer.
        # Let's try a greedy build with random order, but with fast 3-AP check.
        
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        
        S = []
        S_set = set()
        
        for c in candidates:
            if time.time() > deadline:
                break
                
            # Check if adding c creates a 3-term AP with any pair in S
            # A 3-term AP (a, b, c) satisfies 2b = a+c.
            # So if we add c, we check for any a, b in S:
            # 1. 2c = a + b  => c is mean of a, b
            # 2. 2a = c + b  => a is mean of c, b
            # 3. 2b = c + a  => b is mean of c, a
            
            valid = True
            for a in S:
                for b in S:
                    if a == b: continue
                    # Check if c, a, b form a 3-AP
                    # Sort to identify middle element
                    x, y, z = sorted([c, a, b])
                    if 2 * y == x + z:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.append(c)
                S_set.add(c)
                if len(S) == 48:
                    break
        
        # If we got 48 elements, check global validity
        if len(S) >= len(best_set):
            if _is_valid_set(S):
                return S
            else:
                # Try to repair via local search
                # If global validity fails, it's due to a subset of size > 2.
                # This is rare for 3-AP-free sets.
                # We can try removing random elements and re-adding others.
                pass
        
        # Update best set found so far (even if not globally valid, keep the largest 3-AP-free set)
        if len(S) > len(best_set):
            best_set = S[:]
            
    # If we didn't find a valid set of size 48, try to repair the best 3-AP-free set
    # by removing elements that cause higher-order means.
    if best_set:
        S = best_set[:]
        # Simple repair: remove elements that are means of larger subsets.
        # This is expensive, so only do a few iterations.
        for _ in range(50):
            if time.time() > deadline:
                break
            if _is_valid_set(S):
                return S
            # Find a violating element
            # We can just randomly remove an element and try to add a new one.
            # Or, more intelligently, find the element 'a' that is the mean of subset T.
            # For now, random removal.
            idx = rng.randrange(len(S))
            S.pop(idx)
            
    return best_set
