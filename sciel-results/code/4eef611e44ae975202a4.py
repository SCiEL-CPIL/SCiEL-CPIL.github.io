import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # The problem: find max cardinality subset of [1,100] with no subset summing to 200.
    # This is equivalent to finding the maximum size of a subset S of {1,...,100}
    # such that 200 is not in the subset sums of S.
    
    # A key observation: if we take all numbers from 1 to 100, the total sum is 5050.
    # But we just need to avoid sum 200.
    
    # Strategy: Greedy construction with backtracking/local search.
    # Since the target (200) is relatively small compared to the range [1,100],
    # many large subsets will work.
    
    # Let's first try a deterministic greedy approach:
    # Start with all numbers 1..100.
    # While there exists a subset summing to 200, remove one element from such a subset.
    # To maximize cardinality, we want to remove as few elements as possible.
    
    # Alternative: Start empty, add elements one by one if they don't create a sum of 200.
    # This is the standard greedy for "maximal" but not necessarily "maximum" independent set.
    
    # Better approach: 
    # 1. Compute all subset sums of [1,100] that are <= 200.
    # 2. We want the largest subset S such that 200 is not reachable.
    # 3. This is a hitting set problem: we need to hit all subsets that sum to 200.
    #    Actually, we want to AVOID all subsets that sum to 200.
    #    So we want to remove the minimum number of elements such that no remaining subset sums to 200.
    #    This is equivalent to: find the minimum hitting set of all subsets of [1,100] that sum to 200.
    #    Wait, no. If we remove a set R of elements, then no subset of [1,100]\R sums to 200.
    #    This means R must intersect every subset of [1,100] that sums to 200.
    #    So R is a hitting set (transversal) for the family of subsets summing to 200.
    #    We want to minimize |R| to maximize |[1,100]\R|.
    
    # The number of subsets of [1,100] that sum to 200 is huge, but we can work with reachable sums.
    
    # Let's use a different approach:
    # Greedy removal: 
    # 1. Start with S = {1, ..., 100}.
    # 2. While there exists a subset of S summing to 200:
    #    a. Find one such subset C.
    #    b. Remove an element from C that appears in the most other potential conflicts (or just remove one).
    # This is a greedy hitting set algorithm.
    
    # However, finding all conflicts is expensive. Let's use a smarter approach.
    
    # Observation: If we take all numbers > 100, none can be in a subset summing to 200 with positive integers... wait, all numbers are <= 100.
    # If we take all numbers from 101 to 200... no, the set is [1,100].
    
    # Let's think about which elements are "dangerous".
    # An element x is dangerous if there exists a subset of the other elements summing to 200-x.
    
    # A very good heuristic:
    # Start with S = {1, ..., 100}.
    # Repeatedly:
    #   Find a subset C of S summing to 200.
    #   If none, return S.
    #   Remove the element in C that is "most central" (e.g., appears in the most minimal conflicts, or just remove the largest/smallest).
    
    # To speed up, we can precompute or use DP to find conflicts.
    
    # Let's implement a fast conflict finder using bitsets or DP.
    # Since target is 200, we can use a boolean array of size 201.
    
    def find_conflict(S):
        """
        Find a subset of S summing to 200.
        Returns the subset as a set, or None.
        S is a set of integers.
        """
        target = 200
        # reach[sum] = set of elements used to reach that sum
        # We only need one subset.
        # To keep it fast, we track the path.
        reach = {0: frozenset()}
        parent = {0: None}  # Not needed if we store the set directly, but sets are expensive to copy.
        
        # Better: reach[sum] = (parent_sum, element)
        # Then we can backtrack.
        reach = {0: None}  # reach[sum] = (prev_sum, element) or None for 0
        
        for x in S:
            # We need to iterate over a snapshot of current reach keys
            # to avoid using x multiple times in the same "layer" (though x is unique, 
            # we must not use it more than once in a single subset).
            # Actually, since we're building subsets, each x is used at most once per subset.
            # The DP is: for each x, for each sum s currently reachable, s+x becomes reachable.
            # We must not update reach in-place in a way that uses x twice.
            # So we collect new sums first.
            
            new_reach = {}
            for s, _ in reach.items():
                if s + x <= target:
                    if s + x not in reach and s + x not in new_reach:
                        new_reach[s + x] = (s, x)
            
            # Merge
            for s, val in new_reach.items():
                reach[s] = val
            
            if target in reach:
                # Backtrack to find the subset
                subset = set()
                cur = target
                while cur != 0:
                    prev, elem = reach[cur]
                    subset.add(elem)
                    cur = prev
                return subset
        
        return None
    
    # Greedy hitting set
    S = set(range(1, 101))
    
    # We'll use a heuristic to choose which element to remove.
    # Option 1: Remove the largest element in the conflict.
    # Option 2: Remove the element that appears in the most conflicts (expensive).
    # Option 3: Random removal from the conflict.
    
    # Let's try multiple strategies and keep the best (largest) S.
    
    best_S = None
    best_size = 0
    
    strategies = [
        "largest",
        "smallest",
        "random",
        "median"
    ]
    
    for strategy in strategies:
        S = set(range(1, 101))
        iterations = 0
        while time.time() - start_time < deadline - 5.0:
            conflict = find_conflict(S)
            if conflict is None:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
                break
            
            iterations += 1
            
            if strategy == "largest":
                elem = max(conflict)
            elif strategy == "smallest":
                elem = min(conflict)
            elif strategy == "random":
                elem = random.choice(list(conflict))
            elif strategy == "median":
                sorted_conf = sorted(conflict)
                elem = sorted_conf[len(sorted_conf)//2]
            
            S.remove(elem)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
    
    # If we still have time, try local search:
    # Try to add back removed elements or swap.
    # But since we removed elements to break conflicts, adding back might create conflicts.
    # A better local search: start from a random large subset and try to repair.
    
    # Actually, the greedy hitting set is likely very close to optimal.
    # The minimum hitting set for subset sums of 200 in [1,100] is likely small.
    
    # Let's verify with a quick check: 
    # If we remove {200}, but 200 is not in [1,100].
    # The worst case is that we have to remove many elements.
    # But intuitively, removing a few large elements might break many sums.
    
    # Let's also try a randomized restart approach.
    
    for _ in range(100):
        if time.time() - start_time > deadline - 2.0:
            break
        
        S = set(range(1, 101))
        # Randomly remove some elements to start? No, start full.
        # Use random strategy for removal.
        while time.time() - start_time < deadline - 2.0:
            conflict = find_conflict(S)
            if conflict is None:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
                break
            
            elem = random.choice(list(conflict))
            S.remove(elem)
    
    # Final check: ensure best_S is valid
    if best_S is None:
        best_S = set(range(1, 101))
    
    # Verify no subset sums to 200
    # (This is a safety check, find_conflict should have ensured it)
    if find_conflict(best_S) is not None:
        # Fallback: remove elements until valid
        while find_conflict(best_S) is not None:
            conflict = find_conflict(best_S)
            best_S.remove(max(conflict))
    
    return list(best_S)

# Quick test
if __name__ == "__main__":
    import time
    start = time.time()
    result = construct()
    end = time.time()
    print(f"Size: {len(result)}")
    print(f"Time: {end - start:.2f}s")
    # Verify
    def verify(S):
        target = 200
        reach = {0: None}
        for x in S:
            new_reach = {}
            for s, _ in reach.items():
                if s + x <= target:
                    if s + x not in reach and s + x not in new_reach:
                        new_reach[s + x] = (s, x)
            for s, val in new_reach.items():
                reach[s] = val
            if target in reach:
                return False
        return True
    print(f"Valid: {verify(result)}")
