import time

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin
    
    target_diffs = set(range(1, 31))
    
    def is_valid(S):
        """Check if S is Sidon and covers 1..30."""
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return target_diffs.issubset(diffs)
        
    def backtrack(M, k, current_S, current_diffs, idx):
        """
        Recursively pick elements from 1..M to form a set of size k.
        current_S contains 0 and some picked elements (sorted).
        current_diffs is the set of differences so far.
        idx is the minimum index to pick next from 1..M.
        """
        if time.time() - start_time > deadline:
            return None
        
        if len(current_S) == k:
            if target_diffs.issubset(current_diffs):
                return current_S[:]
            return None
        
        remaining_to_pick = k - len(current_S)
        
        # Pruning: If the number of remaining possible diffs is less than missing diffs, prune?
        # Missing diffs = len(target_diffs - current_diffs)
        # Max new diffs = remaining_to_pick * len(current_S) + remaining_to_pick*(remaining_to_pick-1)//2
        # This is an upper bound. If upper bound < missing, prune.
        missing = len(target_diffs - current_diffs)
        max_new = remaining_to_pick * len(current_S) + remaining_to_pick * (remaining_to_pick - 1) // 2
        if max_new < missing:
            return None
            
        for i in range(idx, M + 1):
            x = i
            # Check if x can be added without violating Sidon
            valid = True
            new_diffs = set()
            for y in current_S:
                d = x - y
                if d in current_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Try adding x
            current_S.append(x)
            current_diffs.update(new_diffs)
            
            result = backtrack(M, k, current_S, current_diffs, i + 1)
            
            if result is not None:
                return result
            
            # Backtrack
            current_S.pop()
            current_diffs.difference_update(new_diffs)
            
        return None

    # Try k=9, then k=10, etc.
    # Start M from a reasonable lower bound.
    # Minimum diameter for k=9 Golomb ruler is 42.
    # But we don't need it to be a "Golomb ruler" in the sense of optimal length, just that it covers 1..30.
    # However, if it covers 1..30, it has at least 30 distinct diffs.
    # The smallest possible max element for a set with 30 distinct diffs is related to the Golomb ruler.
    
    for k in range(9, 12):
        # Lower bound for M:
        # We need 30 diffs.
        # If M is too small, we can't fit them.
        # Try M from 30 to 100.
        for M in range(30, 101):
            if time.time() - start_time > deadline:
                break
            
            # Initialize with 0
            current_S = [0]
            current_diffs = set()
            
            # We need to pick k-1 elements from 1..M
            result = backtrack(M, k, current_S, current_diffs, 1)
            
            if result is not None:
                # Found a valid set with max element <= M
                # Since we iterate M upwards, this is the minimal M for this k?
                # Not necessarily, but likely.
                # We can try to optimize further, but the task asks to minimize largest element.
                # Returning the first found for minimal M is good.
                return result
        
        if time.time() - start_time > deadline:
            break

    # Fallback if search fails
    # Return a known valid set, e.g. from a larger M
    # Let's try a hardcoded solution if we find one, or a random one.
    # Actually, if the loop finishes without returning, we need a valid set.
    # Let's try one more time with a larger M in the last few seconds.
    if time.time() - start_time < deadline:
        for M in range(100, 150):
            for k in range(9, 12):
                current_S = [0]
                current_diffs = set()
                result = backtrack(M, k, current_S, current_diffs, 1)
                if result is not None:
                    return result
                    
    # Absolute fallback: A set that is known to work, even if large
    # e.g. {0, 1, 4, 10, 18, 23, 21, 29, 39, 52} ?
    # Let's just return a set that we can verify quickly or a random one.
    # But it must be valid.
    # We can use a greedy construction as a last resort.
    
    S = [0]
    target = set(range(1, 31))
    covered = set()
    while not target.issubset(covered):
        # Add next smallest number that doesn't create duplicate diff
        added = False
        for x in range(1, 1000):
            if x in S:
                continue
            valid = True
            for y in S:
                d = abs(x - y)
                if d in covered:
                    valid = False
                    break
            if valid:
                S.append(x)
                S.sort()
                # Update covered
                new_diffs = set()
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        new_diffs.add(S[j] - S[i])
                covered = new_diffs
                added = True
                break
        if not added:
            break
    return S
