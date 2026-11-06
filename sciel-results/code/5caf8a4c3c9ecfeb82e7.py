import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_set = []
    best_size = 0
    
    # Use a list for counts for speed
    def get_counts(S):
        counts = [0] * 2001
        for i in range(len(S)):
            a = S[i]
            for j in range(i, len(S)):
                b = S[j]
                counts[a + b] += 1
        return counts

    def can_add(x, S, counts):
        for y in S:
            if counts[x + y] >= 2:
                return False
        if counts[2 * x] >= 2:
            return False
        return True

    def can_swap(x, y, S, counts):
        # Decrement y's contributions
        for z in S:
            counts[y + z] -= 1
        
        # Check x
        valid = True
        for z in S:
            if z == y: continue
            if counts[x + z] >= 2:
                valid = False
                break
        if valid and counts[2 * x] >= 2:
            valid = False
            
        # Restore y's contributions if invalid
        if not valid:
            for z in S:
                counts[y + z] += 1
            return False
        
        return True

    # 1. Greedy Initialization
    S = []
    counts = [0] * 2001
    for x in range(1, 1001):
        if can_add(x, S, counts):
            S.append(x)
            for y in S[:-1]:
                counts[x + y] += 1
            counts[2 * x] += 1
    
    best_set = S[:]
    best_size = len(S)
    
    # 2. Local Search
    while time.time() < deadline:
        # Try to add
        added = False
        # Generate candidates outside S
        # Using a set for S membership is O(1)
        S_set = set(S)
        candidates = [x for x in range(1, 1001) if x not in S_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S, counts):
                S.append(x)
                for y in S[:-1]:
                    counts[x + y] += 1
                counts[2 * x] += 1
                added = True
                break
        
        if added:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            continue
        
        # Try to swap
        swapped = False
        # Limit attempts to avoid infinite loops on stuck states
        for _ in range(50):
            if not S or not candidates: break
            y = random.choice(S)
            x = random.choice(candidates)
            if x == y: continue
            
            if can_swap(x, y, S, counts):
                # Perform swap
                S.remove(y)
                S.append(x)
                # Add x to counts (y was already removed from counts by can_swap)
                for z in S:
                    counts[x + z] += 1
                swapped = True
                break
                
        if swapped:
            continue
            
        # Kick: Remove random elements to escape local optimum
        if len(S) > 10:
            kick_size = min(3, len(S) // 5)
            for _ in range(kick_size):
                if S:
                    y = random.choice(S)
                    S.remove(y)
                    # Remove y from counts
                    for z in S:
                        counts[y + z] -= 1
                    # Note: 2*y was counted when z=y. 
                    # When we remove y, we must decrement 2*y.
                    # The loop `for z in S` (after removal) does NOT include y.
                    # So we missed decrementing 2*y.
                    counts[2 * y] -= 1
            
            # Rebuild counts to be safe? 
            # The manual removal is O(n). Rebuild is O(n^2).
            # n is small, so rebuild is safer and not much slower.
            counts = get_counts(S)

    return best_set
