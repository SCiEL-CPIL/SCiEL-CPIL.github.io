import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    N = 1000
    best_size = 0
    best_set = []
    
    # Helper to check if x can be added to S with current sum_counts
    # Returns True if valid, False otherwise
    def can_add(x, S, sum_counts):
        # Check 2*x
        if sum_counts.get(2 * x, 0) >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            s = x + y
            if sum_counts.get(s, 0) >= 2:
                return False
        return True

    # Helper to add x to S and update sum_counts
    def add(x, S, sum_counts):
        S.append(x)
        sum_counts[2 * x] = sum_counts.get(2 * x, 0) + 1
        for y in S[:-1]: # S[:-1] is S before appending x
            s = x + y
            sum_counts[s] = sum_counts.get(s, 0) + 1

    # Helper to remove x from S and update sum_counts
    def remove(x, S, sum_counts):
        S.remove(x)
        sum_counts[2 * x] -= 1
        if sum_counts[2 * x] == 0:
            del sum_counts[2 * x]
        for y in S:
            s = x + y
            sum_counts[s] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]
    
    def local_search(S, sum_counts, max_iters=500):
        """Try swap moves: remove one element, add another to improve size"""
        S = S[:]
        sum_counts = sum_counts.copy()
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try removing each element and see if we can add a new one
            for i in range(len(S)):
                if time.time() >= deadline:
                    break
                x_remove = S[i]
                # Temporarily remove x_remove
                S.remove(x_remove)
                sum_counts[2 * x_remove] -= 1
                if sum_counts[2 * x_remove] == 0:
                    del sum_counts[2 * x_remove]
                for y in S:
                    s = x_remove + y
                    sum_counts[s] -= 1
                    if sum_counts[s] == 0:
                        del sum_counts[s]
                
                # Try to add a random candidate not in S
                candidates = [x for x in range(1, N + 1) if x not in S]
                random.shuffle(candidates)
                added = False
                for x_add in candidates:
                    if can_add(x_add, S, sum_counts):
                        S.append(x_add)
                        sum_counts[2 * x_add] = sum_counts.get(2 * x_add, 0) + 1
                        for y in S[:-1]:
                            s = x_add + y
                            sum_counts[s] = sum_counts.get(s, 0) + 1
                        improved = True
                        added = True
                        break
                
                # Restore x_remove if we didn't add anything
                if not added:
                    S.append(x_remove)
                    sum_counts[2 * x_remove] = sum_counts.get(2 * x_remove, 0) + 1
                    for y in S[:-1]:
                        s = x_remove + y
                        sum_counts[s] = sum_counts.get(s, 0) + 1
                    if improved:
                        break
        return S, sum_counts

    # Run multiple random greedy attempts
    while time.time() < deadline:
        S = []
        sum_counts = {}
        candidates = list(range(1, N + 1))
        
        # Mix of strategies: sometimes random, sometimes biased toward smaller numbers
        strategy = random.random()
        if strategy < 0.3:
            random.shuffle(candidates)
        elif strategy < 0.6:
            # Bias toward smaller numbers
            candidates.sort(key=lambda x: x + random.random() * 100)
        else:
            # Random with some structure
            random.shuffle(candidates)
            # Swap some elements to create variation
            for _ in range(len(candidates) // 10):
                i, j = random.randint(0, len(candidates) - 1), random.randint(0, len(candidates) - 1)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        
        for x in candidates:
            if can_add(x, S, sum_counts):
                add(x, S, sum_counts)
        
        # Apply local search
        S, sum_counts = local_search(S, sum_counts, max_iters=200)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
    return best_set
