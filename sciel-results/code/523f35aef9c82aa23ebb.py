import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    N = 1000
    best_size = 0
    best_set = []
    
    # Helper to validate a set S and return sum_counts
    # S is a list of ints
    def get_sum_counts(S):
        if not S:
            return {}
        S_arr = np.array(S)
        # Compute all pair sums (including i=j)
        # We only need counts for sums <= 2*N
        # Number of pairs is |S|*(|S|+1)/2
        n = len(S)
        # Efficiently compute sums using broadcasting might be memory heavy for large n, 
        # but n ~ 300-400, so n^2 ~ 160k, which is fine.
        # Actually, we can just do a loop in Python for clarity and speed if n is small,
        # but let's try numpy for the check.
        
        # Better: just use a dict loop for accuracy and simplicity in validation,
        # as the heavy lifting is in the search.
        counts = {}
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                counts[s] = counts.get(s, 0) + 1
        return counts

    def is_valid(S):
        if not S:
            return True
        counts = get_sum_counts(S)
        for v in counts.values():
            if v > 2:
                return False
        return True

    # Greedy construction with tightness heuristic
    # We want to pick elements that "fit" well.
    # A common heuristic for packing is to pick elements that create the least "damage" (new conflicts).
    # However, a simpler effective heuristic for B_h sets is often just sorted or random.
    # Let's try a deterministic greedy that iterates 1..N and adds if possible, 
    # but also tries to "repair" by removing conflicting elements if a swap yields a better set?
    # No, let's stick to a robust greedy: 
    # 1. Start with empty set.
    # 2. Iterate candidates in a specific order (e.g., random, or sorted by some property).
    # 3. If adding x is valid, add it.
    
    # Strategy: Multiple restarts with different random orders, but use a faster validity check.
    # The current program's bottleneck is the `can_add` check which is O(|S|).
    # We can maintain `sum_counts` incrementally.
    
    def greedy_random(seed):
        rng = np.random.RandomState(seed)
        candidates = rng.permutation(N) + 1 # 1..1000
        S = []
        sum_counts = {}
        
        for x in candidates:
            # Check if x can be added
            valid = True
            # Check 2*x
            if sum_counts.get(2 * x, 0) >= 2:
                valid = False
            else:
                # Check x + y for y in S
                # Optimization: if S is large, this is slow.
                # But we stop early if found.
                for y in S:
                    s = x + y
                    if sum_counts.get(s, 0) >= 2:
                        valid = False
                        break
            
            if valid:
                S.append(x)
                sum_counts[2 * x] = sum_counts.get(2 * x, 0) + 1
                for y in S[:-1]:
                    s = x + y
                    sum_counts[s] = sum_counts.get(s, 0) + 1
        
        return S, sum_counts

    def local_search_numpy(S, sum_counts):
        """
        Local search using numpy for vectorized conflict detection where possible.
        We will try:
        1. Single element addition (if possible)
        2. Single element removal + addition (swap)
        3. Random perturbations
        """
        S = S[:]
        sum_counts = sum_counts.copy()
        S_set = set(S)
        
        # Helper to check if x can be added to current S/sum_counts
        def can_add(x):
            if x in S_set:
                return False
            if sum_counts.get(2 * x, 0) >= 2:
                return False
            for y in S:
                if sum_counts.get(x + y, 0) >= 2:
                    return False
            return True

        def add(x):
            S.append(x)
            S_set.add(x)
            sum_counts[2 * x] = sum_counts.get(2 * x, 0) + 1
            for y in S[:-1]:
                s = x + y
                sum_counts[s] = sum_counts.get(s, 0) + 1

        def remove(x):
            S.remove(x)
            S_set.remove(x)
            sum_counts[2 * x] -= 1
            if sum_counts[2 * x] == 0:
                del sum_counts[2 * x]
            for y in S:
                s = x + y
                sum_counts[s] -= 1
                if sum_counts[s] == 0:
                    del sum_counts[s]

        max_iter = 100
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try to add any element
            # Instead of iterating 1..N, we can sample or iterate.
            # Iterating 1..N is O(N*|S|). N=1000, |S|~300 -> 300k ops.
            # Doing this 100 times is 30M ops. Might be slow in Python.
            # Let's try a random subset of candidates first.
            
            # Try adding a random candidate
            for _ in range(10):
                x = np.random.randint(1, N + 1)
                if can_add(x):
                    add(x)
                    improved = True
                    break
            
            if not improved:
                # Try swapping: remove one, add another
                # Remove a random element
                idx = np.random.randint(0, len(S))
                x_remove = S[idx]
                
                # Temporarily remove
                # We need to restore if we don't find a good add
                old_sum_counts = sum_counts.copy()
                old_S = S[:]
                old_S_set = S_set.copy()
                
                remove(x_remove)
                
                # Try to add a random element
                success = False
                for _ in range(20):
                    x_add = np.random.randint(1, N + 1)
                    if x_add == x_remove:
                        continue
                    if can_add(x_add):
                        add(x_add)
                        success = True
                        break
                
                if not success:
                    # Restore
                    S = old_S
                    S_set = old_S_set
                    sum_counts = old_sum_counts
                else:
                    improved = True
                    
            if not improved:
                break
                
        return S, sum_counts

    # Main loop
    # We will run several greedy restarts with different seeds
    # and then apply local search to the best ones.
    
    # Pre-compute a few good starting sets?
    # Let's just run the loop.
    
    # To speed up, we can use a list of seeds.
    # Since we have 140s, and each greedy+LS might take ~1-5s?
    # Let's estimate.
    # Greedy: 1000 * 300 = 300k ops. ~0.1s
    # Local Search: 100 iters * (10 adds + 20 swaps * 300 checks) ~ 100 * 4000 = 400k ops?
    # Actually `can_add` is O(|S|).
    # 10 adds: 10 * 300 = 3000
    # 20 swaps: 20 * 300 = 6000
    # Total per iter ~ 9000.
    # 100 iters ~ 900k ops. ~0.1-0.2s.
    # So one full cycle is ~0.3s.
    # We can do ~400 cycles in 140s.
    
    for i in range(500):
        if time.time() > deadline:
            break
            
        seed = i
        S, sum_counts = greedy_random(seed)
        
        # Local search
        S, sum_counts = local_search_numpy(S, sum_counts)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # Optional: If we find a very good set, we can try to refine it further
        # or use it as a base for perturbation.
        # For now, just keep the best.
        
    return best_set

# Quick sanity check / optimization:
# The `can_add` function in local search is called frequently.
# We can optimize it by pre-computing the "forbidden" sums?
# If sum_counts[s] >= 2, then s is "full".
# We need x such that 2x is not full, and x+y is not full for all y in S.
# This is equivalent to: x is not in { (full_sum - y) / 2 ... }
# This is complex. The O(|S|) check is probably fine.

# One potential issue: `S.remove(x)` is O(|S|).
# Using a set for membership is good, but removal from list is slow.
# Since |S| is ~300, it's acceptable.

# Let's refine the local search to be a bit more aggressive with time.
# If we have time, we can do more iterations.

# Re-evaluating the strategy:
# The key insight from the "inspiration" is that deterministic greedy (sorted) might be good?
# "always picking the smallest..."
# Let's add a deterministic greedy (sorted 1..N) as a baseline in the loop.

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    N = 1000
    best_size = 0
    best_set = []
    
    def get_sum_counts(S):
        counts = {}
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                counts[s] = counts.get(s, 0) + 1
        return counts

    def greedy_deterministic():
        S = []
        sum_counts = {}
        for x in range(1, N + 1):
            valid = True
            if sum_counts.get(2 * x, 0) >= 2:
                valid = False
            else:
                for y in S:
                    if sum_counts.get(x + y, 0) >= 2:
                        valid = False
                        break
            if valid:
                S.append(x)
                sum_counts[2 * x] = sum_counts.get(2 * x, 0) + 1
                for y in S[:-1]:
                    sum_counts[x + y] = sum_counts.get(x + y, 0) + 1
        return S, sum_counts

    def greedy_random(seed):
        rng = np.random.RandomState(seed)
        candidates = rng.permutation(N) + 1
        S = []
        sum_counts = {}
        for x in candidates:
            valid = True
            if sum_counts.get(2 * x, 0) >= 2:
                valid = False
            else:
                for y in S:
                    if sum_counts.get(x + y, 0) >= 2:
                        valid = False
                        break
            if valid:
                S.append(x)
                sum_counts[2 * x] = sum_counts.get(2 * x, 0) + 1
                for y in S[:-1]:
                    sum_counts[x + y] = sum_counts.get(x + y, 0) + 1
        return S, sum_counts

    def local_search(S, sum_counts, time_limit=5.0):
        S = S[:]
        sum_counts = sum_counts.copy()
        S_set = set(S)
        
        def can_add(x):
            if x in S_set:
                return False
            if sum_counts.get(2 * x, 0) >= 2:
                return False
            for y in S:
                if sum_counts.get(x + y, 0) >= 2:
                    return False
            return True

        def add(x):
            S.append(x)
            S_set.add(x)
            sum_counts[2 * x] = sum_counts.get(2 * x, 0) + 1
            for y in S[:-1]:
                s = x + y
                sum_counts[s] = sum_counts.get(s, 0) + 1

        def remove(x):
            S.remove(x)
            S_set.remove(x)
            sum_counts[2 * x] -= 1
            if sum_counts[2 * x] == 0:
                del sum_counts[2 * x]
            for y in S:
                s = x + y
                sum_counts[s] -= 1
                if sum_counts[s] == 0:
                    del sum_counts[s]

        start_ls = time.time()
        while time.time() - start_ls < time_limit and time.time() < deadline:
            improved = False
            
            # Try to add
            for _ in range(5):
                x = np.random.randint(1, N + 1)
                if can_add(x):
                    add(x)
                    improved = True
                    break
            
            if not improved:
                # Try swap
                if S:
                    idx = np.random.randint(0, len(S))
                    x_remove = S[idx]
                    old_S = S[:]
                    old_set = S_set.copy()
                    old_counts = sum_counts.copy()
                    
                    remove(x_remove)
                    
                    success = False
                    for _ in range(10):
                        x_add = np.random.randint(1, N + 1)
                        if x_add != x_remove and can_add(x_add):
                            add(x_add)
                            success = True
                            break
                    
                    if not success:
                        S = old_S
                        S_set = old_set
                        sum_counts = old_counts

        return S, sum_counts

    # 1. Deterministic baseline
    S_det, counts_det = greedy_deterministic()
    S_det, counts_det = local_search(S_det, counts_det, time_limit=10.0)
    if len(S_det) > best_size:
        best_size = len(S_det)
        best_set = S_det[:]

    # 2. Random restarts
    while time.time() < deadline:
        seed = int(time.time() * 1000) % 100000
        S, counts = greedy_random(seed)
        # Shorter local search for random restarts to allow more of them
        S, counts = local_search(S, counts, time_limit=2.0)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # If we are close to deadline, do a longer search on the best found so far
        if time.time() > deadline - 20:
            S_best, counts_best = greedy_deterministic() # Re-run det to get fresh counts? 
            # Actually, use the best_set found
            S_best = best_set[:]
            counts_best = get_sum_counts(S_best)
            S_best, counts_best = local_search(S_best, counts_best, time_limit=15.0)
            if len(S_best) > best_size:
                best_size = len(S_best)
                best_set = S_best[:]
            break

    return best_set
