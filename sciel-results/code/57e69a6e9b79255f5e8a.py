import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin before 60s limit
    
    def try_greedy_build(order):
        S = []
        n = 0
        # dp[k] is a bitset of sums for cardinality k
        # Max sum is 256 * 60 = 15360. Python int handles this easily.
        dp = [0] * 61
        dp[0] = 1  # Sum 0 is achievable with cardinality 0
        
        for x in order:
            if x in S:
                continue
            
            # Tentatively add x
            new_dp = [0] * (n + 2)  # Max cardinality becomes n+1
            # Copy old dp
            for k in range(n + 1):
                new_dp[k] = dp[k]
            
            # Add new sums
            for k in range(1, n + 2):
                # New sums of cardinality k come from old sums of cardinality k-1 shifted by x
                new_dp[k] |= (dp[k-1] << x)
            
            # Check validity: No overlap between different cardinalities
            valid = True
            # We only need to check overlaps involving the changed parts?
            # Actually, checking all pairs is safe and simple.
            # O(N^2) bit operations.
            for i in range(n + 2):
                for j in range(i + 1, n + 2):
                    if new_dp[i] & new_dp[j]:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.append(x)
                n += 1
                dp = new_dp
                # Optional: If n is very large, we might stop early? 
                # No, we want to maximize size, so keep trying.
                
        return S

    best_set = []
    
    # Generate diverse orders
    orders = []
    
    # 1. Sorted ascending and descending
    orders.append(list(range(1, 257)))
    orders.append(list(range(256, 0, -1)))
    
    # 2. Alternating small/large
    small = list(range(1, 129))
    large = list(range(256, 127, -1))
    alt = []
    for i in range(128):
        alt.append(small[i])
        alt.append(large[i])
    orders.append(alt)
    
    # 3. Random with different seeds
    rng = random.Random(42)
    
    # We can run many more random restarts now due to bitset speed
    # Estimate: 1 build attempt ~ 5-10ms. 50s / 10ms = 5000 attempts.
    # Let's aim for a few thousand.
    
    max_random_trials = 2000
    
    for trial in range(max_random_trials):
        if time.time() > deadline:
            break
            
        o = list(range(1, 257))
        rng.shuffle(o)
        orders.append(o)
        
        # Periodically check best and maybe adjust strategy?
        # For now, just append.
        
    # 4. Biased random: weight towards certain ranges
    # Hypothesis: Smaller numbers might allow more elements because sums stay smaller?
    # Or larger numbers might separate sums more effectively?
    
    for trial in range(500):
        if time.time() > deadline:
            break
        o = list(range(1, 257))
        # Biased shuffle: sort by random key with bias
        # Or just sample with replacement then dedupe? No, permutation is better.
        # Let's try sorting by a biased random value.
        # e.g., value * random()
        biased = [(v, rng.random()) for v in o]
        biased.sort(key=lambda x: x[1])
        orders.append([v for v, _ in biased])

    # Execute all orders
    for o in orders:
        if time.time() > deadline:
            break
        S = try_greedy_build(o)
        if len(S) > len(best_set):
            best_set = S
            # If we find a very good set, we might want to spend more time on similar strategies?
            # Or just continue.
            
    return [int(x) for x in best_set]
