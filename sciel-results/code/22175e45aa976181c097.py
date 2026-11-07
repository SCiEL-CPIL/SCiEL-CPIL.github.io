import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    def is_valid(S):
        if not S:
            return True
        S = sorted(S)
        n = len(S)
        # Prefix sums for MinSum
        # MinSum(k) is sum of first k elements
        # MaxSum(k) is sum of last k elements
        
        # Check k=0: MinSum(1) > MaxSum(0)=0. x_1 > 0. Always true for 1..256.
        
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + S[i]
            
        # MinSum(k+1) = prefix[k+1]
        # MaxSum(k) = sum of S[n-k:n] = prefix[n] - prefix[n-k]
        
        total = prefix[n]
        
        for k in range(1, n):
            min_sum_next = prefix[k+1]
            max_sum_curr = total - prefix[n-k]
            if min_sum_next <= max_sum_curr:
                return False
        return True

    def count_violations(S):
        if not S:
            return 0
        S = sorted(S)
        n = len(S)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + S[i]
        total = prefix[n]
        violations = 0
        for k in range(1, n):
            min_sum_next = prefix[k+1]
            max_sum_curr = total - prefix[n-k]
            if min_sum_next <= max_sum_curr:
                violations += 1
        return violations

    def local_search(N, max_iter=50000):
        # Initialize with a set close to the known valid 31 set
        # Start with 31 consecutive top elements, add one random, remove one random?
        # Or just random.
        
        # Better init: Start with valid 31, try to insert a 32nd.
        # If N=32:
        base_set = list(range(226, 257)) # 31 elements
        if N == 31:
            return base_set
        
        # For N > 31, start with a perturbed version of a large set
        # Or random.
        # Let's try random first, then maybe refine.
        
        S = random.sample(range(1, 257), N)
        current_violations = count_violations(S)
        best_S = S[:]
        best_violations = current_violations
        
        T = 10.0
        cooling = 0.999
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random element to change
            idx = random.randint(0, N-1)
            old_val = S[idx]
            
            # Option 1: Swap with another element
            if random.random() < 0.5:
                jdx = random.randint(0, N-1)
                if jdx == idx: continue
                S[idx], S[jdx] = S[jdx], S[idx]
                new_violations = count_violations(S)
                delta = new_violations - current_violations
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_S = S[:]
                else:
                    S[idx], S[jdx] = S[jdx], S[idx]
            else:
                # Option 2: Replace with a random unused value
                # Generate a random value not in S
                # Simple way: pick from 1..256, check if in S.
                # For efficiency, maintain a set of used values.
                # But for N=32, S is small.
                while True:
                    new_val = random.randint(1, 256)
                    if new_val not in S:
                        break
                S[idx] = new_val
                new_violations = count_violations(S)
                delta = new_violations - current_violations
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_S = S[:]
                else:
                    S[idx] = old_val
            
            T *= cooling
            
            if best_violations == 0:
                return best_S
        
        return best_S

    # Try to find a valid set of size 32
    # We can run multiple restarts
    best_size = 31
    best_set = list(range(226, 257))
    
    # Check if 31 is valid (it is)
    
    # Try 32
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        S = local_search(32, max_iter=20000)
        if is_valid(S):
            best_size = 32
            best_set = S
            break
        # Keep the best invalid one? No, we need valid.
        # But if we find a set with 0 violations, we return it.
        
    # If we found 32, try 33?
    if best_size == 32:
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            S = local_search(33, max_iter=20000)
            if is_valid(S):
                best_size = 33
                best_set = S
                break

    # Final verification
    if is_valid(best_set):
        return best_set
    else:
        # Fallback to known valid 31
        return list(range(226, 257))
