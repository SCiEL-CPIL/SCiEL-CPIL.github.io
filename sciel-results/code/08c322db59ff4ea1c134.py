import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 12
    MAX_VAL = 10000
    MAX_SUM = 2 * MAX_VAL
    
    # Precompute SPF
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def get_prime_counts(nums):
        counts = {}
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                s = nums[i] + nums[j]
                for p in get_prime_factors(s):
                    counts[p] = counts.get(p, 0) + 1
        return counts

    best_primes = None
    best_nums = None
    best_count = float('inf')
    
    def local_search(start_nums, time_limit):
        current_nums = start_nums.copy()
        if len(set(current_nums)) < N:
            return current_nums, float('inf')
            
        current_counts = get_prime_counts(current_nums)
        current_primes = set(p for p, c in current_counts.items() if c > 0)
        current_prime_count = len(current_primes)
        
        # Candidate pool
        # Small numbers are good for keeping sums small
        # Random numbers help escape local minima
        pool = list(range(1, 1001))
        for _ in range(1000):
            pool.append(random.randint(1, MAX_VAL))
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(N))
            random.shuffle(indices)
            
            for idx in indices:
                old_val = current_nums[idx]
                best_candidate = None
                best_candidate_count = current_prime_count
                best_candidate_counts = None
                best_candidate_primes = None
                
                for y in pool:
                    if y == old_val:
                        continue
                    if y in current_nums:
                        continue
                    
                    # Incremental update
                    # 1. Remove old_val contributions
                    # Sums: old_val + current_nums[k] for k != idx
                    # We need to decrement counts for primes in these sums
                    # 2. Add y contributions
                    # Sums: y + current_nums[k] for k != idx
                    # We need to increment counts for primes in these sums
                    
                    # To avoid modifying the shared dict during iteration and then reverting if not accepted,
                    # we can use a copy or a delta structure.
                    # Given the small number of sums (11), copying the counts dict is cheap.
                    new_counts = current_counts.copy()
                    
                    # Decrement old
                    for k in range(N):
                        if k == idx: continue
                        s_old = old_val + current_nums[k]
                        for p in get_prime_factors(s_old):
                            new_counts[p] -= 1
                            if new_counts[p] == 0:
                                del new_counts[p]
                    
                    # Increment new
                    for k in range(N):
                        if k == idx: continue
                        s_new = y + current_nums[k]
                        for p in get_prime_factors(s_new):
                            new_counts[p] = new_counts.get(p, 0) + 1
                    
                    new_primes = set(new_counts.keys())
                    new_count = len(new_primes)
                    
                    if new_count < best_candidate_count:
                        best_candidate_count = new_count
                        best_candidate = y
                        best_candidate_counts = new_counts
                        best_candidate_primes = new_primes
                        
                if best_candidate is not None:
                    current_nums[idx] = best_candidate
                    current_counts = best_candidate_counts
                    current_primes = best_candidate_primes
                    current_prime_count = best_candidate_count
                    improved = True
                    
        return current_nums, current_prime_count

    # Initializations
    inits = [
        list(range(1, N+1)),
        [2**i for i in range(N)],
        [3**i for i in range(N)],
        [i*100 for i in range(1, N+1)]
    ]
    # Add random inits
    for _ in range(4):
        r = [random.randint(1, MAX_VAL) for _ in range(N)]
        while len(set(r)) < N:
            r.append(random.randint(1, MAX_VAL))
        inits.append(list(set(r))[:N])

    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        time_left = deadline - time.time()
        if time_left < 1:
            break
            
        # Allocate time based on remaining budget
        # Give more time to early runs? Or equal?
        # Let's give 5s to first 4, and 2s to others
        alloc = 5 if i < 4 else 2
        alloc = min(alloc, time_left)
        
        res_nums, res_count = local_search(init, time.time() + alloc)
        
        if res_count < best_count:
            best_count = res_count
            best_nums = res_nums
            # Update best_primes
            best_primes = set()
            for i in range(N):
                for j in range(i+1, N):
                    best_primes.update(get_prime_factors(best_nums[i] + best_nums[j]))
            
            # Polish if good
            if best_count <= 6 and time.time() < deadline - 10:
                 res_nums2, res_count2 = local_search(best_nums, deadline - 10)
                 if res_count2 < best_count:
                     best_count = res_count2
                     best_nums = res_nums2
                     best_primes = set()
                     for i in range(N):
                         for j in range(i+1, N):
                             best_primes.update(get_prime_factors(best_nums[i] + best_nums[j]))

    if best_nums is None:
        return list(range(1, N+1))
        
    return best_nums
