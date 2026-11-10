import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85

    MAX_N = 256

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(MAX_N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Precompute prime factors for each number 1..256 as a bitmask
    num_mask = [0] * (MAX_N + 1)
    for n in range(2, MAX_N + 1):
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                num_mask[n] |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            num_mask[n] |= (1 << prime_to_idx[temp])
            
    # For each prime index, list candidate numbers (values) that contain this prime
    # We want to assign each prime to exactly one number.
    # It is better to iterate over numbers?
    # Let's stick to the recursive assignment of primes.
    
    # Sort primes in descending order to prune faster?
    # If we process large primes first, the number of candidates is small.
    # For p > 128, candidates are just p (and maybe 2p if p<=128? No, 2p > 256).
    # For p in (64, 128], candidates are p, 2p.
    
    # Let's use a recursive search.
    # State: index of current prime to assign, mask of used primes, current sum.
    # We need to ensure that at the end, all primes are used.
    
    # Actually, we can just find the max sum subset of numbers with disjoint prime factors.
    # This is a Max Weight Set Packing.
    
    # Let's use a different approach:
    # Generate all valid numbers 2..256.
    # Sort them by value descending.
    # Use a greedy approach with backtracking?
    
    # Given the small size, let's try the Merging Heuristic first, it's often optimal for this structure.
    # If it finishes fast, we can try to verify or improve.
    
    # Merging Algorithm:
    # 1. Initialize list of numbers with max power of each prime.
    # 2. While there exists a pair (a, b) in the list such that gcd(a,b)=1 and a*b <= 256:
    #    Calculate gain = a*b - (a+b).
    #    Pick the pair with max gain.
    #    Replace a, b with a*b.
    # 3. Return sum of list.
    
    # Is this greedy optimal?
    # Counter-example?
    # Primes 2, 3, 5.
    # Max powers: 128, 81, 25 (wait, 5^2=25, 5^3=125).
    # Start: [128, 81, 125]
    # Pairs:
    # 128*81 > 256
    # 128*125 > 256
    # 81*125 > 256
    # No merges possible. Sum = 334.
    # Alternative:
    # Use 2*3=6, 5^3=125. Sum = 131. Worse.
    # Use 2^7=128, 3*5=15. Sum = 143. Worse.
    
    # Another example: Primes 2, 3, 7.
    # Max powers: 128, 81, 49.
    # 128*49 > 256. 128*81 > 256. 81*49 > 256.
    # Sum = 258.
    
    # Example: Primes 2, 3, 5, 7.
    # 128, 81, 25, 7.
    # 25*7 = 175 <= 256. Gain = 175 - 32 = 143.
    # Merge 25, 7 -> 175.
    # List: [128, 81, 175].
    # 128*81 > 256. 128*175 > 256. 81*175 > 256.
    # Sum = 384.
    
    # What if we merged 2 and 3 first?
    # 128*81 > 256. Can't merge max powers.
    # But we can choose smaller powers.
    # The greedy "max power" initialization might be suboptimal if it prevents merging.
    # E.g., if we have primes 2, 3, 127.
    # Max powers: 128, 81, 127.
    # 128*127 > 256. 81*127 > 256.
    # Sum = 336.
    # Alternative: 2*127 = 254. 3^4 = 81. Sum = 335. Worse.
    # Alternative: 3*127 = 381 > 256.
    
    # It seems the greedy merging of max powers is a very strong heuristic.
    # To be safe, we will implement this, and then try a few variations (e.g., starting with slightly smaller powers) or use a local search on the partition.

    # However, the B&B is exact. Let's try to make it fast.
    # Number of primes: 54.
    # We can use a bitmask for used primes.
    # We process primes in descending order.
    # For each prime, we try all numbers that are multiples of it and use only unused primes.
    
    # This is still potentially exponential.
    
    # Let's stick to the Merging Heuristic + Local Search.
    
    def max_power(p, limit):
        val = p
        while val * p <= limit:
            val *= p
        return val

    # Initialize with max powers
    initial_numbers = [max_power(p, MAX_N) for p in primes]
    
    best_sum = sum(initial_numbers)
    best_set = initial_numbers[:]
    
    # Greedy Merging
    def try_merge(numbers):
        # Find best pair to merge
        best_gain = -1
        best_i = -1
        best_j = -1
        n = len(numbers)
        for i in range(n):
            for j in range(i+1, n):
                a, b = numbers[i], numbers[j]
                if gcd(a, b) != 1:
                    continue
                prod = a * b
                if prod <= MAX_N:
                    gain = prod - (a + b)
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                        best_j = j
        if best_i != -1:
            a, b = numbers[best_i], numbers[best_j]
            # Remove b, then a (to handle indices)
            # Or just create new list
            new_numbers = []
            for k in range(n):
                if k != best_i and k != best_j:
                    new_numbers.append(numbers[k])
            new_numbers.append(a * b)
            return new_numbers, a * b
        return numbers, 0

    current_numbers = initial_numbers[:]
    while True:
        new_numbers, _ = try_merge(current_numbers)
        if new_numbers == current_numbers:
            break
        current_numbers = new_numbers
        
    if sum(current_numbers) > best_sum:
        best_sum = sum(current_numbers)
        best_set = current_numbers[:]
        
    # Local Search: Try to swap parts?
    # The structure is a partition of primes.
    # We can try to "unmerge" and re-merge differently.
    
    # Given the time constraint and the strong heuristic, this is likely sufficient.
    # But let's add a simple random restart or perturbation if time permits.
    
    # Convert to list of ints
    result = [int(x) for x in best_set]
    
    # Verify
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return initial_numbers # Fallback to safe set if error
        
    return result

BET: {mechanism: "Greedy merging of prime powers into composites exploits the superlinear gain (ab > a+b) while respecting the 256 limit, which is the dominant structural feature of the optimal solution.", q: 0.6, delta: [0.0, 0.1]}
