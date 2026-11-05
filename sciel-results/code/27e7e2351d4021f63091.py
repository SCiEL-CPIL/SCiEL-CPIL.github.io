import math
import time
import heapq

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80.0

    def get_primes_up_to(limit):
        if limit < 2:
            return []
        is_prime = bytearray(limit + 1)
        is_prime[2:] = b'\x01' * (limit - 1)
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    def generate_smooth_numbers(primes, limit):
        """
        Generate all numbers <= limit whose prime factors are in 'primes'.
        Returns a sorted list of such numbers.
        """
        if not primes:
            return [1]
        
        n = len(primes)
        # pointers[i] = the smallest exponent such that 1 * primes[i]^exponents not yet added
        # Actually, let's use a different approach:
        # heap contains (value, prime_index)
        # For each prime index, we track the last value that was multiplied by that prime
        
        last = [1] * n  # last[i] = last value that was multiplied by primes[i]
        heap = [(1, i) for i in range(n)]
        heapq.heapify(heap)
        
        result = [1]
        seen = {1}
        
        while heap:
            val, i = heapq.heappop(heap)
            new_val = val * primes[i]
            if new_val > limit:
                continue
            if new_val not in seen:
                seen.add(new_val)
                result.append(new_val)
                heapq.heappush(heap, (new_val, i))
            # Also push back the original val to allow further multiplication
            # Wait, this doesn't work right.
        
        # This approach is flawed. Let me use a simpler approach.
        # Just use the set-based approach but with a list for efficiency.
        
        smooth = {1}
        for p in primes:
            new_smooth = set()
            for s in list(smooth):
                val = s
                while val * p <= limit:
                    val *= p
                    new_smooth.add(val)
            smooth.update(new_smooth)
        
        return sorted(smooth)

    def exists_pair(K):
        """
        Check if there exist a, b such that a+b=N and both are K-smooth.
        Returns (True, a, b) if found, else (False, 0, 0).
        """
        primes = get_primes_up_to(K)
        if not primes:
            if N == 2:
                return True, 1, 1
            return False, 0, 0
            
        smooth_list = generate_smooth_numbers(primes, N)
        
        # Use two-pointer technique on sorted list
        i = 0
        j = len(smooth_list) - 1
        while i <= j:
            s = smooth_list[i] + smooth_list[j]
            if s == N:
                return True, smooth_list[i], smooth_list[j]
            elif s < N:
                i += 1
            else:
                j -= 1
                
        return False, 0, 0

    # Binary search for the smallest K
    lo = 1
    hi = N
    
    best_a = 1
    best_b = N - 1
    found = False
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        exists, a, b = exists_pair(mid)
        
        if exists:
            best_a = a
            best_b = b
            found = True
            hi = mid - 1
        else:
            lo = mid + 1
            
    if not found:
        best_a = 1
        best_b = N - 1

    return [best_a, best_b]
