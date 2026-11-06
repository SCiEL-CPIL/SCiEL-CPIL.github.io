import time
import math
import random

def sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            start = i * i
            step = i
            is_prime[start::step] = bytearray(len(is_prime[start::step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Primes up to 10^6 are sufficient for trial division of numbers up to 10^12
    # sqrt(10^12) = 10^6
    primes = sieve_primes(10**6)
    
    def phi_trial(n):
        """Compute totient using trial division by precomputed primes."""
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
                if temp == 1:
                    break
        if temp > 1:
            result -= result // temp
        return result

    def compute_phi_segment(start, length):
        """
        Compute phi for all numbers in [start, start+length-1].
        Uses segmented sieve logic.
        """
        if start == 1:
            # Handle 1
            res = [1] * length
            # For i > 0, n = 1+i
            for i in range(1, length):
                res[i] = phi_trial(1 + i)
            return res

        # res[i] will hold phi(start+i)
        # temp[i] will hold the remaining part of start+i after dividing small primes
        res = [0] * length
        temp = [0] * length
        
        for i in range(length):
            n = start + i
            res[i] = n
            temp[i] = n
        
        limit = int(math.isqrt(start + length - 1))
        
        for p in primes:
            if p > limit:
                break
            # Find first multiple of p in [start, start+length-1]
            # first = ceil(start / p) * p
            first = ((start + p - 1) // p) * p
            # Ensure first is within range
            if first < start:
                first += p
            
            for m in range(first, start + length, p):
                idx = m - start
                if temp[idx] % p == 0:
                    res[idx] = res[idx] // p * (p - 1)
                    while temp[idx] % p == 0:
                        temp[idx] //= p
        
        # Handle remaining prime factors
        for i in range(length):
            if temp[i] > 1:
                res[i] = res[i] // temp[i] * (temp[i] - 1)
                
        return res

    def find_longest_distinct_run(phi_vals):
        """
        Find the longest contiguous subarray with all distinct values.
        Returns (start_index_relative_to_phi_vals, length)
        """
        n = len(phi_vals)
        if n == 0:
            return 0, 0
        best_len = 1
        best_start = 0
        left = 0
        seen = {}
        for right in range(n):
            v = phi_vals[right]
            if v in seen and seen[v] >= left:
                left = seen[v] + 1
            seen[v] = right
            if right - left + 1 > best_len:
                best_len = right - left + 1
                best_start = left
        return best_start, best_len

    best_s = 1
    best_k = 1
    
    # We search in chunks.
    # Start from near 10^12 and go down.
    # Chunk size: 1,000,000
    chunk_size = 1000000
    
    # Let's run a loop.
    # We'll step down from 10^12.
    # Step size = chunk_size.
    
    for i in range(10000): # Max chunks
        if time.time() > deadline - 10:
            break
            
        end = 10**12 - i * chunk_size
        start = end - chunk_size + 1
        
        if start < 1:
            break
            
        # Compute phi for [start, end]
        phi_vals = compute_phi_segment(start, chunk_size)
        
        # Find longest distinct run
        local_start_idx, local_len = find_longest_distinct_run(phi_vals)
        
        if local_len > best_k:
            best_k = local_len
            best_s = start + local_start_idx
            
            # If we hit the max, we can stop
            if best_k == 200:
                break
            
            # Try to extend
            # Extend right
            right_idx = local_start_idx + local_len
            if right_idx < chunk_size:
                seen_vals = set(phi_vals[local_start_idx:local_start_idx + local_len])
                ext_right = 0
                for j in range(right_idx, chunk_size):
                    v = phi_vals[j]
                    if v in seen_vals:
                        break
                    seen_vals.add(v)
                    ext_right += 1
                    best_k += 1
                    if best_k == 200:
                        break
                if best_k == 200:
                    break
            
            # Extend left
            # This requires phi_trial for numbers < start
            left_idx = local_start_idx
            if left_idx > 0:
                seen_vals = set(phi_vals[local_start_idx:local_start_idx + local_len])
                ext_left = 0
                # We can extend left within the chunk first
                for j in range(left_idx - 1, -1, -1):
                    v = phi_vals[j]
                    if v in seen_vals:
                        break
                    seen_vals.add(v)
                    ext_left += 1
                    best_s -= 1
                    best_k += 1
                    if best_k == 200:
                        break
                
                # Then extend left outside the chunk using phi_trial
                if best_k < 200:
                    current_s = best_s
                    while best_k < 200 and current_s > 1:
                        current_s -= 1
                        v = phi_trial(current_s)
                        if v in seen_vals:
                            break
                        seen_vals.add(v)
                        best_s = current_s
                        best_k += 1
                        if time.time() > deadline - 10:
                            break
                
                if best_k == 200:
                    break

    # If we haven't found a very long run, try some random spots or smaller ranges?
    # The current strategy covers the top 10^12 range downwards.
    # If best_k is small, it might be worth trying the very beginning (1, 2, ...)
    # But phi(1)=1, phi(2)=1 (collision). So start=1 is bad.
    # phi(3)=2, phi(4)=2 (collision).
    # phi(5)=4, phi(6)=2 (collision with 4? No, 4 and 2 are distinct. But 2 is phi(4) and phi(6). So 4,5,6 -> 2,4,2 collision).
    
    # Let's do a quick check of small numbers just in case.
    if best_k < 20:
        # Check first 10000 numbers
        phi_small = compute_phi_segment(1, 10000)
        ls, ll = find_longest_distinct_run(phi_small)
        if ll > best_k:
            best_k = ll
            best_s = 1 + ls

    return {'start': best_s, 'length': best_k}
