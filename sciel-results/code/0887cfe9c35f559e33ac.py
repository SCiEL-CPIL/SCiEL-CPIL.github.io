import math
import numpy as np

def construct():
    # We will search in a segment near 10^11 for a run of distinct phi values.
    # Segment size: 10^6
    segment_size = 10**6
    # Start of the segment
    start = 10**11
    end = start + segment_size - 1
    
    # Ensure end <= 10^12
    if end > 10**12:
        end = 10**12
        segment_size = end - start + 1
    
    # Sieve primes up to sqrt(end)
    limit = int(math.isqrt(end)) + 1
    primes = []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    # Initialize phi array for the segment
    phi = np.arange(start, end + 1, dtype=np.int64)
    
    # For each prime, update phi values in the segment
    for p in primes:
        if p * p > end:
            break
        # Find the first multiple of p in [start, end]
        first_multiple = ((start + p - 1) // p) * p
        for multiple in range(first_multiple, end + 1, p):
            idx = multiple - start
            # Check if multiple is divisible by p
            if phi[idx] % p == 0:
                # Divide by p and multiply by (p-1)
                phi[idx] = phi[idx] // p * (p - 1)
    
    # For numbers that are still prime (or have a large prime factor),
    # we need to handle them. But since we've processed all primes up to sqrt(end),
    # any remaining factor is either 1 or a prime > sqrt(end).
    # If phi[idx] > 1, it means there's a prime factor > sqrt(end) that we haven't processed.
    # But for phi, if n has a prime factor p > sqrt(n), then p appears only once.
    # So we can check if phi[idx] is still equal to the original number, or if it has been reduced.
    
    # Actually, the above loop only processes primes up to sqrt(end).
    # For numbers that have a prime factor > sqrt(end), we need to handle them separately.
    # But since we're sieving up to sqrt(end), any composite number in the segment will have at least one factor <= sqrt(end).
    # So after processing all primes <= sqrt(end), if phi[idx] is still equal to the original number, it means the number is prime.
    # But we can just check: if phi[idx] == original_number, then it's prime, so phi = original_number - 1.
    # But we already applied the formula for primes <= sqrt(end). So if a number is prime, it wasn't touched, so phi[idx] is still the original number.
    # So we need to check for each idx if phi[idx] is still the original number, and if so, set it to original_number - 1.
    
    original = np.arange(start, end + 1, dtype=np.int64)
    # Check which numbers were not touched (i.e., they are prime)
    # But actually, the sieve above only updates phi for multiples of p.
    # So if a number is prime, it is not a multiple of any p < it, so it's not touched.
    # So we can check if phi[idx] == original[idx], and if so, it's prime.
    
    for i in range(segment_size):
        if phi[i] == original[i]:
            phi[i] = phi[i] - 1
    
    # Now phi contains the Euler's totient function for each number in the segment.
    # Use a sliding window to find the longest run of distinct phi values.
    max_len = 0
    max_start_idx = 0
    left = 0
    seen = {}
    
    for right in range(segment_size):
        val = phi[right]
        if val in seen:
            # Move left until we remove the duplicate
            while left <= right and phi[left] != val:
                left += 1
            left += 1
        seen[val] = right
        
        if right - left + 1 > max_len:
            max_len = right - left + 1
            max_start_idx = left
    
    # The actual start number is start + max_start_idx
    actual_start = start + max_start_idx
    actual_length = max_len
    
    # Ensure the length is at most 200
    if actual_length > 200:
        actual_length = 200
        # Adjust start if necessary
        # But since we're taking a subarray of a valid run, it's still valid.
    
    return {'start': actual_start, 'length': actual_length}
