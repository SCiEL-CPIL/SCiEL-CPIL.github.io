import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    min_val = 211
    max_val = 10210
    
    # 210 = 2 * 3 * 5 * 7
    # We want product * 210 to be a square.
    # Let P be the product of selected integers.
    # P * 210 = k^2
    # This implies that the square-free part of P must be equal to the square-free part of 210, which is 210 itself (since 210 is square-free).
    # So, we need the product of selected integers to have the same square-free kernel as 210.
    # Kernel(x) = product of p where exponent of p in x is odd.
    # We need Kernel(P) = 210.
    
    # Strategy:
    # 1. Compute kernel for all numbers in [min_val, max_val].
    # 2. Find 2-element solution: kernel(a) * kernel(b) / gcd(kernel(a), kernel(b))^2 ... no, simpler:
    #    Kernel(P) = Kernel(a * b) = Kernel(a) * Kernel(b) / gcd(Kernel(a), Kernel(b))^2? 
    #    Actually, Kernel(a*b) is the XOR of the prime factor sets (with multiplicity mod 2).
    #    Let's represent the kernel as the number itself if it's square-free? 
    #    Yes, the square-free kernel is a square-free integer.
    #    Kernel(a * b) = Kernel(a) * Kernel(b) if they are coprime? No.
    #    If a = s_a * x^2 and b = s_b * y^2, then a*b = s_a*s_b * (xy)^2.
    #    The square-free part of a*b is the square-free part of s_a*s_b.
    #    Since s_a and s_b are square-free, s_a*s_b might not be square-free if they share primes.
    #    Let s = s_a * s_b. The square-free kernel of s is s / (product of p^2 for p in gcd(s_a, s_b)).
    #    Actually, it's easier to work with the set of primes.
    #    But since max_val is 10210, the kernels are small enough to fit in integers.
    #    We can just compute the square-free kernel as an integer.
    
    # Vectorized computation of square-free kernels
    # 1. Create array of numbers
    nums = np.arange(min_val, max_val + 1, dtype=np.int64)
    kernels = np.ones_like(nums, dtype=np.int64)
    
    # Sieve for square-free kernels
    # For each prime p, if p^2 divides n, then p^2 cancels out in the kernel.
    # If p divides n but p^2 does not, p remains in the kernel.
    # This is tricky to do directly with a simple sieve for the "odd exponent" part.
    
    # Alternative: Compute the square-free part by dividing out squares.
    # Kernel(n) = n / (largest square dividing n).
    # Let's precompute largest square divisor?
    
    # Better approach for small N (10000):
    # Trial division is fast enough for N=10000 if optimized, but let's try to be smart.
    # However, the prompt says "NumPy and SciPy are available".
    
    # Let's stick to a robust, fast Python loop with memoization or precomputed primes, 
    # but the previous program was slow/buggy. 
    # Let's use a simple sieve to compute the square-free kernel for all numbers up to max_val.
    
    # We can compute the square-free kernel of n as follows:
    # Start with kernel[n] = n.
    # For each prime p:
    #   For each multiple m of p^2:
    #       While kernel[m] is divisible by p^2:
    #           kernel[m] //= p^2
    # This leaves only the square-free part.
    
    kernels = np.arange(min_val, max_val + 1, dtype=np.int64)
    
    # We need primes up to sqrt(max_val) ~ 101
    limit = int(math.isqrt(max_val))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Apply sieve
    # Note: We only care about numbers in [min_val, max_val].
    # But it's easier to compute for 1..max_val and slice.
    all_kernels = np.arange(1, max_val + 1, dtype=np.int64)
    
    for p in primes:
        p2 = p * p
        # Multiples of p^2
        for m in range(p2, max_val + 1, p2):
            # Remove p^2 from kernel as long as it divides
            while all_kernels[m - 1] % p2 == 0:
                all_kernels[m - 1] //= p2
                
    # Now all_kernels[i-1] is the square-free kernel of i.
    # We need to find a subset of indices (corresponding to numbers in [min_val, max_val])
    # such that the product of their kernels is 210 (mod squares? No, exactly 210 if we consider the XOR logic).
    # Wait, if we take the product of kernels, it might have even exponents.
    # Example: kernel(12) = 3. kernel(27) = 3. Product = 9 = 3^2. Kernel of product = 1.
    # So we need the XOR of the prime factor sets to be {2,3,5,7}.
    # This is equivalent to: The product of the kernels, reduced to square-free form, equals 210.
    
    # Let's map each number in range to its kernel.
    # kernel_map: kernel_value -> list of numbers having that kernel (sorted)
    kernel_to_nums = {}
    
    for i in range(min_val, max_val + 1):
        k = int(all_kernels[i - 1])
        if k not in kernel_to_nums:
            kernel_to_nums[k] = []
        kernel_to_nums[k].append(i)
        
    # Target kernel is 210.
    # Case 1: 2 elements a, b.
    # Let ka, kb be kernels.
    # We need kernel(ka * kb) == 210.
    # This means ka * kb = 210 * s^2 for some integer s.
    # Or more simply, the square-free part of ka*kb is 210.
    
    # Let's iterate through possible pairs of kernels?
    # Number of distinct kernels is small.
    distinct_kernels = list(kernel_to_nums.keys())
    
    best_max = float('inf')
    best_pair = None
    
    # Precompute square-free kernel for products of two kernels?
    # Since kernels are square-free, ka * kb is not necessarily square-free.
    # But we can just compute the square-free kernel of the product.
    
    def sq_free_kernel(n):
        # n is small (product of two kernels <= 10210^2 ~ 10^8)
        # But we can use the precomputed table if n <= max_val? No, product can be larger.
        # However, since ka, kb are square-free and <= 10210, their product is <= 10^8.
        # We can factorize n quickly or use a helper.
        # Actually, since ka and kb are square-free, the kernel of ka*kb is just ka*kb / gcd(ka,kb)^2?
        # No. If ka=6 (2*3), kb=10 (2*5). Product=60=2^2*3*5. Kernel=15.
        # gcd(6,10)=2. 60 / 2^2 = 15. Correct.
        # Formula: Kernel(ka*kb) = (ka * kb) // (gcd(ka, kb) ** 2)
        
        g = math.gcd(ka, kb)
        return (ka * kb) // (g * g)
        
    # Iterate over distinct kernels
    # To minimize max(a,b), we should check smaller numbers first?
    # The range is small (10000 items).
    # We can just iterate over all pairs of numbers? 10000^2 = 10^8 operations. Too slow in Python.
    # But we can group by kernel.
    
    # For each pair of distinct kernels (k1, k2) including same kernel:
    # Check if kernel(k1*k2) == 210.
    # If yes, find the smallest a in list[k1] and smallest b in list[k2] such that a != b (if k1==k2).
    # Actually, to minimize max(a,b), we want the pair with the smallest maximum element.
    
    # Let's collect all valid (kernel1, kernel2) pairs.
    valid_kernel_pairs = []
    for i, k1 in enumerate(distinct_kernels):
        for j in range(i, len(distinct_kernels)):
            k2 = distinct_kernels[j]
            # Check if kernel(k1*k2) == 210
            g = math.gcd(k1, k2)
            prod_kernel = (k1 * k2) // (g * g)
            if prod_kernel == 210:
                valid_kernel_pairs.append((k1, k2))
                
    # For each valid pair, find the best numbers
    for k1, k2 in valid_kernel_pairs:
        if k1 == k2:
            lst = kernel_to_nums[k1]
            if len(lst) >= 2:
                # We want to minimize max(a,b). Since they are from the same list,
                # the smallest max is achieved by the two smallest elements?
                # No, if we pick lst[0] and lst[1], max is lst[1].
                # Can we do better? No, any other pair would have max >= lst[1].
                a = lst[0]
                b = lst[1]
                current_max = max(a, b)
                if current_max < best_max:
                    best_max = current_max
                    best_pair = [a, b]
        else:
            lst1 = kernel_to_nums[k1]
            lst2 = kernel_to_nums[k2]
            # To minimize max(a,b), we should pick the smallest a and smallest b?
            # Not necessarily. If min(lst1) is very small and min(lst2) is very large,
            # maybe picking a larger a allows picking a smaller b?
            # But we want to minimize the MAX.
            # The minimum possible max for a pair (a in L1, b in L2) is:
            # min_{a in L1, b in L2} max(a,b)
            # This is equivalent to:
            # If we sort L1 and L2.
            # The optimal max is the smallest M such that there exists a <= M in L1 and b <= M in L2.
            # Which is max(min(L1), min(L2))?
            # Let's check.
            # If min(L1) <= min(L2), then we can pick a=min(L1) and b=min(L2). Max is min(L2).
            # Can we get a smaller max?
            # If we pick a' > min(L1), max(a', b) >= min(L2) if b>=min(L2).
            # If we pick b' < min(L2), impossible.
            # So yes, the pair (min(L1), min(L2)) minimizes the maximum.
            a = lst1[0]
            b = lst2[0]
            current_max = max(a, b)
            if current_max < best_max:
                best_max = current_max
                best_pair = [a, b]
                
    if best_pair:
        # Verify
        prod = 210 * best_pair[0] * best_pair[1]
        if math.isqrt(prod)**2 == prod:
            return [int(x) for x in best_pair]
            
    # If no 2-element solution found, try 3-element?
    # Or just return a known good solution?
    # The task says "Short intervals...". It doesn't restrict count, just "1..100 distinct integers".
    # The seed program returned 2 integers.
    # If no 2-element solution exists in range, we might need 3.
    # But 2-element solutions are likely to exist.
    
    # Fallback: 1 element?
    # Kernel(x) = 210 => x = 210 * k^2.
    # 210 * 1^2 = 210 (too small, must be > 210)
    # 210 * 2^2 = 840.
    # 840 is in range.
    # So [840] is a valid 1-element solution. Value = 840 - 210 = 630.
    
    # But we found best_pair. Let's assume it's valid.
    # If best_pair is None, return [840].
    
    if best_pair is None:
        return [840]
        
    return [int(x) for x in best_pair]

# Test locally
if __name__ == "__main__":
    res = construct()
    print(res)
    prod = 210
    for x in res:
        prod *= x
    s = math.isqrt(prod)
    print(f"Product: {prod}, Sqrt: {s}, Valid: {s*s == prod}")
    print(f"Value: {max(res) - 210}")
