import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def generate_powerful_up_to(limit):
        """
        Generate all powerful numbers up to limit.
        A powerful number is of the form prod p_i^{e_i} where e_i >= 2.
        Equivalently, n = s^2 * t^3 where t is squarefree.
        """
        powerful = set([1])  # 1 is powerful
        # Every powerful number can be written as n^2 * m^3 where m is squarefree
        # Actually, the standard form: if n is powerful, n = a^2 * b^3 where b is squarefree
        # Let's use: for each squarefree m, for each n such that n^2 * m^3 <= limit
        # m must be squarefree and m^3 <= limit, so m <= limit^(1/3)
        max_m = int(limit ** (1/3)) + 2
        
        # Generate squarefree numbers up to max_m
        # A number is squarefree if no square of a prime divides it
        is_squarefree = [True] * (max_m + 1)
        is_squarefree[0] = False
        # Sieve for squarefree
        for p in range(2, int(math.isqrt(max_m)) + 1):
            if is_squarefree[p]:  # p is prime (simplified, but we just need squarefree detection)
                p2 = p * p
                for j in range(p2, max_m + 1, p2):
                    is_squarefree[j] = False
        
        # Actually, let's be more careful. We need to generate squarefree numbers.
        # Use a sieve: mark multiples of p^2
        squarefree = [True] * (max_m + 1)
        squarefree[0] = False
        for p in range(2, int(math.isqrt(max_m)) + 1):
            p2 = p * p
            for j in range(p2, max_m + 1, p2):
                squarefree[j] = False
        
        squarefree_nums = [m for m in range(1, max_m + 1) if squarefree[m]]
        
        for m in squarefree_nums:
            m3 = m * m * m
            if m3 > limit:
                break
            max_n = int(math.isqrt(limit // m3))
            for n in range(1, max_n + 1):
                x = n * n * m3
                if x <= limit:
                    powerful.add(x)
        
        return sorted(powerful)

    def count_representations(N, powerful_set, powerful_list):
        """Count the number of pairs (a,b) with a<=b, a+b=N, a,b powerful."""
        count = 0
        pairs = []
        for a in powerful_list:
            if a > N:
                break
            b = N - a
            if b < a:
                continue
            if b in powerful_set:
                count += 1
                pairs.append([a, b])
        return count, pairs

    # Strategy: Generate powerful numbers up to a limit L, then find N that
    # maximizes the number of pairs (a, N-a) both powerful.
    # We'll try increasing limits and use sum-counting to find the best N.
    
    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_count = 1
    
    # Try multiple limits, starting smaller
    limits = [10**6, 5*10**6, 2*10**7, 5*10**7, 10**8, 2*10**8, 5*10**8]
    
    for L in limits:
        if time.time() > deadline - 20:
            break
        
        P = generate_powerful_up_to(L)
        P_set = set(P)
        P_len = len(P)
        
        if P_len < 10:
            continue
        
        # Count all pair sums using a dictionary
        # For efficiency, use numpy if P is large enough, else pure Python
        sum_counts = defaultdict(int)
        sum_pairs = defaultdict(list)
        
        # Only count pairs where a <= b to avoid double counting
        for i in range(P_len):
            a = P[i]
            if time.time() > deadline - 15:
                break
            for j in range(i, P_len):
                b = P[j]
                N = a + b
                if N > 10**18:
                    break
                sum_counts[N] += 1
                sum_pairs[N].append([a, b])
        
        # Find the N with the most representations
        if sum_counts:
            # Get top candidates
            top_N = max(sum_counts, key=sum_counts.get)
            top_count = sum_counts[top_N]
            
            if top_count > best_count:
                best_count = top_count
                best_result = {'N': top_N, 'pairs': sum_pairs[top_N]}
        
        # If we've found a good number, we might want to try larger L
        # But if time is running out, break
        if best_count >= 20:
            break
    
    # Final verification: make sure the pairs are valid
    if best_result['pairs']:
        N = best_result['N']
        verified_pairs = []
        for a, b in best_result['pairs']:
            if a <= b and a + b == N:
                verified_pairs.append([a, b])
        best_result['pairs'] = verified_pairs
        best_count = len(verified_pairs)
    
    return best_result
