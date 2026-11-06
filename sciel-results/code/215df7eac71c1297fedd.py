import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    def is_square_free(n):
        if n <= 1:
            return True
        p = 2
        while p * p <= n:
            if n % (p * p) == 0:
                return False
            p += 1 if p == 2 else 2
        return True
    
    def generate_powerful(limit):
        pw = set([1])
        max_b = int(limit ** (1/3)) + 1
        for b in range(1, max_b + 1):
            if not is_square_free(b):
                continue
            b3 = b ** 3
            if b3 > limit:
                break
            max_a = isqrt(limit // b3)
            for a in range(1, max_a + 1):
                val = a * a * b3
                if val <= limit:
                    pw.add(val)
        return sorted(pw)
    
    # Generate powerful numbers up to 10^9
    limit_pw = 10**9
    powerful = generate_powerful(limit_pw)
    pw_set = set(powerful)
    
    # Find N with max representations using two-pointer
    # Try N values that are sums of two powerful numbers
    # Use the fact that for sorted powerful, we can two-pointer for each N
    
    # Candidate N: try sums involving small powerful numbers
    # Also try N = 2*pw[i] for various i
    candidate_ns = set()
    for a in powerful[:10000]:
        for b in powerful[:10000]:
            s = a + b
            if s <= 10**18:
                candidate_ns.add(s)
    
    # Also add some large N candidates
    for a in powerful[-1000:]:
        for b in powerful[-1000:]:
            s = a + b
            if s <= 10**18:
                candidate_ns.add(s)
    
    best_N = None
    best_count = 0
    best_pairs = []
    
    n_list = sorted(candidate_ns)
    
    for N in n_list:
        if time.time() > deadline - 10:
            break
        
        # Two-pointer to count pairs
        i = 0
        j = len(powerful) - 1
        count = 0
        pairs = []
        while i <= j:
            s = powerful[i] + powerful[j]
            if s == N:
                count += 1
                pairs.append([powerful[i], powerful[j]])
                i += 1
                j -= 1
            elif s < N:
                i += 1
            else:
                j -= 1
        
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    if best_N is None:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}
