import math
import time
import random
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def time_left():
        return deadline - time.time()

    def is_powerful(n):
        if n < 1:
            return False
        if n == 1:
            return True
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                cnt = 0
                while temp % d == 0:
                    temp //= d
                    cnt += 1
                if cnt % 2 != 0:
                    return False
            d += 1 if d == 2 else 2
        return True

    # Generate a diverse set of powerful numbers
    # Form: A * B^2
    # A is square-free
    # B is any integer
    # We want numbers up to 5 * 10^17 (so sums are up to 10^18)
    
    powerfuls = set()
    
    # 1 is powerful
    powerfuls.add(1)
    
    # Small squares
    for i in range(2, 10000):
        sq = i * i
        if sq <= 5 * 10**17:
            powerfuls.add(sq)
    
    # Products of distinct small primes (square-free part)
    # This generates many powerful numbers
    # 2 * (k)^2, 3 * (k)^2, 5 * (k)^2, etc.
    # We'll generate a few thousand more to increase diversity
    
    # Primes for square-free part
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Generate random powerful numbers
    # Target count: ~5000-10000 to make sampling efficient
    target_count = 10000
    current_count = len(powerfuls)
    
    # Add more structured powerfuls
    # p * k^2
    for p in small_primes:
        # k up to sqrt(5*10^17 / p)
        max_k = int(math.isqrt(5 * 10**17 // p))
        # Sample some k values
        # If p is small, max_k is large. We can't add all.
        # We'll add a mix of small k and random k.
        if p <= 7:
            # Add all small k
            for k in range(1, min(max_k, 1000)):
                val = p * k * k
                powerfuls.add(val)
        
        # Add random k
        for _ in range(200):
            k = random.randint(1, max_k)
            val = p * k * k
            powerfuls.add(val)
            
        if len(powerfuls) > target_count:
            break
            
    # Add more random powerfuls of form (prod p_i^e_i) where e_i >= 2
    # This is getting complex. The set above (squares + p*k^2) is likely sufficient 
    # to capture the "dense" regions of powerful numbers.
    
    # Ensure we have enough
    if len(powerfuls) < 5000:
        # Add more squares
        for i in range(10000, 20000):
            powerfuls.add(i*i)

    powerful_list = list(powerfuls)
    random.shuffle(powerful_list)
    
    # Filter to ensure sums are within 10^18
    # Actually, if a > 5*10^17 and b > 5*10^17, sum > 10^18.
    # We should probably filter the list to <= 5*10^17 to be safe for random pairing.
    # Or just check in the loop.
    
    # Sampling Phase
    # We want to find N that appears frequently as a + b.
    # Randomly sample pairs from powerful_list.
    # If the list is size M, pairs are M^2. 
    # Sampling S pairs takes S time.
    # We have ~100s. Python can do ~10^7 simple ops/sec.
    # Let's do 2 * 10^7 samples? That might be tight.
    # Let's do 5 * 10^6 samples first, see how it goes.
    # Or adaptive.
    
    samples = Counter()
    
    # Estimate time per sample
    # List access, addition, dict update. ~1-2 us?
    # 10^7 samples ~ 10-20s.
    # Let's aim for 15s of sampling.
    
    max_samples = 10_000_000
    
    # To speed up, we can restrict the list size if it's too large?
    # No, larger list is better for diversity.
    # But if list is 100k, random access is fast.
    
    # Optimization: Pre-compute nothing? 
    # Just loop.
    
    # We can use numpy for vectorized addition?
    # numpy is available.
    # Generate random indices, add arrays, histogram.
    # This is MUCH faster than Python loop.
    
    import numpy as np
    
    # Convert to numpy array
    # Filter to ensure a+b <= 10^18
    # If we pick a, b from list, and list max is 5*10^17, sum is <= 10^18.
    # Let's ensure list elements are <= 5*10^17.
    powerful_list = [x for x in powerful_list if x <= 5 * 10**17]
    
    if not powerful_list:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    arr = np.array(powerful_list, dtype=np.uint64)
    
    # Sampling
    # We can do batched sampling.
    # Batch size 10^6.
    
    batch_size = 100_000
    total_samples = 0
    target_samples = 20_000_000 # 20 million samples ~ 20-40s with numpy?
    
    # Numpy random choice is fast.
    # Addition is fast.
    # np.unique with return_counts is fast.
    
    # However, storing 20 million integers in a Counter is memory heavy?
    # 20M * 8 bytes = 160MB. Plus overhead. Might be okay.
    # But updating a Python Counter with 20M items is slow.
    # Better to use numpy histogram or unique on the sums.
    
    # Strategy:
    # 1. Generate N1, N2 random indices.
    # 2. Sum = arr[N1] + arr[N2].
    # 3. Use np.unique(sums, return_counts=True).
    # 4. Update a global Counter or just keep the top ones.
    
    # Since we want the GLOBAL frequency, we can accumulate counts.
    # But numpy unique gives local counts.
    # We can sum them into a dictionary.
    
    # Let's do a few large batches.
    
    global_counts = {}
    
    while total_samples < target_samples and time_left() > 30:
        # Generate indices
        # Using np.random.randint is fast
        i1 = np.random.randint(0, len(arr), size=batch_size)
        i2 = np.random.randint(0, len(arr), size=batch_size)
        
        s1 = arr[i1]
        s2 = arr[i2]
        sums = s1 + s2
        
        # Filter sums > 10^18? 
        # If arr max is 5*10^17, max sum is 10^18. So no filter needed.
        
        # Get unique sums and counts
        unique_sums, counts = np.unique(sums, return_counts=True)
        
        # Update global counts
        # This loop is over unique sums in the batch, which is much smaller than batch_size
        for s, c in zip(unique_sums, counts):
            global_counts[s] = global_counts.get(s, 0) + c
            
        total_samples += batch_size
        
        # Check time
        if time_left() < 40:
            break
            
    # Find top candidates
    if not global_counts:
        # Fallback to random
        # Just pick a random sum
        i1 = random.randint(0, len(arr)-1)
        i2 = random.randint(0, len(arr)-1)
        N = arr[i1] + arr[i2]
        candidates = [N]
    else:
        # Sort by count
        # Top 100 candidates
        # Sorting 10k-100k items is fast.
        sorted_items = sorted(global_counts.items(), key=lambda x: x[1], reverse=True)
        candidates = [s for s, c in sorted_items[:100]]
        
    # Verification Phase
    # For each candidate N, find all pairs (a, b) such that a+b=N and a,b powerful.
    # We have the set `powerfuls`.
    # We can iterate through `powerfuls` and check if N-a is in `powerfuls`.
    # This is O(K) per candidate, where K is size of powerfuls.
    # If we have 100 candidates and K=10,000, that's 1,000,000 checks. Fast.
    
    best_N = 0
    best_pairs = []
    
    # To speed up verification, we can pre-filter candidates.
    # Only verify if count is high enough?
    # Or just verify top 20.
    
    # Let's verify top 50.
    for N in candidates[:50]:
        if N < 2 or N > 10**18:
            continue
            
        pairs = []
        # Iterate through powerfuls
        # To avoid duplicates (a,b) and (b,a), and ensure a<=b:
        # We can just collect all, then sort and dedupe.
        # Or iterate and only add if a <= b.
        
        # Since powerfuls is a set, lookup is O(1).
        # But iterating a set is slow in Python?
        # Converting to list is faster for iteration?
        # We already have powerful_list.
        
        # Optimization:
        # If we iterate powerful_list, and for each a, check if b = N-a is in powerfuls (set).
        # If yes, and a <= b, add [a, b].
        
        # To avoid checking all 10k for 50 candidates (500k ops), which is fine.
        
        for a in powerful_list:
            if a > N:
                break # If list is sorted. It's not.
            b = N - a
            if b < a:
                continue
            if b in powerfuls:
                pairs.append([a, b])
                
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_N = N
            
        if time_left() < 10:
            break
            
    # If we didn't find any good pairs, try a few more random candidates
    if not best_pairs:
        for _ in range(10):
            i1 = random.randint(0, len(arr)-1)
            i2 = random.randint(0, len(arr)-1)
            N = arr[i1] + arr[i2]
            pairs = []
            for a in powerful_list:
                b = N - a
                if b < a:
                    continue
                if b in powerfuls:
                    pairs.append([a, b])
            if len(pairs) > len(best_pairs):
                best_pairs = pairs
                best_N = N
                
    # Fallback
    if not best_pairs:
        best_N = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
