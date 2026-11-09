import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        """Generate all powerful numbers <= limit."""
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        sq_free_v = []
        for v in range(1, max_v + 1):
            if is_square_free(v):
                sq_free_v.append(v)
                
        for v in sq_free_v:
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    def count_pairs_for_N(N, P_set, P_list):
        """Count pairs (a,b) with a<=b, a+b=N, both in P_set."""
        pairs = []
        for a in P_list:
            if a > N // 2:
                break
            b = N - a
            if b in P_set:
                pairs.append([a, b])
        return pairs

    def verify_powerful(n):
        """Check if n is powerful by exact factorization."""
        if n < 1:
            return False
        if n == 1:
            return True
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count == 1:
                    return False
            p += 1 if p == 2 else 2
        # temp is now a prime factor
        if temp != 1:
            # temp appears exactly once
            return False
        return True

    def construct_from_bases(bases, max_u):
        """
        For each base v in bases, generate powerful numbers u^2 * v^3 for u=1..max_u.
        Collect all such numbers, then find N with max representations.
        """
        powerful = set()
        for v in bases:
            v3 = v ** 3
            if v3 > 10**18:
                continue
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x > 10**18:
                    break
                powerful.add(x)
        
        P_list = sorted(powerful)
        P_set = set(P_list)
        
        # Use numpy to find sums with max count
        if len(P_list) > 50000:
            # Too many, sample or use a different approach
            # Take a subset
            if len(P_list) > 200000:
                # Random sample
                import random
                random.seed(42)
                sample_size = 200000
                P_list = random.sample(P_list, sample_size)
                P_set = set(P_list)
        
        P_np = np.array(P_list, dtype=np.int64)
        
        # Generate all sums and count
        # To avoid memory issues, do it in chunks
        best_N = None
        best_count = 0
        
        # Chunked approach
        chunk_size = 10000
        n_p = len(P_np)
        
        # Use a dictionary to count sums (more memory efficient for sparse)
        sum_counts = defaultdict(int)
        
        for i in range(0, n_p, chunk_size):
            if time.time() > deadline - 10:
                break
            chunk = P_np[i:i+chunk_size]
            # For each element in chunk, compute sum with all P_np
            sums = chunk[:, None] + P_np[None, :]
            # Count each sum
            flat_sums = sums.ravel()
            # Use bincount for speed
            max_sum = flat_sums.max()
            if max_sum <= 2 * 10**18:
                counts = np.bincount(flat_sums, minlength=max_sum + 1)
                for s, c in enumerate(counts):
                    if c > 0:
                        sum_counts[int(s)] += int(c)
        
        # Find the N with max count
        if sum_counts:
            best_N = max(sum_counts.keys(), key=lambda x: sum_counts[x])
            best_count = sum_counts[best_N]
            
            # Verify best_N is within range
            if best_N > 10**18:
                # Find next best
                sorted_sums = sorted(sum_counts.keys(), key=lambda x: -sum_counts[x])
                best_N = None
                for s in sorted_sums:
                    if s <= 10**18:
                        best_N = s
                        best_count = sum_counts[s]
                        break
            
            if best_N is not None and best_N >= 2:
                # Count distinct pairs
                pairs = count_pairs_for_N(best_N, P_set, P_list)
                return {'N': best_N, 'pairs': pairs}
        
        return None

    # Strategy:
    # 1. Generate powerful numbers up to a reasonable limit using the standard method.
    # 2. Use numpy bincount to find N with max representations.
    # 3. If that fails due to memory, use a chunked/dictionary approach.
    
    best_result = None
    
    # Try different limits
    limits = [500_000_000, 200_000_000, 100_000_000, 50_000_000]
    
    for M in limits:
        if time.time() > deadline - 15:
            break
        
        try:
            P = generate_powerful(M)
            L = len(P)
            
            if L == 0:
                continue
            
            P_np = np.array(P, dtype=np.int64)
            P_set = set(P)
            
            # If L is small enough, do full outer sum
            if L <= 20000:
                # L^2 = 4*10^8, which is ~3.2GB for int64. Might be tight.
                # Let's try with a smaller threshold.
                if L <= 10000:
                    sums = np.add.outer(P_np, P_np)
                    max_sum_val = int(sums.max())
                    counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
                    max_count_idx = int(np.argmax(counts))
                    best_N = max_count_idx
                    C = int(counts[best_N])
                    
                    # Check if N is within range
                    if best_N > 10**18:
                        continue
                    
                    # Count distinct pairs
                    pairs = count_pairs_for_N(best_N, P_set, P)
                    
                    if len(pairs) > 0:
                        current_result = {'N': best_N, 'pairs': pairs}
                        if best_result is None or len(pairs) > len(best_result['pairs']):
                            best_result = current_result
                        # Return immediately if we have a good result
                        return current_result
                else:
                    # L between 10000 and 20000: use chunked approach
                    sum_counts = defaultdict(int)
                    chunk_size = 5000
                    n_p = L
                    
                    for i in range(0, n_p, chunk_size):
                        if time.time() > deadline - 10:
                            break
                        chunk = P_np[i:i+chunk_size]
                        sums = chunk[:, None] + P_np[None, :]
                        flat_sums = sums.ravel()
                        max_sum = int(flat_sums.max())
                        if max_sum <= 2 * 10**18:
                            counts = np.bincount(flat_sums, minlength=max_sum + 1)
                            for s in range(len(counts)):
                                c = int(counts[s])
                                if c > 0:
                                    sum_counts[s] += c
                    
                    if sum_counts:
                        # Find best N within range
                        best_N = None
                        best_count = 0
                        for s, c in sum_counts.items():
                            if s <= 10**18 and c > best_count:
                                best_count = c
                                best_N = s
                        
                        if best_N is not None and best_N >= 2:
                            pairs = count_pairs_for_N(best_N, P_set, P)
                            if len(pairs) > 0:
                                current_result = {'N': best_N, 'pairs': pairs}
                                if best_result is None or len(pairs) > len(best_result['pairs']):
                                    best_result = current_result
                                return current_result
            else:
                # L > 20000: use chunked approach with smaller chunks
                sum_counts = defaultdict(int)
                chunk_size = 2000
                n_p = L
                
                for i in range(0, n_p, chunk_size):
                    if time.time() > deadline - 10:
                        break
                    chunk = P_np[i:i+chunk_size]
                    sums = chunk[:, None] + P_np[None, :]
                    flat_sums = sums.ravel()
                    max_sum = int(flat_sums.max())
                    if max_sum <= 2 * 10**18:
                        counts = np.bincount(flat_sums, minlength=max_sum + 1)
                        for s in range(len(counts)):
                            c = int(counts[s])
                            if c > 0:
                                sum_counts[s] += c
                
                if sum_counts:
                    best_N = None
                    best_count = 0
                    for s, c in sum_counts.items():
                        if s <= 10**18 and c > best_count:
                            best_count = c
                            best_N = s
                    
                    if best_N is not None and best_N >= 2:
                        pairs = count_pairs_for_N(best_N, P_set, P)
                        if len(pairs) > 0:
                            current_result = {'N': best_N, 'pairs': pairs}
                            if best_result is None or len(pairs) > len(best_result['pairs']):
                                best_result = current_result
                            return current_result
        
        except MemoryError:
            continue
        except Exception:
            continue

    # Fallback: use a small set of bases to construct powerful numbers
    if time.time() < deadline - 20:
        try:
            bases = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31]
            max_u = 10000
            result = construct_from_bases(bases, max_u)
            if result and len(result['pairs']) > 0:
                if best_result is None or len(result['pairs']) > len(best_result['pairs']):
                    best_result = result
                return result
        except Exception:
            pass

    # Final fallback
    if best_result:
        return best_result
    
    try:
        P = generate_powerful(10000)
        P_np = np.array(P, dtype=np.int64)
        P_set = set(P)
        sums = np.add.outer(P_np, P_np)
        counts = np.bincount(sums.ravel(), minlength=int(sums.max()) + 1)
        best_N = int(np.argmax(counts))
        pairs = count_pairs_for_N(best_N, P_set, P)
        return {'N': best_N, 'pairs': pairs}
    except:
        return {'N': 2, 'pairs': [[1, 1]]}

