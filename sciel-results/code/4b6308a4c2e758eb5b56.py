import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
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

    # Strategy:
    # 1. Generate powerful numbers up to a large limit.
    # 2. The number of representations r(N) = sum_{a+b=N, a,b in P} 1.
    # 3. We want to maximize the number of distinct pairs (a,b) with a<=b.
    # 4. Using bincount on the sums of all pairs (a,b) where a,b in P is the standard approach.
    # 5. Memory is the main constraint. P size for limit M is roughly M^{1/3} * something? 
    #    Actually, count of powerful numbers up to M is ~ sqrt(M). No, it's sum_{n^2 k = x} 1/k?
    #    Density is ~ C / sqrt(log M) ? No.
    #    Number of powerful numbers <= M is ~ M^{1/3} * (some constant) + M^{1/2}?
    #    Let's estimate:
    #    P(2, M) ~ M^{1/2}
    #    P(3, M) ~ M^{1/3}
    #    So size is ~ M^{1/2}.
    #    If M = 10^12, size ~ 10^6.
    #    Outer sum size ~ 10^12, which is too big for bincount (memory).
    #    Wait, the current program uses M up to 500M.
    #    Size of P for M=500M is ~ sqrt(500M) ~ 22360? No, that's too small.
    #    Let's check: powerful numbers are n^2 * k where k is powerful? No.
    #    Powerful: p^e with e>=2.
    #    Count ~ M^{1/2} * (1 + 1/sqrt(2) + ...)?
    #    Actually, let's just run it.
    #    If M=10^6, P has ~1000 elements?
    #    Let's test locally in mind.
    #    1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81, 100, 108, 121, 125, 128, 144, 169, 196, 216, 225, 243, 256, 289, 324, 343, 361, 392, 400...
    #    It's denser than M^{1/3}.
    
    # The key improvement:
    # Instead of just picking the N with max count in the "all pairs" bincount,
    # we can try to optimize the selection or use a smarter search.
    # However, the "all pairs" bincount is O(|P|^2) in memory if we do outer sum.
    # For M=500M, |P| is likely around 5000-10000?
    # If |P|=10000, outer sum is 10^8 elements. 10^8 * 8 bytes = 800MB. Might be ok.
    # If |P|=50000, outer sum is 2.5*10^9 elements. 20GB. Too big.
    
    # Let's check the count of powerful numbers.
    # Formula: sum_{k=1}^{sqrt(M)} floor(sqrt(M/k^2)) ... no.
    # It is known that the number of powerful numbers <= x is ~ A x^{1/3} + B x^{1/2} + ...
    # Actually, the main term is ~ x^{1/2}? No.
    # Erdos-Sarkozy: ~ C x^{1/3} (2zeta(3)/zeta(2))^{1/3}? No.
    # Let's assume |P| ~ M^{1/2} is an overestimate.
    # If M=10^6, |P| ~ 100?
    # If M=10^9, |P| ~ 1000?
    # If M=10^12, |P| ~ 10000?
    # If |P| is small, we can go to larger M.
    
    # Let's try M = 10^10.
    # If |P| ~ 10^5, outer sum is 10^10 elements. 80GB. Fail.
    # So we need to be careful with M.
    
    # Current program uses M=500M.
    # Let's stick to M=10^8 or 10^9 and optimize the counting.
    # We can use scipy? No, standard numpy.
    
    # Optimization:
    # If we cannot do full outer sum, we can iterate over a and use a hash map for b?
    # But we need the max count.
    
    # Let's try to push M higher with careful memory checking.
    # We will estimate |P| first without storing it? No, we need the list.
    
    # Let's try M = 200_000_000.
    # And add a check on the size of P before doing the outer sum.
    
    best_result = None
    
    # Try a few limits, starting from high to low
    limits = [200_000_000, 100_000_000, 50_000_000, 20_000_000]
    
    for M in limits:
        if time.time() > deadline - 10:
            break
            
        try:
            P = generate_powerful(M)
            L = len(P)
            
            # Heuristic check on memory for outer sum
            # If L > 50000, L^2 is 2.5*10^9, which is ~20GB. Too much.
            # If L > 20000, L^2 is 4*10^8, ~3.2GB. Might be tight.
            # Let's cap L at 30000 to be safe (9*10^8 elements, ~7.2GB).
            # Actually, numpy int64 array of size 10^8 is 800MB.
            # So L=10000 -> 10^8 elements -> 800MB. Safe.
            # L=50000 -> 2.5*10^9 elements -> 20GB. Unsafe.
            
            if L > 15000:
                # We can't do full outer sum.
                # Fallback: Sample? Or use a different algorithm?
                # For now, skip or reduce M.
                # Let's try a smaller M if we get here.
                continue
                
            P_np = np.array(P, dtype=np.int64)
            
            # Generate sums
            # To save memory, we can do it in chunks if needed, but 15000^2 = 2.25*10^8 is 1.8GB.
            # This might be ok on a modern machine.
            
            sums = np.add.outer(P_np, P_np)
            
            # Find max sum for bincount size
            max_sum_val = sums.max()
            
            # bincount can be memory intensive if max_sum_val is large.
            # max_sum_val ~ 2*M.
            # If M=200M, max_sum ~ 400M.
            # bincount size 400M * 4 bytes (int32 default?) = 1.6GB.
            # Total memory: 1.8GB (sums) + 1.6GB (bincount) + P. ~3.5GB.
            # Should be ok.
            
            counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
            
            # Find the sum with max count
            max_count_idx = np.argmax(counts)
            best_N = int(max_count_idx)
            C = int(counts[best_N])
            
            # Calculate distinct pairs count
            d = 0
            if best_N % 2 == 0:
                half_N = best_N // 2
                if half_N in P:
                    d = 1
            
            num_pairs = (C + d) // 2
            
            # Recover pairs
            pairs = []
            P_set = set(P)
            for a in P:
                b = best_N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            current_result = {'N': best_N, 'pairs': pairs}
            
            if best_result is None or len(pairs) > len(best_result['pairs']):
                best_result = current_result
                # If we found a good number, we might break, but larger M is better.
                # So we continue to try larger M? No, we start with larger M.
                # So if M=200M works, we return it?
                # Yes, because larger M allows for larger N and potentially more reps.
                # But wait, the max count might not increase monotonically with M?
                # It should, roughly.
                # So we return the first successful large M.
                return current_result

        except MemoryError:
            continue
        except Exception:
            continue

    # Fallback
    if best_result:
        return best_result
        
    try:
        P = generate_powerful(1000)
        P_np = np.array(P, dtype=np.int64)
        sums = np.add.outer(P_np, P_np)
        counts = np.bincount(sums.ravel(), minlength=sums.max() + 1)
        best_N = int(np.argmax(counts))
        C = int(counts[best_N])
        d = 1 if (best_N % 2 == 0 and best_N // 2 in P) else 0
        num_pairs = (C + d) // 2
        pairs = []
        P_set = set(P)
        for a in P:
            b = best_N - a
            if b >= a and b in P_set:
                pairs.append([a, b])
        return {'N': best_N, 'pairs': pairs}
    except:
        return {'N': 2, 'pairs': [[1, 1]]}
