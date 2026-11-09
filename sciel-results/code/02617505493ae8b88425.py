import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        """Generate all powerful numbers <= limit using u^2 * v^3 form."""
        powerful = set()
        # v must be square-free and v^3 <= limit
        max_v = int(limit ** (1.0/3.0)) + 1
        # Precompute square-free numbers up to max_v
        sq_free = []
        for v in range(1, max_v + 1):
            if is_square_free(v):
                sq_free.append(v)
        
        for v in sq_free:
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        
        return sorted(powerful)

    def count_representations(N, P_set, P_list):
        """Count number of pairs (a,b) with a<=b, a+b=N, a,b in P."""
        count = 0
        for a in P_list:
            if a > N // 2:
                break
            b = N - a
            if b >= a and b in P_set:
                count += 1
        return count

    best_result = None
    best_count = 0

    # Phase 1: Generate powerful numbers up to a moderate limit
    # and find good candidate N values
    # Phase 2: For top candidates, verify with larger P if time permits

    # Try several limits, starting with a reasonable one
    limits_to_try = [500_000_000, 200_000_000, 100_000_000]

    for M in limits_to_try:
        if time.time() > deadline - 15:
            break

        try:
            P = generate_powerful(M)
            L = len(P)
            P_set = set(P)
            P_np = np.array(P, dtype=np.int64)

            # If L is manageable, do full outer sum
            if L <= 8000:
                # Full outer sum approach
                sums = np.add.outer(P_np, P_np)
                max_sum_val = int(sums.max())
                counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
                
                # Find top few candidates
                top_indices = np.argsort(counts)[::-1][:20]
                
                for idx in top_indices:
                    N = int(idx)
                    C = int(counts[N])
                    if C < 2:
                        continue
                    
                    # Count actual distinct pairs
                    d = 0
                    if N % 2 == 0:
                        half_N = N // 2
                        if half_N in P_set:
                            d = 1
                    
                    num_pairs = (C + d) // 2
                    
                    if num_pairs > best_count:
                        # Recover pairs
                        pairs = []
                        for a in P:
                            b = N - a
                            if b < a:
                                continue
                            if b in P_set:
                                pairs.append([a, b])
                        
                        best_count = num_pairs
                        best_result = {'N': N, 'pairs': pairs}
            else:
                # L too large for full outer sum. Use sampling + targeted search.
                # Sample random pairs to estimate distribution of sums
                n_samples = min(500000, L * L // 10)
                if n_samples < 1000:
                    n_samples = 1000
                
                # Use numpy for fast random sampling
                idx_i = np.random.randint(0, L, size=n_samples)
                idx_j = np.random.randint(0, L, size=n_samples)
                sample_sums = P_np[idx_i] + P_np[idx_j]
                
                # Find most common sums in sample
                unique_sums, sample_counts = np.unique(sample_sums, return_counts=True)
                top_sample_indices = np.argsort(sample_counts)[::-1][:50]
                
                # For each top candidate, count exact representations
                candidates = set()
                for si in top_sample_indices:
                    candidates.add(int(unique_sums[si]))
                
                # Also try some structured candidates: near multiples of small numbers
                # Powerful numbers tend to cluster around certain values
                for c in list(candidates)[:30]:
                    if time.time() > deadline - 5:
                        break
                    num_pairs = count_representations(c, P_set, P)
                    if num_pairs > best_count:
                        pairs = []
                        for a in P:
                            b = c - a
                            if b < a:
                                continue
                            if b in P_set:
                                pairs.append([a, b])
                        best_count = num_pairs
                        best_result = {'N': c, 'pairs': pairs}

            # If we found a good result with large M, we can stop
            if best_count >= 10:
                break

        except MemoryError:
            continue
        except Exception:
            continue

    # Fallback: try a small limit
    if best_result is None:
        try:
            P = generate_powerful(100000)
            P_set = set(P)
            P_np = np.array(P, dtype=np.int64)
            sums = np.add.outer(P_np, P_np)
            counts = np.bincount(sums.ravel(), minlength=int(sums.max()) + 1)
            best_N = int(np.argmax(counts))
            C = int(counts[best_N])
            d = 1 if (best_N % 2 == 0 and best_N // 2 in P_set) else 0
            num_pairs = (C + d) // 2
            pairs = []
            for a in P:
                b = best_N - a
                if b >= a and b in P_set:
                    pairs.append([a, b])
            best_result = {'N': best_N, 'pairs': pairs}
        except:
            best_result = {'N': 2, 'pairs': [[1, 1]]}

    return best_result
