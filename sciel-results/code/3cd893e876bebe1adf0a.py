import time
import numpy as np
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 10s safety margin
    deadline = start_time + 140.0

    def generate_powerful(limit):
        """Generate all powerful numbers <= limit."""
        pw = set()
        # Powerful numbers are of the form u^2 * v^3
        # v^3 <= limit => v <= limit^(1/3)
        max_v = int(limit ** (1/3)) + 2
        
        for v in range(1, max_v + 1):
            v_cubed = v ** 3
            if v_cubed > limit:
                break
            # u^2 <= limit / v^3 => u <= sqrt(limit / v^3)
            max_u = isqrt(limit // v_cubed)
            for u in range(1, max_u + 1):
                p = u * u * v_cubed
                pw.add(p)
        return sorted(pw)

    # Strategy:
    # 1. Generate powerful numbers up to M.
    # 2. Instead of one huge FFT of size 2*M (which might OOM for large M),
    #    we bucket the powerful numbers by value.
    # 3. For each pair of buckets (i, j), we compute the convolution.
    #    The max sum from bucket i and j is max_val_i + max_val_j.
    #    We can choose M such that the largest bucket pair FFT fits in memory.
    #    Actually, a simpler memory-safe approach for large M:
    #    If M is too large for a single FFT, we can split the *range of sums* we care about?
    #    No, we want the max over all N.
    #    
    #    Let's refine the "Bucketed Convolution" idea:
    #    If we have powerful numbers $p_1, \dots, p_K$.
    #    We want $C_N = \sum_{i \le j} [p_i + p_j = N]$.
    #    
    #    If we split the powerful numbers into two sets $A$ and $B$ (e.g., by index or value),
    #    $C_N = C^A_N + C^B_N + C^{AB}_N$? No, that's not right because pairs can be within A or within B.
    #    
    #    Correct decomposition:
    #    Let $S$ be the set of powerful numbers.
    #    Split $S$ into $S_1$ and $S_2$ such that $S_1 \cup S_2 = S$ and $S_1 \cap S_2 = \emptyset$.
    #    Total pairs for N:
    #    Pairs from $S_1 \times S_1$ (including diagonal)
    #    Pairs from $S_2 \times S_2$ (including diagonal)
    #    Pairs from $S_1 \times S_2$
    #    
    #    This doesn't reduce the FFT size if $S_1$ and $S_2$ span the full range $[1, M]$.
    #    The FFT size is determined by the max value in the set.
    #    
    #    However, if we bucket by *value*:
    #    Bucket $k$ contains powerful numbers in $[L_k, R_k]$.
    #    Sum of two numbers from bucket $i$ and $j$ is in $[L_i+L_j, R_i+R_j]$.
    #    The FFT size required for this specific pair of buckets is $R_i + R_j + 1$.
    #    The largest FFT size will be for the buckets containing the largest numbers.
    #    If we split the range $[1, M]$ into buckets, the largest bucket is near $M$.
    #    So the peak memory is still determined by the largest bucket pair.
    #    
    #    Is there a way to avoid the large FFT?
    #    Yes, if we notice that for very large $N$, the number of representations is likely small.
    #    But we want the MAX. The max is likely at moderate $N$.
    #    
    #    Alternative Structural Change: **Hybrid Deterministic Search**
    #    1. Use FFT for a "core" range of powerful numbers $M_{core}$ (e.g., $10^7$).
    #       This gives exact counts for $N \le 2 M_{core}$.
    #    2. For $N > 2 M_{core}$, we can't use the core FFT.
    #       But we can generate powerful numbers up to $M_{large}$ (e.g., $5 \times 10^7$).
    #    3. For the "tail" powerful numbers $P_{tail} = \{p \in P \mid p > M_{core}\}$,
    #       the number of such numbers is small.
    #       $K_{core} \approx M_{core}^{1/3}$. $K_{large} \approx M_{large}^{1/3}$.
    #       $K_{tail} = K_{large} - K_{core}$.
    #       If $M_{core} = 10^7$, $K_{core} \approx 215$.
    #       If $M_{large} = 5 \times 10^7$, $K_{large} \approx 368$.
    #       $K_{tail} \approx 153$.
    #       
    #    We can compute the contributions of pairs involving at least one tail number
    #    using direct iteration, which is fast because $K_{tail}$ is small.
    #    
    #    Let $P = P_{core} \cup P_{tail}$.
    #    Pairs $(a, b)$ with $a, b \in P$.
    #    Case 1: $a, b \in P_{core}$. Handled by FFT on $P_{core}$.
    #    Case 2: $a \in P_{core}, b \in P_{tail}$ (or vice versa).
    #    Case 3: $a, b \in P_{tail}$.
    #    
    #    For Case 2 and 3, we can iterate over the smaller set ($P_{tail}$) and check against the full set?
    #    Or just iterate over all pairs in $P_{tail}$ and pairs between $P_{core}$ and $P_{tail}$.
    #    
    #    Number of pairs in Case 2: $K_{core} \times K_{tail} \approx 215 \times 153 \approx 33k$.
    #    Number of pairs in Case 3: $K_{tail}^2 / 2 \approx 153^2 / 2 \approx 11k$.
    #    Total direct checks: ~45k. Very fast.
    #    
    #    This allows us to extend $M$ to $5 \times 10^7$ or even $10^8$ easily,
    #    because the FFT only needs to handle $M_{core}$, and the tail is handled by O($K_{tail} K$) direct checks.
    #    We can increase $M_{core}$ to improve the FFT resolution and capture more "dense" regions.
    #    Let's set $M_{core} = 2 \times 10^7$. $K_{core} \approx 271$.
    #    FFT size $4 \times 10^7$. Float32 -> 160MB. Safe.
    #    Let's set $M_{large} = 10^8$. $K_{large} \approx 464$.
    #    $K_{tail} \approx 193$.
    #    Direct checks: $271 \times 193 + 193^2/2 \approx 52k + 18k = 70k$. Fast.
    
    M_core = 20_000_000
    M_large = 100_000_000
    
    # Check if we have time for M_large generation
    if time.time() > deadline - 50:
        M_large = M_core
        
    powerful_list_large = generate_powerful(M_large)
    powerful_set_large = set(powerful_list_large)
    
    # Identify core and tail
    # Core: powerful numbers <= M_core
    # Tail: powerful numbers > M_core and <= M_large
    core_list = [p for p in powerful_list_large if p <= M_core]
    tail_list = [p for p in powerful_list_large if p > M_core]
    
    K_core = len(core_list)
    K_tail = len(tail_list)
    
    # Initialize counts dictionary
    # We will accumulate counts for N.
    # N can be up to 2 * M_large.
    counts = defaultdict(int)
    
    # Part 1: FFT for Core-Core pairs
    if K_core > 0:
        # Create indicator array for core_list
        # Size needs to be at least max(core_list) + 1 for the input,
        # but FFT size should be >= 2 * max(core_list) to avoid circular convolution.
        max_core = core_list[-1]
        fft_size = 1
        while fft_size < 2 * max_core:
            fft_size *= 2
        
        # Use float32 for memory efficiency
        core_arr = np.zeros(fft_size, dtype=np.float32)
        for p in core_list:
            core_arr[p] = 1.0
            
        # FFT
        fft_core = np.fft.rfft(core_arr)
        conv_core = np.fft.irfft(fft_core * fft_core, n=fft_size)
        
        # The result conv_core[n] is the number of ordered pairs (a, b) in core_list
        # such that a + b = n.
        # We want unordered pairs a <= b.
        # For a != b, count in conv is 2 * unordered_count.
        # For a == b, count in conv is 1 * unordered_count (since (a,a) is one pair).
        # Actually, conv[n] = sum_{a+b=n} 1.
        # If a=b, it contributes 1.
        # If a!=b, it contributes 2 (a,b) and (b,a).
        # So unordered_count(n) = (conv[n] + delta_{n, 2a}) / 2?
        # Let's just extract the integer counts and adjust.
        # conv_core is float32, might have small errors.
        # Round to nearest integer.
        
        # We only care about N values that are sums of two core numbers.
        # Max N from core-core is 2 * max_core.
        for n in range(2, 2 * max_core + 1):
            c = int(np.round(conv_core[n]))
            if c > 0:
                # c is the number of ordered pairs.
                # If n is even, and n/2 is in core_list, then one pair is (n/2, n/2).
                # The number of unordered pairs is (c + 1) // 2 if (n/2 in core) else c // 2.
                # Actually, simpler:
                # c = 2 * (# pairs a<b) + 1 * (# pairs a=b)
                # We want # pairs a<=b = # pairs a<b + # pairs a=b
                # Let x = # pairs a=b (0 or 1).
                # c = 2 * (total_unordered - x) + x = 2 * total_unordered - x.
                # total_unordered = (c + x) / 2.
                # x is 1 if n/2 is in core_list and n is even, else 0.
                
                x = 0
                if n % 2 == 0:
                    half = n // 2
                    if half in powerful_set_large: # Check if it's in core (it must be <= M_core if n <= 2*M_core and half <= M_core)
                        # Need to ensure half is in core_list
                        if half <= M_core:
                            x = 1
                
                unordered_c = (c + x) // 2
                if unordered_c > 0:
                    counts[n] += unordered_c

    # Part 2: Direct iteration for pairs involving Tail
    # Pairs (a, b) where at least one is in tail_list.
    # To avoid double counting, we can iterate:
    # 1. All pairs (a, b) with a, b in tail_list, a <= b.
    # 2. All pairs (a, b) with a in core_list, b in tail_list.
    #    Note: a + b = N.
    
    # Optimization: 
    # For tail-tail:
    for i in range(K_tail):
        a = tail_list[i]
        for j in range(i, K_tail):
            b = tail_list[j]
            n = a + b
            counts[n] += 1
            
    # For core-tail:
    # a in core, b in tail.
    # Since core and tail are disjoint, a != b is always true?
    # Not necessarily, but values are distinct.
    # We just add 1 for each pair.
    for a in core_list:
        for b in tail_list:
            n = a + b
            counts[n] += 1

    # Find the N with the maximum count
    if not counts:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Get top candidates
    # Sort by count descending
    # We need to return the pairs for the best N.
    # We don't have the pairs stored, only counts.
    # We need to reconstruct the pairs for the best N.
    
    # Get the best N
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Reconstruct pairs for best_N
    pairs = []
    # Iterate through all powerful numbers a <= best_N // 2
    # Check if b = best_N - a is in powerful_set_large
    # And a <= b
    
    # Since powerful_list_large is sorted, we can break early.
    for a in powerful_list_large:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set_large:
            # a <= b is guaranteed by the loop condition a <= best_N // 2
            # Also a+b=best_N
            # We must ensure a and b are distinct? The task says "distinct pairs a <= b".
            # Usually "distinct pairs" means the set of pairs is unique.
            # If a==b, it's one pair.
            pairs.append([a, b])
            
    # Verify count matches
    if len(pairs) != best_count:
        # This shouldn't happen, but if it does, we might have a bug in counting logic.
        # We return the pairs we found. The score is len(pairs).
        pass
        
    return {'N': best_N, 'pairs': pairs}

