import math
import time
from collections import Counter

def construct():
    """
    Find N with many representations as a sum of two powerful numbers.
    
    Strategy:
    1. Generate all powerful numbers up to a reasonable limit.
    2. Use a sparse dictionary-based accumulation (Counter) over pairs of powerful numbers
       to count how many pairs sum to each N.
    3. Find the top candidates by count.
    4. Verify the top candidates and extract their pairs.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit

    def generate_powerful_numbers(limit):
        """Generate all powerful numbers up to limit."""
        powerful = set()
        max_y = int(math.ceil(limit ** (1.0/3.0))) + 2
        
        # Sieve for square-free numbers up to max_y
        sq_free = [True] * (max_y + 1)
        sq_free[0] = False
        p = 2
        while p * p <= max_y:
            if sq_free[p]:
                sq = p * p
                for j in range(sq, max_y + 1, sq):
                    sq_free[j] = False
            p += 1
        
        for y in range(1, max_y + 1):
            if not sq_free[y]:
                continue
            y3 = y * y * y
            if y3 > limit:
                break
            max_x = int(math.isqrt(limit // y3))
            for x in range(1, max_x + 1):
                n = x * x * y3
                powerful.add(n)
        
        return sorted(powerful)

    # Generate powerful numbers up to a limit
    # We want to cover sums up to N_max. If we generate powerful numbers up to L,
    # we can find sums up to 2*L.
    # Let's try L = 5 * 10^7. This gives ~200k powerful numbers.
    # Pairs: ~200k^2 / 2 = 2 * 10^10 pairs. This is too many for a naive double loop in Python.
    # We need to be smarter.
    
    # Alternative: Use a smaller L but be very efficient, or use a hybrid approach.
    # Let's try L = 10^6 first to see the performance, then scale up.
    # For L = 10^6, #powerful ~ 10^4. Pairs ~ 5 * 10^7. This is feasible in Python with optimization.
    
    # Let's use a moderate limit that balances search space and computation time.
    # L = 5 * 10^6: #powerful ~ 3 * 10^5? No, L^(2/3) = (5*10^6)^(2/3) ~ 30000.
    # Pairs ~ 30000^2 / 2 = 4.5 * 10^8. This might be too slow in pure Python.
    
    # Let's use NumPy for the pair generation to speed it up.
    # Or, stick to a smaller limit for the sparse method.
    
    # Actually, the FFT method was scoring 5.8. Let's see if we can beat it with a different approach.
    # The sparse method is O(P^2) where P is the number of powerful numbers.
    # For L = 10^6, P ~ 10^4. P^2 = 10^8. In Python, 10^8 operations is ~10-30 seconds.
    # For L = 10^7, P ~ 4.6 * 10^4. P^2 = 2 * 10^9. Too slow.
    
    # So we are limited to L ~ 10^6 for the sparse method in Python.
    # This limits N to 2 * 10^6.
    # The FFT method was using L = 5 * 10^7, N up to 10^8.
    # The sparse method will likely find fewer representations because the search space is smaller.
    
    # Let's try a different structural change:
    # Instead of generating all powerful numbers, generate them in a way that allows
    # us to focus on N that are likely to have many representations.
    
    # Key insight: N = 2 * Q where Q is powerful has at least 1 representation (Q, Q).
    # N = Q1 + Q2 where Q1, Q2 are powerful.
    
    # Let's try the sparse method with L = 2 * 10^6 and see.
    L = 2 * 10**6
    P = generate_powerful_numbers(L)
    P_set = set(P)
    n_p = len(P)
    
    # Use a Counter to count pairs
    pair_counts = Counter()
    
    # Optimize the double loop
    # We only need a <= b, so a <= N/2.
    # But we don't know N in advance.
    # We can just count all pairs (a, b) with a <= b and a + b <= 2*L.
    
    # To speed this up, we can break early if a > 2*L - a (i.e., a > L), but a is always <= L.
    # Actually, a and b are both <= L, so a + b <= 2*L.
    
    # Let's use a list of P and iterate.
    # For each a in P:
    #   For each b in P with b >= a:
    #     if a + b > 2*L: break
    #     pair_counts[a + b] += 1
    
    # This is O(P^2) in the worst case.
    # For L = 2 * 10^6, P ~ 40000. P^2 = 1.6 * 10^9. Too slow.
    
    # We need to reduce P. Let's use L = 5 * 10^5.
    # P ~ (5*10^5)^(2/3) ~ 6000. P^2 = 3.6 * 10^7. This is feasible.
    
    L = 5 * 10**5
    P = generate_powerful_numbers(L)
    P_set = set(P)
    n_p = len(P)
    
    pair_counts = Counter()
    
    # Convert P to a list for faster indexing
    P_list = P
    
    for i, a in enumerate(P_list):
        # b must be >= a and a + b <= 2*L
        # b <= 2*L - a
        # Find the largest index j such that P_list[j] <= 2*L - a
        # Since P_list is sorted, we can binary search.
        max_b = 2 * L - a
        # Find the rightmost index where P_list[idx] <= max_b
        # Use bisect
        import bisect
        j_max = bisect.bisect_right(P_list, max_b)
        
        for j in range(i, j_max):
            b = P_list[j]
            s = a + b
            pair_counts[s] += 1
    
    # Get top candidates
    top_candidates = pair_counts.most_common(50)
    
    best_n = 0
    best_pairs = []
    
    for N, count in top_candidates:
        if N < 2:
            continue
        
        # Verify and extract pairs
        pairs = []
        for a in P_list:
            if a > N // 2:
                break
            b = N - a
            if b in P_set:
                pairs.append([a, b])
        
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_n = N
    
    # If we didn't find anything good, try a slightly larger L with a timeout check
    if time.time() < deadline - 30 and len(best_pairs) < 5:
        L2 = 10**6
        P2 = generate_powerful_numbers(L2)
        P2_set = set(P2)
        P2_list = P2
        n_p2 = len(P2)
        
        pair_counts2 = Counter()
        
        import bisect
        for i, a in enumerate(P2_list):
            max_b = 2 * L2 - a
            j_max = bisect.bisect_right(P2_list, max_b)
            for j in range(i, j_max):
                b = P2_list[j]
                s = a + b
                pair_counts2[s] += 1
        
        top_candidates2 = pair_counts2.most_common(50)
        
        for N, count in top_candidates2:
            if N < 2:
                continue
            
            pairs = []
            for a in P2_list:
                if a > N // 2:
                    break
                b = N - a
                if b in P2_set:
                    pairs.append([a, b])
            
            if len(pairs) > len(best_pairs):
                best_pairs = pairs
                best_n = N
    
    if not best_pairs:
        best_n = 5
        best_pairs = [[1, 4]]
    
    return {'N': best_n, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
