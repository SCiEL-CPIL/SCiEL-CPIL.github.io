"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from collections import Counter

def generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit."""
    if limit < 1:
        return []
    
    powerful = set()
    
    # A powerful number is of the form prod(p_i^{e_i}) where e_i >= 2
    # We can generate them by iterating over possible "square part" and "cube part"
    # Actually, a simpler approach: powerful numbers are numbers where for every prime p dividing n, p^2 divides n.
    # Equivalently, n = a^2 * b^3 where a and b are integers and b is square-free? No.
    # The standard generation: n = prod p_i^{k_i} with k_i >= 2.
    # We can write n = x^2 * y^3 where y is square-free? Not exactly.
    # Let's use the recursive generation or the a^2 * b^3 method.
    # Any powerful number can be written as a^2 * b^3 where b is square-free.
    # Let's verify: if n = prod p_i^{k_i}, let k_i = 2*m_i + 3*n_i? No.
    # If k_i >= 2, we can write k_i = 2 + r_i. Then n = (prod p_i) * (prod p_i^{r_i}).
    # This doesn't help directly.
    
    # Let's use the property: n is powerful iff n = a^2 * b^3 for some a, b?
    # Example: 8 = 2^3. 8 = 2^2 * 2^1. Here a=2, b=2? 2^2*2^3 = 16*8=128. No.
    # 8 = 1^2 * 2^3. So a=1, b=2.
    # 12 = 2^2 * 3. Not powerful (3^1).
    # 72 = 2^3 * 3^2. 72 = (2*3)^2 * 2^1 = 36*2. Not a^2*b^3 form easily?
    # 72 = 6^2 * 2. 2 is not a cube.
    # Wait, the representation n = a^2 * b^3 is not unique and not all powerful numbers fit simply.
    
    # Better algorithm:
    # Generate all squares a^2 <= limit.
    # For each square a^2, generate b^3 such that a^2 * b^3 <= limit.
    # But this generates numbers like 4 * 8 = 32 (2^5, powerful), 4 * 27 = 108 (2^2*3^3, powerful).
    # Does this cover all powerful numbers?
    # Let n be powerful. n = prod p_i^{k_i}, k_i >= 2.
    # Let k_i = 2 + 2*u_i + 3*v_i? No.
    # We can always write k_i = 2*m_i + 3*n_i?
    # If k_i = 2, m_i=1, n_i=0.
    # If k_i = 3, m_i=0, n_i=1.
    # If k_i = 4, m_i=2, n_i=0.
    # If k_i = 5, m_i=1, n_i=1.
    # If k_i = 6, m_i=3, n_i=0 or m_i=0, n_i=2.
    # So yes, any k_i >= 2 can be represented as 2*m + 3*n with m,n >= 0.
    # Thus n = (prod p_i^{m_i})^2 * (prod p_i^{n_i})^3 = a^2 * b^3.
    # So generating all a^2 * b^3 <= limit covers all powerful numbers.
    
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= limit:
                powerful.add(n)
    
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s buffer for 120s limit
    
    # Strategy:
    # 1. Generate powerful numbers up to a limit L.
    # 2. Compute all pairwise sums a+b (with a <= b) and count frequencies.
    # 3. Find N with highest frequency.
    # 4. Verify N <= 10^18.
    
    # How large can L be?
    # If we generate up to L, we can find N up to 2L.
    # We want N <= 10^18, so L can be up to 5*10^17.
    # But generating powerful numbers up to 5*10^17 might be too many.
    # Number of powerful numbers up to X is approx O(sqrt(X)).
    # For X = 10^18, sqrt(X) = 10^9. Too many.
    # For X = 10^12, sqrt(X) = 10^6. ~1M numbers.
    # Pairs: 10^6 * 10^6 / 2 = 5*10^11. Too many.
    
    # We need a balance.
    # Let's try L = 10^9.
    # Powerful numbers up to 10^9: approx sqrt(10^9) = 31622.
    # Actually, let's estimate better.
    # Count of powerful numbers up to N is ~ c * N^{1/2}.
    # For N=10^9, count is around 30k-40k.
    # Pairs: 40k * 40k / 2 = 800M. This is feasible in C/Python with optimization?
    # Python loops are slow. 800M iterations will take too long.
    
    # We need to be smarter or smaller.
    # Let's try L = 10^7.
    # Powerful numbers up to 10^7: approx 3000-4000.
    # Pairs: 4000 * 4000 / 2 = 8M. Very fast.
    # Max N found: 2 * 10^7 = 2*10^7.
    
    # Can we do better?
    # Maybe L = 10^8?
    # Powerful numbers up to 10^8: approx 10000.
    # Pairs: 10000 * 10000 / 2 = 50M.
    # In Python, 50M dictionary increments might take 10-30 seconds.
    # This is risky but possible with the 100s deadline.
    
    # Let's try to maximize L within time.
    # We can generate powerful numbers, then use a Counter for sums.
    # To speed up, we can use NumPy?
    # NumPy broadcasting for sums of two lists of size N is N^2 memory.
    # If N=10000, N^2 = 10^8 integers. 800MB memory. Might be too much.
    # If N=5000, N^2 = 25M integers. 200MB. Feasible.
    
    # Let's stick to Python loops with Counter for safety, but optimize the generation.
    # Or use a hybrid: generate list, then nested loops.
    
    # Let's try L = 5 * 10^7 first.
    # Estimate powerful count: ~ sqrt(5*10^7) * c.
    # Let's just generate and see how many we get.
    
    L = 10**7
    powerful = generate_powerful_numbers(L)
    n_pow = len(powerful)
    
    # If n_pow is large, we might need to reduce L or use a different strategy.
    # Let's assume n_pow is manageable (e.g., < 10000).
    
    # Count sums
    sum_counts = Counter()
    
    # Optimization: use local variables for speed
    pow_list = powerful
    n = n_pow
    
    # Nested loop
    for i in range(n):
        a = pow_list[i]
        # b must be >= a
        for j in range(i, n):
            b = pow_list[j]
            s = a + b
            sum_counts[s] += 1
    
    # Find best N
    # We want max count, and N <= 10^18.
    # Since L is small, all sums are <= 2L which is << 10^18.
    best_N, best_count = sum_counts.most_common(1)[0]
    
    # Verify and extract pairs
    pairs = []
    for i in range(n):
        a = pow_list[i]
        b = best_N - a
        if b >= a and b in sum_counts: # b must be powerful
             # Check if b is in powerful list
             # Since we generated powerful up to L, and b <= 2L, b might be > L?
             # No, a >= 1, so b <= best_N - 1 < 2L.
             # But our powerful list only goes up to L.
             # If b > L, it's not in our list.
             # However, if a <= L and b <= L, then a+b <= 2L.
             # If b > L, it wasn't generated.
             # But wait, if b > L, then b is not in `powerful`.
             # So we must ensure b is in `powerful`.
             if b <= L:
                 # Check if b is actually powerful
                 # Since `powerful` is a list, checking membership is O(n).
                 # Better to use a set for membership.
                 pass
    
    # Let's redo the pair extraction properly.
    pow_set = set(powerful)
    pairs = []
    for a in powerful:
        b = best_N - a
        if b >= a and b in pow_set:
            pairs.append([a, b])
    
    # Sort pairs for consistency
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
