"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120.0

    def elapsed():
        return time.time() - start_time

    def remaining():
        return deadline - time.time()

    # Generate powerful numbers up to L
    # Powerful numbers: every prime factor appears with exponent >= 2
    # A powerful number has the form: product of p_i^e_i where e_i >= 2
    # We can generate them by: for each square a^2, multiply by cubes of primes?
    # Actually, a powerful number n can be written uniquely as a^2 * b^3 where b is squarefree.
    # So we generate all a^2 * b^3 <= L where b is squarefree.

    # Strategy:
    # 1. Generate all squares a^2 <= L
    # 2. For each squarefree b, generate b^3 * a^2 <= L for a = 1, 2, ...
    # But b^3 grows fast. For b=1: a^2 <= L, so a <= sqrt(L)
    # For b=2: 8a^2 <= L, a <= sqrt(L/8)
    # etc.

    # First, find all squarefree numbers up to L^(1/3) since b^3 <= L means b <= L^(1/3)
    L = 50_000_000  # We'll adjust based on timing

    # Actually, let's use a different approach:
    # Generate powerful numbers directly using the characterization.
    # A number n is powerful iff for every prime p dividing n, p^2 divides n.
    # Equivalently, n = a^2 * b where b is squarefree and gcd(a,b) might not be 1... 
    # Wait, the standard form is: n = a^2 * b^3 where b is squarefree. This is unique.
    # Proof: write n = product p_i^e_i. Let e_i = 2*f_i + g_i where g_i in {0,1,2}.
    # If g_i = 0: p_i^(2f_i) = (p_i^f_i)^2
    # If g_i = 1: p_i^(2f_i+1) = (p_i^f_i)^2 * p_i^1... but then p_i appears in b with exponent 1, 
    #   but b must be squarefree, so this works: a gets p_i^f_i, b gets p_i^1.
    #   Wait no: n = a^2 * b^3. If p_i^e_i || n, then e_i = 2*v_p(a) + 3*v_p(b).
    #   Since b is squarefree, v_p(b) is 0 or 1.
    #   If v_p(b) = 0: e_i = 2*v_p(a), so e_i is even.
    #   If v_p(b) = 1: e_i = 2*v_p(a) + 3, so e_i >= 3 and e_i - 3 is even.
    #   So e_i mod 6 determines v_p(b): if e_i mod 6 is 0,2,4 then v_p(b)=0; if 3,5,1 then v_p(b)=1? 
    #   Let's check: e_i=3: 3=2*0+3*1, v_p(b)=1. e_i=5: 5=2*1+3*1, v_p(b)=1. 
    #   e_i=1: not powerful. e_i=2: 2=2*1+3*0, v_p(b)=0. e_i=4: 4=2*2+3*0, v_p(b)=0.
    #   e_i=6: 6=2*3+3*0, v_p(b)=0. e_i=7: 7=2*2+3*1, v_p(b)=1.
    #   So v_p(b) = 1 iff e_i mod 6 is in {3,5,1}? No: e_i=1 is not powerful. 
    #   For powerful numbers, e_i >= 2. So e_i mod 6 in {2,4,0} -> v_p(b)=0; {3,5,1} but e_i>=2 so {3,5} -> v_p(b)=1.
    #   Actually e_i mod 6 = 1 would mean e_i=1,7,13,... e_i=1 not powerful, e_i=7: 7=2*2+3*1, v_p(b)=1. OK.

    # So to generate powerful numbers up to L:
    # For each squarefree b with b^3 <= L, and each a with a^2 * b^3 <= L, 
    #   n = a^2 * b^3 is powerful.

    # Number of squarefree b with b^3 <= L: b <= L^(1/3) ~ 37 for L=50M. That's very few.
    # For each such b, a <= sqrt(L / b^3).

    # Let's generate all powerful numbers up to L this way.
    def gen_powerful(L):
        max_b = int(round(L ** (1/3))) + 2
        powerful = set()
        # Find squarefree numbers up to max_b
        # Use sieve-like approach
        is_sqfree = [True] * (max_b + 1)
        is_sqfree[0] = False
        p = 2
        while p * p <= max_b:
            if is_sqfree[p]:
                p2 = p * p
                for j in range(p2, max_b + 1, p2):
                    is_sqfree[j] = False
            p += 1
        
        sqfree = [i for i in range(1, max_b + 1) if is_sqfree[i]]
        
        for b in sqfree:
            b3 = b ** 3
            if b3 > L:
                break
            max_a = int((L // b3) ** 0.5) + 1
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerful.add(n)
        return sorted(powerful)

    # Start with a moderate L
    L = 20_000_000
    powerful = gen_powerful(L)
    print(f"Generated {len(powerful)} powerful numbers up to {L}")

    powerful_set = set(powerful)
    n_powerful = len(powerful)

    # Now, find N with many representations as sum of two powerful numbers.
    # Use numpy for efficiency: compute all pairwise sums? That's O(n^2) which is too large.
    # Instead, for each a in powerful, compute b = N - a and check if b is in powerful_set.
    # But we want to find the best N. We can't iterate over all N up to 10^18.
    # 
    # Alternative: generate all sums a+b where a <= b and a,b are powerful, and count frequencies.
    # The number of powerful numbers up to L is roughly L^(2/3) or so. For L=20M, that's maybe ~10000.
    # 10000^2 / 2 = 50M sums. That's feasible with numpy.

    # Let's convert to numpy array
    pw = np.array(powerful, dtype=np.int64)
    
    # Generate all pairs (i, j) with i <= j, compute sums, count.
    # To avoid O(n^2) memory, we can do it in chunks or use a different approach.
    # For n=10000, n^2/2 = 50M. Each sum is 8 bytes, so 400MB. That's tight.
    # Let's try with smaller L first if needed.

    if n_powerful > 15000:
        # Too many, reduce L
        L = 10_000_000
        powerful = gen_powerful(L)
        pw = np.array(powerful, dtype=np.int64)
        n_powerful = len(powerful)
        powerful_set = set(powerful)

    # Generate all pairwise sums i <= j
    # Use broadcasting in chunks to save memory
    best_N = 2
    best_count = 0
    best_pairs = []

    # Chunk-based approach
    chunk_size = 2000
    counts = {}

    for i_start in range(0, n_powerful, chunk_size):
        i_end = min(i_start + chunk_size, n_powerful)
        pw_i = pw[i_start:i_end]
        
        for j_start in range(i_start, n_powerful, chunk_size):
            j_end = min(j_start + chunk_size, n_powerful)
            pw_j = pw[j_start:j_end]
            
            # Compute all sums
            sums = pw_i[:, None] + pw_j[None, :]
            
            # Only keep i <= j, i.e., sums where the index constraint holds
            # Since i_start <= j_start, all i in [i_start, i_end) and j in [j_start, j_end) 
            # satisfy i <= j only if i <= j. But within the chunk, we need i <= j.
            # Actually, since we're iterating i_start <= j_start, for any i in [i_start,i_end) and j in [j_start,j_end),
            # we have i >= i_start and j >= j_start >= i_start, but i could be > j if i_end > j_start.
            # Let's just filter: keep only where pw_i[:,None] <= pw_j[None,:]
            
            mask = pw_i[:, None] <= pw_j[None, :]
            sums = sums[mask]
            
            if len(sums) == 0:
                continue
            
            # Count occurrences
            # Use np.unique with return_counts
            unique_sums, cnts = np.unique(sums, return_counts=True)
            
            for s, c in zip(unique_sums, cnts):
                s_int = int(s)
                if s_int not in counts:
                    counts[s_int] = 0
                counts[s_int] += c
            
            if elapsed() > deadline - 30:
                break
        
        if elapsed() > deadline - 30:
            break

    # Find the best N
    for n_val, cnt in counts.items():
        if cnt > best_count:
            best_count = cnt
            best_N = n_val

    # Now generate the pairs for best_N
    pairs = []
    for a in powerful:
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs.append([a, b])
            if len(pairs) >= best_count:
                break

    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
