import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin from 150s limit
    
    # We want to find M with many solutions k such that k * sigma(k) = M.
    # Let's try to construct M from small primes.
    # If M is a product of small primes, it might have many divisors k 
    # such that sigma(k) is also a divisor of M (or rather, M/k).
    
    # Strategy:
    # 1. Generate candidate M values that are "smooth" (product of small primes).
    # 2. For each candidate M, find all k <= sqrt(M) such that k * sigma(k) = M.
    #    We can do this by iterating through divisors k of M? No, k doesn't have to divide M.
    #    Wait, if k * sigma(k) = M, then k must divide M.
    #    Proof: sigma(k) is an integer, so k divides M.
    #    So, for a fixed M, the solutions k must be divisors of M.
    #    This is a huge insight! We don't need to scan all k up to sqrt(M).
    #    We only need to scan the divisors of M.
    
    # So the algorithm is:
    # 1. Generate candidate M values.
    # 2. For each M, find all divisors k of M such that k <= sqrt(M) and sigma(k) == M/k.
    # 3. Count such k's.
    # 4. Return the M with the most such k's.
    
    # How to generate good candidate M's?
    # M should have many divisors k such that sigma(k) is an integer (always true) 
    # and specifically sigma(k) = M/k.
    # This implies M/k = sigma(k).
    # So k * sigma(k) = M.
    
    # Let's try to find M by searching over k's that are "sigma-abundant" or similar?
    # No, let's stick to generating M.
    # If we pick M to be a highly composite number, it has many divisors.
    # But we need k to be a divisor AND satisfy the sigma condition.
    
    # Alternative: Generate k's that are "good" and compute M = k * sigma(k).
    # Then group by M.
    # This is what the current program does, but it's slow because it computes sigma for ALL k up to N.
    # Can we be smarter about which k's to check?
    # k must be <= sqrt(M). If M <= 10^18, k <= 10^9.
    # We can't iterate up to 10^9.
    
    # Let's look at the constraint: k * sigma(k) = M.
    # If we pick k to be a prime p, M = p * (p+1).
    # If we pick k to be p^a, M = p^a * (p^{a+1}-1)/(p-1).
    
    # Let's try a hybrid:
    # Generate a set of "seed" k values that are likely to produce collisions or high counts.
    # Actually, the current program found M=333312 with 3 solutions.
    # 333312 = 2^5 * 3^2 * 7 * 11 * 13?
    # Let's factor 333312.
    # 333312 / 2 = 166656
    # 166656 / 2 = 83328
    # 83328 / 2 = 41664
    # 41664 / 2 = 20832
    # 20832 / 2 = 10416
    # 10416 / 2 = 5208
    # 5208 / 2 = 2604
    # 2604 / 2 = 1302
    # 1302 / 2 = 651
    # 651 / 3 = 217
    # 217 = 7 * 31
    # So 333312 = 2^9 * 3 * 7 * 31.
    
    # The solutions were 336, 372, 434.
    # 336 = 2^4 * 3 * 7. sigma(336) = (1+2+4+8+16)*(1+3)*(1+7) = 31 * 4 * 8 = 992.
    # 336 * 992 = 333312. Correct.
    # 372 = 2^2 * 3 * 31. sigma(372) = (1+2+4)*(1+3)*(1+31) = 7 * 4 * 32 = 896.
    # 372 * 896 = 333312. Correct.
    # 434 = 2 * 7 * 31. sigma(434) = (1+2)*(1+7)*(1+31) = 3 * 8 * 32 = 768.
    # 434 * 768 = 333312. Correct.
    
    # Notice that the prime factors of M are {2, 3, 7, 31}.
    # The k's use subsets of these primes.
    
    # New Strategy:
    # 1. Pick a set of small primes P.
    # 2. Generate M as a product of powers of these primes.
    # 3. For each M, enumerate all divisors k of M.
    # 4. Check if k * sigma(k) == M.
    # 5. To make this fast, we need to generate M's that have many divisors, 
    #    and we need to compute sigma(k) fast for divisors k.
    
    # Let's generate M's from the first few primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47.
    # We will iterate over combinations of exponents for these primes to form M.
    # We want M <= 10^18.
    
    # To avoid generating too many M's, we can be selective.
    # We know that k must be a divisor of M.
    # If M has d(M) divisors, we check d(M) candidates for k.
    # We want d(M) to be large, but M to be small enough to have many valid k's.
    
    # Let's try to construct M from primes up to 47.
    # We'll use a priority queue or just nested loops for the first few primes.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3
    
    # We will generate candidate M values.
    # Since M <= 10^18, and we use small primes, we can have high exponents.
    # Let's iterate over the number of distinct prime factors used.
    
    # Helper to compute sigma of a number given its prime factorization
    def sigma_from_factorization(factors):
        # factors is dict {p: a}
        s = 1
        for p, a in factors.items():
            s *= (p**(a+1) - 1) // (p - 1)
        return s
    
    # Helper to get divisors of M from its factorization
    def get_divisors(factors):
        divs = [1]
        for p, a in factors.items():
            new_divs = []
            p_pow = 1
            for i in range(a + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        return divs
    
    # We need to be careful with time.
    # Let's generate M's by combining small prime powers.
    # We'll use a list of candidate M's.
    
    # Let's try specific structures.
    # The example M=333312 used primes 2,3,7,31.
    # Let's try to find M's with more solutions by searching over prime sets.
    
    # We'll generate M's from the first 10 primes.
    # To keep the number of M's manageable, we'll limit the exponents.
    
    # Let's try a random search over the exponents of the first 10 primes.
    # But we need to be systematic to find the best.
    
    # Let's try to iterate over M values that are "smooth" with respect to small primes.
    # We'll generate all M <= 10^12 that are 5-smooth? No, too many.
    
    # Let's go back to the divisor idea.
    # If we pick M, we check its divisors.
    # The number of divisors d(M) can be large (e.g., >1000).
    # Checking sigma(k) for 1000 divisors is fast if we have the factorization.
    
    # Let's generate a list of candidate M's.
    # We'll pick M's that are products of the first 6-8 primes with small exponents.
    
    # Generate candidates
    candidates = []
    
    # We'll use a recursive generation for the first 8 primes.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19
    # Max exponent for 2: ~60 (2^60 ~ 10^18)
    # But we want to mix primes.
    
    # Let's try a different approach:
    # Iterate k from 1 to some limit L (e.g., 100,000).
    # Compute M = k * sigma(k).
    # Store in a dict: M -> list of k's.
    # This is what the current program does, but it uses numpy arrays which are slow for dynamic sizing or large N.
    # Using a pure Python dict might be faster for smaller N, or we can optimize the numpy version.
    
    # The current program's bottleneck is likely the SPF sieve and sigma DP for N=100,000,000.
    # 100M iterations in Python/NumPy might take 10-30s.
    # If we can reduce N to 10M or 5M, it will be much faster.
    # But does N=10M yield more solutions?
    # The best found was k=434. So N=1000 was enough for that M.
    # Maybe there are M's with solutions k > 1000 that have more solutions?
    # If k ~ 1000, M ~ 1000 * 1000 * log(1000) ~ 10^7.
    # If k ~ 10^5, M ~ 10^5 * 10^6 ~ 10^11.
    # If k ~ 10^6, M ~ 10^6 * 10^7 ~ 10^13.
    
    # Let's try to search for M's with k up to 1,000,000.
    # We can do this in chunks to manage memory.
    
    # Let's implement a chunked search.
    # Chunk size: 1,000,000.
    # For each chunk, compute sigma for k in [start, end].
    # Compute M = k * sigma(k).
    # Add to a global dictionary: M -> count (or list of k's if count > best).
    
    # To compute sigma fast, we need SPF up to end.
    # We can compute SPF up to 1,000,000 once.
    # Then for each k, factor it using SPF and compute sigma.
    # This is O(log k) per k. Total O(N log N).
    # For N=1,000,000, this is fast.
    
    # Let's do this.
    
    MAX_K = 2_000_000  # Let's try up to 2M
    
    # 1. Sieve SPF up to MAX_K
    spf = list(range(MAX_K + 1))
    for i in range(2, int(MAX_K**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_K + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # 2. Function to compute sigma(k) using SPF
    def calc_sigma(k):
        if k == 1:
            return 1
        res = 1
        while k > 1:
            p = spf[k]
            power_sum = 1
            power_val = 1
            while k % p == 0:
                k //= p
                power_val *= p
                power_sum += power_val
            res *= power_sum
        return res
    
    # 3. Iterate k, compute M, store in dict
    # To avoid storing all k's for all M (memory), we can store counts first, 
    # and if a count exceeds best, we re-scan or store the k's.
    # But we need the list of k's for the final answer.
    # Let's store: M -> [k1, k2, ...]
    # If the list gets too big, we might run out of memory, but M is bounded by 10^18.
    # Number of distinct M's will be large, but most will have 1 solution.
    # We can filter: only keep M if count >= 2?
    # No, we don't know the best count yet.
    # Let's store counts, and if count > best_count, we store the k's.
    # Actually, we can just store the list of k's for all M, but prune if list length > 1000?
    # Or just store counts, and at the end, re-scan for the best M to get the k's?
    # Re-scanning 2M k's is fast.
    
    m_to_k_list = {}
    
    for k in range(1, MAX_K + 1):
        if time.time() > deadline:
            break
        s = calc_sigma(k)
        M = k * s
        if M > 10**18:
            continue
        if M not in m_to_k_list:
            m_to_k_list[M] = []
        m_to_k_list[M].append(k)
        
    # Find the M with the most k's
    # Sort by length of list
    # To avoid sorting a huge dict, we can track the max during iteration?
    # But we need to check all.
    # Let's find the max length.
    
    max_count = 0
    best_M_candidate = None
    
    for M, k_list in m_to_k_list.items():
        if len(k_list) > max_count:
            max_count = len(k_list)
            best_M_candidate = M
    
    if best_M_candidate is not None and max_count > best_count:
        best_M = best_M_candidate
        best_k_list = m_to_k_list[best_M]
        best_count = max_count
        
    # If we have time, try larger MAX_K?
    # But 2M is likely enough to find good M's.
    # The previous best was k=434.
    # If there are solutions with k > 2M, we might miss them.
    # But k * sigma(k) grows fast.
    # If k ~ 10^6, M ~ 10^13.
    # If k ~ 10^7, M ~ 10^15.
    # If k ~ 10^8, M ~ 10^17.
    # If k ~ 10^9, M ~ 10^19 (too big).
    # So k up to 10^8 is possible for M <= 10^18.
    # 2M might be too small.
    
    # Let's try to extend if time permits.
    # But the SPF sieve up to 10^8 is 400MB (int64) or 100MB (int32).
    # And the loop is 10^8 iterations. Might be too slow.
    
    # Let's stick to 2M for now, or maybe 5M.
    # Let's try 5M if time allows.
    
    if time.time() < deadline - 20:
        MAX_K2 = 5_000_000
        # Extend SPF? No, just re-sieve or use a larger sieve.
        # Let's just re-run with larger MAX_K if we haven't found a very good score.
        # But re-running is expensive.
        # Let's just use the result from 2M.
        
    # Ensure types
    return {'M': int(best_M), 'k': best_k_list}

if __name__ == "__main__":
    construct()
