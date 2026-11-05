import time
from math import gcd, lcm
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for safety
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCMs for all pairs in [1, 128]
    # lcm[i][j] is the LCM of i+1 and j+1
    lcm_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = i + 1, j + 1
            lcm_table[i][j] = lcm(a, b)
            lcm_table[j][i] = lcm_table[i][j]
            
    def is_valid_triple(i, j, k):
        """
        Returns True if the triple of elements (i+1, j+1, k+1) is VALID,
        i.e., their pairwise LCMs are NOT all equal.
        """
        l_ij = lcm_table[i][j]
        l_ik = lcm_table[i][k]
        l_jk = lcm_table[j][k]
        return not (l_ij == l_ik == l_jk)
    
    def is_compatible_with_set(elem_idx, current_set_indices):
        """
        Check if adding elem_idx to current_set_indices creates any bad triple.
        A bad triple is formed if there exist a, b in current_set_indices
        such that lcm(a, b) == lcm(a, elem) == lcm(b, elem).
        """
        for idx_a in current_set_indices:
            l_ae = lcm_table[idx_a][elem_idx]
            for idx_b in current_set_indices:
                if idx_b == idx_a:
                    continue
                l_be = lcm_table[idx_b][elem_idx]
                l_ab = lcm_table[idx_a][idx_b]
                if l_ae == l_be == l_ab:
                    return False
        return True

    # Strategy:
    # 1. Identify "prime powers" (numbers of the form p^k).
    #    These are "structurally safe" because if a, b, c are distinct prime powers,
    #    their prime factorizations involve disjoint sets of primes or nested primes.
    #    It is a known result that sets of prime powers contain no equal-LCM triples.
    #    Proof sketch: If lcm(a,b)=lcm(a,c)=lcm(b,c)=L, then a,b,c must divide L.
    #    If a=p^x, b=q^y, c=r^z with p,q,r distinct primes, L=p^x q^y r^z.
    #    lcm(a,b)=p^x q^y, lcm(a,c)=p^x r^z. These are equal only if q^y=r^z,
    #    which implies q=r (contradiction) or y=z=0 (not prime powers >1).
    #    If p=q, say a=p^x, b=p^y, c=q^z (q!=p).
    #    lcm(a,b)=p^max(x,y). lcm(a,c)=p^x q^z. Equal => p^max(x,y) = p^x q^z => q^z = p^max(x,y)-x.
    #    This requires q^z to be a power of p, so q=p, contradiction.
    #    Thus, distinct prime powers are mutually compatible in this sense.
    
    prime_powers = set()
    for i in range(1, n + 1):
        num = i
        base = num
        while num % base == 0:
            num //= base
        if num == 1:
            prime_powers.add(i)
            
    # Start with all prime powers
    selected_indices = [i - 1 for i in prime_powers]
    
    # 2. Add remaining numbers greedily.
    #    We want to maximize cardinality.
    #    Sort remaining numbers by some heuristic? 
    #    Numbers with fewer prime factors might be "safer" or "more dangerous"?
    #    Actually, numbers with many small prime factors have many divisors and might create more LCM conflicts.
    #    Let's just try adding them in natural order, or maybe descending order of "complexity"?
    #    Natural order is fine for a first pass.
    
    remaining_indices = [i for i in range(n) if i not in selected_indices]
    
    # To improve results, we can try multiple orders or local search.
    # But first, let's just do a solid greedy.
    # We can sort remaining by number of prime factors to see if that helps.
    
    def count_prime_factors(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count
        
    # Heuristic: Numbers with fewer distinct prime factors are "closer" to prime powers?
    # Actually, let's just iterate.
    
    for idx in remaining_indices:
        if time.time() > deadline:
            break
        if is_compatible_with_set(idx, selected_indices):
            selected_indices.append(idx)
            
    # 3. Local Search / Swap Optimization
    #    Try to swap an included element with an excluded one to increase size?
    #    Or just try to add more? The greedy already added all compatible ones.
    #    Maybe we can remove one "bad" element to allow two "good" ones?
    #    This is complex. Let's stick to the constructive set.
    #    The set of prime powers is size ~50.
    #    Greedy adds more.
    
    # Convert indices back to values
    result = [i + 1 for i in selected_indices]
    
    # Final verification (optional, but good for safety)
    # If time permits, verify no bad triples.
    if time.time() < deadline - 1.0:
        res_set = set(result)
        valid = True
        # Check all triples in result
        # This is O(N^3) which for N~60 is ~36k checks, very fast.
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                for k in range(j + 1, len(result)):
                    a, b, c = result[i], result[j], result[k]
                    if lcm(a, b) == lcm(a, c) == lcm(b, c):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            # Fallback to just prime powers if greedy created a bug (unlikely)
            result = [i for i in prime_powers]
            
    return result if result else [1]

# Quick test logic to ensure function signature is correct
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Sample: {res[:10]}")
