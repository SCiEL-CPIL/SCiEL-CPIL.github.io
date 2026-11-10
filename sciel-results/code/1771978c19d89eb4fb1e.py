"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

P = {'n': 40, 'maximum_size': 18}

def _prime_factors(n):
    """Return dictionary of prime -> exponent for n."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill
    
    # 1. Generate candidate integers in [2, 40]
    candidates = list(range(2, 41))
    
    # 2. Compute prime factorization vectors for each candidate
    #    We use a fixed set of primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    vectors = []
    for num in candidates:
        facs = _prime_factors(num)
        vec = []
        for p in primes:
            vec.append(facs.get(p, 0))
        vectors.append(vec)
    
    # 3. Branch and Bound to find maximum size linearly independent subset
    #    We sort candidates by "complexity" or just use natural order.
    #    Actually, for linear independence, the order doesn't affect the max size,
    #    but it affects the search path.
    #    A common heuristic is to try including smaller numbers first? 
    #    Or maybe numbers with more unique prime factors?
    #    Let's stick to natural order 2..40 for simplicity, or maybe sort by number of prime factors.
    #    Numbers with many distinct prime factors (like 30, 42...) are "rich" but might block others.
    #    Primes are "pure" and never block each other (unless same prime, which is impossible here).
    #    Actually, primes are the safest to include.
    #    Let's sort by: number of distinct prime factors (ascending), then value.
    #    This prioritizes primes and prime powers, which are "cheap" in terms of span usage?
    #    Wait, a prime vector is e_i. A composite like 6 is e_2 + e_3.
    #    If we include 6, we use up dimensions 2 and 3.
    #    If we include 2 and 3, we also use up dimensions 2 and 3.
    #    So 6 is equivalent to {2,3} in terms of dimension cost, but uses 1 slot instead of 2.
    #    Therefore, we should PREFER composites?
    #    If we pick 6, we get 1 vector. If we pick 2 and 3, we get 2 vectors.
    #    We want to MAXIMIZE the number of vectors.
    #    So we want vectors that are "sparse" or "independent" in a way that doesn't waste dimension?
    #    No, we want to pack as many independent vectors as possible.
    #    The maximum number of independent vectors is bounded by the dimension of the space (12 primes).
    #    But we can have more if we use composite exponents?
    #    Wait, the vectors are in Q^12. The max size of a linearly independent set is 12.
    #    However, the task allows up to 18 integers.
    #    Is it possible to have > 12 integers with distinct subset products?
    #    Yes, if they are not linearly independent?
    #    No! The condition "all 2^k subset products are distinct" is EQUIVALENT to the vectors being linearly independent over Q.
    #    Proof: If vectors are linearly dependent, there exist coefficients c_i in {-1, 0, 1} (actually integers) such that sum c_i v_i = 0.
    #    This implies product of x_i^{c_i} = 1.
    #    If c_i are in {-1, 0, 1}, this means product of some elements equals product of some others.
    #    Since elements are distinct, this implies a collision in subset products (unless the subsets are disjoint and empty, but 1 is not in the set).
    #    Wait, if the subsets are disjoint, say A and B, and prod(A) = prod(B), then prod(A) / prod(B) = 1.
    #    This is a linear dependence with coefficients +1 for A and -1 for B.
    #    So, distinct subset products <=> linear independence over Q.
    #    Therefore, the maximum size k is at most the dimension of the vector space.
    #    The primes are 2, 3, ..., 37. There are 12 primes.
    #    So the max possible size is 12?
    #    Let's check the inspirations.
    #    Inspiration 3 says: "measured 1 -> 1.333".
    #    Current program returns 16? No, current program score is 1.3333.
    #    If seed score is 1.0, and value is count, then seed count = 1.0 * seed_value?
    #    The prompt says "search score is value/seed".
    #    If seed program returns N, score = N/N_seed? No, "value/seed" usually means value / (baseline_value).
    #    If current score is 1.3333, and it's an improvement over 1.0, it likely returns 16 items if seed returned 12?
    #    16/12 = 1.333.
    #    So the current program finds 16 items.
    #    But I just argued max is 12.
    #    Let me re-read the linear independence condition.
    #    "All 2^k subset products ... must be distinct."
    #    This implies that no non-trivial combination of products is equal.
    #    This is equivalent to the vectors being linearly independent.
    #    Max independent set in Q^12 is 12.
    #    Why does the current program get 16?
    #    Maybe the primes are not the only basis?
    #    The vectors are formed by prime exponents. The dimension is the number of distinct primes involved.
    #    Primes in 2..40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37. Total 12.
    #    So dimension is 12.
    #    Max independent set size is 12.
    #    Is it possible that "distinct subset products" does NOT imply linear independence?
    #    If vectors are dependent, there is a relation sum c_i v_i = 0 with c_i integers.
    #    This means prod x_i^{c_i} = 1.
    #    If c_i are not just -1, 0, 1, e.g., 2*v_1 - v_2 = 0 => v_2 = 2*v_1.
    #    x_2 = x_1^2.
    #    Subset products: {x_1, x_2}. Products: 1, x_1, x_2, x_1*x_2.
    #    If x_2 = x_1^2, are they distinct?
    #    x_1 != 1, x_1 != 2 (since x_1 >= 2).
    #    x_1^2 != x_1.
    #    x_1^2 != 1.
    #    x_1 * x_1^2 = x_1^3.
    #    Are 1, x_1, x_1^2, x_1^3 distinct? Yes.
    #    So {2, 4} has distinct subset products (1, 2, 4, 8).
    #    But {2, 4} is linearly DEPENDENT (v_4 = 2*v_2).
    #    So my equivalence was wrong.
    #    Distinct subset products <=> No subset product equals another subset product.
    #    This allows for dependencies where the coefficients are not in {-1, 0, 1}.
    #    Specifically, it forbids "multiplicative relations" where the product of some subset equals the product of a disjoint subset.
    #    This is equivalent to saying the set is "multiplicatively independent".
    #    Wait, if x_2 = x_1^2, then x_2 = x_1 * x_1.
    #    Subset {2} product is 4. Subset {1, 1} is not a subset of the set {2, 4} because we can only pick each element once.
    #    The subsets are of the SET {2, 4}.
    #    Subsets: {}, {2}, {4}, {2, 4}.
    #    Products: 1, 2, 4, 8.
    #    All distinct.
    #    So {2, 4} is valid.
    #    {2, 4, 16}?
    #    Subsets: 1, 2, 4, 8, 16, 32, 64, 128.
    #    All distinct.
    #    So we can include powers of the same prime!
    #    This changes everything.
    #    The condition is that the set is "sum-free" in the exponent space?
    #    No, it's that no sum of a subset of vectors equals the sum of another disjoint subset of vectors.
    #    This is equivalent to: All subset sums are distinct.
    #    This is the "distinct subset sums" problem.
    #    For a set of vectors, when are all subset sums distinct?
    #    This is true if the vectors are such that no vector is a sum of other vectors?
    #    No, {1, 2} has sums 0, 1, 2, 3. Distinct.
    #    {1, 1} has sums 0, 1, 1, 2. Not distinct.
    #    So we need a set of vectors such that all subset sums are unique.
    #    This is equivalent to the set being "superincreasing"? No.
    #    It is equivalent to the set being a "Sidon set" in the group? No, that's for pairs.
    #    It is simply that the map S -> sum(S) is injective.
    #    This is possible if the vectors are "linearly independent" AND we restrict to coefficients 0/1?
    #    No, linear independence over Q ensures that no integer combination is 0.
    #    If we have a collision sum(A) = sum(B) with A != B, then sum(A \ B) = sum(B \ A).
    #    Let C = A \ B and D = B \ A. C and D are disjoint.
    #    sum(C) = sum(D).
    #    sum(C) - sum(D) = 0.
    #    This is a linear dependence with coefficients in {-1, 0, 1}.
    #    So, distinct subset products <=> No linear dependence with coefficients in {-1, 0, 1}.
    #    This is weaker than linear independence.
    #    We can have dependencies like 2*v1 = v2, as long as v2 != v1 + v1 (which is not a subset sum of distinct elements).
    #    So, we can pack more elements than the dimension!
    #    Max size is bounded by log2(product of elements) roughly?
    #    Or rather, the number of distinct subset products is 2^k.
    #    The max product is 40^18.
    #    The number of possible products is bounded by 40^18.
    #    So 2^k <= 40^18 => k <= 18 * log2(40) ~ 18 * 5.3 = 95.
    #    So 18 is easily possible.
    #    
    #    Strategy:
    #    We want to select 18 integers from 2..40 such that all subset products are distinct.
    #    This is equivalent to: No two distinct subsets have the same product.
    #    
    #    Let's use a greedy approach with backtracking or just a smart greedy.
    #    Sort candidates.
    #    Try to add them one by one.
    #    When adding x, check if it creates a collision.
    #    Collision: exists subset S of current set such that prod(S) == x * prod(T) for some subset T?
    #    Actually, just check if x * prod(S) is already in the set of products, or if prod(S) is in the set of products of x * ...?
    #    Simpler: Maintain the set of all subset products.
    #    When adding x, new products are {p * x for p in products}.
    #    Check if any of these are already in `products`.
    #    If yes, collision.
    #    If no, add them.
    #    
    #    This is O(2^k) per step. k=18 => 2^18 = 262,144 products.
    #    39 candidates.
    #    Total operations ~ 39 * 262,144 ~ 10^7. Very fast.
    #    
    #    To maximize size, we should try to fit as many as possible.
    #    Greedy might fail to find the max if it picks "bad" numbers early.
    #    "Bad" numbers are those that create many collisions or block many others.
    #    Numbers with many small prime factors might be "bad"?
    #    Or maybe numbers that are powers of 2 are "good" because they only interact with other powers of 2?
    #    Actually, powers of 2: {2, 4, 8, 16, 32}.
    #    Subset products of {2, 4, 8, 16, 32} are {2^k}. All distinct.
    #    So we can take all 5 powers of 2.
    #    Similarly for 3: {3, 9, 27}. 3 items.
    #    5: {5, 25}. 2 items.
    #    7: {7}. 1 item.
    #    11, 13, 17, 19, 23, 29, 31, 37. 8 items.
    #    Total so far: 5 + 3 + 2 + 1 + 8 = 19.
    #    Wait, are they independent?
    #    Set: {2, 4, 8, 16, 32, 3, 9, 27, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37}
    #    Count: 19.
    #    But max is 18.
    #    Let's check if this set has distinct subset products.
    #    Any subset product is of the form 2^a * 3^b * 5^c * 7^d * 11^e * ...
    #    Since the exponents for each prime are determined uniquely by the subset (because the elements are powers of that prime, and each prime appears in disjoint groups), the product uniquely determines the subset.
    #    Wait, is that true?
    #    For prime 2, we have elements 2, 4, 8, 16, 32.
    #    Subset of these gives 2^k where k is sum of exponents {1, 2, 3, 4, 5}.
    #    Can we get the same k from different subsets?
    #    Subsets of {1, 2, 3, 4, 5}.
    #    Sums:
    #    0: {}
    #    1: {1}
    #    2: {2}
    #    3: {3} or {1, 2} -> COLLISION!
    #    3 = 3 and 3 = 1+2.
    #    So {2, 4, 8} has collision: 8 = 2*4.
    #    So we cannot have {2, 4, 8}.
    #    We can have {2, 4} (sums 0, 1, 2, 3). Distinct.
    #    We can have {2, 8} (sums 0, 1, 3, 4). Distinct.
    #    We can have {4, 8} (sums 0, 2, 3, 5). Distinct.
    #    We can have {2, 4, 16}? Sums of {1, 2, 4}: 0, 1, 2, 3, 4, 5, 6, 7. Distinct.
    #    We can have {2, 4, 16, 32}? Sums of {1, 2, 4, 8}: 0..15. Distinct.
    #    So for prime p, we want a subset of powers {p^1, p^2, ..., p^m} such that all subset sums of exponents are distinct.
    #    This is equivalent to finding a set of integers (exponents) with distinct subset sums.
    #    Max size of such a set from {1, ..., M} is roughly log2(M)?
    #    Actually, if we pick powers of 2: {1, 2, 4, 8, ...}, subset sums are 0..2^k-1. All distinct.
    #    So we can pick exponents that are powers of 2.
    #    For p=2, powers <= 40: 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32.
    #    Exponents: 1, 2, 3, 4, 5.
    #    We want a subset of {1, 2, 3, 4, 5} with distinct subset sums.
    #    Best is {1, 2, 4} (from 1, 2, 4) -> sums 0..7.
    #    Or {1, 2, 5} -> sums 0, 1, 2, 3, 5, 6, 7, 8. Distinct.
    #    Can we do 4?
    #    {1, 2, 4, 8} requires 8, which is 2^8=256 > 40.
    #    So max size for p=2 is 3? {2, 4, 16} (exponents 1, 2, 4).
    #    Wait, 2^4=16. Exponent is 4.
    #    Set {2, 4, 16}. Exponents {1, 2, 4}.
    #    Subset sums: 0, 1, 2, 3, 4, 5, 6, 7. Distinct.
    #    Can we add 32 (exp 5)?
    #    {1, 2, 4, 5}. Sums:
    #    0, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 12.
    #    Collision: 5 (from {5}) and 5 (from {1, 4}).
    #    So {2, 4, 16, 32} is invalid.
    #    So max for p=2 is 3.
    #    
    #    For p=3, powers <= 40: 3, 9, 27. Exponents 1, 2, 3.
    #    Subset of {1, 2, 3} with distinct sums.
    #    {1, 2} -> 0, 1, 2, 3. Distinct.
    #    {1, 3} -> 0, 1, 3, 4. Distinct.
    #    {2, 3} -> 0, 2, 3, 5. Distinct.
    #    {1, 2, 3} -> 0, 1, 2, 3, 3, 4, 5, 6. Collision (3).
    #    So max for p=3 is 2.
    #    
    #    For p=5, powers <= 40: 5, 25. Exponents 1, 2.
    #    {1, 2} -> distinct. Max 2.
    #    
    #    For p=7, 7. Max 1.
    #    For p=11, 11. Max 1.
    #    ...
    #    For p=37, 37. Max 1.
    #    
    #    Total max size:
    #    p=2: 3
    #    p=3: 2
    #    p=5: 2
    #    p=7: 1
    #    p=11: 1
    #    p=13: 1
    #    p=17: 1
    #    p=19: 1
    #    p=23: 1
    #    p=29: 1
    #    p=31: 1
    #    p=37: 1
    #    Sum: 3+2+2+1+1+1+1+1+1+1+1+1 = 15.
    #    
    #    But we can mix primes!
    #    We are not restricted to pure powers.
    #    We can use composites like 6 = 2*3.
    #    This allows us to "share" dimensions?
    #    Actually, mixing primes makes it HARDER to keep subset sums distinct, because the exponents of different primes become coupled.
    #    Wait, if we use 6, we add 1 to exp(2) and 1 to exp(3).
    #    If we already have 2 and 3, adding 6 creates a relation 2*3=6.
    #    Subset {2, 3} product 6. Subset {6} product 6. Collision.
    #    So we cannot have {2, 3, 6}.
    #    
    #    So the "pure power" strategy is actually quite strong because it decouples the primes.
    #    If we decouple, the problem splits into independent subproblems for each prime.
    #    Max size is sum of max sizes for each prime.
    #    We calculated 15.
    #    Can we do better by mixing?
    #    Mixing allows us to use "wasted" capacity?
    #    No, mixing creates dependencies.
    #    However, maybe we can use numbers that are not pure powers to fill gaps?
    #    The current program gets 16.
    #    My pure power count is 15.
    #    So there is a way to get 16.
    #    
    #    Let's implement the greedy check for distinct subset products.
    #    It's robust and doesn't rely on my theoretical decomposition.
    #    
    #    Algorithm:
    #    1. Start with empty set.
    #    2. Iterate through candidates in a specific order.
    #    3. For each candidate, try to add it.
    #    4. If adding it causes a collision, skip it.
    #    5. If not, add it.
    #    
    #    To maximize size, we should choose the order carefully.
    #    Candidates that are "easier" to add should come first?
    #    Or candidates that are "more useful"?
    #    Actually, since we want max size, and the condition is local (subset products),
    #    a greedy approach might work well if we sort by "complexity".
    #    Primes are simple. Powers are simple.
    #    Let's sort by number of prime factors (with multiplicity)?
    #    Or just try all permutations? No, 39! is too big.
    #    
    #    Let's try a greedy approach with a few different orderings and pick the best.
    #    Ordering 1: Ascending value.
    #    Ordering 2: Descending value.
    #    Ordering 3: Primes first, then prime powers, then composites.
    #    
    #    Actually, let's just use the greedy check.
    #    It is O(N * 2^k).
    #    N=39, k=18.
    #    
    #    To ensure we find 16 or more, we can use a simple backtracking if greedy fails to reach a target?
    #    No, greedy is likely sufficient if the order is good.
    #    Let's try sorting by "prime signature" or just value.
    #    
    #    One improvement: If greedy gets stuck, we can try to swap out a "bad" element for a "good" one?
    #    Too complex.
    #    
    #    Let's stick to the greedy with a smart order.
    #    Order: Primes, then prime powers, then others.
    #    Within each group, ascending.
    
    def _is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True

    def _is_prime_power(n):
        if _is_prime(n): return True
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            x = n
            while x % p == 0:
                x //= p
            if x == 1:
                return True
        return False

    # Categorize candidates
    primes_list = []
    prime_powers_list = []
    others_list = []
    
    for num in candidates:
        if _is_prime(num):
            primes_list.append(num)
        elif _is_prime_power(num):
            prime_powers_list.append(num)
        else:
            others_list.append(num)
    
    # Sort each group
    primes_list.sort()
    prime_powers_list.sort()
    others_list.sort()
    
    # Try different orders
    orders = [
        primes_list + prime_powers_list + others_list,
        others_list + prime_powers_list + primes_list,
        sorted(candidates),
        sorted(candidates, reverse=True)
    ]
    
    best_set = []
    best_products = set()
    
    for order in orders:
        current_set = []
        products = {1}
        
        for x in order:
            if time.time() > deadline:
                break
                
            # Check if adding x causes collision
            new_products = set()
            collision = False
            for p in products:
                np = p * x
                if np in products:
                    collision = True
                    break
                new_products.add(np)
            
            if not collision:
                current_set.append(x)
                products.update(new_products)
                
        # Update best
        if len(current_set) > len(best_set):
            best_set = current_set
            best_products = products
            
        if len(best_set) >= 18:
            break
            
    # Verify best_set
    # (Already verified by construction, but double check for safety)
    if len(best_set) > 0:
        prods = {1}
        for x in best_set:
            new_prods = {p * x for p in prods}
            if prods.intersection(new_prods):
                # Fallback to safe set
                return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
            prods.update(new_prods)
            
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
