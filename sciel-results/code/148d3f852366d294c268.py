import time
from math import gcd
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 78  # safety margin before 90s limit
    
    n = 256
    all_nums = list(range(2, n + 1))
    
    # Precompute factorizations
    def factorize(x):
        factors = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors[d] = factors.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            factors[x] = factors.get(x, 0) + 1
        return factors
    
    facs = {x: factorize(x) for x in range(2, n + 1)}
    
    # Precompute conflict matrix: conflict[a][b] = True if a | b*c for some c, 
    # but that's not quite right. Let's think differently.
    # 
    # The constraint: for every triple (a,b,c) of distinct elements in S, a ∤ b*c
    # Equivalently: for every a in S, and every pair (b,c) in S\{a} with b≠c, a ∤ b*c
    # 
    # This is a 3-uniform hypergraph independent set problem.
    # Hyperedge: {a,b,c} is "bad" if a|b*c or b|a*c or c|a*b
    #
    # For branch and bound, we need efficient conflict checking.
    # Let's precompute: for each pair (b,c), the set of a values such that a|b*c
    # i.e., a divides b*c. Since a ≤ 256, we can compute divisors of b*c up to 256.
    
    # For each a, precompute which pairs (b,c) with b<c cause a|b*c
    # Actually for the B&B, we need: given a candidate a and current set S,
    # check if adding a creates any conflict.
    # Conflict if ∃ b,c in S, b≠c, such that a|b*c OR b|a*c OR c|a*b
    
    # Precompute: for each a, and each b≠a, the set of c values where a|b*c
    # a|b*c ⟺ (a/gcd(a,b)) | c
    # So for given a,b: let m = a/gcd(a,b), then c must be a multiple of m.
    
    # For efficiency in B&B, let's precompute for each a:
    # divisors_info[a] = for each b≠a, the modulus m = a/gcd(a,b)
    # Then a|b*c iff c % m == 0
    
    # Let's precompute a bitset representation for speed
    # For each a, create a set of (b, m) where m = a//gcd(a,b), m > 1
    # (if m==1, then a|b*c for ALL c, which means a|b, so a and b can't both be in S with any third element)
    
    # Actually if m==1, that means a|b, so a divides b. Then for any c, a|b*c.
    # So if a|b and both are in S, then for ANY third element c, the triple (a,b,c) is bad.
    # This means: if a|b, we can't have both a and b in S unless S has size ≤ 2.
    # Since we want large sets, this is very restrictive.
    
    # Let me precompute: for each a, the set of b's such that a|b (a divides b)
    # These are "strong conflicts" - having both in S is nearly impossible for |S|>2
    
    # For the B&B approach, let's use a different strategy:
    # 1. Start with a good greedy solution as lower bound
    # 2. Use branch and bound to try to improve
    
    # First, get a good initial solution using greedy with multiple orderings
    def greedy_solution(order):
        selected = []
        for a in order:
            valid = True
            for i in range(len(selected)):
                b = selected[i]
                for j in range(i + 1, len(selected)):
                    c = selected[j]
                    if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                selected.append(a)
        return selected
    
    # Try several greedy orderings
    best_greedy = []
    
    # Order 1: by number of divisors in [2,256] ascending (fewer divisors = less likely to divide products)
    def num_divisors_in_range(x):
        return sum(1 for y in range(2, n+1) if y % x == 0)
    
    order1 = sorted(all_nums, key=lambda x: (num_divisors_in_range(x), x))
    sol1 = greedy_solution(order1)
    if len(sol1) > len(best_greedy):
        best_greedy = sol1
    
    # Order 2: descending value (larger numbers less likely to divide products)
    order2 = sorted(all_nums, reverse=True)
    sol2 = greedy_solution(order2)
    if len(sol2) > len(best_greedy):
        best_greedy = sol2
    
    # Order 3: by conflict count (precomputed)
    conflict_counts = [0] * (n + 1)
    for a in range(2, n + 1):
        cnt = 0
        for b in range(2, n + 1):
            if b == a:
                continue
            g = gcd(a, b)
            m = a // g
            if m == 1:
                cnt += n - 2
            else:
                count_c = n // m
                if a % m == 0:
                    count_c -= 1
                if b % m == 0:
                    count_c -= 1
                cnt += max(0, count_c)
        conflict_counts[a] = cnt
    
    order3 = sorted(all_nums, key=lambda x: (conflict_counts[x], x))
    sol3 = greedy_solution(order3)
    if len(sol3) > len(best_greedy):
        best_greedy = sol3
    
    # Order 4: primes first (they're "safe" - only divide products if the product contains their prime factor)
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                return False
        return True
    
    primes = [x for x in range(2, n+1) if is_prime(x)]
    composites = [x for x in range(2, n+1) if not is_prime(x)]
    order4 = sorted(primes, reverse=True) + sorted(composites, key=lambda x: (conflict_counts[x], x))
    sol4 = greedy_solution(order4)
    if len(sol4) > len(best_greedy):
        best_greedy = sol4
    
    # Order 5: random restarts
    import random
    random.seed(42)
    for _ in range(5):
        random_order = all_nums[:]
        random.shuffle(random_order)
        # Bias towards larger numbers
        random_order.sort(key=lambda x: random.random() * 0.5 + conflict_counts[x] / max(1, conflict_counts[-1]) * 0.5)
        sol = greedy_solution(random_order)
        if len(sol) > len(best_greedy):
            best_greedy = sol
    
    best_size = len(best_greedy)
    
    if time.time() > deadline - 10:
        return list(best_greedy)
    
    # Now try branch and bound to improve
    # Convert to 0-indexed for bitset operations
    # Map: num -> index, index -> num
    # We'll work with the full set [2..256]
    
    # Precompute for each a: list of (b, m) where m = a//gcd(a,b) > 1, b != a
    # This tells us: a | b*c iff c is a multiple of m
    # Also need: b | a*c iff c is a multiple of b//gcd(a,b)
    # And: c | a*b iff c divides a*b
    
    # For B&B, the key operation is: can we add vertex v to current set S?
    # Check: for all pairs (b,c) in S, b≠c: v ∤ b*c AND b ∤ v*c AND c ∤ v*b
    
    # Let's precompute a faster check structure
    # For each v, and each b: 
    #   v|b*c ⟺ c % (v//gcd(v,b)) == 0
    #   b|v*c ⟺ c % (b//gcd(v,b)) == 0
    # So for adding v to S: for each pair (b,c) in S, check if c % m1 == 0 or c % m2 == 0
    # where m1 = v//gcd(v,b), m2 = b//gcd(v,b)
    # Wait, that's not quite right either. Let me re-derive.
    
    # Triple (a,b,c) is bad if:
    #   a | b*c, OR b | a*c, OR c | a*b
    # 
    # When adding new element v to existing set S:
    # For each pair (b,c) in S:
    #   Check if v | b*c: need (b*c) % v == 0
    #   Check if b | v*c: need (v*c) % b == 0  
    #   Check if c | v*b: need (v*b) % c == 0
    
    # For efficiency, precompute for each v and b:
    #   v_divides_b_times_c: v | b*c ⟺ c % (v//gcd(v,b)) == 0  [when v//gcd(v,b) > 1]
    #   b_divides_v_times_c: b | v*c ⟺ c % (b//gcd(v,b)) == 0  [when b//gcd(v,b) > 1]
    
    # Let's just do the direct check but with early termination and good ordering.
    
    # B&B: we'll try to add elements from a candidate pool to the current set
    # Upper bound: |current| + |remaining candidates that don't conflict with current|
    
    # Actually, let me try a different approach: local search / hill climbing
    # Start from best_greedy, try to swap in elements or add elements
    
    current = list(best_greedy)
    
    # Try to add elements
    def can_add(elem, s):
        for i in range(len(s)):
            b = s[i]
            for j in range(i+1, len(s)):
                c = s[j]
                if (b*c) % elem == 0 or (elem*c) % b == 0 or (elem*b) % c == 0:
                    return False
        return True
    
    # Repeatedly try to add the "easiest" remaining element
    remaining = [x for x in all_nums if x not in current]
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        # Try adding each remaining element, pick one that works
        for elem in sorted(remaining, key=lambda x: conflict_counts[x]):
            if can_add(elem, current):
                current.append(elem)
                remaining.remove(elem)
                improved = True
                break
    
    if len(current) > best_size:
        best_greedy = current
        best_size = len(current)
    
    # Try removal + re-addition: remove one element, try to add two
    if time.time() < deadline - 20:
        for _ in range(10):  # limited iterations
            if time.time() > deadline - 20:
                break
            current = list(best_greedy)
            remaining = [x for x in all_nums if x not in current]
            # Try removing each element and see if we can add more
            for idx in range(len(current)):
                if time.time() > deadline - 20:
                    break
                removed = current.pop(idx)
                # Try to add elements
                added = 0
                for elem in sorted(remaining, key=lambda x: conflict_counts[x]):
                    if elem == removed:
                        continue
                    if can_add(elem, current):
                        current.append(elem)
                        added += 1
                        if added >= 2:
                            break
                if len(current) > len(best_greedy):
                    best_greedy = list(current)
                    best_size = len(current)
                else:
                    current.insert(idx, removed)
    
    # Final verification
    def verify(s):
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                for k in range(j+1, len(s)):
                    a, b, c = s[i], s[j], s[k]
                    if (b*c) % a == 0 or (a*c) % b == 0 or (a*b) % c == 0:
                        return False
        return True
    
    if not verify(best_greedy):
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        best_greedy = [i for i in range(2, n+1) if is_prime[i]]
    
    return [int(x) for x in best_greedy]
