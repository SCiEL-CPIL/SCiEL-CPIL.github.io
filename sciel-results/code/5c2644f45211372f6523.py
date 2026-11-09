"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

P = {'n': 256}
_DEADLINE = None

def _build_compatibility():
    n = P['n']
    nums = list(range(2, n + 1))
    m = len(nums)
    # Precompute prime factorization exponents for each number
    # We need to check: for distinct a, b, c in set: a does not divide b*c
    # Equivalently: for each a in set, and for each pair (b,c) in set with b != c, b != a, c != a:
    #   a ∤ b*c
    # 
    # Key insight: a divides b*c iff for every prime p, v_p(a) <= v_p(b) + v_p(c)
    # So a does NOT divide b*c iff there exists a prime p such that v_p(a) > v_p(b) + v_p(c)
    #
    # For a fixed a, define the "deficit" of a as: for each prime p dividing a, 
    # we need v_p(b) + v_p(c) < v_p(a).
    #
    # Actually, let's think about compatibility between pairs.
    # Two numbers x, y are "compatible" in the sense that they can coexist 
    # without violating the constraint with respect to each other and any third element.
    #
    # Wait, the constraint is: for ALL distinct triples (a,b,c) in the set, a ∤ b*c.
    # This is a 3-uniform hypergraph constraint, not just pairwise.
    #
    # However, we can think of it this way: 
    # A set S is valid iff for every a in S, and every pair {b,c} ⊆ S \ {a} with b≠c, 
    # we have a ∤ b*c.
    #
    # Equivalently: for every a in S, if we look at all pairs in S\{a}, none of them 
    # should have product divisible by a.
    #
    # Let's define: for a fixed a, let F(a) = { x in 2..n : x shares some prime factor with a 
    # such that v_p(x) < v_p(a) for some p }... this is getting complex.
    #
    # Alternative approach: For a fixed a, define the set of "forbidden pairs" for a:
    # P_a = { (b,c) : b≠c, b,c ≠ a, a | b*c }
    # A set S is valid iff for every a in S, S\{a} contains no pair from P_a.
    #
    # This means: for every a in S, the set S\{a} is an independent set in the graph G_a 
    # where G_a has vertices 2..n \ {a} and edges between b,c if a|b*c.
    #
    # This is still a 3-uniform constraint. Let's try a different angle.
    #
    # Observation: If a is square-free, then a|b*c means that for each prime p|a, p|b or p|c.
    # So a|b*c iff b and c together cover all prime factors of a.
    #
    # Let's try: a set S is valid iff for every a in S, and every b in S\{a}, 
    # there is no c in S\{a,b} such that a|b*c.
    #
    # Actually, let's just use the direct check for small sets and build a compatibility 
    # matrix where two numbers are "compatible" if adding them to a set doesn't create 
    # a violation. But that's not pairwise...
    #
    # Let me reconsider. The constraint a ∤ b*c for all distinct a,b,c in S.
    # 
    # Key structural insight: 
    # If a is a prime p, then p|b*c means p|b or p|c. So for prime p in S, 
    # no two other elements in S can both be multiples of p... wait, that's not right.
    # p|b*c iff (p|b) or (p|c). So to have p ∤ b*c, we need p∤b AND p∤c.
    # So if p is in S, then for any b,c in S\{p}, we need p∤b AND p∤c.
    # This means: if p is in S, then NO other element in S can be divisible by p.
    #
    # Wait, that's a strong constraint! Let me verify:
    # If p ∈ S and q ∈ S, p≠q, and p|q, then for any r ∈ S distinct from p,q:
    # p | q*r (since p|q), so p divides q*r. Violation!
    # So: if p is a prime in S, then no other element of S can be a multiple of p.
    #
    # More generally: if a ∈ S and b ∈ S, a≠b, and a|b, then for any c ∈ S distinct 
    # from a,b: a | b*c. Violation.
    # So: S must be an antichain under divisibility!
    #
    # But that's necessary, not sufficient. Even if no element divides another, 
    # we still need a ∤ b*c for all triples.
    #
    # Example: S = {6, 10, 15}. 6∤10, 6∤15, 10∤15, 10∤6, 15∤6, 15∤10. Antichain.
    # Check: 6 | 10*15 = 150? 150/6 = 25. Yes! So 6 divides 10*15. Violation!
    #
    # So antichain is necessary but not sufficient.
    #
    # Let's think about what makes a divide b*c.
    # a | b*c iff for all primes p, v_p(a) <= v_p(b) + v_p(c).
    #
    # For the set to be valid, for every a in S and every pair b,c in S\{a}:
    # there exists some prime p such that v_p(a) > v_p(b) + v_p(c).
    #
    # This is a complex 3-uniform constraint. Let's just do a search.
    #
    # Given n=256, the maximum set size is probably around 20-40.
    # Let's use a greedy + local search approach, or branch and bound.
    
    # Let's precompute for each a in 2..256, the set of pairs (b,c) such that a|b*c.
    # Actually, let's define: for a fixed a, bad_pairs[a] = set of frozensets {b,c} where b≠c, 
    # b,c in 2..256, b≠a, c≠a, and a|b*c.
    #
    # Then a set S is valid iff for every a in S, S\{a} contains no bad pair from bad_pairs[a].
    #
    # Equivalently: for every a in S, and every b in S\{a}, the set (S\{a,b}) ∩ 
    # {c : a|b*c} must be empty.
    #
    # Let's precompute: for each a, for each b≠a, the set of c such that a|b*c.
    # comp[a][b] = {c : c≠a, c≠b, a|b*c}
    #
    # Then when building S, if we have current set S and want to add x:
    # For each a in S: we need that for each b in S\{a}, x is not in comp[a][b].
    #   i.e., x ∉ ∪_{b in S\{a}} comp[a][b]
    # And for each b in S: we need that for each a in S\{b}, x is not in comp[b][a].
    #   i.e., x ∉ ∪_{a in S\{b}} comp[b][a]
    # (These are the same condition by symmetry of the triple constraint.)
    #
    # Actually, the constraint is: for all distinct a,b,c in S∪{x}, a∤b*c.
    # New violations can only involve x. So:
    # 1. x ∤ a*b for all distinct a,b in S.
    # 2. a ∤ x*b for all distinct a,b in S.
    # 3. a ∤ b*x for all distinct a,b in S. (same as 2)
    #
    # So: x is compatible with S iff:
    #   (i) For all distinct a,b in S: x ∤ a*b
    #   (ii) For all distinct a,b in S: a ∤ x*b
    #
    # Let's precompute these.

    n = P['n']
    # For condition (i): x ∤ a*b for all distinct a,b in S.
    # Precompute: for each x, the set of pairs (a,b) such that x|a*b.
    # div_pairs[x] = set of frozensets {a,b} where a≠b, x|a*b
    
    # For condition (ii): a ∤ x*b for all distinct a,b in S.
    # For each a, the set of (x,b) pairs such that a|x*b.
    # Or: for each a, for each b, the set of x such that a|x*b.
    # a_div_xb[a][b] = {x : a|x*b}
    
    # This might be too memory intensive. Let's compute on the fly or use a different approach.
    
    # Alternative: just check validity of a set directly, and use branch and bound.
    # For n=256, if the answer is ~30, branch and bound with good pruning should work.
    
    return None  # We'll do direct checking

def _valid_set(S):
    """Check if set S is valid: for all distinct a,b,c in S, a ∤ b*c."""
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            for k in range(j+1, m):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(S, x):
    """Check if x can be added to set S without violating the constraint."""
    # Check all triples involving x
    S_list = list(S)
    m = len(S_list)
    # x ∤ a*b for all distinct a,b in S
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
    # a ∤ x*b for all distinct a,b in S
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    return True

def _greedy_order(nums, order_func):
    """Greedy: add numbers in the order given by order_func."""
    S = set()
    for x in order_func(nums):
        if _can_add(S, x):
            S.add(x)
    return S

def _local_search(S, candidates, time_limit):
    """Try to improve S by removing one element and adding others."""
    global _DEADLINE
    best = set(S)
    best_size = len(best)
    start = time.time()
    
    # Try all removals
    for rem in list(best):
        if time.time() > _DEADLINE:
            break
        S2 = best - {rem}
        # Try to add candidates in various orders
        for x in candidates:
            if x in S2:
                continue
            if _can_add(S2, x):
                S2.add(x)
                # Try to add more
                for y in candidates:
                    if y in S2:
                        continue
                    if _can_add(S2, y):
                        S2.add(y)
                if len(S2) > best_size:
                    best = set(S2)
                    best_size = len(best)
    
    return best

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 50  # 50 second budget
    
    n = P['n']
    nums = list(range(2, n + 1))
    
    # Start with primes (always a valid set)
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                return False
        return True
    
    primes = [x for x in nums if is_prime(x)]
    S = set(primes)
    
    # Try greedy with different orderings
    orderings = [
        lambda nums: sorted(nums, key=lambda x: -x),  # descending
        lambda nums: sorted(nums, key=lambda x: x),    # ascending
        lambda nums: sorted(nums, key=lambda x: (x, -x)),  # natural
    ]
    
    best_set = set(S)
    best_size = len(S)
    
    for order_func in orderings:
        S_greedy = _greedy_order(nums, order_func)
        if len(S_greedy) > best_size:
            best_set = S_greedy
            best_size = len(S_greedy)
    
    # Local search from best
    candidates = [x for x in nums if x not in best_set]
    if time.time() < _DEADLINE:
        best_set = _local_search(best_set, candidates, _DEADLINE)
    
    # Verify
    S_list = sorted(best_set)
    if not _valid_set(S_list):
        # Fallback to primes
        S_list = sorted(primes)
    
    return list(S_list)
# EVOLVE-BLOCK-END
