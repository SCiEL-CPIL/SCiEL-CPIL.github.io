"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM for all pairs
    lcm = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            lcm[i][j] = i * j // gcd(i, j)
    
    # Precompute prime factorizations for all numbers 1..N
    def prime_factors(n):
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
    
    factorizations = [None] * (N + 1)
    for i in range(1, N + 1):
        factorizations[i] = prime_factors(i)
    
    # Partition numbers by their 2-adic valuation (v2 = highest power of 2 dividing n)
    # v2(n) = k means n = 2^k * m where m is odd
    # Numbers with the same v2 have special LCM properties
    v2_groups = defaultdict(list)
    for x in range(1, N+1):
        v = 0
        tmp = x
        while tmp % 2 == 0:
            v += 1
            tmp //= 2
        v2_groups[v].append(x)
    
    # For each group, sort by the odd part
    for v in v2_groups:
        v2_groups[v].sort(key=lambda x: x // (2**v))
    
    # Build conflict structure: for each element x, store which pairs (y,z) form a bad triple
    # Use a more efficient representation: for each x, store a set of "forbidden pairs"
    # But we need to check if BOTH y AND z are in the set
    
    # Key insight: A triple (a,b,c) with a<b<c is bad iff:
    # lcm(a,b) = lcm(a,c) = lcm(b,c) = L
    # This is equivalent to: for each prime p, max(vp(a),vp(b)) = max(vp(a),vp(c)) = max(vp(b),vp(c))
    # Let's precompute all bad triples using the algebraic structure
    
    # For efficiency, we'll use the following approach:
    # Group numbers by their "signature" in terms of prime exponents
    # Two numbers x,y have the same "exponent vector" if for all primes p, vp(x)=vp(y)
    # But this is too fine-grained. Instead, let's directly compute conflicts.
    
    # Precompute: for each number x, which numbers y have the property that
    # there exists z such that {x,y,z} is a bad triple
    # We'll store conflicts as: conflicts[x] = set of frozensets {y,z}
    
    conflicts = defaultdict(set)
    
    # Iterate over all triples a<b<c and check if they're bad
    # This is O(N^3) = 128^3 ≈ 2M operations, which is fast
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            L_ab = lcm[a][b]
            for c in range(b+1, N+1):
                # Check if lcm(a,c) == L_ab and lcm(b,c) == L_ab
                # Since c > b > a, and L_ab = lcm(a,b), we need c | L_ab
                if L_ab % c != 0:
                    continue
                # Check LCM conditions
                if a * c // gcd(a, c) == L_ab and b * c // gcd(b, c) == L_ab:
                    # Bad triple found
                    conflicts[a].add(frozenset([b, c]))
                    conflicts[b].add(frozenset([a, c]))
                    conflicts[c].add(frozenset([a, b]))
    
    # Now we need to find the maximum subset with no bad triples
    # This is a maximum independent set problem in a 3-uniform hypergraph
    # We'll use branch-and-bound with the following structure:
    # - Order elements by their "conflict degree" (number of bad triples they participate in)
    # - Use upper bound: remaining elements + current size
    # - Prune when upper bound <= best found
    
    # Compute conflict degree for each element
    degree = [0] * (N+1)
    for x in range(1, N+1):
        degree[x] = len(conflicts.get(x, set()))
    
    # Sort elements by degree (descending) for branching
    ordered_elements = sorted(elements, key=lambda x: -degree[x])
    
    best_set = set()
    best_size = 0
    nodes_explored = 0
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a bad triple."""
        for pair in conflicts.get(x, set()):
            if pair.issubset(S):
                return False
        return True
    
    def upper_bound(S, remaining):
        """Compute an upper bound on the final set size."""
        # Simple bound: current size + number of remaining elements
        # We can improve this by checking which remaining elements can be added
        # For now, use a simple bound
        return len(S) + len(remaining)
    
    def branch_and_bound(idx, S, remaining):
        """
        Branch and bound search.
        idx: index in ordered_elements
        S: current set (as a set for fast membership)
        remaining: list of elements from idx onwards that haven't been decided
        """
        nonlocal best_set, best_size, nodes_explored
        
        if time.time() > deadline - 5:
            return
        
        nodes_explored += 1
        
        if not remaining:
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            return
        
        # Upper bound check
        if len(S) + len(remaining) <= best_size:
            return
        
        x = ordered_elements[idx]
        
        # Option 1: Include x
        if can_add(x, S):
            S.add(x)
            branch_and_bound(idx + 1, S, remaining[1:])
            S.remove(x)
        
        # Option 2: Exclude x
        branch_and_bound(idx + 1, S, remaining[1:])
    
    # The full branch-and-bound on 128 elements is intractable
    # We need a smarter approach. Let's use a hybrid:
    # 1. Use the algebraic structure to identify "safe" groups
    # 2. Within each group, the problem is smaller
    # 3. Use local search for refinement
    
    # Alternative approach: Use the fact that numbers with large 2-adic valuation
    # have fewer conflicts with other numbers
    
    # Let's try a different structural approach:
    # Numbers can be classified by their "odd part" and 2-adic valuation
    # Two numbers x = 2^a * m, y = 2^b * n (m,n odd)
    # lcm(x,y) = 2^max(a,b) * lcm(m,n)
    
    # For three numbers to have equal pairwise LCMs, they must satisfy:
    # 2^max(va,vb) * lcm(odd_a, odd_b) = 2^max(va,vc) * lcm(odd_a, odd_c) = 2^max(vb,vc) * lcm(odd_b, odd_c)
    
    # This implies constraints on the valuations and odd parts
    
    # Let's use a greedy approach with multiple random restarts, but with
    # a better ordering based on the algebraic structure
    
    def greedy_fill(S, order):
        """Add elements from 'order' to S if possible."""
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
    
    # Local search: try to swap elements to improve
    def local_search(S, max_iters=100):
        """Attempt to improve S by removing one element and re-filling."""
        current_size = len(S)
        if current_size == 0:
            return False, S
            
        S_list = list(S)
        random.shuffle(S_list)
        
        for u in S_list:
            if time.time() > deadline - 10:
                break
                
            S.remove(u)
            
            # Try to re-fill with random order
            best_completion = set(S)
            best_completion_size = len(S)
            
            for trial in range(5):
                if time.time() > deadline - 10:
                    break
                S_trial = set(S)
                candidates = [x for x in range(1, N+1) if x not in S_trial]
                random.shuffle(candidates)
                
                for x in candidates:
                    if can_add(x, S_trial):
                        S_trial.add(x)
                
                if len(S_trial) > best_completion_size:
                    best_completion_size = len(S_trial)
                    best_completion = set(S_trial)
            
            S = set(best_completion)
            
            if len(S) > current_size:
                return True, S
        
        return False, S
    
    # Main search: multiple restarts with different strategies
    num_restarts = 500
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        S = set()
        
        # Different ordering strategies
        if r % 3 == 0:
            # Random order
            order = elements[:]
            random.shuffle(order)
        elif r % 3 == 1:
            # Order by 2-adic valuation (descending)
            order = sorted(elements, key=lambda x: -x.bit_length() + (x & -x).bit_length() - 1)
        else:
            # Order by odd part
            order = sorted(elements, key=lambda x: x // (x & -x))
        
        greedy_fill(S, order)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 10:
            improved, S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Final polish
    if time.time() < deadline - 3:
        for x in elements:
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                
    if not best_set:
        best_set = {1}
        
    return list(best_set)
# EVOLVE-BLOCK-END
