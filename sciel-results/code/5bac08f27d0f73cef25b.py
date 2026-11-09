"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 512] such that lcm(a, b) <= 512 for every pair (a, b).
    
    Approach:
    1. The seed solution returns [1, 2, ..., 22] (since isqrt(512) = 22), size 22.
    2. We can do much better. Note that if all elements divide 512 = 2^9, then 
       lcm(a,b) divides 512, so lcm(a,b) <= 512. The divisors of 512 are 
       {1, 2, 4, 8, 16, 32, 64, 128, 256}, size 9. That's worse.
    
    3. Better approach: Use the fact that lcm(a,b) = a*b/gcd(a,b) <= 512.
       A key insight: if we take all numbers that are powers of 2 up to 512, 
       plus carefully chosen other numbers.
    
    4. Actually, let's think more carefully. The condition lcm(a,b) <= 512 
       means that for every prime p, the maximum of v_p(a) and v_p(b) is at most 
       the exponent of p in 512. Since 512 = 2^9, only prime 2 matters for the 
       bound... wait, no. lcm(a,b) <= 512 means the actual integer lcm(a,b) <= 512,
       not just the 2-adic valuation.
    
    5. Let's use a greedy/incremental approach or exploit structure.
    
    Key observation: If a | 512 and b | 512, then lcm(a,b) | 512, so lcm(a,b) <= 512.
    But we can include numbers that don't divide 512.
    
    For example, 3: lcm(3, 2^k) = 3*2^k for k where 3*2^k <= 512, i.e., 2^k <= 170, 
    so k <= 7 (2^7=128, 3*128=384<=512). But lcm(3, 256) = 768 > 512. So 3 conflicts with 256.
    
    Let me build the conflict graph and find a large independent set.
    """
    N = 512
    # Build conflict graph: edge between a,b if lcm(a,b) > 512
    # We want maximum independent set.
    
    # For efficiency, let's think about which numbers can coexist.
    # 
    # Strategy: Start with all numbers 1..512, and greedily remove conflicts.
    # But greedy might not be optimal. Let's try a smarter approach.
    #
    # Observation: If we restrict to numbers whose prime factorization only involves
    # primes where the product of max exponents stays <= 512...
    #
    # Actually, let's just build the graph and use a heuristic for max independent set.
    # The graph has 512 nodes, which is manageable.
    
    from math import gcd
    
    # Build compatibility graph: edge if lcm(a, b) <= 512
    # We want Maximum Clique in this graph.
    # Node i represents number i+1.
    
    comp_adj = [set() for _ in range(N)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            lcm = a * b // g
            if lcm <= N:
                comp_adj[a-1].add(b-1)
                comp_adj[b-1].add(a-1)
                
    # Bron-Kerbosch with pivot to find Maximum Clique
    best_clique = []
    
    # Order nodes by degree descending to help pruning?
    # Actually, for BK, we just need adjacency.
    
    # To improve performance, we can sort nodes initially or use a specific order.
    # Let's stick to standard BK with pivot.
    
    # Use a non-recursive version or ensure recursion limit is high enough?
    # Max clique size is likely < 100, so recursion depth is fine.
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Pruning: if current size + potential size <= best, return
        if len(R) + len(P) <= len(best_clique):
            return
            
        # Pivot selection: choose u in P union X with max degree in P
        # This reduces the number of branches
        # We want to iterate over P \ N(u)
        
        # For efficiency, pick u from P if P is not empty, else from X
        # Standard heuristic: max degree in P
        max_deg = -1
        pivot = -1
        for u in P:
            deg = len(comp_adj[u] & P)
            if deg > max_deg:
                max_deg = deg
                pivot = u
        if pivot == -1 and X:
             # If P is empty, we would have returned? No, if P empty and X not, we return?
             # If P is empty, we check X. If X not empty, no new cliques.
             # The condition "if not P and not X" handles the base case.
             # If P is empty, we should return immediately because no more vertices can be added.
             # Wait, the standard BK algorithm says:
             # if P and X are both empty, report R.
             # So if P is empty, we return.
             pass
        
        if P:
            # Pivot chosen from P
            # Candidates are P \ N(pivot)
            candidates = P - comp_adj[pivot]
        else:
            # If P is empty, we should have returned?
            # Actually, if P is empty, we can't add any more vertices to R.
            # So we just check if R is a valid clique (it is) and return.
            # The base case `if not P and not X` covers the maximal clique reporting.
            # If P is empty but X is not, we don't report R as a *maximal* clique in the original graph?
            # No, R is a maximal clique if P and X are empty.
            # If P is empty and X is not, R is not maximal?
            # Actually, BK finds *maximal* cliques.
            # We want *maximum* clique.
            # So we just need to visit all maximal cliques and keep the largest.
            # If P is empty, we stop branching.
            return

        for v in candidates:
            # Branch on v
            new_R = R | {v}
            new_P = P & comp_adj[v]
            new_X = X & comp_adj[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            P.discard(v)
            X.add(v)
            
    # Initial call
    # R = empty, P = all nodes, X = empty
    # To speed up, we can order nodes?
    # Let's just run it.
    # Note: P and X are sets, so they are mutable. We must copy them for the initial call
    # or ensure we don't modify the original if we need to restart.
    # Since we only run once, it's fine.
    
    initial_P = set(range(N))
    initial_X = set()
    bron_kerbosch(set(), initial_P, initial_X)
    
    return [x + 1 for x in best_clique]

# EVOLVE-BLOCK-END
