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
    
    # Create conflict matrix
    # conflict[i][j] = True if lcm(i+1, j+1) > 512
    # Use 0-indexed: node i represents number i+1
    
    # For memory efficiency with 512 nodes, use a list of bitsets or sets
    conflicts = [set() for _ in range(N)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            lcm = a * b // g
            if lcm > N:
                conflicts[a-1].add(b-1)
                conflicts[b-1].add(a-1)
    
    # Now find a large independent set.
    # Greedy approach: repeatedly pick the node with smallest degree, add to independent set,
    # remove it and its neighbors.
    
    # Convert to adjacency list for dynamic removal
    # Use a set of remaining nodes
    remaining = set(range(N))
    independent_set = []
    
    # Greedy: pick node with min degree among remaining
    while remaining:
        # Find node with minimum degree in remaining subgraph
        min_deg = float('inf')
        best_node = -1
        for node in remaining:
            deg = len(conflicts[node] & remaining)
            if deg < min_deg:
                min_deg = deg
                best_node = node
                if deg == 0:
                    break
        
        independent_set.append(best_node + 1)
        remaining.discard(best_node)
        # Remove neighbors of best_node from remaining
        for neighbor in conflicts[best_node]:
            if neighbor in remaining:
                remaining.discard(neighbor)
    
    return independent_set

# EVOLVE-BLOCK-END
