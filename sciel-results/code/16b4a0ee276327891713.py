"""Construct a linear triple system on 16 points with small independence number."""
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _generate_triples():
    """
    Generate a set of triples on 16 points (0..15) such that:
    - Any two triples share at most one point (linear).
    - The independence number (largest subset with no triple) is minimized.
    
    Approach:
    Use the 4-bit vector space GF(2)^4 representation. Points are 0..15.
    We construct triples using affine lines and structured patterns to ensure
    high coverage and low independence number.
    
    We'll use a greedy approach with backtracking or a known construction.
    Since we need determinism and speed, we'll use a precomputed or algorithmic
    construction based on the geometry of GF(2)^4.
    
    A known approach: use all lines of the affine geometry AG(4,2) that have
    size 2? No, lines in AG(4,2) have size 2. We need triples.
    
    Alternative: Use a Steiner system S(2,3,16)? That doesn't exist (16 not ≡1,3 mod 6).
    
    Let's use a greedy algorithm with careful ordering to minimize independence number.
    We want to cover as many 3-subsets as possible while maintaining linearity.
    
    Actually, let's use a specific construction:
    Partition 16 points into 4 groups of 4. Use triples that span groups.
    
    Better approach: Use a computer search / greedy with random restarts, but since we need
    determinism and <120s, let's use a fixed heuristic.
    
    Let's try: All triples {a,b,c} where a,b,c are distinct and a+b+c = 0 in GF(2)^4?
    That gives lines through origin. There are (2^4-1)/3 = 5 such lines. Not enough.
    
    Let's use a known good construction or run a fast greedy:
    1. Generate all 560 triples.
    2. Sort by some heuristic (e.g., points with high degree first).
    3. Greedily add triples if they don't conflict (share ≤1 point with existing).
    4. Evaluate independence number.
    
    But we need to minimize independence number. Let's use a more structured approach.
    
    Construction using 4x4 grid:
    Points are (i,j) for i,j in 0..3.
    Triples:
    - Rows: { (i,0), (i,1), (i,2) }, { (i,1), (i,2), (i,3) }, { (i,2), (i,3), (i,0) }? 
      This creates cycles. Let's use simple triples.
    
    Let's use a different approach: 
    We know that a random linear hypergraph with enough edges has small independence number.
    Let's generate a large set of random triples, filter for linearity, and check.
    
    For determinism, we'll use a fixed seed.
    """
    import random
    random.seed(42)
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    random.shuffle(all_triples)
    
    # We'll use a greedy approach with multiple restarts to find a good solution
    best_triples = []
    best_indep = float('inf')
    
    def independence_number(triples):
        """Compute the independence number of the hypergraph."""
        # A subset is independent if it contains no triple.
        # We want the largest such subset.
        # This is NP-hard in general, but for n=16 we can use inclusion-exclusion or brute force.
        # Brute force over all 2^16 = 65536 subsets is feasible.
        
        triple_set = set(frozenset(t) for t in triples)
        
        max_indep = 0
        # Iterate over all subsets
        for mask in range(1 << n):
            # Check if this subset contains any triple
            subset = set()
            for i in range(n):
                if mask & (1 << i):
                    subset.add(i)
            
            # Check if subset contains any triple
            contains_triple = False
            for t in triple_set:
                if t.issubset(subset):
                    contains_triple = True
                    break
            
            if not contains_triple:
                max_indep = max(max_indep, len(subset))
        
        return max_indep
    
    # Greedy with multiple random orderings
    for restart in range(50):
        random.seed(42 + restart)
        random.shuffle(all_triples)
        
        triples = []
        # Track point pairs to ensure linearity
        pair_count = {}
        
        for t in all_triples:
            a, b, c = t
            # Check if adding this triple violates linearity
            # A triple (a,b,c) has pairs (a,b), (a,c), (b,c)
            # We can add it if none of these pairs have already been used in another triple
            # Wait, linearity means any two triples share at most one point.
            # So a pair of points can appear in at most one triple.
            
            pairs = [(a,b), (a,c), (b,c)]
            ok = True
            for p in pairs:
                if p in pair_count:
                    ok = False
                    break
            
            if ok:
                triples.append(t)
                for p in pairs:
                    pair_count[p] = 1
        
        # Evaluate independence number
        indep = independence_number(triples)
        if indep < best_indep:
            best_indep = indep
            best_triples = triples.copy()
        
        # Early stop if we find a very good solution
        if best_indep <= 5:
            break
    
    return best_triples

def construct():
    triples = _generate_triples()
    # Convert to list of lists
    return [list(t) for t in triples]

# EVOLVE-BLOCK-END
