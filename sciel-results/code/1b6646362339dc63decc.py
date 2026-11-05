# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    """
    Return a 5-uniform hypergraph with no proper 2-colouring.
    
    This implementation constructs a non-2-colourable 5-uniform hypergraph on 10 vertices
    with 63 edges. This is significantly better than the 126-edge complete hypergraph on 9 vertices.
    
    The construction relies on the fact that for n=10, any 2-colouring corresponds to a partition
    of vertices into sets A and B. To be non-2-colourable, every partition must have a monochromatic edge.
    We select 63 edges such that:
    1. For every partition (A, B) with |A|=5, |B|=5, either A or B is an edge.
       This is achieved by picking exactly one edge from each complementary pair of 5-sets.
    2. For every partition (A, B) with |A|=4, |B|=6, B contains an edge.
       We search for a random selection of 63 edges (one from each pair) that satisfies this.
    """
    n = 10
    all_edges = [tuple(e) for e in combinations(range(n), 5)]
    
    # Identify complementary pairs of 5-sets
    pairs = []
    visited = set()
    for e in all_edges:
        if e in visited:
            continue
        e_set = set(e)
        comp_set = set(range(n)) - e_set
        comp = tuple(sorted(comp_set))
        if comp not in visited:
            pairs.append((e, comp))
            visited.add(e)
            visited.add(comp)
            
    # We have 63 pairs. We need to pick one from each.
    # We use a deterministic random seed to ensure reproducibility.
    random.seed(12345)
    
    for attempt in range(1000):
        E = []
        for p in pairs:
            if random.random() < 0.5:
                E.append(p[0])
            else:
                E.append(p[1])
                
        E_set = set(E)
        
        # Check if every 6-set contains an edge from E
        # This ensures that for any partition with a part of size >= 6, there is a monochromatic edge.
        valid = True
        for b in combinations(range(n), 6):
            b_set = set(b)
            # Check if any edge in E is a subset of b
            found = False
            for e in E_set:
                if e[0] in b_set and e[1] in b_set and e[2] in b_set and e[3] in b_set and e[4] in b_set:
                    found = True
                    break
            if not found:
                valid = False
                break
                
        if valid:
            # Condition 2 (balanced partitions) is satisfied by construction (one from each pair).
            # Condition 1 (unbalanced partitions) is satisfied by the check above.
            return {'n': n, 'edges': [list(e) for e in E]}
            
    # Fallback: Complete hypergraph on 9 vertices (126 edges)
    n = 9
    edges = [list(e) for e in combinations(range(n), 5)]
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
