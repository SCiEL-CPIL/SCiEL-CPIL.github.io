import time
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Constructs a maximal 3-uniform hypergraph on 16 vertices without a K(2,2,2) subgraph.
    Uses Integer Linear Programming (ILP) via scipy.optimize.milp.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    # Generate all possible triples (edges)
    all_triples = list(itertools.combinations(range(n), 3))
    m = len(all_triples) # 560
    
    # Map triple to index for quick lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Identify all K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs of vertices.
    # Let the pairs be P1=(a,b), P2=(c,d), P3=(e,f).
    # The 8 transversal triples are formed by picking one vertex from each pair.
    # We must ensure that NOT all 8 of these triples are present.
    # This is equivalent to: sum(x_i for i in transversals) <= 7
    
    constraints = []
    constraints_list = []
    
    # We need to iterate over all sets of 3 disjoint pairs.
    # Number of ways to choose 6 distinct vertices from 16: C(16,6) = 8008
    # Number of ways to partition 6 vertices into 3 pairs: 15
    # Total configurations: 8008 * 15 = 120120
    # This is manageable.
    
    vertices = range(n)
    
    # Precompute constraints as sparse matrices or lists of indices
    # Since m=560 is small, we can build a dense matrix if needed, 
    # but scipy.milp prefers sparse or linear constraints objects.
    # Let's collect constraint rows.
    
    A_rows = []
    b_ub = []
    b_lb = []
    
    for verts in itertools.combinations(vertices, 6):
        # Partition these 6 vertices into 3 pairs
        # Pairs can be formed by: (v0,v1),(v2,v3),(v4,v5)
        # There are 15 distinct perfect matchings on 6 vertices.
        # We can generate them by permuting the first 3 pairs? 
        # Easier: iterate over all partitions.
        # A standard way to generate perfect matchings of 2k points:
        # Fix v0. Pair it with one of the other 5. 
        # Recursively pair the rest.
        
        def get_matchings(vs):
            if len(vs) == 0:
                yield []
                return
            if len(vs) == 2:
                yield [(vs[0], vs[1])]
                return
            first = vs[0]
            rest = vs[1:]
            for i in range(len(rest)):
                partner = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for sub_match in get_matchings(new_rest):
                    yield [(first, partner)] + sub_match
        
        for matching in get_matchings(list(verts)):
            # matching is a list of 3 pairs
            p1, p2, p3 = matching
            
            # Generate the 8 transversal triples
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        idx = triple_to_idx[t]
                        transversals.append(idx)
                        
            # Constraint: sum(x_i for i in transversals) <= 7
            # In scipy.milp, we define LinearConstraint(A, lb, ub)
            # A is a matrix. We can build it row by row.
            
            row = np.zeros(m)
            for idx in transversals:
                row[idx] = 1.0
            
            A_rows.append(row)
            b_lb.append(0)
            b_ub.append(7)
            
            # Check time
            if time.time() > deadline - 20: # Leave 20s for solver
                break
        
        if time.time() > deadline - 20:
            break
            
    if not A_rows:
        # Fallback if constraint generation was too slow (unlikely)
        # Return seed solution
        return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
        
    A = np.array(A_rows)
    
    # Objective: maximize sum(x_i)
    # milp minimizes, so minimize -sum(x_i)
    c = -np.ones(m)
    
    # Variables are binary
    integrality = np.ones(m)
    
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # Constraints
    # A @ x >= 0
    # A @ x <= 7
    # We can combine into one LinearConstraint or use two.
    # LinearConstraint(A, 0, 7) means 0 <= A @ x <= 7
    
    cons = LinearConstraint(A, 0, 7)
    
    # Run solver
    try:
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            # res.x contains the optimal values
            x = res.x
            selected_triples = []
            for i in range(m):
                if x[i] > 0.5:
                    selected_triples.append(list(all_triples[i]))
            return selected_triples
        else:
            # If solver fails, fallback
            pass
    except Exception as e:
        pass
    
    # Fallback: Greedy construction
    # This ensures we always return something valid, even if ILP fails or times out
    current_triples = []
    triple_set = set()
    
    # Heuristic: Sort triples by some metric or just iterate
    # To be fast, we can use a simple greedy approach with random restarts if time permits
    # But for safety, let's do a deterministic greedy based on degree or just order
    
    # A better fallback: Use the seed solution (105 triples)
    # Or try a quick greedy
    # Let's try a quick greedy: add triples that don't create K(2,2,2)
    
    # Precompute conflicts for greedy? Too slow.
    # Just return the seed solution if ILP fails.
    # Seed solution: all triples containing 0.
    return [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]

if __name__ == "__main__":
    # Quick test
    t = construct()
    print(f"Number of triples: {len(t)}")
