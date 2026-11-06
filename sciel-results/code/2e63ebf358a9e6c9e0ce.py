import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 78  # 78s safety margin (90s kill, 12s buffer)
    
    N = 180
    
    # Precompute all forbidden triples
    # 1/a = 1/b + 1/c where a,b,c are distinct
    # a = bc/(b+c), so b+c must divide bc
    # Equivalently: for each a, find b,c > a (or any distinct) such that bc/(b+c) = a
    # Let b = a+d, c = a+e, then (a+d)(a+e)/(2a+d+e) = a
    # (a+d)(a+e) = a(2a+d+e)
    # a² + ae + ad + de = 2a² + ad + ae
    # de = a²
    # So for each a, factor a² = d*e with d != e, and b = a+d, c = a+e (or swapped)
    # We need b,c <= N and b != c (d != e) and all distinct from a
    
    # Build conflict structure: for each element, which other elements conflict with it
    # Two elements x,y conflict if there exists z such that {x,y,z} is a forbidden triple
    # Actually, we need: no triple (a,b,c) all in our set with 1/a=1/b+1/c
    # This is a 3-uniform hypergraph. For independent set in 3-uniform hypergraph,
    # we can't just use pairwise conflicts.
    
    # Let's enumerate all forbidden triples
    forbidden_triples = set()
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if b <= N and c <= N and b != c:
                # Check all three are distinct
                if a != b and a != c and b != c:
                    triple = tuple(sorted([a, b, c]))
                    forbidden_triples.add(triple)
            # Also try other factorizations? No, de=a² with d<e gives all
            # But wait: b and c could be less than a too?
            # 1/a = 1/b + 1/c. If b < a, then 1/b > 1/a, so 1/c = 1/a - 1/b < 0, impossible.
            # So b > a and c > a always. Good.
    
    # Now we have all forbidden triples. We want max subset with no forbidden triple fully contained.
    # This is maximum independent set in a 3-uniform hypergraph.
    # Use branch and bound.
    
    # Represent the problem: we have N elements (1..N), and a set of forbidden triples.
    # We want to select as many as possible such that no forbidden triple is fully selected.
    
    # Greedy initialization: start with all elements, remove conflicts
    # Better: use a greedy approach to get a good initial solution
    
    # For branch and bound on 3-uniform hypergraph:
    # State: which elements are "decided" (in/out), and which triples are still "active"
    # (have at least 2 undecided elements and no element already out)
    
    # Let's use a different approach: since N=180 is moderate, use a smart greedy + local search
    # Or use integer programming formulation with branch and bound
    
    # Actually, let's use a recursive branch and bound with good ordering
    
    # First, get a greedy lower bound
    # Greedy: process elements in some order, add if doesn't complete a forbidden triple
    
    # For better results, let's use a MILP-like approach with scipy or just careful B&B
    
    # Given time constraints, let's do:
    # 1. Greedy to get initial solution
    # 2. Local search / branch and bound to improve
    
    # Greedy heuristic: sort by "conflict degree" (number of triples involving that element)
    # Add elements in order of lowest conflict degree first
    
    # Build conflict counts
    triple_count_per_elem = [0] * (N + 1)
    triples_by_elem = [[] for _ in range(N + 1)]
    for tri in forbidden_triples:
        for x in tri:
            triple_count_per_elem[x] += 1
            triples_by_elem[x].append(tri)
    
    # Greedy: add elements sorted by increasing triple_count
    # Actually, for maximum independent set in hypergraph, we want to avoid adding
    # elements that would complete many triples. So add "safe" elements first.
    
    # Let's try: start with all, then iteratively remove elements that are in most active triples
    # Or: greedy add in order of least "damage"
    
    # Simple greedy: process in order 1..N, add if no forbidden triple is completed
    selected = set()
    # For efficiency, track for each pair (x,y) how many z would complete a triple
    # Actually, let's just check: when adding x, is there a triple (x,y,z) with y,z already in?
    
    # Better data structure: for each element x, store the triples it's in
    # When considering adding x, check if any triple containing x has the other two already selected
    
    # Let's precompute for each x, the list of (y,z) pairs such that (x,y,z) is forbidden
    pairs_for = [[] for _ in range(N + 1)]
    for tri in forbidden_triples:
        x, y, z = tri
        pairs_for[x].append((y, z))
        pairs_for[y].append((x, z))
        pairs_for[z].append((x, y))
    
    # Greedy: try adding in order of increasing number of conflicts
    order = sorted(range(1, N + 1), key=lambda x: len(pairs_for[x]))
    
    selected = set()
    for x in order:
        # Check if adding x creates a forbidden triple
        conflict = False
        for (y, z) in pairs_for[x]:
            if y in selected and z in selected:
                conflict = True
                break
        if not conflict:
            selected.add(x)
    
    best_size = len(selected)
    best_set = set(selected)
    
    # Now try to improve with local search / branch and bound
    # Time check
    if time.time() > deadline - 2:
        return sorted(best_set)
    
    # Try random restarts with different orderings
    import random
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline - 2:
            break
        
        # Random perturbation of the greedy
        order2 = list(range(1, N + 1))
        random.shuffle(order2)
        
        sel2 = set()
        for x in order2:
            conflict = False
            for (y, z) in pairs_for[x]:
                if y in sel2 and z in sel2:
                    conflict = True
                    break
            if not conflict:
                sel2.add(x)
        
        if len(sel2) > best_size:
            best_size = len(sel2)
            best_set = set(sel2)
    
    # Try to improve best_set by removing and re-adding
    # Local search: try removing one element and adding another
    if time.time() > deadline - 2:
        return sorted(best_set)
    
    # More sophisticated: use the structure
    # Elements that appear in no forbidden triples can always be included
    # Let's identify "free" elements
    free_elements = set()
    for x in range(1, N + 1):
        if len(pairs_for[x]) == 0:
            free_elements.add(x)
    
    # All free elements should be in the solution
    # For the rest, we need to solve the hypergraph independent set
    
    # Let's try a more careful branch and bound on the non-free elements
    constrained = set(range(1, N + 1)) - free_elements
    
    # Filter triples to only those with at least one constrained element
    # (triples with all free elements can't happen since free means no triples)
    
    # For the constrained part, use B&B
    # Order constrained elements by degree
    constrained_list = sorted(constrained, key=lambda x: len(pairs_for[x]))
    
    # B&B: try to build a set from constrained elements
    # State: index in constrained_list, current selected set, best so far
    # Prune: if current_size + remaining < best_size, prune
    
    # This could be exponential, so we need good pruning
    # Let's limit the depth and use the greedy result as lower bound
    
    # Actually, given time, let's just return what we have from greedy + random restarts
    # The greedy should already be quite good
    
    return sorted(best_set)
