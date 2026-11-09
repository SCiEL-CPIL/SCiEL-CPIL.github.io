"""Maximum independent set in 3-uniform hypergraph for sunflower-free families."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # safety margin before 90s kill
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
    
    total = len(all_sets)  # 1820
    
    # Precompute pairwise intersections
    inter = [[0]*total for _ in range(total)]
    for i in range(total):
        for j in range(i+1, total):
            inter[i][j] = all_sets[i] & all_sets[j]
            inter[j][i] = inter[i][j]
    
    # Build hypergraph: hyperedges are forbidden sunflower triples
    # For each triple (i,j,l), check if it forms a sunflower
    # A sunflower: three sets where all three pairwise intersections are equal
    # (including empty intersection)
    
    # For efficiency, precompute for each pair (i,j) the intersection mask
    # and the set of all l such that (i,j,l) forms a sunflower
    
    # A triple (i,j,l) is a sunflower iff:
    # inter[i][j] == inter[i][l] == inter[j][l]
    
    # For each pair (i,j), the intersection is inter[i][j]
    # For l to form a sunflower with i,j, we need:
    # inter[i][l] == inter[i][j] AND inter[j][l] == inter[i][j]
    
    # Build adjacency: for each vertex, list of hyperedges containing it
    # But hyperedges are triples, so we need a different structure
    
    # Alternative: build a graph where two vertices are "conflicting" if they
    # appear together in some forbidden triple with a third vertex
    # Actually, for MIS in hypergraph, we need: no hyperedge is fully contained in solution
    
    # Let's use a different approach: 
    # Build a 3-uniform hypergraph H where vertices = all 4-subsets
    # Hyperedges = all triples (i,j,l) that form a sunflower
    # We want maximum independent set in H (no hyperedge fully in solution)
    
    # For branch-and-bound, we need efficient conflict checking
    # Two vertices i,j are in conflict if there exists l such that (i,j,l) is a hyperedge
    # i.e., inter[i][j] == inter[i][l] == inter[j][l]
    
    # Precompute for each pair (i,j), the list of l that complete a sunflower
    # This is expensive: 1820^2 pairs, each checking 1820 l's = ~6 billion ops
    
    # Better: for each pair (i,j), inter[i][j] is fixed.
    # We need l such that inter[i][l] == inter[i][j] AND inter[j][l] == inter[i][j]
    
    # Let's precompute for each vertex i and each possible intersection value v,
    # the list of vertices j such that inter[i][j] == v
    
    # Possible intersection values: subsets of 4 elements, so 0 to 4 elements
    # There are 2^4 = 16 possible intersection patterns for a fixed 4-set
    
    # Actually, inter[i][j] is a bitmask of up to 4 bits (subset of the 4 elements of i)
    
    # Let's use a different strategy: local search + greedy
    
    # Strategy: 
    # 1. Start with a good initial solution (greedy)
    # 2. Use local search (add/remove/swap) to improve
    # 3. Use branch-and-bound on reduced problem if time permits
    
    # First, let's build a fast conflict check
    # For a given set of chosen vertices, can we add vertex v?
    # v conflicts with pair (i,j) in chosen if inter[i][j] == inter[i][v] == inter[j][v]
    
    # Precompute for each vertex v, and each pair of "intersection types"
    
    # Let me use a smarter approach: 
    # For each vertex v, precompute a "signature" for each other vertex
    # sig[v][u] = inter[v][u]
    
    # Then v can be added to chosen set S iff for all pairs (i,j) in S:
    # inter[i][j] != inter[i][v] OR inter[i][j] != inter[j][v]
    # i.e., NOT (inter[i][j] == inter[i][v] == inter[j][v])
    
    # This is O(|S|^2) per insertion check, which is expensive for large S
    
    # Alternative: precompute for each vertex v, the set of pairs (i,j) that form
    # a sunflower with v. Then v cannot be added if any such pair is both in S.
    
    # For vertex v, a pair (i,j) forms a sunflower with v iff:
    # inter[i][j] == inter[i][v] == inter[j][v]
    
    # Let's precompute for each v, the list of (i,j) pairs that conflict with v
    # This is still expensive: for each v (1820), for each pair (i,j) (1820^2/2)
    
    # Let's try a different angle: use the structure of 4-subsets
    # Two 4-subsets can intersect in 0, 1, 2, 3, or 4 elements
    # If they intersect in 4, they're the same set
    # If they intersect in 3, they share 3 elements
    
    # For a sunflower with core C (|C| = c), the petals are C ∪ {a_i}, C ∪ {b_i}, C ∪ {c_i}
    # where a_i, b_i, c_i are distinct elements not in C
    # The pairwise intersections are all C
    
    # So for a given core C (size c), the sunflower triples are triples of 4-sets
    # that all contain C and have distinct "petal" elements
    
    # Let's enumerate by core size:
    # c=0: empty core, three 4-sets with empty intersection
    # c=1: one common element
    # c=2: two common elements
    # c=3: three common elements
    
    # For each core C, count how many 4-sets contain C
    # If m(C) 4-sets contain C, then the number of sunflower triples with core C is C(m(C),3)
    # But we need to be careful: a triple might have a larger core
    
    # Actually, the core of a sunflower is the intersection of all three sets
    # So for a triple (A,B,C), the core is A∩B∩C
    # The triple is a sunflower iff A∩B = B∩C = A∩C = core
    
    # Let's just build the hypergraph explicitly, but efficiently
    
    # For each triple, check if it's a sunflower
    # Number of triples: C(1820,3) ≈ 1 billion - too many
    
    # We need a smarter approach
    
    # Key insight: for a fixed pair (i,j), the intersection inter[i][j] determines
    # which l's can form a sunflower with i,j
    # l must satisfy: inter[i][l] == inter[i][j] AND inter[j][l] == inter[i][j]
    
    # Let's precompute for each vertex i, group other vertices by intersection with i
    # groups[i][v] = list of j such that inter[i][j] == v
    
    # Then for pair (i,j), the candidates for l are in groups[i][inter[i][j]] ∩ groups[j][inter[i][j]]
    
    # Let's build this structure
    from collections import defaultdict
    
    # groups[i] = dict mapping intersection_mask -> list of vertex indices
    groups = [defaultdict(list) for _ in range(total)]
    for i in range(total):
        for j in range(total):
            if i != j:
                v = inter[i][j]
                groups[i][v].append(j)
    
    # Now for each pair (i,j), the conflict partners are:
    # l in groups[i][inter[i][j]] ∩ groups[j][inter[i][j]]
    # But we need l != i and l != j
    
    # For branch-and-bound, let's use a different strategy:
    # Greedy construction with lookahead, then local search
    
    # Greedy: order vertices, add if no conflict with existing pairs
    # For efficiency, maintain for each pair in chosen set, the "forbidden" vertices
    
    # Actually, let's use a simpler but effective approach:
    # 1. Greedy construction with multiple random orderings
    # 2. Local search: try to add one vertex, remove one, or swap
    
    # For conflict check, we need: can we add vertex v to chosen set S?
    # v conflicts if there exist i,j in S such that inter[i][j] == inter[i][v] == inter[j][v]
    
    # Precompute for each v, a bitset of vertices that are "compatible" with v
    # Two vertices i,j are compatible with v if adding v doesn't create a sunflower with i,j
    # But this depends on the pair (i,j), not just individual vertices
    
    # Let's just do the O(|S|^2) check but optimize with early termination
    
    def can_add_fast(chosen, v):
        """Check if vertex v can be added to chosen set."""
        n_chosen = len(chosen)
        # Precompute inter[v][*] for all in chosen
        for i_idx in range(n_chosen):
            i = chosen[i_idx]
            iv = inter[i][v]
            for j_idx in range(i_idx + 1, n_chosen):
                j = chosen[j_idx]
                if inter[i][j] == iv and inter[j][v] == iv:
                    return False
        return True
    
    # Greedy with multiple restarts
    best_size = 0
    best_family = []
    
    rng = random.Random(123)
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        order = list(range(total))
        rng.shuffle(order)
        
        chosen = []
        
        for idx in order:
            if time.time() > deadline - 5:
                break
            if can_add_fast(chosen, idx):
                chosen.append(idx)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
    
    # Local search: try to add vertices not in best_family
    # Try each vertex not in best_family, see if it can be added
    chosen_set = set(best_family)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for v in range(total):
            if v in chosen_set:
                continue
            if can_add_fast(list(best_family), v):
                best_family.append(v)
                chosen_set.add(v)
                improved = True
    
    # Try swaps: remove one, add two
    for i in range(len(best_family)):
        if time.time() > deadline - 3:
            break
        saved = best_family[i]
        best_family.pop(i)
        chosen_set.remove(saved)
        
        added = 0
        for v in range(total):
            if v in chosen_set or v == saved:
                continue
            if can_add_fast(best_family, v):
                best_family.append(v)
                chosen_set.add(v)
                added += 1
                if added >= 2:
                    break
        
        if added <= 1:
            # Undo
            if added == 1:
                last = best_family[-1]
                best_family.pop()
                chosen_set.remove(last)
            best_family.insert(i, saved)
            chosen_set.add(saved)
    
    # Convert to result
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = [i+1 for i in range(n) if mask & (1 << i)]
        result.append(elems)
    
    return result
