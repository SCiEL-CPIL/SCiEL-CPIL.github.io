"""Maximum 4-uniform family on [16] with no three-petal sunflower (all pairwise intersections equal)."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin (evaluator kills at 90s)
    
    n = 16
    k = 4
    # Generate all 4-subsets of [1, 16]
    all_sets = list(combinations(range(1, n + 1), k))
    num_sets = len(all_sets)  # C(16,4) = 1820
    
    # Represent each set as a bitmask over 16 elements
    set_masks = []
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        set_masks.append(mask)
    
    # Build conflict graph: edge between i and j if there exists a third set l
    # such that |A_i ∩ A_j| = |A_i ∩ A_l| = |A_j ∩ A_l|
    # Two sets can form a sunflower with a third if their intersection size is
    # "compatible" with some third set.
    # Precompute intersection sizes between all pairs.
    # inter[i][j] = popcount(mask_i & mask_j)
    
    # For efficiency, precompute popcount for all 16-bit masks
    # Actually, let's precompute intersection sizes as a list of arrays
    # inter[i] is a list of length num_sets with intersection sizes
    
    # Better approach: For each pair (i,j), the intersection size d = |A_i ∩ A_j|.
    # A third set l forms a sunflower with i,j if |A_i ∩ A_l| = d AND |A_j ∩ A_l| = d.
    # So for each pair (i,j) with intersection size d, we need to check if there
    # exists l ≠ i,j such that |A_i ∩ A_l| = d and |A_j ∩ A_l| = d.
    
    # Precompute: for each set i and each possible intersection size d (0..4),
    # the list of sets l such that |A_i ∩ A_l| = d.
    
    # Actually, let's just precompute the full intersection matrix.
    # 1820 x 1820 = ~3.3 million entries, each 0-4. Use bytearray.
    
    inter = [bytearray(num_sets) for _ in range(num_sets)]
    
    for i in range(num_sets):
        mi = set_masks[i]
        for j in range(i + 1, num_sets):
            d = bin(mi & set_masks[j]).count('1')
            inter[i][j] = d
            inter[j][i] = d
        inter[i][i] = k
    
    # Now build the conflict graph.
    # Edge (i,j) exists if there's a sunflower triple containing i and j.
    # For each pair (i,j), let d = inter[i][j].
    # We need to check if any l (l≠i,j) has inter[i][l] == d AND inter[j][l] == d.
    
    # To make this faster, for each set i and each d, precompute the set of l's
    # where inter[i][l] == d.
    
    # neighbors_by_d[i][d] = frozenset of l such that inter[i][l] == d
    neighbors_by_d = [[set() for _ in range(k + 1)] for _ in range(num_sets)]
    for i in range(num_sets):
        for j in range(num_sets):
            d = inter[i][j]
            neighbors_by_d[i][d].add(j)
    
    # Now for each pair (i,j), check if intersection of neighbors_by_d[i][d] and
    # neighbors_by_d[j][d] contains any l not in {i,j}.
    
    # conflict[i] = set of j such that (i,j) is an edge in the conflict graph.
    conflict = [set() for _ in range(num_sets)]
    
    for i in range(num_sets):
        for j in range(i + 1, num_sets):
            d = inter[i][j]
            # Check if there's l in neighbors_by_d[i][d] ∩ neighbors_by_d[j][d] \ {i,j}
            # This is expensive to do naively. Let's optimize.
            # The intersection of two sets of l's...
            # neighbors_by_d[i][d] has size = number of sets with intersection d with set i.
            # For d=0: sets disjoint from i. C(12,4) = 495
            # For d=1: 12 * C(11,3) = 12*165 = 1980... wait no.
            # |A_i ∩ A_l| = 1: choose 1 from A_i (4 ways), 3 from complement (12 choose 3 = 220), total 4*220=880
            # For d=2: C(4,2)*C(12,2) = 6*66 = 396
            # For d=3: C(4,3)*C(12,1) = 4*12 = 48
            # For d=4: 1 (itself)
            
            # The intersection could be large. Let's just iterate over the smaller set.
            si = neighbors_by_d[i][d]
            sj = neighbors_by_d[j][d]
            if len(si) > len(sj):
                si, sj = sj, si
            found = False
            for l in si:
                if l == i or l == j:
                    continue
                if l in sj:
                    found = True
                    break
            if found:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Now we have the conflict graph. We want the maximum independent set.
    # This is NP-hard, but with 1820 vertices we can use branch and bound.
    
    # Convert to bitsets for efficiency
    # Use Python integers as bitsets (1820 bits fits in a Python int)
    
    # adj[i] = bitmask of neighbors of i
    adj = [0] * num_sets
    for i in range(num_sets):
        mask = 0
        for j in conflict[i]:
            mask |= (1 << j)
        adj[i] = mask
    
    # Non-neighbors (including self) for independent set expansion
    # non_adj[i] = bitmask of vertices not adjacent to i (and not i itself)
    all_mask = (1 << num_sets) - 1
    non_adj = [0] * num_sets
    for i in range(num_sets):
        # non_adj[i] = all_mask ^ adj[i] ^ (1 << i)
        non_adj[i] = all_mask & ~(adj[i] | (1 << i))
    
    # Branch and bound for maximum independent set
    # Use a greedy coloring bound on the remaining candidates.
    
    best_set = 0  # bitmask of best independent set found
    best_size = 0
    
    # For the bound, we use a simple greedy coloring of the subgraph induced by candidates.
    # The number of colors is an upper bound on the clique number, which is a lower bound...
    # Wait, we want independent set. The coloring of the complement graph gives a bound.
    # Actually, for maximum independent set, we can use the maximum clique in the complement.
    # The complement of the conflict graph: edge (i,j) if NOT in conflict.
    # An independent set in conflict graph = clique in complement.
    # We can use branch and bound for maximum clique.
    
    # Let's reformulate: find maximum clique in the complement graph.
    # comp_adj[i] = bitmask of non-neighbors of i in conflict graph (excluding i)
    # = non_adj[i]
    
    # Branch and bound for maximum clique with greedy coloring bound.
    # We'll use a recursive approach with bitmasks.
    
    # Greedy coloring for bound: given a set of candidate vertices (bitmask),
    # greedily color the subgraph. The max color class size gives an upper bound
    # on the clique size in that subgraph... actually the number of colors is an
    # upper bound on the clique size.
    
    # Let's implement a simple branch and bound.
    
    def greedy_color_bound(candidates):
        """Return an upper bound on the max clique size in the subgraph induced by candidates.
        Use greedy coloring: the number of colors used is an upper bound on clique size."""
        if candidates == 0:
            return 0
        # Greedy coloring: repeatedly find a maximal independent set in the remaining
        # subgraph (which is a maximal clique in the complement, i.e., maximal set of
        # mutually non-adjacent in original... wait this is getting confusing.
        
        # Let me think again. We're finding max clique in comp_adj graph.
        # comp_adj[i] has edges between non-conflicting pairs.
        # A clique in this graph = set of mutually non-conflicting vertices = independent set in conflict graph.
        
        # For the bound: greedily color the comp_adj subgraph. Each color class is an
        # independent set in comp_adj = a clique in conflict graph. The number of colors
        # is an upper bound on the clique size in comp_adj.
        
        # Actually simpler: just count how many vertices we can pack if we greedily
        # pick vertices. A simple bound: the size of a maximal independent set in the
        # conflict subgraph induced by candidates.
        
        # Let's use a simpler approach: iterative removal.
        # Bound = greedy: repeatedly pick a vertex, remove it and its neighbors in conflict graph.
        # This gives a lower bound on independent set, not upper.
        
        # For upper bound on clique in comp_adj: use the fact that if we color the comp_adj
        # graph, clique size ≤ number of colors.
        
        # Greedy coloring of comp_adj subgraph:
        remaining = candidates
        color_count = 0
        while remaining != 0:
            # Find a maximal independent set in comp_adj[remaining]
            # = maximal clique in conflict[remaining]
            # Greedily: pick a vertex, remove it and its non-neighbors in comp_adj
            # (i.e., its neighbors in conflict graph)
            
            # Pick the first set bit
            v_bit = remaining & (-remaining)
            v = v_bit.bit_length() - 1
            # In comp_adj, neighbors of v are comp_adj[v] = non_adj[v]
            # Independent set in comp_adj: vertices not connected to v in comp_adj
            # = vertices that ARE neighbors of v in conflict graph = adj[v]
            # So remove v and its comp_adj neighbors:
            # Remove v and (non_adj[v] & remaining)
            remaining &= ~(v_bit | (non_adj[v] & remaining))
            color_count += 1
        
        return color_count
    
    # This bound is expensive. Let's use a simpler one: just the number of candidates.
    # Or use a precomputed ordering.
    
    # Actually, for 1820 vertices, a full branch and bound might be too slow.
    # Let's use a different approach: local search / greedy with restarts.
    
    # Strategy: 
    # 1. Greedy construction: repeatedly pick a vertex with minimum degree in the
    #    remaining conflict graph, add it to the independent set, remove it and its neighbors.
    # 2. Local improvement: try to swap vertices in and out.
    # 3. Multiple random restarts.
    
    # Let's implement greedy with degree heuristic + local search.
    
    def greedy_iset(order=None):
        """Greedy independent set. order: optional list of vertex indices for consideration order."""
        if order is None:
            order = list(range(num_sets))
        
        chosen = set()
        removed = set()
        
        for v in order:
            if v in removed:
                continue
            # Check if v conflicts with any chosen vertex
            can_add = True
            for u in chosen:
                if v in conflict[u]:
                    can_add = False
                    break
            if can_add:
                chosen.add(v)
                # Mark v and its neighbors as removed
                removed.add(v)
                for u in conflict[v]:
                    removed.add(u)
        
        return chosen
    
    def local_search(initial_set, time_limit):
        """Improve an independent set via local search (swap operations)."""
        current = set(initial_set)
        current_size = len(current)
        in_set = [False] * num_sets
        for v in current:
            in_set[v] = True
        
        # For each vertex in the set, compute its "cost" = number of neighbors it blocks
        # that are not in the set. Swapping might help.
        
        # Simpler: try adding a vertex not in the set, if it conflicts with some in-set
        # vertices, remove those and add it if the net change is positive.
        
        not_in_set = [v for v in range(num_sets) if not in_set[v]]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle for randomness
            random.shuffle(not_in_set)
            for v in not_in_set:
                if in_set[v]:
                    continue
                # Find conflicts of v with current set
                conflicts = [u for u in current if v in conflict[u]]
                if len(conflicts) < 1:
                    # Can add v without removing anything
                    current.add(v)
                    in_set[v] = True
                    current_size += 1
                    improved = True
                elif len(conflicts) == 1:
                    # Swap: remove 1, add 1. No net gain, skip.
                    pass
                # For len(conflicts) > 1, removing multiple to add one is not helpful
                # unless we also add others. Skip for now.
        
        return current, current_size
    
    # Run multiple greedy + local search restarts
    best = set()
    best_size = 0
    
    num_restarts = 0
    while time.time() < deadline - 5.0:
        num_restarts += 1
        # Random order for greedy
        order = list(range(num_sets))
        random.shuffle(order)
        
        # Greedy
        candidate = greedy_iset(order)
        
        # Local search
        candidate, size = local_search(candidate, deadline - 5.0)
        
        if size > best_size:
            best_size = size
            best = set(candidate)
        
        # Also try a different greedy: prioritize low-degree vertices
        # Compute degrees
        if time.time() < deadline - 5.0:
            degrees = [len(conflict[v]) for v in range(num_sets)]
            order2 = sorted(range(num_sets), key=lambda v: degrees[v])
            random.shuffle(order2)  # Add some randomness
            # Actually, sort by degree but with random tie-breaking
            order2 = sorted(range(num_sets), key=lambda v: (degrees[v], random.random()))
            candidate2 = greedy_iset(order2)
            candidate2, size2 = local_search(candidate2, deadline - 5.0)
            if size2 > best_size:
                best_size = size2
                best = set(candidate2)
    
    # Convert best set to list of sorted 4-tuples
    result = []
    for v in best:
        result.append(list(all_sets[v]))
    
    # Sort for determinism
    result.sort()
    
    return result

# Pre-compute at module level for speed? No, construct() should be self-contained.
# But we can cache.
_cache = None

def construct():
    global _cache
    if _cache is not None:
        return _cache
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 16
    k = 4
    all_sets = list(combinations(range(1, n + 1), k))
    num_sets = len(all_sets)
    
    set_masks = []
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        set_masks.append(mask)
    
    # Precompute intersection sizes
    inter = [bytearray(num_sets) for _ in range(num_sets)]
    for i in range(num_sets):
        mi = set_masks[i]
        for j in range(i + 1, num_sets):
            d = bin(mi & set_masks[j]).count('1')
            inter[i][j] = d
            inter[j][i] = d
        inter[i][i] = k
    
    # neighbors_by_d[i][d] = set of l with inter[i][l] == d
    neighbors_by_d = [[set() for _ in range(k + 1)] for _ in range(num_sets)]
    for i in range(num_sets):
        for j in range(num_sets):
            d = inter[i][j]
            neighbors_by_d[i][d].add(j)
    
    # Build conflict graph
    conflict = [set() for _ in range(num_sets)]
    for i in range(num_sets):
        for j in range(i + 1, num_sets):
            d = inter[i][j]
            si = neighbors_by_d[i][d]
            sj = neighbors_by_d[j][d]
            if len(si) > len(sj):
                si, sj = sj, si
            found = False
            for l in si:
                if l == i or l == j:
                    continue
                if l in sj:
                    found = True
                    break
            if found:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Greedy + local search
    best = set()
    best_size = 0
    
    def greedy_iset(order):
        chosen = set()
        removed = set()
        for v in order:
            if v in removed:
                continue
            can_add = True
            for u in chosen:
                if v in conflict[u]:
                    can_add = False
                    break
            if can_add:
                chosen.add(v)
                removed.add(v)
                for u in conflict[v]:
                    removed.add(u)
        return chosen
    
    def local_search(initial_set, time_limit):
        current = set(initial_set)
        in_set = [False] * num_sets
        for v in current:
            in_set[v] = True
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            not_in_set = [v for v in range(num_sets) if not in_set[v]]
            random.shuffle(not_in_set)
            for v in not_in_set:
                if in_set[v]:
                    continue
                conflicts = [u for u in current if v in conflict[u]]
                if len(conflicts) == 0:
                    current.add(v)
                    in_set[v] = True
                    improved = True
        
        return current, len(current)
    
    while time.time() < deadline - 5.0:
        order = list(range(num_sets))
        random.shuffle(order)
        candidate = greedy_iset(order)
        candidate, size = local_search(candidate, deadline - 5.0)
        if size > best_size:
            best_size = size
            best = set(candidate)
        
        if time.time() < deadline - 5.0:
            degrees = [len(conflict[v]) for v in range(num_sets)]
            order2 = sorted(range(num_sets), key=lambda v: (degrees[v], random.random()))
            candidate2 = greedy_iset(order2)
            candidate2, size2 = local_search(candidate2, deadline - 5.0)
            if size2 > best_size:
                best_size = size2
                best = set(candidate2)
    
    result = [list(all_sets[v]) for v in best]
    result.sort()
    _cache = result
    return result
