"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    n = 256
    # Build conflict graph: vertex i (0-indexed, representing value i+1)
    # conflicts with vertex j if (i+1) + (j+1) is a perfect square
    # Also self-conflict if 2*(i+1) is a perfect square
    
    # Precompute which values are "self-conflicting" (2a is perfect square)
    self_conflict = [False] * n
    for i in range(n):
        val = 2 * (i + 1)
        r = math.isqrt(val)
        if r * r == val:
            self_conflict[i] = True
    
    # Build adjacency as bitsets (using Python integers as bitsets)
    # adj[i] is a bitmask of all vertices j that conflict with i
    adj = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            s = (i + 1) + (j + 1)
            r = math.isqrt(s)
            if r * r == s:
                mask |= (1 << j)
        adj[i] = mask
    
    # Remove self-conflicting vertices from consideration
    # They can never be in any valid set
    available_mask = 0
    for i in range(n):
        if not self_conflict[i]:
            available_mask |= (1 << i)
    
    # Count available vertices
    avail_count = available_mask.bit_count()
    
    best_set_mask = 0  # bitset of best solution found
    best_size = 0
    time_exceeded = [False]
    
    def count_bits(x):
        return x.bit_count()
    
    def greedy_coloring_bound(candidates):
        """
        Upper bound on max independent set size using greedy coloring on the
        conflict subgraph induced by 'candidates'.
        The number of colors needed >= size of max independent set's complement...
        Actually: we want max independent set in G[candidates].
        Equivalently, max clique in complement of G[candidates].
        Greedy coloring of complement gives upper bound on max clique.
        
        Simpler approach: greedy coloring of the conflict graph G[candidates].
        Each color class is an independent set. The max independent set size
        is at most the sum of... no that's not right.
        
        Better: Use the standard B&B upper bound for MIS:
        - Greedily color the conflict graph. The chromatic number is an upper
          bound on the size of any clique, but we want independent set.
        
        Actually for maximum independent set, a valid upper bound is:
        Greedily pick vertices, each time picking a vertex and removing it
        plus all its neighbors. The count is a LOWER bound (greedy IS size).
        
        For an UPPER bound: use the fact that if we color the graph, each
        color class is an independent set, so alpha(G) <= sum over... no.
        
        Let me use a different approach: the upper bound is the size of the
        maximum matching complement... 
        
        Simplest effective upper bound: greedy independent set (lower bound)
        + (remaining vertices that could still be added) - but that's trivial.
        
        Better upper bound: For the subgraph induced by candidates, compute
        a greedy coloring. The number of vertices / max_degree_in_coloring
        gives an upper bound. Actually, the standard approach:
        
        Upper bound = number of vertices in candidates that are not yet
        "covered" by the greedy process, where we greedily select vertices
        of minimum degree. This is actually a lower bound.
        
        Let me use: upper bound = count of candidates (trivial) minus
        vertices that must be excluded due to conflicts with already-selected
        vertices. But at the root, no vertices are selected.
        
        The most practical fast upper bound: greedy coloring of the conflict
        graph. If we use k colors, then the independence number alpha <= n/k
        is NOT true. What IS true: alpha(G) >= n/k where k is chromatic number.
        And alpha(G) <= n - tau(G) where tau is min vertex cover.
        
        For a practical upper bound in B&B for MIS:
        - Compute a greedy maximal matching in the conflict subgraph.
        - Upper bound = |candidates| - |matching| (since each edge in matching
          means at most one of the two endpoints can be in the IS).
        This is a valid upper bound and fast to compute.
        """
        if candidates == 0:
            return 0
        
        # Greedy maximal matching upper bound
        remaining = candidates
        matching_size = 0
        while remaining:
            # Pick the first vertex
            lb = remaining & (-remaining)
            v = lb.bit_length() - 1
            # Find a neighbor of v in remaining
            neighbors = adj[v] & remaining
            if neighbors:
                # Pick the neighbor with lowest index
                lb2 = neighbors & (-neighbors)
                u = lb2.bit_length() - 1
                matching_size += 1
                remaining &= ~(lb | lb2)
            else:
                # v has no neighbors in remaining, it's isolated
                # It can always be in the IS, so it doesn't reduce the bound
                # But for the matching bound: isolated vertices don't form edges
                remaining &= ~lb
        return remaining.bit_count() + matching_size  # Wait, this isn't right either
    
    # Let me reconsider. The standard upper bound for MIS in B&B:
    # At a node with candidate set C, the upper bound on the total IS size
    # is: current_size + alpha(G[C])
    # We need a fast upper bound on alpha(G[C]).
    # 
    # Valid upper bounds:
    # 1. |C| (trivial)
    # 2. |C| - max_matching(G[C]) (each matched edge contributes at most 1)
    #    Actually: alpha(G) = |V| - tau(G) >= |V| - |matching|
    #    So |V| - |matching| is a LOWER bound on alpha, not upper.
    # 3. For each vertex v in C: alpha(G[C]) <= 1 + alpha(G[C - N[v]])
    #    This is the recursion itself.
    # 4. Greedy coloring: if G[C] can be colored with k colors, then
    #    alpha(G[C]) <= |C| (trivial). Not helpful.
    # 5. alpha(G[C]) <= |C| / (1 + avg_degree/(|C|-1)) ... not standard.
    #
    # The most practical fast upper bound used in practice:
    # - Compute a greedy maximal matching M in G[C].
    # - Then alpha(G[C]) <= |C| - |M| is WRONG (it's a lower bound).
    # - Correct: since each edge in M has at most one endpoint in IS,
    #   and unmatched vertices can all be in IS:
    #   alpha(G[C]) <= |C| - |M| is actually a valid UPPER bound?
    #   No: consider a path of 3 vertices a-b-c. Matching = {a-b}.
    #   |C| - |M| = 3 - 1 = 2. alpha = 2 (pick a,c). OK.
    #   Consider a triangle. Matching = 1. |C|-|M| = 2. alpha = 1. 2 >= 1. OK.
    #   Consider K4. Matching = 2. |C|-|M| = 2. alpha = 1. 2 >= 1. OK.
    #   Consider two disjoint edges: a-b, c-d. Matching = 2. |C|-|M| = 2.
    #   alpha = 2. OK.
    #   Consider a-b, b-c, c-d (path of 4). Matching could be {a-b, c-d}, size 2.
    #   |C|-|M| = 2. alpha = 2. OK.
    #   Hmm, is |C| - |max_matching| always >= alpha? 
    #   alpha = |V| - tau, and tau >= |max_matching|, so alpha <= |V| - |max_matching|.
    #   YES! This is a valid upper bound.
    #
    # But computing max matching is expensive. A greedy maximal matching gives
    # a valid (weaker) bound: alpha <= |V| - |greedy_maximal_matching|.
    # Wait no: tau >= |any_matching|, so alpha = |V| - tau <= |V| - |any_matching|.
    # So even a non-maximal matching gives a valid upper bound!
    # A greedy maximal matching is fine.
    
    def upper_bound(candidates):
        """Fast upper bound on alpha(G[candidates]) using greedy maximal matching."""
        if candidates == 0:
            return 0
        remaining = candidates
        matching_size = 0
        while remaining:
            lb = remaining & (-remaining)
            v = lb.bit_length() - 1
            neighbors = adj[v] & remaining
            if neighbors:
                lb2 = neighbors & (-neighbors)
                matching_size += 1
                remaining &= ~(lb | lb2)
            else:
                remaining &= ~lb
        return remaining.bit_count() + matching_size
        # Wait: unmatched vertices (remaining) can ALL be in the IS.
        # Matched vertices: at most one per edge.
        # So upper bound = |remaining_unmatched| + |matching|
        # Because for each matched edge, at most 1 vertex, plus all unmatched.
        # |remaining_unmatched| + |matching| = (|C| - 2*|matching|) + |matching|
        # = |C| - |matching|
        # This matches the formula above. Good.
    
    def branch_and_bound(candidates, current_mask, current_size):
        nonlocal best_set_mask, best_size, time_exceeded
        
        if time_exceeded[0]:
            return
        if time.time() > DEADLINE:
            time_exceeded[0] = True
            return
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
        
        # Upper bound
        ub = current_size + upper_bound(candidates)
        if ub <= best_size:
            return
        
        # Choose vertex to branch on: maximum degree in candidates
        # (branching on high-degree vertex prunes more)
        best_v = -1
        best_deg = -1
        temp = candidates
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = (adj[v] & candidates).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        new_candidates = candidates & ~(adj[best_v] | (1 << best_v))
        branch_and_bound(new_candidates, current_mask | (1 << best_v), current_size + 1)
        
        if time_exceeded[0]:
            return
        
        # Branch 2: Exclude best_v
        new_candidates = candidates & ~(1 << best_v)
        branch_and_bound(new_candidates, current_mask, current_size)
    
    # Initialize with greedy solution to get a good lower bound
    def greedy_init():
        remaining = available_mask
        sol_mask = 0
        while remaining:
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lb
            if best_v == -1:
                break
            sol_mask |= (1 << best_v)
            remaining &= ~(adj[best_v] | (1 << best_v))
        return sol_mask
    
    init_mask = greedy_init()
    best_size = init_mask.bit_count()
    best_set_mask = init_mask
    
    # Run branch and bound
    branch_and_bound(available_mask, 0, 0)
    
    # Extract solution
    result = []
    for i in range(n):
        if (best_set_mask >> i) & 1:
            result.append(i + 1)
    
    if not result:
        # Fallback: return something valid
        result = [i + 1 for i in range(n) if not self_conflict[i]]
    
    return result
# EVOLVE-BLOCK-END
