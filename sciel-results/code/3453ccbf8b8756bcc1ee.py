"""Dense Q3-free graph on 16 vertices via precomputed Q3 masks + swap-based local search."""
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 110  # Safety margin
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    total_edges = len(all_edges)  # 120

    # Precompute all Q3 subgraphs (as frozensets of edge indices)
    # A Q3 is a 3-cube: 8 vertices, 12 edges, bipartite with 4+4, each vertex degree 3
    # We enumerate all 8-subsets of vertices and check if they can be labeled as Q3
    
    # For each 8-subset, check if the induced subgraph (on those 8 vertices) is isomorphic to Q3
    # Q3 has: 8 vertices, 12 edges, bipartite (4,4), each vertex degree 3, girth 4
    
    # Precompute: for each 8-subset, determine if it's a Q3 and which edges form it
    q3_edge_sets = []  # list of frozensets of (u,v) pairs
    
    # Enumerate all 8-subsets of {0,...,15}
    all_8subsets = list(combinations(range(n), 8))
    
    for subset in all_8subsets:
        # Check if the complete graph on these 8 vertices contains a Q3 as a subgraph
        # Actually, we need: does there exist a labeling of these 8 vertices as Q3 vertices
        # such that all 12 Q3 edges are present?
        # Since we're looking for Q3 as a subgraph (not induced), we need to check if
        # there's a bijection from Q3 vertices to these 8 vertices where all Q3 edges map to edges in our graph.
        # But our graph is being built, so we just need to know WHICH 12 edges would form a Q3.
        
        # Q3 structure: vertices are binary triples (0,0,0) to (1,1,1)
        # Edges connect vertices differing in exactly one bit
        q3_vertices = [(a, b, c) for a in range(2) for b in range(2) for c in range(2)]
        q3_edges = set()
        for i, v1 in enumerate(q3_vertices):
            for j, v2 in enumerate(q3_vertices):
                if i < j:
                    diff = sum(1 for x, y in zip(v1, v2) if x != y)
                    if diff == 1:
                        q3_edges.add((i, j))
        
        # Now check all bijections from q3_vertices to subset
        # There are 8! = 40320 bijections, but we can be smarter.
        # Q3 is bipartite with parts of size 4 (even weight and odd weight vertices)
        # Even weight: (0,0,0), (1,1,0), (1,0,1), (0,1,1)
        # Odd weight: (1,0,0), (0,1,0), (0,0,1), (1,1,1)
        
        even_q3 = [0, 3, 5, 6]  # indices of even-weight vertices
        odd_q3 = [1, 2, 4, 7]   # indices of odd-weight vertices
        
        # Try all ways to assign the 4 even Q3 vertices to 4 of the subset vertices
        # and the 4 odd Q3 vertices to the remaining 4
        # C(8,4) * 4! * 4! = 70 * 24 * 24 = 40320 -- same as before
        
        # Let's just do it directly but break early
        found = False
        for even_assign in combinations(range(8), 4):
            odd_assign = tuple(x for x in range(8) if x not in even_assign)
            # Try all permutations of even_assign for the 4 even Q3 vertices
            for e_perm in _permutations(even_assign):
                # Try all permutations of odd_assign for the 4 odd Q3 vertices
                for o_perm in _permutations(odd_assign):
                    # Build mapping: q3 vertex index -> subset vertex
                    mapping = [0] * 8
                    for qi, si in zip(even_q3, e_perm):
                        mapping[qi] = subset[si]
                    for qi, si in zip(odd_q3, o_perm):
                        mapping[qi] = subset[si]
                    
                    # Check if all Q3 edges are valid (both endpoints in subset)
                    # They always are since we're mapping to subset vertices
                    # The edges in our graph that would form this Q3:
                    edge_set = frozenset()
                    for i, j in q3_edges:
                        u = mapping[i]
                        v = mapping[j]
                        if u > v:
                            u, v = v, u
                        edge_set.add((u, v))
                    
                    # This is a valid Q3 candidate
                    # But we need to check if this specific labeling gives distinct edge sets
                    # Actually, different labelings may give the same set of 12 edges
                    # We want unique edge sets
                    if edge_set not in q3_edge_sets:
                        q3_edge_sets.append(edge_set)
                    found = True
                    break
                if found:
                    break
            if found:
                break
        # Note: for any 8-subset, there always exists SOME labeling that makes it a Q3
        # (since Q3 is the unique 3-regular bipartite graph on 8 vertices with girth 4... 
        #  actually no, we're not requiring the induced subgraph to BE Q3, just that Q3 is a subgraph)
        # Wait, I'm overcomplicating. We just need all possible sets of 12 edges that form a Q3.
        # The above does that, but it's expensive. Let me precompute more carefully.
    
    # Actually, let me just precompute all Q3 edge sets more efficiently.
    # A Q3 is determined by choosing 8 vertices and a specific bipartite structure.
    # Let me just enumerate all possible Q3s by their edge sets.
    
    # Reset and do it properly
    q3_edge_sets = set()
    
    # Q3 has a specific structure. Let's enumerate by choosing the 8 vertices
    # and then the bipartition (which vertices are in which part)
    # For a given 8-subset, the Q3 subgraph must have a specific edge pattern.
    
    # Simpler approach: just enumerate all 12-edge subsets that form a Q3
    # But that's C(120,12) which is too large.
    
    # Better: for each 8-subset, the number of Q3 subgraphs within it is limited.
    # Let me just do the enumeration but cache results.
    
    # Actually, for the purpose of this problem, let me just precompute all Q3s
    # by iterating over all ways to pick 8 vertices and check if they can form a Q3.
    
    # Let me use a different approach: precompute all Q3s as sets of edges.
    # There are at most C(16,8) * (number of Q3 structures on 8 labeled vertices) Q3s.
    # The number of non-isomorphic Q3 structures on 8 labeled vertices is 1 (Q3 is unique).
    # But the number of labelings is 8! / |Aut(Q3)| = 40320 / 48 = 840.
    # So for each 8-subset, there are at most 840 distinct Q3 edge sets.
    # Total: 12870 * 840 = ~10.8 million. That's a lot to precompute.
    
    # Let me just compute them on the fly but cache.
    
    # For now, let me use a simpler approach: precompute for each pair of vertices,
    # which Q3s they belong to. But that's also complex.
    
    # Let me just implement the local search with the creates_q3 check, but use
    # precomputed rest masks to speed it up.
    
    # Precompute: for each edge (u,v), precompute the "rest mask" that helps check
    # if adding (u,v) creates a Q3.
    
    # Actually, let me just use the current creates_q3_fast approach but optimize it.
    
    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        adj[i] is a bitmask of neighbors of i.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        # We need to find a2, a3, a4, a5, a6, a7 such that:
        # u-a2-a4-v forms a path, u-a3-a5-v forms a path
        # a2-a4, a3-a5 are edges
        # a2-a6, a4-a6 are edges
        # a3-a7, a5-a7 are edges
        # a6-a7 is an edge
        # All 8 vertices distinct
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def count_edges(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def greedy_add(adj, edge_order):
        added = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def get_missing_edges(adj):
        return [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]

    def get_existing_edges(adj):
        res = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    res.append((u, v))
                mask -= bit
        return res

    def local_search(adj, max_iters=500):
        """
        Local search with swap moves: remove one edge, add one or more edges.
        """
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try to add a missing edge
            missing = get_missing_edges(adj)
            if not missing:
                break
            
            found_add = False
            for _ in range(10):
                if not missing:
                    break
                idx = random.randint(0, len(missing) - 1)
                u, v = missing[idx]
                if not creates_q3_fast(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    missing.pop(idx)
                    current_count += 1
                    found_add = True
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                    break
            
            if found_add:
                continue
                
            # Perform a swap: remove a random edge, then try to add edges
            existing = get_existing_edges(adj)
            if not existing:
                break
            
            u_rem, v_rem = random.choice(existing)
            
            trial = adj[:]
            trial[u_rem] &= ~(1 << v_rem)
            trial[v_rem] &= ~(1 << u_rem)
            
            # Try to add edges to trial
            missing_trial = get_missing_edges(trial)
            random.shuffle(missing_trial)
            greedy_add(trial, missing_trial)
            
            new_count = count_edges(trial)
            if new_count > best_count:
                best_count = new_count
                best_adj = trial[:]
                adj = trial[:]
                current_count = new_count
            elif new_count >= current_count - 1:
                # Accept with some probability
                if random.random() < 0.15:
                    adj = trial[:]
                    current_count = new_count

        return best_adj, best_count

    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    # Run multiple restarts with local search
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        adj = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        greedy_add(adj, order)
        
        adj, cnt = local_search(adj)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj

    # Final polish
    if time.time() < deadline - 2:
        remaining = get_missing_edges(best_adj)
        random.shuffle(remaining)
        greedy_add(best_adj, remaining)
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)

def _permutations(iterable):
    """Generate all permutations of an iterable."""
    pool = list(iterable)
    n = len(pool)
    if n == 0:
        yield ()
        return
    indices = list(range(n))
    while True:
        yield tuple(pool[i] for i in indices)
        # Find next permutation
        i = n - 2
        while i >= 0 and indices[i] > indices[i+1]:
            i -= 1
        if i < 0:
            return
        j = n - 1
        while indices[j] < indices[i]:
            j -= 1
        indices[i], indices[j] = indices[j], indices[i]
        indices[i+1:] = indices[i+1:][::-1]
