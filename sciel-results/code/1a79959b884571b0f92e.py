"""C5-free graph with no independent 9-set. Maximize n."""
import time
import itertools
from collections import defaultdict

_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def _check_c5_bitmask(adj_masks, n):
    """Check C5 using bitmask adjacency. adj_masks[i] is a bitmask of neighbors of i."""
    # C5: v0-v1-v2-v3-v4-v0
    # For each ordered pair (v0, v1) with edge, find v2 in N(v1)\{v0},
    # v3 in N(v2)\{v0,v1}, v4 in N(v3)\{v0,v1,v2} ∩ N(v0)
    for v0 in range(n):
        mask0 = adj_masks[v0]
        for v1 in range(n):
            if v1 == v0 or not (mask0 >> v1) & 1:
                continue
            mask1 = adj_masks[v1]
            # v2 in N(v1) \ {v0}
            candidates_v2 = mask1 & ~(1 << v0)
            while candidates_v2:
                v2 = (candidates_v2 & -candidates_v2).bit_length() - 1
                candidates_v2 &= candidates_v2 - 1
                mask2 = adj_masks[v2]
                # v3 in N(v2) \ {v0, v1}
                candidates_v3 = mask2 & ~(1 << v0) & ~(1 << v1)
                while candidates_v3:
                    v3 = (candidates_v3 & -candidates_v3).bit_length() - 1
                    candidates_v3 &= candidates_v3 - 1
                    mask3 = adj_masks[v3]
                    # v4 in N(v3) \ {v0, v1, v2} ∩ N(v0)
                    candidates_v4 = mask3 & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & mask0
                    if candidates_v4:
                        return True
    return False

def _has_c5(adj, n):
    """Check if graph has a 5-cycle. adj is list of sets."""
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m
    return _check_c5_bitmask(masks, n)

def _has_independent_set_9(adj, n):
    """Check if graph has an independent set of size 9."""
    # Independent set in G = clique in complement
    # Use branch and bound on complement
    comp = [0] * n
    for i in range(n):
        m = 0
        for j in range(n):
            if j != i and j not in adj[i]:
                m |= (1 << j)
        comp[i] = m
    
    best = 0
    found = [False]
    
    def bronk(R_size, P_mask, X_mask):
        nonlocal best, found
        if found[0]:
            return
        if P_mask == 0 and X_mask == 0:
            best = max(best, R_size)
            if best >= 9:
                found[0] = True
            return
        if P_mask == 0:
            return
        # Upper bound: R_size + popcount(P_mask)
        if R_size + bin(P_mask).count('1') <= best:
            return
        # Pivot
        pivot = (P_mask & -P_mask).bit_length() - 1
        candidates = P_mask & ~comp[pivot]
        while candidates:
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            candidates ^= v_bit
            if found[0]:
                return
            P2 = P_mask & comp[v]
            X2 = X_mask & comp[v]
            bronk(R_size + 1, P2, X2)
            P_mask ^= v_bit
            X_mask |= v_bit
        return
    
    bronk(0, (1 << n) - 1, 0)
    return best >= 9

def _is_valid(adj, n):
    """Check C5-free and alpha <= 8."""
    if _has_c5(adj, n):
        return False
    if _has_independent_set_9(adj, n):
        return False
    return True

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0
    
    # Start with 8 disjoint K4s: n=32, C5-free, alpha=8
    n = 32
    adj = [set() for _ in range(n)]
    for b in range(8):
        for i in range(4):
            for j in range(i+1, 4):
                u = 4*b + i
                v = 4*b + j
                adj[u].add(v)
                adj[v].add(u)
    
    # Verify base
    if not _is_valid(adj, n):
        # Fallback to 7 K4s
        n = 28
        adj = [set() for _ in range(n)]
        for b in range(7):
            for i in range(4):
                for j in range(i+1, 4):
                    u = 4*b + i
                    v = 4*b + j
                    adj[u].add(v)
                    adj[v].add(u)
        return {'n': n, 'edges': [[min(u,v), max(u,v)] for u in range(n) for v in sorted(adj[u])]}
    
    best_n = n
    best_adj = [s.copy() for s in adj]
    
    # Try to add vertices one by one
    # Key insight for C5-free with K4 blocks:
    # A new vertex v connected to set S creates a C5 iff there exist a,d in S
    # and a path of length 3 from a to d in G (through vertices not in {v}).
    # In the 8xK4 graph, paths of length 3 between vertices in the same K4 exist.
    # So S can contain at most 1 vertex from each K4 block to avoid C5 within a block.
    # For paths between different blocks: in 8xK4, there are no edges between blocks,
    # so no path of length >= 1 between different blocks. Thus no C5 can span blocks.
    # So: connect v to at most 1 vertex from each K4 block, and no C5 is created.
    
    # For alpha: v is non-adjacent to all vertices except those in S.
    # An independent set containing v can include v plus any independent set from
    # V\S. Since v is adjacent to exactly one vertex per selected block,
    # the remaining vertices in those blocks (3 per block) can form independent sets.
    # To keep alpha <= 8, we need: 1 + max independent set in V\S <= 8
    # i.e., max independent set in V\S <= 7.
    # In 8xK4, alpha = 8 (one from each K4). If S has one vertex from k blocks,
    # then in those k blocks, 3 vertices remain (alpha contribution up to 3 each? No,
    # in K4 minus one vertex = K3, alpha=1). In the other (8-k) blocks, alpha=1 each.
    # So max independent set in V\S = k*1 + (8-k)*1 = 8. So 1+8=9 > 8. Bad!
    # We need max independent set in V\S <= 7.
    # This means we need to remove enough vertices from independent sets.
    # If S has one vertex from each of all 8 blocks: V\S has 3 vertices per block (K3s).
    # Alpha of V\S = 8 (one from each K3). Still 1+8=9.
    # If S has 2 vertices from some blocks: those blocks have 2 remaining (edge, alpha=1).
    # Still alpha of V\S = 8.
    # Hmm, seems hard to get alpha <= 7 in V\S with 32 vertices.
    # 
    # Alternative: maybe we can modify the base graph. Instead of 8 K4s, use a different
    # C5-free graph with alpha <= 8 that has more "room" for an extra vertex.
    # 
    # Let's try: 7 K4s + 1 K3 = 28+3 = 31 vertices, alpha = 7+1 = 8.
    # Then add a vertex connected to 1 vertex from each K4 and maybe the K3.
    # Actually, let's just try the systematic search.
    
    # Systematic approach: try adding vertex v with neighbors = one vertex from each of k blocks
    # Check if valid. If yes, update.
    
    # Precompute: for the 8xK4 graph, what subsets S of one-vertex-per-block work?
    # C5 constraint: automatically satisfied (no edges between blocks).
    # Alpha constraint: need alpha(V\S) <= 7.
    # In 8xK4, if we remove one vertex from k blocks and zero from (8-k) blocks:
    #   k blocks have K3 (alpha 1), (8-k) blocks have K4 (alpha 1).
    #   Total alpha = k*1 + (8-k)*1 = 8. Always 8!
    # So 1 + 8 = 9 > 8. Never works with this structure.
    #
    # We need a different base. Let's try: modify some K4s to have fewer edges.
    # E.g., replace one K4 with a path P4 (4 vertices, 3 edges, alpha=2).
    # Then alpha = 7*1 + 2 = 9. Too high.
    # Replace one K4 with C4 (4-cycle, alpha=2). Same issue.
    # 
    # What about: 7 K4s + 2 K3s? n=28+6=34, alpha=7+2=9. Too high.
    # 7 K4s + 1 K3 + 1 K2? n=28+3+2=33, alpha=7+1+1=9. Too high.
    # 6 K4s + 2 K3s + 1 K2? n=24+6+2=32, alpha=6+2+1=9. Too high.
    # 
    # It seems alpha=8 requires exactly 8 components each with alpha=1, i.e., 8 cliques of size >=2.
    # For n=33 with alpha<=8, we need the "average" to work out differently.
    # 
    # Let's try a completely different approach: use a known Ramsey graph.
    # R(C5, K9) > 32 is known. Can we find a 33-vertex example?
    # 
    # Let me try: start with 8 K4s and try to add a vertex with specific connections,
    # then possibly remove some edges to fix alpha.
    
    # Actually, let me try a random search with the fast C5 check.
    import random
    
    for attempt in range(500):
        if time.time() > _DEADLINE - 30:
            break
        
        # Start from current best
        new_n = best_n + 1
        if new_n > 35:
            break
        new_adj = [s.copy() for s in best_adj]
        new_adj.append(set())
        
        random.seed(attempt * 7919 + 13)
        
        # Try different neighbor sets
        for edge_count in [1, 2, 3, 4, 5]:
            if time.time() > _DEADLINE - 30:
                break
            
            neighbors = set(random.sample(range(best_n), min(edge_count, best_n)))
            for nb in neighbors:
                new_adj[best_n].add(nb)
                new_adj[nb].add(best_n)
            
            if _has_c5(new_adj, new_n):
                continue
            
            if not _has_independent_set_9(new_adj, new_n):
                # Valid!
                best_n = new_n
                best_adj = new_adj
                break
    
    # Build edge list
    edges = []
    for u in range(best_n):
        for v in sorted(best_adj[u]):
            if u < v:
                edges.append([u, v])
    
    return {'n': best_n, 'edges': edges}

# EVOLVE-BLOCK-END
