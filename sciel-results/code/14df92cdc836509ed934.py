"""Improved program: efficient C5 detection and alpha bound for finding larger valid graphs."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    def remaining():
        return deadline - time.time()

    # ---- Guaranteed valid fallback: 8 disjoint K4s -> n=32, alpha=8, no C5 ----
    def k4_blocks(nblocks):
        n = 4 * nblocks
        edges = []
        for b in range(nblocks):
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([4 * b + i, 4 * b + j])
        return {'n': n, 'edges': edges}

    fallback = k4_blocks(8)  # n = 32

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5_fast(adj, n):
        """Check for C5 using a smarter approach: for each pair of non-adjacent vertices,
        check if they share a common neighbor that can form a path of length 3 between them."""
        # A C5 exists iff there exist vertices a,b,c,d,e such that a-b-c-d-e-a
        # Equivalently: there's a vertex v, two neighbors x,y of v, and a path x-...-y of length 3
        # through vertices not including v.
        
        # Better: for each ordered pair (s, t) with s != t and not adjacent,
        # check if there's a path of length 4 from s to t (s-v1-v2-v3-t)
        # This is equivalent to checking if s and t have a common neighbor in the "square" sense.
        
        # Most efficient: iterate over all 5-cycles by fixing the start vertex
        # and doing bounded DFS with early termination.
        
        for s in range(n):
            sbit = 1 << s
            # DFS to find paths of length 4 from s back to s
            # Path: s -> v1 -> v2 -> v3 -> s (all distinct except s)
            stack = [(s, sbit, 0)]
            found = False
            while stack:
                v, used, depth = stack.pop()
                if depth == 3:
                    if adj[v] & sbit:
                        return True
                else:
                    cand = adj[v] & ~used
                    while cand:
                        lb = cand & (-cand)
                        w = lb.bit_length() - 1
                        stack.append((w, used | lb, depth + 1))
                        cand ^= lb
        return False

    def alpha_at_least_9_fast(adj, n):
        """Check if alpha(G) >= 9 by checking if complement has a K9.
        Use branch and bound with better pruning."""
        comp = [((1 << n) - 1) ^ (adj[i] | (1 << i)) for i in range(n)]
        
        # Sort vertices by degree in complement (descending) for better pruning
        order = list(range(n))
        order.sort(key=lambda i: -bin(comp[i]).count('1'))
        
        # Map to new order
        pos = {v: i for i, v in enumerate(order)}
        comp2 = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if comp[order[i]] & (1 << order[j]):
                    mask |= (1 << pos[order[j]])
            comp2[i] = mask
        
        t0 = time.time()
        
        def rec(cand, k):
            if k >= 9:
                return True
            if time.time() - t0 > 5.0:
                return None
            if bin(cand).count('1') + k < 9:
                return False
            
            # Greedy upper bound: iteratively pick vertex with max degree in cand
            cb = cand
            gc = 0
            while cb:
                # Pick vertex with max degree in current candidate set
                best_v = -1
                best_deg = -1
                cb2 = cb
                while cb2:
                    lb = cb2 & (-cb2)
                    v = lb.bit_length() - 1
                    deg = bin(cb & comp2[v]).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    cb2 ^= lb
                if best_v < 0:
                    break
                gc += 1
                cb &= comp2[best_v]
            
            if gc + k < 9:
                return False
            
            # Pick vertex with highest degree in cand
            lb = cand & (-cand)
            v = lb.bit_length() - 1
            r = rec(cand & comp2[v], k + 1)
            if r is True:
                return True
            if r is None:
                return None
            return rec(cand ^ lb, k)

        return rec((1 << n) - 1, 0)

    def is_valid(n, edges):
        adj = build_adj(n, edges)
        if has_c5_fast(adj, n):
            return False
        a9 = alpha_at_least_9_fast(adj, n)
        if a9 is True:
            return False
        if a9 is None:
            return False
        return True

    best = fallback  # always valid

    # ---- Search: add a 33rd vertex to 8xK4 ----
    # For n=33: add vertex 32 connected to one vertex in each K4 block.
    # The new vertex must not create a C5. Since K4 blocks are cliques and disconnected,
    # a C5 would need to go through the new vertex. The only way is:
    # new - v1 - ... - v5 - new where v1,v5 are in different blocks.
    # But since blocks are disconnected, any path between blocks goes through new.
    # So a C5 would be: new - a - (path in block) - b - new, but a and b are in same block
    # and adjacent, so this is a triangle, not C5.
    # Actually: new connects to one vertex per block. A C5 would need 5 vertices in a cycle.
    # If new connects to a in block A and b in block B, then for a C5 we'd need a path of
    # length 3 from a to b not using new. But blocks are disconnected, so no such path exists.
    # Therefore ANY choice of one vertex per block gives a valid C5-free graph!
    
    # Now check alpha: the new vertex is adjacent to one vertex in each block.
    # An independent set can contain new only if it doesn't contain any of its neighbors.
    # If new is in the independent set, we can pick at most 3 from each block (excluding the neighbor).
    # Max independent set with new: 1 + 8*3 = 25... wait that's too high.
    # Actually: from each K4, if we exclude one vertex (the neighbor of new), we can pick at most
    # 3 from the remaining 3 (which form a K3, so max independent set is 1).
    # So max independent set containing new = 1 + 8*1 = 9. That's exactly 9, which is NOT allowed.
    # So we need alpha < 9, meaning no independent set of size 9.
    
    # Without new: alpha = 8 (one from each K4).
    # With new: if new is in IS, we can pick at most 1 from each block (the non-neighbor vertices
    # in each block form a K3, so max IS in K3 is 1). Total = 1 + 8 = 9. BAD.
    # So we need to add MORE edges to reduce alpha.
    
    # Alternative: connect new to MULTIPLE vertices per block to reduce the independent set size.
    # If new connects to k vertices in a block, the remaining (4-k) vertices form a clique of size (4-k),
    # so max IS from that block is 1 if 4-k >= 1, else 0.
    # To get total < 9: 1 + sum(min(1, 4-k_i) for i in 8) < 9
    # If new connects to at least 1 in each block: 1 + 8*1 = 9. Still bad.
    # If new connects to at least 2 in each block: 1 + 8*1 = 9 (remaining 2 form K2, IS=1). Still bad.
    # If new connects to at least 3 in each block: 1 + 8*1 = 9 (remaining 1, IS=1). Still bad.
    # If new connects to all 4 in some block: that block contributes 0.
    # So we need: 1 + sum(contrib_i) < 9 where contrib_i = 1 if new connects to <4 in block i, else 0.
    # We need sum(contrib_i) <= 7, so at least one block must have new connected to all 4 vertices.
    
    # Let's try: new connected to all 4 in block 0, and 1 in each other block.
    # Then IS containing new: 1 + 0 + 7*1 = 8 < 9. 
    # IS not containing new: max 1 per block = 8 < 9. 
    # Check C5: new is connected to all of block 0 and one vertex in each other block.
    # A C5 through new: new - a - ... - b - new. a in block 0, b in block 0 (since only block 0 has multiple connections).
    # Path from a to b in block 0 of length 3: a - c - d - b. In K4, any two vertices are adjacent,
    # so a-b is an edge. For a path of length 3, we need a-c-d-b where c,d are other vertices.
    # In K4={v0,v1,v2,v3}, a=v0, b=v1: path v0-v2-v3-v1 exists. So new-v0-v2-v3-v1-new is a C5!
    # This is BAD.
    
    # So connecting to all 4 in a block creates C5s. We need a different strategy.
    
    # Let's try: new connected to 2 vertices in each block.
    # IS with new: 1 + 8*1 = 9. Still bad.
    
    # It seems hard to get n=33 with this construction. Let me try a different approach.
    
    # For n=33, use a different base graph. How about 7 K4s + 1 K5?
    # 7*4 + 5 = 33. K5 has alpha=1, K4 has alpha=1. Total alpha = 8 < 9. 
    # C5-free? K5 contains C5! So that doesn't work.
    
    # How about using K4s and adding cross edges carefully?
    # This is getting complex. Let me just try random graphs with efficient verification.
    
    # ---- Random search for n=33,34,35 ----
    for target in (33, 34, 35):
        if remaining() < 20.0:
            break
        
        t0 = time.time()
        limit = min(remaining() - 15.0, 15.0)
        found = None
        
        for attempt in range(5000):
            if time.time() - t0 > limit:
                break
            
            n = target
            # Generate a random graph with edge probability ~0.3
            # Then verify
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.3:
                        edges.append([i, j])
            
            if is_valid(n, edges):
                found = {'n': n, 'edges': edges}
                break
        
        if found is not None:
            best = found
            break

    if time.time() > deadline - 2.0:
        return fallback

    return best
# EVOLVE-BLOCK-END
