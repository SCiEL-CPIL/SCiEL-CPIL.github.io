# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    n = 31
    m = 91  # number of edges required
    q = 5
    
    # Build PG(2,5) points as 1D subspaces of GF(5)^3
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                inv = pow(vec[first_nonzero], q-2, q)
                vec = [(x * inv) % q for x in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    # There should be q^2 + q + 1 = 25 + 5 + 1 = 31 points
    if len(points) != 31:
        # Fallback
        random.seed(42)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        return [[i, j] for i, j in all_possible[:m]]
    
    # Polarity: a point p=(x,y,z) is perpendicular to q'=(x',y',z') if xx'+yy'+zz' = 0 mod 5
    # The polarity graph connects p and q if p is perpendicular to q (and p != q)
    
    def perp(p, qpt):
        return (p[0]*qpt[0] + p[1]*qpt[1] + p[2]*qpt[2]) % q == 0
    
    # Build adjacency for the full polarity graph using bitsets for efficiency
    adj = [0] * n  # bitset representation
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if perp(points[i], points[j]):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    
    # Count edges in polarity graph
    polarity_edges = len(edges)
    # Should be 93 (each vertex has degree 6, so 31*6/2 = 93)
    
    if polarity_edges < m:
        # Fallback: add random edges
        random.seed(42)
        existing = set(edges)
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for e in all_possible:
            if e not in existing:
                edges.append(e)
                if len(edges) == m:
                    break
        return [[u, v] for u, v in edges]
    
    # Helper to count C4s in the current graph using bitsets
    def count_c4(adj_bitsets):
        c4_count = 0
        for u in range(n):
            for v in range(u+1, n):
                # Find common neighbors using bitset AND
                common = adj_bitsets[u] & adj_bitsets[v]
                # Remove u and v from common if they are neighbors
                common &= ~(1 << u)
                common &= ~(1 << v)
                # For each pair of common neighbors (w, x), if w~x, then u-w-x-v-u is a C4
                # Count pairs of set bits in common that are adjacent
                # Extract set bits
                common_bits = []
                tmp = common
                while tmp:
                    lsb = tmp & (-tmp)
                    w = lsb.bit_length() - 1
                    common_bits.append(w)
                    tmp ^= lsb
                # For each pair, check if adjacent
                for i in range(len(common_bits)):
                    for j in range(i+1, len(common_bits)):
                        w = common_bits[i]
                        x = common_bits[j]
                        if (adj_bitsets[w] >> x) & 1:
                            c4_count += 1
        return c4_count
    
    # Helper to compute the number of C4s an edge (u,v) belongs to using bitsets
    def count_c4s_for_edge(adj_bitsets, u, v):
        count = 0
        # A C4 containing edge (u,v) is of the form u-w-x-v-u
        # w is a neighbor of u (w != v)
        # x is a neighbor of v (x != u)
        # w and x are distinct and adjacent
        neighbors_u = adj_bitsets[u] & ~(1 << v)
        neighbors_v = adj_bitsets[v] & ~(1 << u)
        
        # For each w in neighbors_u, for each x in neighbors_v where w != x and w~x
        tmp_w = neighbors_u
        while tmp_w:
            lsb_w = tmp_w & (-tmp_w)
            w = lsb_w.bit_length() - 1
            tmp_w ^= lsb_w
            
            # For each x in neighbors_v where x != w and w~x
            tmp_x = neighbors_v & ~(1 << w)
            while tmp_x:
                lsb_x = tmp_x & (-tmp_x)
                x = lsb_x.bit_length() - 1
                if (adj_bitsets[w] >> x) & 1:
                    count += 1
                tmp_x ^= lsb_x
        return count
    
    # Initial C4 count
    current_adj = adj[:]
    current_edges = set(edges)
    initial_c4 = count_c4(current_adj)
    
    # Strategy: perform edge swaps (remove one edge, add one non-edge) to minimize C4s
    # We need to go from 93 edges to 91 edges, so we can:
    # 1. Remove 2 edges (no additions)
    # 2. Remove 1 edge, add 1 edge, then remove 1 more edge
    # 3. Or do a sequence of swaps
    
    # First, try all single edge removals and find the best
    best_single_removal = None
    best_single_c4 = initial_c4
    
    for e in current_edges:
        u, v = e
        # Temporarily remove edge
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        c4 = count_c4(current_adj)
        # Restore
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        
        if c4 < best_single_c4:
            best_single_c4 = c4
            best_single_removal = e
    
    if best_single_removal is not None:
        u, v = best_single_removal
        current_adj[u] &= ~(1 << v)
        current_adj[v] &= ~(1 << u)
        current_edges.remove(best_single_removal)
    
    # Now try to improve with swaps: remove one edge, add one non-edge
    # But we still need to remove one more edge to get to 91
    # So: from current state (92 edges), try swap (remove e1, add e2) then remove e3
    # Or: from current state, just remove one more edge
    
    # Let's try: from the 92-edge graph, try all possible swaps (remove e, add non-edge f)
    # and see if we can get a better 92-edge graph, then remove one edge
    best_swap = None
    best_swap_c4 = count_c4(current_adj)
    
    # Get all non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    non_edges = [e for e in all_pairs if e not in current_edges]
    
    # For each edge to remove and each non-edge to add, compute the C4 change
    # This is expensive, so let's be smart: only consider promising swaps
    
    # Precompute C4 counts for each edge in current graph
    edge_c4_counts = {}
    for e in current_edges:
        u, v = e
        edge_c4_counts[e] = count_c4s_for_edge(current_adj, u, v)
    
    # For each non-edge, compute how many C4s it would create if added
    # A non-edge (u,v) creates C4s with pairs of common neighbors
    non_edge_c4_counts = {}
    for e in non_edges:
        u, v = e
        # Count C4s that would be created by adding edge (u,v)
        # This is the number of paths of length 2 from u to v where the middle vertices are adjacent
        common = current_adj[u] & current_adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        # For each pair of common neighbors (w,x) that are adjacent, adding (u,v) creates a C4 u-w-x-v-u
        # But wait, u-w-x-v-u is a C4 if w~x. We already have (u,w), (v,w), (u,x), (v,x).
        # Adding (u,v) creates the cycle u-v-x-w-u? No, that's not right.
        # Actually, adding edge (u,v) creates C4s of the form u-v-x-w-u where w~x and w,x in N(u)∩N(v)
        # Wait, no. A C4 containing the new edge (u,v) is u-v-x-y-u where x~y, x in N(v), y in N(u), x!=y
        # So: for each x in N(v)\{u}, for each y in N(u)\{v}, if x~y and x!=y, then u-v-x-y-u is a C4
        c4_created = 0
        neighbors_v = current_adj[v] & ~(1 << u)
        neighbors_u = current_adj[u] & ~(1 << v)
        
        tmp_x = neighbors_v
        while tmp_x:
            lsb_x = tmp_x & (-tmp_x)
            x = lsb_x.bit_length() - 1
            tmp_x ^= lsb_x
            
            tmp_y = neighbors_u & ~(1 << x)
            while tmp_y:
                lsb_y = tmp_y & (-tmp_y)
                y = lsb_y.bit_length() - 1
                if (current_adj[x] >> y) & 1:
                    c4_created += 1
                tmp_y ^= lsb_y
        
        non_edge_c4_counts[e] = c4_created
    
    # Now try swaps: remove edge e1, add non-edge e2
    # The change in C4 count is approximately: -c4s_containing_e1 + c4s_created_by_e2
    # But this is approximate because removing e1 might destroy some C4s that e2 would create
    # For now, use the approximation to find promising swaps
    
    # Sort edges by C4 count (highest first) and non-edges by C4 created (lowest first)
    sorted_edges = sorted(current_edges, key=lambda e: edge_c4_counts[e], reverse=True)
    sorted_non_edges = sorted(non_edges, key=lambda e: non_edge_c4_counts[e])
    
    # Try top K swaps
    K_edges = min(20, len(sorted_edges))
    K_non_edges = min(20, len(sorted_non_edges))
    
    best_swap_result = None
    best_swap_score = float('inf')
    
    for e1 in sorted_edges[:K_edges]:
        for e2 in sorted_non_edges[:K_non_edges]:
            # Check if e2 is not already an edge (should be true by construction)
            if e2 in current_edges:
                continue
            
            # Approximate score: -edge_c4_counts[e1] + non_edge_c4_counts[e2]
            score = -edge_c4_counts[e1] + non_edge_c4_counts[e2]
            
            if score < best_swap_score:
                best_swap_score = score
                best_swap_result = (e1, e2)
    
    # Apply the best swap if it's promising
    if best_swap_result is not None:
        e1, e2 = best_swap_result
        u1, v1 = e1
        u2, v2 = e2
        
        # Apply swap
        current_adj[u1] &= ~(1 << v1)
        current_adj[v1] &= ~(1 << u1)
        current_edges.remove(e1)
        
        current_adj[u2] |= (1 << v2)
        current_adj[v2] |= (1 << u2)
        current_edges.add(e2)
        
        # Now we have 92 edges, need to remove one more
        # Try all single removals and pick the best
        best_final_removal = None
        best_final_c4 = count_c4(current_adj)
        
        for e in current_edges:
            u, v = e
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            c4 = count_c4(current_adj)
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
            if c4 < best_final_c4:
                best_final_c4 = c4
                best_final_removal = e
        
        if best_final_removal is not None:
            u, v = best_final_removal
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            current_edges.remove(best_final_removal)
    
    # Verify we have exactly m edges
    if len(current_edges) != m:
        # Fallback: just use the polarity graph with 2 edges removed
        current_adj = adj[:]
        current_edges = set(edges)
        # Remove 2 edges greedily
        for _ in range(2):
            best_edge = None
            best_count = -1
            for e in current_edges:
                u, v = e
                count = count_c4s_for_edge(current_adj, u, v)
                if count > best_count:
                    best_count = count
                    best_edge = e
            if best_edge is not None:
                u, v = best_edge
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                current_edges.remove(best_edge)
    
    # Build final edge list
    final_edges = list(current_edges)
    
    # Verify all edges are distinct and valid
    for e in final_edges:
        u, v = e
        if not (0 <= u < n and 0 <= v < n and u != v):
            # Fallback
            random.seed(42)
            all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_possible)
            final_edges = all_possible[:m]
            break
    
    # Convert to list of lists
    result = [[u, v] for u, v in final_edges]
    return result

# EVOLVE-BLOCK-END
