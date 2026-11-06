import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    vertices = list(range(n))
    all_edges = set((u, v) for u in range(16) for v in range(u+1, 16))
    all_edges_list = list(all_edges)
    
    # Precompute all Q3 embeddings as frozensets of 12 edges
    # Q3 vertices: {0,1,2,3,4,5,6,7} with edges based on binary representation
    # Vertex i has neighbors i^1, i^2, i^4 (xor with 1, 2, 4)
    q3_edges_template = set()
    for i in range(8):
        for bit in [1, 2, 4]:
            j = i ^ bit
            if i < j:
                q3_edges_template.add((i, j))
    # q3_edges_template has 12 edges
    
    # Precompute all Q3 embeddings on 16 vertices
    # Each embedding is a mapping from {0..7} to 8 distinct vertices of {0..15}
    # Two embeddings are equivalent under relabeling, but we need all distinct edge sets
    # Number of ways: C(16,8) * 8! / (symmetries of Q3) but we just enumerate all injective maps
    # That's 16!/(16-8)! = 16*15*...*9 which is too many (518,918,400)
    # Instead: for each 8-subset, enumerate all 8! bijections from {0..7} to the subset
    # But that's 40320 * 12870 = 519M, still too many
    
    # Better approach: precompute the 12 edge patterns in terms of positions
    # For a given 8-subset s = (s0,...,s7), an embedding maps q3 vertex i to s[i]
    # The edges are: for each i, edges (s[i], s[i^1]), (s[i], s[i^2]), (s[i], s[i^4])
    # We need to check if all 12 of these edges are present in the graph
    
    # Key insight: Q3 is a specific graph. Two embeddings give the same edge set iff
    # they differ by an automorphism of Q3. Q3 has 48 automorphes.
    # So for each 8-subset, there are 8!/48 = 840 distinct edge sets.
    # Total: C(16,8) * 840 = 12870 * 840 = 10,810,800 edge sets
    # That's manageable if we store them as bitmasks!
    
    # Represent each edge set as a 120-bit integer (edge index in all_edges_list)
    # Build edge index map
    edge_to_idx = {e: i for i, e in enumerate(all_edges_list)}
    
    # Precompute all Q3 edge sets as 120-bit integers
    # For speed, we'll precompute for each 8-subset the list of Q3 edge bitmasks
    # But 10.8M entries is a lot of memory. Let's use a different approach.
    
    # Alternative: for the Q3 check, iterate over 8-subsets and check if any
    # permutation of Q3 fits. Use the bipartite degree check which is faster.
    
    # Actually, let's use a smarter Q3 check:
    # Q3 is bipartite with parts of size 4. In any Q3 subgraph, the 8 vertices
    # form a bipartite graph with min degree 3 on both sides.
    # But not every such bipartite graph contains Q3 as a subgraph.
    # However, if the bipartite graph between X and Y (|X|=|Y|=4) has min degree >= 3,
    # it contains a 3-regular bipartite subgraph on 8 vertices, which is Q3 or a
    # union of cycles. A 3-regular bipartite graph on 8 vertices is either Q3 or
    # K_{3,3} plus... no, 3-regular bipartite on 8 vertices: each side has 4 vertices
    # each of degree 3. The possible graphs are: Q3 (which is the unique 3-regular
    # bipartite graph on 4+4 vertices that is connected). Actually there are multiple:
    # Q3 and the disjoint union of... no, 3-regular bipartite on 8 vertices must be
    # connected (since min degree 3 > 8/2 - 1). Actually it could be two K_{2,2} with
    # extra edges? No. Let me think: 4 vertices on each side, each degree 3.
    # Total edges = 4*3 = 12. The bipartite graph is 3-regular.
    # The only 3-regular bipartite graphs on 4+4 vertices are Q3 and... 
    # Actually, there are exactly 2: Q3 and the graph obtained by removing a perfect
    # matching from K_{4,4} and adding... no.
    # K_{4,4} has degree 4. Removing a perfect matching gives degree 3. That's Q3!
    # Because Q3 = K_{4,4} minus a perfect matching (the "opposite" pairs).
    # Wait, is that right? In Q3, vertex 0 is connected to 1,2,4 (not 3,5,6,7).
    # In K_{4,4} with parts {0,2,4,6} and {1,3,5,7}, vertex 0 connects to all of {1,3,5,7}.
    # In Q3, vertex 0 connects to {1,2,4} which are not all in one part.
    # So Q3 is NOT K_{4,4} minus a matching.
    
    # Let me just use the direct check: for each 8-subset and each partition into X,Y
    # of size 4, check if the bipartite graph has a 3-regular subgraph.
    # A bipartite graph has a 3-regular subgraph iff... this is hard.
    
    # Simplest correct approach: for each 8-subset s and each bijection from {0..7} to s,
    # check if all 12 Q3 edges are present. But 8! = 40320 per subset, times 12870 subsets
    # = 519M checks. Too slow.
    
    # Better: precompute the 12 edges of Q3 as pairs of positions (0..7).
    # For a given 8-subset s, the Q3 edges under identity mapping are:
    # (s[0],s[1]), (s[0],s[2]), (s[0],s[4]), (s[1],s[3]), (s[1],s[5]), (s[1],s[7]),
    # (s[2],s[3]), (s[2],s[6]), (s[2],s[7]), (s[4],s[5]), (s[4],s[6]), (s[5],s[7])
    # But we need to check ALL permutations.
    
    # Key optimization: Q3 has 48 automorphisms. So for each 8-subset, we only need
    # to check 8!/48 = 840 distinct edge sets.
    # Precompute these 840 edge sets for a canonical 8-subset, then apply permutations.
    
    # Actually, let's just precompute all Q3 edge bitmasks once.
    # Use numpy for speed.
    
    # Precompute Q3 edge bitmasks for all 8-subsets
    # This is expensive but done once.
    
    def time_check():
        return time.time() > deadline - 5
    
    # Build all Q3 edge sets as frozensets of edge tuples
    # For memory efficiency, store as 120-bit integers
    
    # Precompute edge index for fast lookup
    edge_idx = {}
    for i, (u, v) in enumerate(all_edges_list):
        edge_idx[(u, v)] = i
        edge_idx[(v, u)] = i  # normalize
    
    # Precompute Q3 edge sets
    # For each 8-subset, compute all distinct Q3 embeddings (mod automorphisms)
    # and store the 120-bit mask
    
    # This is the expensive part. Let's do it efficiently.
    
    # First, get the 48 automorphisms of Q3 as permutations of {0..7}
    # Q3 automorphism group is the hyperoctahedral group B3, order 48
    # Generated by: coordinate permutations (6) and bit flips (8)
    
    def generate_q3_autos():
        """Generate all 48 automorphisms of Q3 as tuples of 8 elements."""
        autos = set()
        for perm in itertools.permutations([0, 1, 2]):  # 6 coordinate perms
            for flips in itertools.product([0, 1], repeat=3):  # 8 bit flips
                def transform(x):
                    # Apply bit flips
                    y = x
                    for i in range(3):
                        if flips[i]:
                            y ^= (1 << i)
                    # Apply coordinate permutation
                    # perm[i] means new coordinate i = old coordinate perm[i]
                    # Actually, let's think of it as: new_bit_i = old_bit_{perm[i]} ^ flips[i]
                    new = 0
                    for i in range(3):
                        bit = (y >> perm[i]) & 1
                        bit ^= flips[i]
                        new |= (bit << i)
                    return new
                auto = tuple(transform(i) for i in range(8))
                autos.add(auto)
        return list(autos)
    
    q3_autos = generate_q3_autos()
    # Should have 48 elements
    
    # Q3 edges as position pairs
    q3_pos_edges = []
    for i in range(8):
        for bit in [0, 1, 2]:
            j = i ^ (1 << bit)
            if i < j:
                q3_pos_edges.append((i, j))
    # 12 edges
    
    # Precompute all Q3 edge masks
    # For each 8-subset, compute the set of distinct Q3 edge masks
    # Store as a list of 120-bit integers
    
    # For efficiency, precompute for the canonical subset (0,1,...,7)
    # Then for other subsets, apply the vertex permutation
    
    # Actually, let's just precompute everything. 12870 * 840 = 10.8M masks.
    # Each mask is a Python int (arbitrary precision), so ~28 bytes each = ~300MB.
    # That's too much. Let's use a different approach.
    
    # Alternative: don't precompute. Instead, for the Q3 check, iterate over
    # 8-subsets and use a fast bipartite check.
    
    # Fast Q3 check using bitmasks:
    # adj_mask[v] is a 16-bit int. For an 8-subset s, build a local adjacency matrix.
    
    def has_q3(adj_mask):
        """Check if graph contains Q3 as a subgraph. adj_mask is list of 16-bit ints."""
        for s in subsets_8:
            # Build local adjacency: 8x8 bitmask
            # local_adj[i] has bit j set if s[i] and s[j] are adjacent
            local_adj = [0] * 8
            for i in range(8):
                m = adj_mask[s[i]]
                for j in range(8):
                    if i != j and (m >> s[j]) & 1:
                        local_adj[i] |= (1 << j)
            
            # Check if this 8-vertex graph contains Q3
            # Q3 has 12 edges. Count edges first.
            edge_count = sum(local_adj[i] >> (j+1) for i in range(8) for j in range(i))
            # Actually: sum of bits in upper triangle
            edge_count = 0
            for i in range(8):
                edge_count += bin(local_adj[i] & ((1 << 8) - 1) & ~((1 << (i+1)) - 1)).count('1')
            if edge_count < 12:
                continue
            
            # Now check if Q3 is a subgraph
            # Try all 840 distinct embeddings (8!/48)
            # For speed, use the fact that Q3 is 3-regular
            # Each vertex in Q3 has degree 3, so in the subgraph, each of the 8
            # vertices must have degree >= 3 in the local graph.
            local_deg = [bin(local_adj[i]).count('1') for i in range(8)]
            if min(local_deg) < 3:
                continue
            
            # Now we need to check if there's a 3-regular bipartite subgraph
            # with the specific structure of Q3.
            # Q3 is the unique (up to isomorphism) 3-regular bipartite graph
            # on 8 vertices that is not a disjoint union of smaller components.
            # Actually, there are multiple 3-regular bipartite graphs on 8 vertices.
            # Let's just check all 840 embeddings.
            
            # For each permutation p of {0..7}, check if edges (s[p[i]], s[p[j]])
            # exist for all Q3 edges (i,j).
            # To avoid checking all 40320 permutations, use the 48 automorphisms:
            # Two permutations give the same edge set iff they differ by an automorphism.
            # So we fix p[0]=0 (by symmetry) and enumerate the rest.
            # Actually, let's just enumerate all 40320 and use early termination.
            
            # Optimize: check vertex degrees first. In Q3, each vertex has degree 3.
            # So we need a matching of Q3 vertices to local vertices such that
            # each local vertex used has degree >= 3 (already checked) and the
            # edges exist.
            
            # Use backtracking with pruning
            # Map q3_vertex -> local_index, or -1
            mapping = [-1] * 8
            q3_adj = [0] * 8
            for i, j in q3_pos_edges:
                q3_adj[i] |= (1 << j)
                q3_adj[j] |= (1 << i)
            
            def backtrack(k):
                if time_check():
                    return False
                if k == 8:
                    return True
                for li in range(8):
                    if mapping[li] != -1:
                        continue
                    # Check if assigning q3 vertex k to local vertex li is consistent
                    ok = True
                    for q in range(k):
                        if mapping[q] != -1 and (q3_adj[k] >> q) & 1:
                            # q3 vertices k and q are adjacent, so local vertices
                            # mapping[k] and mapping[q] must be adjacent
                            if not (local_adj[li] >> mapping[q]) & 1:
                                ok = False
                                break
                        elif (q3_adj[k] >> q) & 1:
                            # Not adjacent in Q3, no constraint
                            pass
                    if not ok:
                        continue
                    # Also check: if q3 vertex k is NOT adjacent to q, but local
                    # vertices are adjacent, that's fine (subgraph, not induced)
                    mapping[li] = k
                    if backtrack(k + 1):
                        return True
                    mapping[li] = -1
                return False
            
            if backtrack(0):
                return True
        return False
    
    subsets_8 = list(itertools.combinations(range(16), 8))
    
    def build_adj_mask(edge_set):
        adj_mask = [0] * 16
        for u, v in edge_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask
    
    # Try Shrikhande graph complement
    def shrikhande_adj(u, v):
        a, b = divmod(u, 4)
        c, d = divmod(v, 4)
        da = (a - c) % 4
        db = (b - d) % 4
        if da == 0 and (db == 1 or db == 3):
            return True
        if db == 0 and (da == 1 or da == 3):
            return True
        if da == 2 and db == 2:
            return True
        return False
    
    shrikhande_edges = set()
    for u in range(16):
        for v in range(u+1, 16):
            if shrikhande_adj(u, v):
                shrikhande_edges.add((u, v))
    
    complement_edges = all_edges - shrikhande_edges
    
    adj_mask = build_adj_mask(complement_edges)
    if not has_q3(adj_mask):
        best_edges = set(complement_edges)
        best_count = len(best_edges)
    else:
        # Try other constructions
        best_edges = None
        best_count = 0
        
        # Try random dense graphs and check
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            # Start with a random subset of edges
            random.shuffle(all_edges_list)
            # Try different densities
            for frac in [0.7, 0.65, 0.6, 0.55, 0.5]:
                num_edges = int(len(all_edges_list) * frac)
                candidate = set(all_edges_list[:num_edges])
                adj_m = build_adj_mask(candidate)
                if not has_q3(adj_m):
                    if len(candidate) > best_count:
                        best_count = len(candidate)
                        best_edges = set(candidate)
                    break  # Found a valid graph at this density, try lower
    
        if best_edges is None:
            # Fallback: K_{3,13} + triangle
            best_edges = set()
            for a in range(3):
                for b in range(3, 16):
                    best_edges.add((a, b))
            for u, v in [(0,1), (1,2), (0,2)]:
                best_edges.add((u, v))
            best_count = len(best_edges)
    
    # Local search: greedy addition + edge swaps
    candidate_edges = list(all_edges - best_edges)
    
    for restart in range(30):
        if time.time() > deadline - 20:
            break
        
        random.shuffle(candidate_edges)
        current_edges = set(best_edges)
        adj_mask = build_adj_mask(current_edges)
        
        # Phase 1: greedy addition
        for u, v in candidate_edges:
            if time.time() > deadline - 10:
                break
            if (u, v) in current_edges:
                continue
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            current_edges.add((u, v))
            if has_q3(adj_mask):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                current_edges.discard((u, v))
        
        # Phase 2: edge swaps (remove one, add two)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_list = list(current_edges)
            random.shuffle(current_list)
            for rem_u, rem_v in current_list:
                if time.time() > deadline - 10:
                    break
                # Try removing (rem_u, rem_v) and adding two new edges
                adj_mask[rem_u] &= ~(1 << rem_v)
                adj_mask[rem_v] &= ~(1 << rem_u)
                current_edges.discard((rem_u, rem_v))
                
                # Find two edges to add
                added = []
                for u2, v2 in candidate_edges:
                    if time.time() > deadline - 10:
                        break
                    if (u2, v2) in current_edges:
                        continue
                    if u2 == rem_u and v2 == rem_v:
                        continue
                    adj_mask[u2] |= (1 << v2)
                    adj_mask[v2] |= (1 << u2)
                    current_edges.add((u2, v2))
                    if has_q3(adj_mask):
                        adj_mask[u2] &= ~(1 << v2)
                        adj_mask[v2] &= ~(1 << u2)
                        current_edges.discard((u2, v2))
                    else:
                        added.append((u2, v2))
                        if len(added) == 2:
                            break
                
                if len(added) == 2:
                    improved = True
                else:
                    # Revert
                    for u2, v2 in added:
                        adj_mask[u2] &= ~(1 << v2)
                        adj_mask[v2] &= ~(1 << u2)
                        current_edges.discard((u2, v2))
                    adj_mask[rem_u] |= (1 << rem_v)
                    adj_mask[rem_v] |= (1 << rem_u)
                    current_edges.add((rem_u, rem_v))
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = set(current_edges)
            candidate_edges = list(all_edges - best_edges)
    
    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
