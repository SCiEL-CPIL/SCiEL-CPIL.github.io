"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    all_possible = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    all_possible_set = set(all_possible)

    # Precompute all Q3 subgraphs on 16 vertices as edge-bitmasks
    # Q3: 8 vertices, 12 edges, 3-regular bipartite (4+4)
    # Generate all ways to embed Q3 into 16 vertices
    # Q3 has 8! = 40320 labelings, but many are isomorphic
    # Better: enumerate all 8-subsets and check which induce Q3-like structure
    
    # Actually, let's precompute all Q3 edge sets as frozensets of edges
    # For each 8-subset of {0..15}, check if it can host a Q3
    # Q3 structure: vertices 0-7 with edges forming the 3-cube
    
    q3_vertex_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }
    
    # Generate all distinct Q3 edge patterns on 8 abstract vertices
    # Then map to all 8-subsets of 16 vertices
    # Key insight: two embeddings are the same Q3 if they have the same edge set
    
    q3_edge_sets = set()
    # Generate all bijections from Q3 vertices to 8-subset
    # But that's too many. Instead, generate all 8-subsets and for each,
    # try all 8! permutations to find valid Q3 embeddings
    # This is 12870 * 40320 = too many
    
    # Better approach: for each 8-subset, check if the induced subgraph
    # on those vertices in a candidate graph contains Q3
    # Use the fact that Q3 is uniquely determined by its bipartition
    
    # Fast check: for a given graph (as adjacency bitmask), check if it contains Q3
    # Strategy: for each pair of 4-sets (A, B) partitioning an 8-set,
    # check if the bipartite graph between A and B is exactly Q3
    
    # Precompute all 8-subsets
    all_8sets = list(combinations(range(16), 8))
    
    # For each 8-set, precompute all ways to split into two 4-sets
    # and the required edge pattern for Q3
    # Q3 bipartition: even parity vs odd parity vertices
    # In standard Q3: {0,3,5,6} and {1,2,4,7} (or similar)
    
    # Let's just precompute all Q3 edge sets explicitly
    # For each 8-subset S, and each permutation p of S,
    # check if p maps Q3 edges to a valid subgraph
    # Store unique edge sets
    
    # Actually, the fastest approach: precompute all Q3 edge sets as 120-bit masks
    # (120 = C(16,2) possible edges)
    # Then checking if graph contains Q3 = check if any Q3 mask is subset of graph mask
    
    # Number of distinct Q3 subgraphs on 16 vertices:
    # Choose 8 vertices: C(16,8) = 12870
    # For each, number of Q3 embeddings / automorphisms
    # Q3 has 48 automorphisms (hyperoctahedral group)
    # 8! / 48 = 840 distinct embeddings per 8-set
    # Total: 12870 * 840 = ~10.8 million - too many to precompute all
    
    # Alternative: don't precompute all, but use fast checking
    # For a given graph, check if it contains Q3 by:
    # 1. For each 8-set, check if induced subgraph contains Q3
    # 2. Use bipartite structure: Q3 is 3-regular bipartite 4+4
    
    # Even faster: precompute for each edge, which Q3s it belongs to
    # Then when adding edge (u,v), only check Q3s containing (u,v)
    
    # Let me use a different approach: precompute all Q3 edge sets
    # but use a hash-based lookup
    
    # Generate Q3 edge sets for all 8-subsets
    # For each 8-subset, generate all 8! permutations is too slow
    # Instead, for each 8-subset, try all 70 ways to choose 4 vertices for one side
    # and check if the bipartite graph is 3-regular
    
    q3_edge_bitmasks = set()
    
    def get_edge_bitmask(edge_set):
        """Convert set of edges to a bitmask over C(16,2) positions."""
        # Precompute edge-to-index mapping
        mask = 0
        for (u, v) in edge_set:
            if u > v:
                u, v = v, u
            idx = edge_index[(u, v)]
            mask |= (1 << idx)
        return mask
    
    # Build edge index mapping
    edge_index = {}
    idx = 0
    for (u, v) in all_possible:
        edge_index[(u, v)] = idx
        idx += 1
    num_edges = 120
    
    # For each 8-set, find all Q3 embeddings
    for vs in all_8sets:
        vs_set = set(vs)
        # Try all ways to partition into two 4-sets
        # The 4-set determines the bipartition
        four_sets = list(combinations(vs, 4))
        for A in four_sets:
            B = tuple(v for v in vs if v not in A)
            # Check if bipartite graph between A and B is 3-regular
            # Each vertex in A must connect to exactly 3 in B, and vice versa
            # This means the edge set between A and B has exactly 12 edges
            # and each vertex has degree 3
            
            # For a valid Q3, the specific pattern matters
            # Q3 is the unique 3-regular bipartite graph on 4+4 vertices
            # Wait, no - there are multiple 3-regular bipartite graphs on 4+4
            # Q3 is one specific one
            
            # Let me just check: is the bipartite graph isomorphic to Q3?
            # Q3 adjacency: each vertex connects to 3 specific others
            # In Q3, the bipartite graph between {0,3,5,6} and {1,2,4,7}:
            # 0-1, 0-2, 0-4
            # 3-1, 3-2, 3-7
            # 5-1, 5-4, 5-7
            # 6-2, 6-4, 6-7
            
            # Actually, let me just generate all 3-regular bipartite graphs on 4+4
            # and check which is Q3
            
            # Simpler: just check if the edge set forms a Q3
            # by checking the cycle structure
            # Q3 has 6 squares (4-cycles)
            
            # For now, let's use a simpler check:
            # A 3-regular bipartite graph on 4+4 is Q3 iff it's connected
            # and has girth 4 (no triangles, which is automatic for bipartite)
            # Actually all 3-regular bipartite 4+4 graphs are isomorphic to Q3?
            # No, there are multiple. Let me just enumerate.
            
            pass
    
    # This is getting complex. Let me use a direct approach:
    # Precompute all Q3 edge sets by generating them explicitly
    
    # Q3 on vertices 0-7:
    q3_base_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    # Generate all distinct embeddings into 16 vertices
    # Use the fact that Q3 automorphism group has 48 elements
    # So for each 8-set, there are 8!/48 = 840 distinct embeddings
    
    q3_masks = set()
    
    # To speed up, only generate a subset or use smart enumeration
    # For now, generate all and see if it's fast enough
    
    from itertools import permutations
    
    # This will be slow: 12870 * 840 = 10.8M permutations to check
    # Let's limit or use a different strategy
    
    # Alternative: don't precompute all Q3 masks. Instead, use fast checking.
    # For a given graph, to check if it contains Q3:
    # 1. For each 8-set, check if induced subgraph contains Q3
    # 2. Use backtracking with pruning
    
    # Let me implement a fast contains_q3 check
    
    def contains_q3(adj_bits):
        """Check if graph with adjacency bitmasks contains a Q3 subgraph."""
        # For each 8-set, check if it induces a Q3
        # Use the bipartite structure for pruning
        
        for vs in all_8sets:
            # Quick degree check: each vertex must have degree >= 3 within vs
            local_deg = [0] * 8
            for i in range(8):
                u = vs[i]
                mask = adj_bits[u]
                for j in range(8):
                    if i != j and (mask >> vs[j]) & 1:
                        local_deg[i] += 1
                if local_deg[i] < 3:
                    break
            else:
                # All degrees >= 3, now check if Q3 is present
                # Use backtracking
                if _check_q3_in_8set(vs, adj_bits):
                    return True
        return False
    
    def _check_q3_in_8set(vs, adj_bits):
        """Check if the induced subgraph on vs contains a Q3."""
        # Build local adjacency
        local_adj = [0] * 8
        for i in range(8):
            u = vs[i]
            mask = adj_bits[u]
            for j in range(8):
                if i != j and (mask >> vs[j]) & 1:
                    local_adj[i] |= (1 << j)
        
        # Backtracking: assign Q3 vertices to vs vertices
        assignment = [-1] * 8
        used = [False] * 8
        
        def backtrack(q3_v):
            if q3_v == 8:
                return True
            # Try each unused vs vertex
            for vs_i in range(8):
                if used[vs_i]:
                    continue
                # Check compatibility with already assigned Q3 neighbors
                ok = True
                for nb in q3_vertex_adj[q3_v]:
                    if nb < q3_v:
                        if assignment[nb] == -1:
                            ok = False
                            break
                        if not ((local_adj[vs_i] >> assignment[nb]) & 1):
                            ok = False
                            break
                if ok:
                    assignment[q3_v] = vs_i
                    used[vs_i] = True
                    if backtrack(q3_v + 1):
                        return True
                    used[vs_i] = False
                    assignment[q3_v] = -1
            return False
        
        return backtrack(0)
    
    # Even faster: precompute which 8-sets contain a given edge
    pair_to_8sets = {}
    for vs in all_8sets:
        for i in range(8):
            for j in range(i+1, 8):
                pair = (vs[i], vs[j])
                if pair not in pair_to_8sets:
                    pair_to_8sets[pair] = []
                pair_to_8sets[pair].append(vs)
    
    def check_edge_creates_q3(u, v, adj_bits):
        """Check if adding edge (u,v) creates a Q3."""
        if u > v:
            u, v = v, u
        relevant_8sets = pair_to_8sets.get((u, v), [])
        for vs in relevant_8sets:
            if _check_q3_in_8set(vs, adj_bits):
                return True
        return False
    
    def edges_to_adj_bits(edges):
        adj = [0] * 16
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    # Main search: greedy with multiple restarts
    best_edges = set()
    best_count = 0
    
    random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Start with a random sparse graph
        shuffled = all_possible[:]
        random.shuffle(shuffled)
        
        # Start with a small random set
        initial_size = random.randint(5, 20)
        initial_edges = set(shuffled[:initial_size])
        
        # Verify no Q3
        adj = edges_to_adj_bits(initial_edges)
        valid = True
        for vs in all_8sets:
            if _check_q3_in_8set(vs, adj):
                valid = False
                break
        
        if not valid:
            # Remove edges until valid
            for e in list(initial_edges):
                if time.time() > deadline - 10:
                    break
                initial_edges.discard(e)
                adj = edges_to_adj_bits(initial_edges)
                has_q3 = False
                for vs in all_8sets:
                    if _check_q3_in_8set(vs, adj):
                        has_q3 = True
                        break
                if not has_q3:
                    break
        
        if not valid:
            continue
        
        local_adj = edges_to_adj_bits(initial_edges)
        local_edges = set(initial_edges)
        
        # Greedy add edges
        missing = [e for e in all_possible if e not in local_edges]
        random.shuffle(missing)
        
        for edge in missing:
            if time.time() > deadline - 8:
                break
            u, v = edge
            new_adj = local_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            if not check_edge_creates_q3(u, v, new_adj):
                local_adj = new_adj
                local_edges.add(edge)
        
        if len(local_edges) > best_count:
            best_count = len(local_edges)
            best_edges = set(local_edges)
        
        # Local search: try to add more edges
        if time.time() > deadline - 5:
            break
        
        missing = [e for e in all_possible if e not in local_edges]
        random.shuffle(missing)
        
        improved = False
        for edge in missing:
            if time.time() > deadline - 5:
                break
            u, v = edge
            new_adj = local_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
            if not check_edge_creates_q3(u, v, new_adj):
                local_adj = new_adj
                local_edges.add(edge)
                improved = True
                if len(local_edges) > best_count:
                    best_count = len(local_edges)
                    best_edges = set(local_edges)
        
        if not improved:
            # Try removing and re-adding
            if time.time() > deadline - 3:
                break
            
            # Remove a random edge and try to add others
            if local_edges:
                to_remove = random.choice(list(local_edges))
                local_edges.discard(to_remove)
                local_adj = edges_to_adj_bits(local_edges)
                
                missing = [e for e in all_possible if e not in local_edges]
                random.shuffle(missing)
                
                for edge in missing:
                    if time.time() > deadline - 3:
                        break
                    u, v = edge
                    new_adj = local_adj[:]
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                    
                    if not check_edge_creates_q3(u, v, new_adj):
                        local_adj = new_adj
                        local_edges.add(edge)
                        if len(local_edges) > best_count:
                            best_count = len(local_edges)
                            best_edges = set(local_edges)
    
    result = [[u, v] for (u, v) in best_edges]
    return result
# EVOLVE-BLOCK-END
