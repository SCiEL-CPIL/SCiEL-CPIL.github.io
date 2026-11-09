import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Precompute all 48 (partition, permutation) Q3 edge patterns
    # For each pattern, we store:
    # - a_mask: 4-bit mask indicating which of the 4 A-vertices are "even" in the cube
    # - For each of the 4 A-vertices, a 4-bit mask of which B-vertices it must connect to
    # - For each of the 4 B-vertices, a 4-bit mask of which A-vertices it must connect to
    
    # Actually, let's precompute for each of the 70 ways to split 8 positions into 4+4,
    # and for each of 24 permutations, the required edges as a set of (a_idx, b_idx) pairs.
    # But we'll store them as bitmasks for fast checking.
    
    # Better approach: precompute for each of the 48 unique "cube edge patterns" 
    # (accounting for the fact that swapping A and B gives the same structure)
    # the required edges in terms of position indices.
    
    # Let's precompute all valid (A_positions, B_positions, perm) combinations
    # and for each, store the 12 required (a_pos, b_pos) pairs.
    
    idx4_of_8 = list(itertools.combinations(range(8), 4))
    perms24 = list(itertools.permutations(range(4)))
    
    # For each combination of 4 positions for A and each permutation of B,
    # compute the required edges
    q3_patterns = []  # List of (a_positions, b_positions, required_edges)
    for a_pos in idx4_of_8:
        b_pos = tuple(p for p in range(8) if p not in a_pos)
        for perm in perms24:
            # A[0] (even=000) connects to B positions: perm[0], perm[1], perm[3]
            # A[1] (even=011) connects to B positions: perm[0], perm[1], perm[2]
            # A[2] (even=101) connects to B positions: perm[0], perm[2], perm[3]
            # A[3] (even=110) connects to B positions: perm[1], perm[2], perm[3]
            required = [
                (0, perm[0]), (0, perm[1]), (0, perm[3]),
                (1, perm[0]), (1, perm[1]), (1, perm[2]),
                (2, perm[0]), (2, perm[2]), (2, perm[3]),
                (3, perm[1]), (3, perm[2]), (3, perm[3])
            ]
            q3_patterns.append((a_pos, b_pos, required))
    
    # For fast checking, precompute for each pattern:
    # - a_mask: bitmask of the 4 A vertex positions (in the 8-set)
    # - For each A vertex i (0-3), a 4-bit mask of which B positions it connects to
    # - For each B vertex j (0-3), a 4-bit mask of which A positions it connects to
    pattern_data = []
    for a_pos, b_pos, required in q3_patterns:
        a_mask = 0
        for i in range(4):
            a_mask |= (1 << a_pos[i])
        
        # Build adjacency masks for A and B
        a_adj = [0, 0, 0, 0]  # a_adj[i] = bitmask of B positions that A[i] connects to
        b_adj = [0, 0, 0, 0]  # b_adj[j] = bitmask of A positions that B[j] connects to
        for (ai, bi) in required:
            a_adj[ai] |= (1 << bi)
            b_adj[bi] |= (1 << ai)
        
        pattern_data.append((a_mask, a_adj, b_adj))
    
    # Helper function to check if a specific 8-vertex set contains Q3
    def contains_q3_in_set(v8, adj_mask):
        """Check if the induced subgraph on v8 contains Q3 as a subgraph."""
        # Fast filter: count edges in the induced subgraph
        edge_count = 0
        for i in range(8):
            for j in range(i+1, 8):
                if (adj_mask[v8[i]] >> v8[j]) & 1:
                    edge_count += 1
        if edge_count < 12:
            return False
        
        # Try all 48 patterns
        for a_mask, a_adj, b_adj in pattern_data:
            # Check if the A positions in this pattern are a subset of the 8 vertices
            # We need to map pattern positions to actual vertices
            # a_mask tells us which of the 8 positions are in A
            # b_mask = ~a_mask & 0xFF tells us which are in B
            
            # Extract A vertices and B vertices from v8 based on a_mask
            A = []
            B = []
            for pos in range(8):
                if (a_mask >> pos) & 1:
                    A.append(v8[pos])
                else:
                    B.append(v8[pos])
            
            # Now check if the bipartite graph G[A,B] contains the required edges
            # For each A vertex, check if it connects to all required B vertices
            ok = True
            for i in range(4):
                m = adj_mask[A[i]]
                for j in range(4):
                    if (a_adj[i] >> j) & 1:
                        if not (m >> B[j]) & 1:
                            ok = False
                            break
                if not ok:
                    break
            
            if ok:
                return True
        
        return False

    # Initial graph: K_{3,13}
    U = {0, 1, 2}
    V = set(range(3, 16))
    edges = set()
    for u in U:
        for v in V:
            edges.add((u, v))
    
    # Build adjacency masks
    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    all_pairs = set(itertools.combinations(range(16), 2))
    
    def degree(v):
        return bin(adj_mask[v]).count('1')
    
    def try_add_edge(u, v):
        """Try to add edge (u,v). Returns True if safe."""
        if (u, v) in edges:
            return True
        if (v, u) in edges:
            return True
        
        # Temporarily add
        edges.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        # Check all 8-sets containing u and v
        others = [x for x in range(16) if x != u and x != v]
        for combo6 in itertools.combinations(others, 6):
            v8 = (u, v) + tuple(combo6)
            if contains_q3_in_set(v8, adj_mask):
                # Remove and return False
                edges.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                return False
        
        return True

    def try_remove_edge(u, v):
        """Remove edge (u,v)."""
        edges.remove((u, v))
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    # Local search: try edge swaps to escape local optima
    def local_search(edges_set, adj_mask_local, max_iters=100):
        """Try to improve by swapping edges."""
        best_count = len(edges_set)
        best_edges = set(edges_set)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adding a non-edge
            non_edges = [e for e in all_pairs if e not in edges_set]
            random.shuffle(non_edges)
            
            for e in non_edges:
                if time.time() > deadline - 5:
                    break
                u, v = e
                if try_add_edge(u, v):
                    improved = True
                    if len(edges_set) > best_count:
                        best_count = len(edges_set)
                        best_edges = set(edges_set)
                    break
            
            if not improved:
                # Try removing an edge and adding two others
                edge_list = list(edges_set)
                random.shuffle(edge_list)
                
                for e1 in edge_list:
                    if time.time() > deadline - 5:
                        break
                    
                    u1, v1 = e1
                    # Remove e1
                    edges_set.remove(e1)
                    adj_mask_local[u1] &= ~(1 << v1)
                    adj_mask_local[v1] &= ~(1 << u1)
                    
                    # Try to add two new edges
                    added = []
                    non_edges = [e for e in all_pairs if e not in edges_set]
                    random.shuffle(non_edges)
                    
                    for e2 in non_edges:
                        if len(added) == 2:
                            break
                        if time.time() > deadline - 5:
                            break
                        u2, v2 = e2
                        if try_add_edge(u2, v2):
                            added.append(e2)
                    
                    # If we added 2 edges, we improved
                    if len(added) == 2:
                        improved = True
                        if len(edges_set) > best_count:
                            best_count = len(edges_set)
                            best_edges = set(edges_set)
                    else:
                        # Revert: re-add e1, remove added edges
                        for e in added:
                            try_remove_edge(e[0], e[1])
                        edges_set.add(e1)
                        adj_mask_local[u1] |= (1 << v1)
                        adj_mask_local[v1] |= (1 << u1)
                    
                    if improved:
                        break
        
        return best_edges

    # Multiple restarts with different initial structures
    best_edges = set(edges)
    best_count = len(edges)
    
    # Restart 1: Current greedy
    for pass_num in range(10):
        if time.time() > deadline - 10:
            break
        
        non_edges = [e for e in all_pairs if e not in edges]
        
        # Sort by combined degree
        non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]), reverse=True)
        
        changed = False
        for e in non_edges:
            if time.time() > deadline - 10:
                break
            u, v = e
            if try_add_edge(u, v):
                changed = True
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = set(edges)
        
        if not changed:
            break
    
    # Restart 2: Local search from best
    if time.time() < deadline - 15:
        adj_mask_local = [0] * 16
        for (u, v) in best_edges:
            adj_mask_local[u] |= (1 << v)
            adj_mask_local[v] |= (1 << u)
        
        improved_edges = local_search(set(best_edges), adj_mask_local, max_iters=50)
        if len(improved_edges) > best_count:
            best_count = len(improved_edges)
            best_edges = improved_edges
    
    # Restart 3: Different initial structure - K_{4,12}
    if time.time() < deadline - 20:
        edges2 = set()
        U2 = {0, 1, 2, 3}
        V2 = set(range(4, 16))
        for u in U2:
            for v in V2:
                edges2.add((u, v))
        
        adj_mask2 = [0] * 16
        for (u, v) in edges2:
            adj_mask2[u] |= (1 << v)
            adj_mask2[v] |= (1 << u)
        
        # Greedy on edges2
        for pass_num in range(5):
            if time.time() > deadline - 15:
                break
            
            non_edges = [e for e in all_pairs if e not in edges2]
            non_edges.sort(key=lambda e: bin(adj_mask2[e[0]]).count('1') + bin(adj_mask2[e[1]]).count('1'), reverse=True)
            
            changed = False
            for e in non_edges:
                if time.time() > deadline - 15:
                    break
                u, v = e
                if (u, v) in edges2 or (v, u) in edges2:
                    continue
                
                # Temporarily add
                edges2.add((u, v))
                adj_mask2[u] |= (1 << v)
                adj_mask2[v] |= (1 << u)
                
                # Check
                others = [x for x in range(16) if x != u and x != v]
                safe = True
                for combo6 in itertools.combinations(others, 6):
                    v8 = (u, v) + tuple(combo6)
                    if contains_q3_in_set(v8, adj_mask2):
                        safe = False
                        break
                
                if safe:
                    changed = True
                    if len(edges2) > best_count:
                        best_count = len(edges2)
                        best_edges = set(edges2)
                else:
                    edges2.remove((u, v))
                    adj_mask2[u] &= ~(1 << v)
                    adj_mask2[v] &= ~(1 << u)
            
            if not changed:
                break
    
    # Final polish: try to add any remaining edges
    if time.time() < deadline - 5:
        non_edges = [e for e in all_pairs if e not in best_edges]
        random.shuffle(non_edges)
        
        for e in non_edges:
            if time.time() > deadline - 5:
                break
            u, v = e
            if (u, v) in best_edges or (v, u) in best_edges:
                continue
            
            # Temporarily add
            best_edges.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            others = [x for x in range(16) if x != u and x != v]
            safe = True
            for combo6 in itertools.combinations(others, 6):
                v8 = (u, v) + tuple(combo6)
                if contains_q3_in_set(v8, adj_mask):
                    safe = False
                    break
            
            if safe:
                best_count = len(best_edges)
            else:
                best_edges.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
    
    return [list(e) for e in best_edges]
