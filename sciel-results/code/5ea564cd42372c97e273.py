import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    rng = random.Random(42)
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def bit_count(x):
        return x.bit_count()
        
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) to a valid graph adj keeps it valid.
        Assumes (u,v) is not currently an edge.
        Complexity: O(n^2) but with small constant.
        """
        # Check triples containing u: (u, y, z)
        # Count increases if v is in N(y) and N(z).
        # Need old_count(u, y, z) <= 1.
        for y in range(n):
            if y == u or y == v:
                continue
            ny = adj[y]
            # Check if v is neighbor of y
            if (ny >> v) & 1:
                for z in range(y + 1, n):
                    if z == u or z == v:
                        continue
                    nz = adj[z]
                    # Check if v is neighbor of z
                    if (nz >> v) & 1:
                        # v is common neighbor of y and z.
                        # Adding (u,v) makes v a common neighbor of u,y,z?
                        # No.
                        # Common neighbors of (u,y,z) are N(u) & N(y) & N(z).
                        # N(u) gains v.
                        # So new intersection = (N(u) & N(y) & N(z)) | ({v} & N(y) & N(z)).
                        # Since v is in N(y) and N(z), the second term is {v}.
                        # So count increases by 1.
                        # We need new count < 3 => old count <= 1.
                        if bit_count(adj[u] & ny & nz) > 1:
                            return False
            
            # Check if u is neighbor of y
            if (ny >> u) & 1:
                for z in range(y + 1, n):
                    if z == u or z == v:
                        continue
                    nz = adj[z]
                    # Check if u is neighbor of z
                    if (nz >> u) & 1:
                        # u is common neighbor of y and z.
                        # Adding (u,v) makes u a common neighbor of v,y,z?
                        # Common neighbors of (v,y,z) are N(v) & N(y) & N(z).
                        # N(v) gains u.
                        # New intersection = (N(v) & N(y) & N(z)) | ({u} & N(y) & N(z)).
                        # Since u is in N(y) and N(z), second term is {u}.
                        # Count increases by 1.
                        # Need old count <= 1.
                        if bit_count(adj[v] & ny & nz) > 1:
                            return False
        
        return True

    def count_edges(adj):
        return sum(bit_count(adj[i]) for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def is_graph_valid(adj):
        """
        Full check for K3,3 freedom.
        Used to verify the initial cleanup.
        """
        for i in range(n):
            for j in range(i + 1, n):
                if not ((adj[i] >> j) & 1):
                    continue
                # Check triples containing i, j, k
                # Common neighbors of i, j, k
                common_ij = adj[i] & adj[j]
                for k in range(j + 1, n):
                    if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1):
                        continue
                    # i, j, k form a triangle.
                    # Check if they have >= 3 common neighbors.
                    common_ijk = common_ij & adj[k]
                    # Exclude i, j, k themselves from the count (though they are not neighbors of each other in the common set usually, 
                    # wait, in a simple graph, a vertex is not its own neighbor. 
                    # But i is in N(j) and N(k) if triangle. 
                    # The definition of common neighbors of {i,j,k} is N(i) ∩ N(j) ∩ N(k).
                    # Does this set include i, j, or k?
                    # i ∈ N(i)? No. So i is not in the intersection.
                    # So we just count bits.
                    if bit_count(common_ijk) >= 3:
                        return False
        return True

    best_edges = 0
    best_adj = None
    
    num_starts = 10
    per_start_time = 10.0
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        # 1. Random Dense Start
        adj = [0] * n
        p = 0.6 + 0.05 * (start_idx % 3) # Vary density
        for (u, v) in all_edges:
            if rng.random() < p:
                add_edge(adj, u, v)
        
        # 2. Cleanup: Remove edges until valid
        # We remove an edge if it is part of a violation.
        # To find a violation, we check all triples.
        # If a violation is found, we remove one of the edges connecting the triple to the common neighbors.
        # Actually, simpler: just iterate edges in random order and remove if removal makes it "more valid"?
        # No, standard approach: Find a K3,3 subgraph, remove an edge from it.
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Find a violation
            violation_found = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((adj[i] >> j) & 1):
                        continue
                    common_ij = adj[i] & adj[j]
                    for k in range(j + 1, n):
                        if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1):
                            continue
                        common_ijk = common_ij & adj[k]
                        if bit_count(common_ijk) >= 3:
                            # Found K3,3: {i,j,k} and 3 common neighbors
                            # Remove a random edge from the K3,3
                            # Edges in K3,3 are between {i,j,k} and common_ijk
                            # Pick one such edge
                            # Get list of common neighbors
                            common_list = []
                            for v in range(n):
                                if (common_ijk >> v) & 1:
                                    common_list.append(v)
                            if len(common_list) >= 3:
                                # Pick 3 common neighbors
                                c1, c2, c3 = common_list[0], common_list[1], common_list[2]
                                # Pick a random edge from the bipartite part
                                # e.g., (i, c1)
                                u, v = rng.choice([(i, c1), (j, c1), (k, c1), (i, c2), (j, c2), (k, c2)])
                                remove_edge(adj, u, v)
                                changed = True
                                violation_found = True
                                break
                    if violation_found:
                        break
                if violation_found:
                    break
            if not violation_found:
                break
        
        if time.time() > deadline - 5:
            break
            
        # Verify validity (should be valid if cleanup worked)
        if not is_graph_valid(adj):
            continue
            
        cur_edges = count_edges(adj)
        if cur_edges > best_edges:
            best_edges = cur_edges
            best_adj = [a for a in adj]
            
        # 3. Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Add-only
            rng.shuffle(all_edges)
            for (u, v) in all_edges:
                if (adj[u] >> v) & 1:
                    continue
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    cur_edges += 1
                    improved = True
            
            # Phase 2: Swaps (1-for-1)
            if time.time() > deadline - 5:
                break
            
            # Try all pairs of (remove_edge, add_edge)
            # To speed up, we can just iterate.
            # E ~ 100. E^2 ~ 10,000.
            # is_valid_add is fast.
            
            for (u1, v1) in all_edges:
                if time.time() > deadline - 5:
                    break
                if not ((adj[u1] >> v1) & 1):
                    continue
                
                remove_edge(adj, u1, v1)
                
                for (u2, v2) in all_edges:
                    if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                        continue
                    if (adj[u2] >> v2) & 1:
                        continue
                    
                    if is_valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        improved = True
                        # cur_edges remains same
                        break
                else:
                    # No valid add found, revert
                    add_edge(adj, u1, v1)
                    continue
                
                # If we broke out of inner loop, we made a swap. 
                # Continue to next remove edge? 
                # Usually better to restart the add phase after a swap to see if more adds are possible.
                # But for now, just mark improved and continue searching for more swaps in this pass.
                # Actually, breaking out of the outer loop after a swap is safer to avoid complex state management.
                # Let's just break out of outer loop to restart the Add phase.
                break
            
            if cur_edges > best_edges:
                best_edges = cur_edges
                best_adj = [a for a in adj]
                
            if not improved:
                # Try 2-swaps?
                # If no 1-swap works, try 2-swaps.
                # This is expensive. Only do a few random attempts.
                two_swap_improved = False
                for _ in range(50): # Limit attempts
                    if time.time() > deadline - 5:
                        break
                    # Pick 2 edges to remove
                    e1 = rng.choice(all_edges)
                    if not ((adj[e1[0]] >> e1[1]) & 1):
                        continue
                    e2 = rng.choice(all_edges)
                    if (e2[0], e2[1]) == (e1[0], e1[1]) or (e2[0], e2[1]) == (e1[1], e1[0]):
                        continue
                    if not ((adj[e2[0]] >> e2[1]) & 1):
                        continue
                    
                    remove_edge(adj, e1[0], e1[1])
                    remove_edge(adj, e2[0], e2[1])
                    
                    # Pick 2 edges to add
                    f1 = rng.choice(all_edges)
                    if (adj[f1[0]] >> f1[1]) & 1:
                        # Revert
                        add_edge(adj, e1[0], e1[1])
                        add_edge(adj, e2[0], e2[1])
                        continue
                    f2 = rng.choice(all_edges)
                    if (f2[0], f2[1]) == (f1[0], f1[1]) or (f2[0], f2[1]) == (f1[1], f1[0]):
                        add_edge(adj, e1[0], e1[1])
                        add_edge(adj, e2[0], e2[1])
                        continue
                    if (adj[f2[0]] >> f2[1]) & 1:
                        add_edge(adj, e1[0], e1[1])
                        add_edge(adj, e2[0], e2[1])
                        continue
                    # Also ensure f1, f2 are not the ones we just removed (already checked by existence)
                    # And f1 != e1, f1 != e2, etc.
                    if (f1[0], f1[1]) in [(e1[0], e1[1]), (e2[0], e2[1]), (e1[1], e1[0]), (e2[1], e2[0])]:
                         add_edge(adj, e1[0], e1[1])
                         add_edge(adj, e2[0], e2[1])
                         continue
                    if (f2[0], f2[1]) in [(e1[0], e1[1]), (e2[0], e2[1]), (e1[1], e1[0]), (e2[1], e2[0])]:
                         add_edge(adj, e1[0], e1[1])
                         add_edge(adj, e2[0], e2[1])
                         continue

                    # Try adding f1
                    if is_valid_add(adj, f1[0], f1[1]):
                        add_edge(adj, f1[0], f1[1])
                        # Try adding f2
                        if is_valid_add(adj, f2[0], f2[1]):
                            add_edge(adj, f2[0], f2[1])
                            two_swap_improved = True
                            improved = True
                            # cur_edges remains same
                            break
                        else:
                            remove_edge(adj, f1[0], f1[1])
                    
                    # Revert removals
                    add_edge(adj, e1[0], e1[1])
                    add_edge(adj, e2[0], e2[1])
                    
                if cur_edges > best_edges:
                    best_edges = cur_edges
                    best_adj = [a for a in adj]
                    
                if not two_swap_improved and not improved:
                    break

    if best_adj is None:
        # Fallback: return empty or a known small graph
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
