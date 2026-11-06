import time
import itertools
import random
import math

def construct():
    deadline = time.time() + 135  # Safety margin for 150s limit
    n = 16
    
    # Precompute all disjoint pairs of 4-sets
    all_4sets = list(itertools.combinations(range(n), 4))
    num_4sets = len(all_4sets)
    
    # Represent 4-sets as bitmasks
    set_masks = [0] * num_4sets
    for i, s in enumerate(all_4sets):
        mask = 0
        for v in s:
            mask |= (1 << v)
        set_masks[i] = mask
            
    disjoint_pairs = []
    for i in range(num_4sets):
        for j in range(i+1, num_4sets):
            if (set_masks[i] & set_masks[j]) == 0:
                disjoint_pairs.append((i, j))
    
    num_disjoint = len(disjoint_pairs)
    
    # For fast checking, for each vertex v, list of 4-set indices containing v
    sets_containing = [[] for _ in range(n)]
    for i, s in enumerate(all_4sets):
        for v in s:
            sets_containing[v].append(i)
    
    # Precompute for each disjoint pair (i,j), the list of (u,v) edges between A and B
    # This allows fast checking
    pair_edges = []
    for i, j in disjoint_pairs:
        A = all_4sets[i]
        B = all_4sets[j]
        edges_between = []
        for u in A:
            for v in B:
                edges_between.append((u, v))
        pair_edges.append(edges_between)
    
    # For each edge (u,v), precompute which disjoint pairs it affects
    # edge_to_pairs[e] = list of pair indices where (u,v) is between A and B
    all_edge_set = set()
    for i, j in disjoint_pairs:
        A = all_4sets[i]
        B = all_4sets[j]
        for u in A:
            for v in B:
                all_edge_set.add((u, v))
                all_edge_set.add((v, u))
    
    edge_to_pairs = {}
    for pair_idx, (i, j) in enumerate(disjoint_pairs):
        A = all_4sets[i]
        B = all_4sets[j]
        for u in A:
            for v in B:
                key = (u, v)
                if key not in edge_to_pairs:
                    edge_to_pairs[key] = []
                edge_to_pairs[key].append(pair_idx)
    
    # Fast check: for a given pair, count non-edges in H between A and B
    # and check if they form a matching
    def check_pair_fast(adj, pair_idx):
        """
        Returns True if H[A,B] is NOT a matching (i.e., G is safe).
        Returns False if H[A,B] IS a matching (i.e., G contains Q3).
        """
        edges_between = pair_edges[pair_idx]
        
        non_edges = []
        for u, v in edges_between:
            if not (adj[u] & (1 << v)):
                non_edges.append((u, v))
                
        if len(non_edges) >= 5:
            return True  # Safe
            
        # Check if non_edges form a matching
        used = set()
        for u, v in non_edges:
            if u in used or v in used:
                return True  # Not a matching, so Safe
            used.add(u)
            used.add(v)
            
        # If we get here, they are all disjoint = matching
        return False  # Unsafe
    
    def is_q3_free(adj):
        for pair_idx in range(num_disjoint):
            if not check_pair_fast(adj, pair_idx):
                return False
        return True
    
    # Greedy initialization: start with K16 and remove edges to make Q3-free
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    # Remove edges to eliminate Q3 subgraphs
    # For each Q3 found, remove an edge
    removed_count = 0
    for _ in range(500):  # Max iterations
        # Find a Q3
        found = False
        for pair_idx in range(num_disjoint):
            if not check_pair_fast(adj, pair_idx):
                # Found a Q3, remove an edge from it
                edges_between = pair_edges[pair_idx]
                non_edges = []
                for u, v in edges_between:
                    if not (adj[u] & (1 << v)):
                        non_edges.append((u, v))
                
                # Remove one of the non-edges (add it to G)
                # Actually, we need to remove an edge from G that's part of the Q3
                # The Q3 consists of the non-edges in H, which are edges in G
                # Wait, I'm confused. Let me reconsider.
                
                # H is the complement of G.
                # A Q3 in H means there's a Q3 subgraph in H.
                # To make G Q3-free, we need H to be Q3-free.
                # So we need to remove edges from H (add to G) to break all Q3s.
                
                # But we want to MAXIMIZE edges in G, so we want to MINIMIZE edges in H.
                # Start with H = empty (G = K16), which has no Q3.
                # That's already optimal if K16 is Q3-free!
                
                # Wait, is K16 Q3-free? No, K16 definitely contains Q3.
                # So we need to remove edges from G (add to H) to break Q3s.
                
                # Let me reconsider the approach.
                # We want G to be Q3-free, maximizing |E(G)|.
                # Start with G = K16 (120 edges).
                # While G contains a Q3, remove an edge from G.
                
                # Find which edges of G are in a Q3
                # For this pair (A,B), the Q3 in H means H[A,B] is a matching.
                # The edges of G between A and B are the complement of H[A,B].
                # To break this Q3, we need to add at least one edge to H[A,B]
                # i.e., remove an edge from G between A and B.
                
                # Pick an edge from G between A and B to remove
                A = all_4sets[disjoint_pairs[pair_idx][0]]
                B = all_4sets[disjoint_pairs[pair_idx][1]]
                
                # Find an edge in G between A and B
                removed = False
                for u in A:
                    for v in B:
                        if adj[u] & (1 << v):
                            # Remove this edge from G
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            removed_count += 1
                            removed = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        if not found:
            break
    
    # Now G is Q3-free
    # Verify
    if not is_q3_free(adj):
        # Should not happen, but just in case
        # Try harder to remove edges
        for _ in range(1000):
            found = False
            for pair_idx in range(num_disjoint):
                if not check_pair_fast(adj, pair_idx):
                    A = all_4sets[disjoint_pairs[pair_idx][0]]
                    B = all_4sets[disjoint_pairs[pair_idx][1]]
                    for u in A:
                        for v in B:
                            if adj[u] & (1 << v):
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
            if not found:
                break
    
    best_adj = adj[:]
    best_count = sum(bin(a).count('1') for a in adj) // 2
    
    # Now try to add edges back using simulated annealing
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_affected_pairs(u, v):
        """Returns list of pair indices where (u,v) is between A and B."""
        if (u, v) in edge_to_pairs:
            return edge_to_pairs[(u, v)]
        if (v, u) in edge_to_pairs:
            return edge_to_pairs[(v, u)]
        return []
    
    def try_add_edge(adj, u, v):
        """Try adding edge (u,v) to G. Returns True if valid."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        affected = get_affected_pairs(u, v)
        for pair_idx in affected:
            if not check_pair_fast(adj, pair_idx):
                # Invalid, revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                return False
        return True
    
    def try_remove_edge(adj, u, v):
        """Remove edge (u,v) from G. Returns True if edge existed."""
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return True
        return False
    
    random.seed(42)
    
    # Simulated annealing
    temperature = 50.0
    cooling = 0.995
    min_temp = 0.1
    
    current_adj = best_adj[:]
    current_count = best_count
    
    while temperature > min_temp and time.time() < deadline:
        # Pick a random edge to flip
        u, v = random.choice(all_edges)
        
        if current_adj[u] & (1 << v):
            # Edge is present, try removing it
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
            new_count = current_count - 1
            
            # Always accept removal if it helps (but we want to maximize, so rarely remove)
            # Actually, removing an edge decreases our score, so only do it if it enables more additions
            # For now, accept with low probability
            if random.random() < math.exp(-10 / temperature):
                current_count = new_count
            else:
                # Revert
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
        else:
            # Edge is absent, try adding it
            if try_add_edge(current_adj, u, v):
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
            # If invalid, the edge was not added (try_add_edge reverts)
        
        temperature *= cooling
    
    # Final hill climbing: try to add any missing edge
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if not (best_adj[u] & (1 << v)):
                # Try adding to a copy
                test_adj = best_adj[:]
                if try_add_edge(test_adj, u, v):
                    best_adj = test_adj
                    best_count += 1
                    improved = True
    
    # Final verification
    if not is_q3_free(best_adj):
        # Should not happen, but fix if needed
        for pair_idx in range(num_disjoint):
            if not check_pair_fast(best_adj, pair_idx):
                A = all_4sets[disjoint_pairs[pair_idx][0]]
                B = all_4sets[disjoint_pairs[pair_idx][1]]
                for u in A:
                    for v in B:
                        if best_adj[u] & (1 << v):
                            best_adj[u] &= ~(1 << v)
                            best_adj[v] &= ~(1 << u)
                            best_count -= 1
                            break
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
