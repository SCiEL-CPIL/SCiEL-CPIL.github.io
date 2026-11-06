import random
import itertools
import time

def is_k4_free(edges, n=18):
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Check for K4: every set of 4 vertices must not be a clique
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        if b in adj[a] and c in adj[a] and d in adj[a] and \
           c in adj[b] and d in adj[b] and \
           d in adj[c]:
            return False
    return True

def random_k4_free_graph(n=18, p=0.5, seed=None):
    if seed is not None:
        random.seed(seed)
    
    # Generate random graph
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < p:
                edges.append([u, v])
    
    # Remove edges to ensure K4-free
    # We'll iteratively find K4s and remove edges
    changed = True
    while changed:
        changed = False
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find a K4
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            if b in adj[a] and c in adj[a] and d in adj[a] and \
               c in adj[b] and d in adj[b] and \
               d in adj[c]:
                # Remove a random edge from this K4
                # The edges in the K4 are (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                # Pick one at random
                candidate_edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                # Filter to those actually in edges
                valid = []
                for e in candidate_edges:
                    if [e[0], e[1]] in edges or [e[1], e[0]] in edges:
                        valid.append(e)
                if valid:
                    e = random.choice(valid)
                    # Remove edge
                    if [e[0], e[1]] in edges:
                        edges.remove([e[0], e[1]])
                    else:
                        edges.remove([e[1], e[0]])
                    changed = True
                    break
        if changed:
            continue
    
    return edges

def max_triangle_free_size(edges, n=18):
    # Build adjacency bitsets
    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    
    # Check for triangle-free subsets of size k, starting from n down to 1
    for k in range(n, 0, -1):
        for subset in itertools.combinations(range(n), k):
            subset_mask = 0
            for v in subset:
                subset_mask |= (1 << v)
            
            # Check if induced subgraph on subset is triangle-free
            is_tf = True
            for v in subset:
                # Neighbors of v within subset
                neighbors_in_subset = adj_bits[v] & subset_mask
                # Check if neighbors_in_subset contains any edge
                # For each neighbor u in neighbors_in_subset, check if adj_bits[u] & neighbors_in_subset is non-zero
                # But we can do: if there exists u in neighbors_in_subset such that adj_bits[u] & neighbors_in_subset != 0
                # Actually, we can check if the subgraph induced by neighbors_in_subset has any edge.
                # This is equivalent to: for each u in neighbors_in_subset, if adj_bits[u] & neighbors_in_subset != 0, then there's a triangle.
                # But we can optimize: if (adj_bits[u] & neighbors_in_subset) != 0 for any u in neighbors_in_subset, then not triangle-free.
                # However, note that u is in neighbors_in_subset, so adj_bits[u] & neighbors_in_subset includes u's neighbors in the subset.
                # We need to check if there is any edge among the neighbors.
                # A faster way: let N = neighbors_in_subset. The subgraph on N is triangle-free iff for every u in N, (adj_bits[u] & N) == 0? No, that's not right.
                # Actually, we just need to check if there is any edge in N.
                # We can do: for each u in N, if adj_bits[u] & N != 0, then there is an edge incident to u in N. But this might double count.
                # Simpler: iterate over u in N, and for each, check if any neighbor of u is also in N (other than itself).
                # But since N is a set, we can do:
                # for u in N:
                #     if adj_bits[u] & N:
                #         # There is a neighbor of u in N. But we need to ensure it's not just u itself, but since adj_bits[u] doesn't include u, it's fine.
                #         # However, this would be true if u has any neighbor in N. But we want to know if there is any edge in N.
                #         # Actually, if u has a neighbor in N, then that edge exists. So if for any u in N, adj_bits[u] & N != 0, then there is an edge.
                #         # But wait, if u has a neighbor v in N, then the edge (u,v) exists. So yes, if any u in N has a neighbor in N, then there is an edge.
                #         # However, this is always true if |N| >= 2 and there is at least one edge. But if N is an independent set, then for all u in N, adj_bits[u] & N == 0.
                #         # So we can check: if any u in N has adj_bits[u] & N != 0, then N is not independent.
                #         # But we are checking if the induced subgraph on N has any edge. So if N is not independent, then there is a triangle (v, u, w) where u,w in N and v is the center.
                #         # So yes, if N is not independent, then the subset is not triangle-free.
                #         # Therefore, we can break early if we find any u in N with adj_bits[u] & N != 0.
                #         # But note: we are iterating v in subset, and for each v, we look at N = adj_bits[v] & subset_mask.
                #         # If N is not independent, then there is a triangle.
                #         # So we can do:
                #             for u in N:
                #                 if adj_bits[u] & N:
                #                     is_tf = False
                #                     break
                #             if not is_tf:
                #                 break
                #         # But this is O(|N|) per v, which is fine.
                #         # However, we can optimize by precomputing or using bit operations.
                #         # For now, let's do the simple check.
                #         pass
                # Let's do the simple check
                # Convert N to list of vertices
                # But we can use bit operations:
                # We want to know if the subgraph induced by N has any edge.
                # This is equivalent to: for each u in N, if adj_bits[u] & N != 0, then there is an edge.
                # But we can do: 
                #   temp = N
                #   while temp:
                #       u = (temp & -temp).bit_length() - 1  # get lowest set bit
                #       temp &= temp - 1
                #       if adj_bits[u] & N:
                #           is_tf = False
                #           break
                #   if not is_tf:
                #       break
                # This is efficient.
                temp = neighbors_in_subset
                while temp:
                    u = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    if adj_bits[u] & neighbors_in_subset:
                        is_tf = False
                        break
                if not is_tf:
                    break
            if is_tf:
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds
    best_edges = None
    best_score = 18  # worst case
    
    # Generate multiple random graphs
    for seed in range(1000):
        if time.time() > deadline - 10:  # leave 10 seconds margin
            break
        
        edges = random_k4_free_graph(18, 0.5, seed)
        
        # Verify K4-free (should be, but just in case)
        if not is_k4_free(edges, 18):
            continue
        
        score = max_triangle_free_size(edges, 18)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            
            # If we reach the theoretical lower bound (which is likely 8 or 9), we can stop early
            if best_score <= 8:
                break
    
    if best_edges is None:
        # Fallback to the original
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        best_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    return best_edges
