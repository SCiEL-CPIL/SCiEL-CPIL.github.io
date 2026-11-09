import random
import time
import sys

def construct():
    start = time.time()
    deadline = 145  # Safety margin before 150s kill

    def can_add_edge(n, adj, u, v):
        """
        Checks if adding edge (u, v) maintains C4-freeness.
        Condition: For all pairs (a, b), |N(a) & N(b)| <= 1.
        Adding (u, v) affects pairs involving u or v.
        """
        # 1. Check u and v themselves
        # New common neighbors of u and v: N(u) & N(v) (unchanged, as u not in N(u) etc)
        # Wait, N_new(u) = N_old(u) U {v}. N_new(v) = N_old(v) U {u}.
        # Intersection: (N_old(u) U {v}) & (N_old(v) U {u})
        # = (N_old(u) & N_old(v)) U (N_old(u) & {u}) U ({v} & N_old(v)) U ({v} & {u})
        # = N_old(u) & N_old(v)  (since u not in N_old(u), v not in N_old(v))
        # So common neighbors of u and v remain the same.
        if len(adj[u] & adj[v]) > 1:
            return False
        # Actually, if it was > 1 before, graph was already invalid. 
        # But we assume current graph is C4-free, so it must be <= 1.
        
        # 2. Check pairs (u, w) where w in N(v)
        # N_new(u) = N_old(u) U {v}
        # N_new(w) = N_old(w)
        # Intersection: (N_old(u) U {v}) & N_old(w) = (N_old(u) & N_old(w)) U ({v} & N_old(w))
        # If v in N_old(w), then v is added to the intersection.
        # So new size = old_size + 1 (if v in N(w) and v not in N_old(u)&N_old(w))
        # We know v not in N_old(u) (edge not present).
        # So if v in N(w), new size = |N_old(u) & N_old(w)| + 1.
        # We need new size <= 1, so |N_old(u) & N_old(w)| must be 0.
        
        for w in adj[v]:
            if w == u: continue
            # Check if N(u) and N(w) have any common neighbors
            if adj[u] & adj[w]:
                return False

        # 3. Check pairs (v, z) where z in N(u)
        # Symmetric to above.
        for z in adj[u]:
            if z == v: continue
            if adj[v] & adj[z]:
                return False

        return True

    def is_c4_free(n, adj):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if len(adj[u] & adj[v]) > 1:
                        return False
        return True

    def min_degree(n, adj):
        return min(len(adj[i]) for i in range(n))

    def to_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    best = {'n': 39, 'edges': []}

    # Try descending n from 47 to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        found = False
        attempts = 0
        max_attempts = 2000  # Greedy is fast, so we can do many attempts

        while attempts < max_attempts and time.time() < deadline and not found:
            attempts += 1
            
            adj = [set() for _ in range(n)]
            
            # Generate random order of edges
            edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(edges_to_try)
            
            # Greedy addition
            for u, v in edges_to_try:
                if time.time() > deadline:
                    break
                if can_add_edge(n, adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
            
            if time.time() > deadline:
                break

            # Check if min degree is sufficient
            target_deg = n - 39
            if min_degree(n, adj) >= target_deg:
                # Verify C4-free just in case
                if is_c4_free(n, adj):
                    best = {'n': n, 'edges': to_edges(adj, n)}
                    found = True
                    break
            
            # If min degree is too low, try local repair
            # This is a fallback if greedy didn't reach the degree
            # We can try to add edges to low-degree vertices specifically
            if min_degree(n, adj) < target_deg:
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    # Sort vertices by degree, lowest first
                    vertices = sorted(range(n), key=lambda x: len(adj[x]))
                    for u in vertices:
                        if len(adj[u]) >= target_deg:
                            continue
                        # Try to add edges to u
                        candidates = [v for v in range(n) if v != u and v not in adj[u]]
                        random.shuffle(candidates)
                        for v in candidates:
                            if time.time() > deadline:
                                break
                            if can_add_edge(n, adj, u, v):
                                adj[u].add(v)
                                adj[v].add(u)
                                improved = True
                                if len(adj[u]) >= target_deg:
                                    break
                        if time.time() > deadline:
                            break
                    
                    if min_degree(n, adj) >= target_deg and is_c4_free(n, adj):
                        best = {'n': n, 'edges': to_edges(adj, n)}
                        found = True
                        break

        if found:
            break

    return best

# EVOLVE-BLOCK-END
