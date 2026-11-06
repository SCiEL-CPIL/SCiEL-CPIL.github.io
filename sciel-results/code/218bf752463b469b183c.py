import time
import random

# The evaluator kills at 150s. We must return well before that.
# We aim for a valid graph on n=40 (which would be optimal if it exists, but R(3,10)<=41 implies n<=40).
# Known best is 39.
# Strategy:
# 1. Hardcode a known valid graph on n=39 (if we can find one or generate one fast) as a safe fallback.
# 2. Attempt to find a valid graph on n=40 using a structured construction (Clebsch product idea) or fast local search.
# 3. If n=40 fails, return the n=39 graph.

# Let's try to generate a valid graph on n=39 quickly to ensure we don't return the seed (n=26).
# We can use the "Andrasfai" style or random bipartite + repair.
# Actually, a random graph on 39 vertices with p~0.4 is likely triangle-free? No, p~0.4 has many triangles.
# Triangle-free max density is 1/4. So p~0.2-0.3 is safer for triangle-free.
# To keep alpha < 10, we need enough edges.
# For n=39, alpha <= 9.

def _is_triangle_free(adj, n):
    """Check if graph is triangle-free."""
    for i in range(n):
        # For each neighbor j of i, check if j has any neighbor in adj[i]
        # We can just check if adj[j] & adj[i] has any bits in common with adj[i]?
        # No, triangle is i-j-k-i.
        # For each i, for each j in N(i), check if N(j) intersects N(i).
        # Specifically, if there is k in N(j) such that k in N(i), then i-j-k-i is a triangle?
        # Wait, if k in N(i) and k in N(j), then i-k and j-k are edges.
        # So i, j, k form a triangle if i~j, j~k, k~i.
        # Since we check for i, we look at neighbors j of i.
        # If j has a neighbor k that is also a neighbor of i, then i-j-k-i is a triangle.
        # Note: k != i and k != j.
        # adj[i] contains i's neighbors.
        # adj[j] contains j's neighbors.
        # common = adj[i] & adj[j].
        # If common is not empty, does it imply a triangle?
        # If k is in common, then i~k and j~k. And we know i~j.
        # So yes, {i, j, k} is a triangle.
        
        ai = adj[i]
        # Iterate through neighbors of i
        j_mask = ai
        while j_mask:
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            if adj[j] & ai:
                return False
            j_mask ^= lsb
    return True

def _has_independent_set_10(adj, n):
    """
    Check if there is an independent set of size 10.
    Returns True if such a set exists.
    Uses branch and bound with bitsets.
    """
    if n < 10:
        return False
        
    # We want to find a clique of size 10 in the complement graph.
    # Complement adjacency: comp[i] = ~adj[i] & ((1<<n)-1) & ~(1<<i)
    full_mask = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full_mask ^ adj[i] ^ (1 << i)

    # Branch and bound for clique of size 10 in comp.
    # State: candidates (bitset), current_size
    # Pruning: if current_size + candidates.bit_count() < 10, return False.
    
    def dfs(candidates, size):
        if size >= 10:
            return True
        if candidates == 0:
            return False
        if size + candidates.bit_count() < 10:
            return False
            
        # Heuristic: pick vertex with max degree in candidate subgraph?
        # Or just pick lowest bit?
        # To speed up, ordering matters.
        # Let's pick the vertex with the highest degree in the induced subgraph on candidates.
        # This is expensive to calculate every time.
        # Simple heuristic: pick lowest bit.
        # Better: Greedy coloring bound?
        # For now, simple branch and bound with "include/exclude" logic on the lowest bit.
        
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        
        # Option 1: Include v
        # New candidates: candidates & comp[v]
        # We must remove v and any vertices not connected to v in comp (i.e., connected in original).
        # In comp, neighbors of v are comp[v].
        # So new candidates are candidates & comp[v].
        new_candidates = candidates & comp[v]
        if dfs(new_candidates, size + 1):
            return True
            
        # Option 2: Exclude v
        new_candidates = candidates & ~v_bit
        if dfs(new_candidates, size):
            return True
            
        return False

    return dfs(full_mask, 0)

def _is_valid(n, edges):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    if not _is_triangle_free(adj, n):
        return False
    if _has_independent_set_10(adj, n):
        return False
    return True

def _edges_to_adj(edges, n):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        mask = adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
    return edges

def _generate_valid_graph(n, deadline):
    """
    Attempt to find a valid graph on n vertices.
    Returns edge list or None.
    """
    if time.time() > deadline:
        return None

    # Strategy: Randomized local search starting from a random bipartite graph.
    # Bipartite graphs are triangle-free.
    # We want to minimize independence number.
    
    for restart in range(20): # Fewer restarts for larger n, more for smaller?
        if time.time() > deadline:
            return None
            
        # Split into two sets
        mid = n // 2
        # Random partition
        vertices = list(range(n))
        random.shuffle(vertices)
        A = set(vertices[:mid])
        B = set(vertices[mid:])
        
        # Start with random bipartite graph
        adj = [0] * n
        for u in A:
            for v in B:
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        # Local search: Add edges to break independent sets, while maintaining triangle-free.
        # Since it's bipartite, we can only add edges between A and B.
        # Adding an edge (u,v) with u in A, v in B creates a triangle if there exists w in A such that w~u and w~v?
        # No, triangle is u-v-w-u.
        # If u in A, v in B, w must be in A (since v only connects to A).
        # So triangle u-v-w-u exists if u~v, v~w, w~u.
        # u~v is the new edge.
        # So we need v~w and w~u.
        # w is in A.
        # So adding (u,v) creates a triangle if N_A(v) intersects N_A(u).
        # N_A(v) is the set of neighbors of v in A.
        # N_A(u) is the set of neighbors of u in A? No, u is in A, so N_A(u) is empty (no edges within A).
        # Wait.
        # Triangle u-v-w.
        # u in A, v in B.
        # w must be neighbor of v (so w in A) and neighbor of u (so w in B? No, u in A, neighbors of u are in B).
        # So w must be in A (neighbor of v) AND w must be in B (neighbor of u).
        # Contradiction.
        # Therefore, ANY edge added between A and B in a bipartite graph creates NO triangles?
        # Let's re-verify.
        # Graph is bipartite with parts A, B.
        # Edges only between A and B.
        # Path u-v-w. u in A, v in B, w in A.
        # If we add edge (u, w), both in A, that's not allowed in bipartite.
        # If we add edge (u, v) where u in A, v in B.
        # Does it create a triangle?
        # Triangle requires 3 vertices x, y, z with x~y, y~z, z~x.
        # If x=u (A), y=v (B), z must be neighbor of v (so z in A) and neighbor of u (so z in B).
        # z cannot be both in A and B.
        # So yes, adding any edge between A and B to a bipartite graph keeps it bipartite and thus triangle-free.
        
        # So the constraint is just: keep it bipartite?
        # No, we are not restricted to bipartite. We can add edges within A or within B?
        # If we add an edge within A, say (u, w) with u, w in A.
        # Triangle u-w-v-u?
        # v must be neighbor of u (in B) and neighbor of w (in B).
        # So if N_B(u) intersects N_B(w), adding (u,w) creates a triangle.
        
        # So we can add edges within A or within B, provided they don't create triangles.
        # This allows us to increase density and reduce alpha.
        
        # Algorithm:
        # 1. Start with random bipartite.
        # 2. Find independent set of size 10.
        # 3. If found, try to add an edge within that set that doesn't create a triangle.
        # 4. If no such edge, try to "move" a vertex between A and B? Or just restart?
        
        for step in range(500):
            if time.time() > deadline:
                return None
                
            # Check validity
            if not _has_independent_set_10(adj, n):
                # Verify triangle-free just in case
                if _is_triangle_free(adj, n):
                    return _adj_to_edges(adj, n)
                else:
                    break # Should not happen if we maintain invariants, but safety
            
            # Find an independent set of size 10
            # We can use the same DFS but return the set?
            # For speed, just use the boolean check and a greedy finder for the set.
            
            # Greedy independent set
            remaining = (1 << n) - 1
            indep = []
            while remaining:
                # Pick vertex with min degree in remaining?
                # Or just first.
                v_bit = remaining & -remaining
                v = v_bit.bit_length() - 1
                indep.append(v)
                remaining &= ~adj[v] & ~(1 << v)
                
            if len(indep) < 10:
                # Greedy failed to find 10, but exact check said there is one?
                # This can happen if the independent set is "hidden".
                # We need a better way to find the set to repair.
                # Or just restart if we can't find a set to repair.
                break
                
            # Try to add an edge within indep
            added = False
            # Shuffle indep to randomize
            random.shuffle(indep)
            for i in range(len(indep)):
                for j in range(i+1, len(indep)):
                    u = indep[i]
                    v = indep[j]
                    # Check if adding (u,v) creates a triangle
                    # Triangle if N(u) & N(v) is not empty.
                    if (adj[u] & adj[v]) == 0:
                        # Safe to add
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added = True
                        break
                if added:
                    break
                    
            if not added:
                # Stuck. Try a random perturbation?
                # Remove a random edge?
                # Or restart.
                break
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try n=40 first
    # If we find a valid graph on 40, that's the best possible (since R(3,10)<=41).
    # If not, try 39.
    
    # To ensure we don't return a bad result, we pre-compute or generate a 39 graph.
    # Generating a 39 graph might take some time.
    # Let's try to find a 40 graph quickly. If not, we must find a 39 graph.
    
    # We will try to find a valid graph for n=40.
    # If that fails, we will try n=39.
    
    # We'll use a time slice for each.
    # 50s for n=40, 50s for n=39.
    
    # Try n=40
    t0 = time.time()
    edges_40 = _generate_valid_graph(40, t0 + 50)
    if edges_40:
        return {'n': 40, 'edges': edges_40}
        
    # Try n=39
    t1 = time.time()
    edges_39 = _generate_valid_graph(39, t1 + 50)
    if edges_39:
        return {'n': 39, 'edges': edges_39}
        
    # Fallback: Try n=38, 37...
    for n in range(38, 25, -1):
        if time.time() > deadline:
            break
        edges_n = _generate_valid_graph(n, deadline)
        if edges_n:
            return {'n': n, 'edges': edges_n}
            
    # Absolute fallback: The seed graph (n=26)
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    return {'n': n, 'edges': edges}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Result n={res['n']}")
    # Verify
    if _is_valid(res['n'], res['edges']):
        print("Valid")
    else:
        print("INVALID")
