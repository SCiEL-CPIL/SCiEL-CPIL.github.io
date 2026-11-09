"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 16
    
    # 1. Generate the Grötzsch graph (Mycielski of C5) on vertices 0..10
    def get_groetzsch_edges():
        edges = []
        # C5: 0-1-2-3-4-0
        for i in range(5):
            edges.append((i, (i + 1) % 5))
        # Mycielski: 5..9
        for i in range(5):
            neighbors = [(i + 1) % 5, (i - 1) % 5]
            for n_idx in neighbors:
                edges.append((5 + i, n_idx))
        # Vertex 10
        for i in range(5):
            edges.append((10, 5 + i))
        return edges

    # Initialize graph
    edge_set = set()
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        if u == v: return False
        if (u, v) in edge_set or (v, u) in edge_set: return False
        # Check triangle free
        if adj[u] & adj[v]:
            return False
        # Add
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        return True

    def remove_edge(u, v):
        edge_set.discard((u, v))
        edge_set.discard((v, u))
        adj[u].discard(v)
        adj[v].discard(u)

    def save_state():
        return (set(edge_set), [set(s) for s in adj])
    
    def restore_state(state):
        nonlocal edge_set, adj
        edge_set, adj = state[0], state[1]
        # Rebuild adj from edge_set to be safe
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

    # Load Grötzsch
    for u, v in get_groetzsch_edges():
        add_edge(u, v)

    best_edges_count = len(edge_set)
    best_edge_list = [(u, v) for u, v in edge_set]
    
    # Helper to check 3-colorability with better pruning using bitmasks
    def is_not_3_colorable():
        # Backtracking 3-coloring with forward checking using bitmasks for speed
        color = [-1] * n
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        # Precompute neighbor bitmasks
        neighbor_mask = [0] * n
        for v in range(n):
            mask = 0
            for u in adj[v]:
                mask |= (1 << u)
            neighbor_mask[v] = mask
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Available colors: check neighbors
            used_colors = 0
            v_mask = 1 << v
            for c in range(3):
                # Check if any neighbor has color c
                # We need to check all neighbors
                for u in range(n):
                    if (neighbor_mask[v] >> u) & 1:
                        if color[u] == c:
                            used_colors |= (1 << c)
                            break
            
            # Forward checking: check if any uncolored neighbor has no available color
            for u in range(n):
                if (neighbor_mask[v] >> u) & 1:
                    if color[u] == -1:
                        used_u = 0
                        for w in range(n):
                            if (neighbor_mask[u] >> w) & 1:
                                if color[w] != -1:
                                    used_u |= (1 << color[w])
                        if used_u == 0b111:
                            return False
            
            for c in range(3):
                if not (used_colors >> c) & 1:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return not backtrack(0)

    # Find a maximum independent set in the current graph (among active vertices)
    def find_max_independent_set(active_vertices):
        """Find a large independent set using greedy + local search."""
        # Greedy: repeatedly pick vertex with minimum degree in remaining graph
        remaining = set(active_vertices)
        ind_set = set()
        
        while remaining:
            # Find vertex with minimum degree within remaining
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg_in_rem = len(adj[v] & remaining)
                if deg_in_rem < min_deg:
                    min_deg = deg_in_rem
                    min_v = v
            
            ind_set.add(min_v)
            remaining.discard(min_v)
            # Remove neighbors of min_v from remaining
            for u in adj[min_v]:
                remaining.discard(u)
        
        return ind_set

    # Phase 1: Add vertices 11-15 using MIS-based strategy
    # For each new vertex, connect it to a large independent set of the current graph
    for new_v in range(11, n):
        if time.time() > deadline - 10.0:
            break
        
        # Get active vertices (0..new_v-1)
        active = list(range(new_v))
        
        # Find a large independent set
        # Try multiple random orderings to find a good MIS
        best_is = set()
        best_is_size = 0
        
        for attempt in range(20):
            # Greedy MIS with random tie-breaking
            remaining = set(active)
            ind_set = set()
            
            while remaining:
                # Find vertices with minimum degree in remaining
                min_deg = min(len(adj[v] & remaining) for v in remaining)
                candidates = [v for v in remaining if len(adj[v] & remaining) == min_deg]
                
                # Randomly pick from candidates
                v = random.choice(candidates)
                ind_set.add(v)
                remaining.discard(v)
                for u in adj[v]:
                    remaining.discard(u)
            
            if len(ind_set) > best_is_size:
                best_is_size = len(ind_set)
                best_is = ind_set
        
        # Connect new_v to all vertices in the independent set
        for u in best_is:
            add_edge(new_v, u)
        
        if len(edge_set) > best_edges_count:
            best_edges_count = len(edge_set)
            best_edge_list = [(u, v) for u, v in edge_set]

    # Phase 2: Greedy edge addition - add all possible edges that maintain triangle-free
    while time.time() < deadline - 20.0:
        current_edges_count = len(edge_set)
        
        # Find all valid non-edges
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set:
                    # Check if valid (triangle free)
                    if not (adj[u] & adj[v]):
                        cost = len(adj[u]) + len(adj[v])
                        candidates.append((cost, u, v))
        
        if not candidates:
            break
            
        # Sort by cost (lowest first) - add edges that block fewer future edges
        candidates.sort()
        
        # Try to add the cheapest one
        added = False
        for cost, u, v in candidates:
            if add_edge(u, v):
                added = True
                break
        
        if not added:
            break
            
        if len(edge_set) > best_edges_count:
            best_edges_count = len(edge_set)
            best_edge_list = [(u, v) for u, v in edge_set]

    # Phase 3: Intensive local search with edge swaps
    # Strategy: remove one edge, try to add one or two new edges that maintain triangle-free
    # and check 3-colorability
    
    for iteration in range(10000):
        if time.time() > deadline - 5.0:
            break
        
        # Pick a random edge to remove
        if not edge_set:
            break
        e1 = random.choice(list(edge_set))
        u1, v1 = e1
        
        # Remove it
        remove_edge(u1, v1)
        
        # Find all edges that can now be added (only those involving u1 or v1, since
        # removing (u1,v1) only unblocks edges where one endpoint was previously 
        # adjacent to both u1 and v1)
        potential_new_edges = []
        
        # Check edges involving u1
        for x in range(n):
            if x != u1 and x != v1:
                if (u1, x) not in edge_set and (x, u1) not in edge_set:
                    # u1 and x must not share a common neighbor
                    if not (adj[u1] & adj[x]):
                        potential_new_edges.append((u1, x))
        
        # Check edges involving v1
        for x in range(n):
            if x != u1 and x != v1:
                if (v1, x) not in edge_set and (x, v1) not in edge_set:
                    # v1 and x must not share a common neighbor
                    if not (adj[v1] & adj[x]):
                        potential_new_edges.append((v1, x))
        
        # Also check if (u1, v1) itself can be re-added with different context
        # (it's already removed, but other edges might have changed)
        
        if not potential_new_edges:
            add_edge(u1, v1)
            continue
        
        # Try to add up to 2 new edges that maximize total edge count
        # Sort by degree sum (higher is better for connectivity)
        potential_new_edges = list(set(potential_new_edges))
        potential_new_edges.sort(key=lambda e: -(len(adj[e[0]]) + len(adj[e[1]])))
        
        # Try adding the top candidate
        added_edges = []
        for u2, v2 in potential_new_edges[:5]:  # Try top 5
            if add_edge(u2, v2):
                added_edges.append((u2, v2))
                break
        
        # Try adding one more
        if added_edges:
            for u2, v2 in potential_new_edges[:10]:
                if (u2, v2) not in [e for e in added_edges]:
                    if add_edge(u2, v2):
                        added_edges.append((u2, v2))
                        break
        
        if added_edges:
            # Check 3-colorability
            if is_not_3_colorable():
                if len(edge_set) > best_edges_count:
                    best_edges_count = len(edge_set)
                    best_edge_list = [(u, v) for u, v in edge_set]
                # Accept the change
            else:
                # Reject, revert
                for u2, v2 in added_edges:
                    remove_edge(u2, v2)
                add_edge(u1, v1)
        else:
            # Revert
            add_edge(u1, v1)

    # Phase 4: Final polish - try to add any remaining edges
    while time.time() < deadline - 2.0:
        progress = False
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set and not (adj[u] & adj[v]):
                    if add_edge(u, v):
                        progress = True
                        if len(edge_set) > best_edges_count:
                            best_edges_count = len(edge_set)
                            best_edge_list = [(u, v) for u, v in edge_set]
        
        if not progress:
            break
    
    # Phase 5: Final 3-colorability verification and edge maximization
    # Try to add edges in a specific order that might help maintain non-3-colorability
    if time.time() < deadline - 1.0:
        # Try all possible edges in order of how many future edges they block
        all_non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set:
                    # Count how many other non-edges this would block
                    blocked = 0
                    for w in range(n):
                        if w != u and w != v:
                            if (u, w) not in edge_set and (v, w) not in edge_set:
                                if adj[u] & adj[w] and adj[v] & adj[w]:
                                    blocked += 1
                    all_non_edges.append((blocked, u, v))
        
        all_non_edges.sort()
        
        for blocked, u, v in all_non_edges:
            if time.time() > deadline - 0.5:
                break
            if not (adj[u] & adj[v]):
                if add_edge(u, v):
                    if len(edge_set) > best_edges_count:
                        best_edges_count = len(edge_set)
                        best_edge_list = [(u, v) for u, v in edge_set]
    
    return best_edge_list
# EVOLVE-BLOCK-END
