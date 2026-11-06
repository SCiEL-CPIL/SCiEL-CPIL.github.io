"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _check_validity(edges):
    """Check that every 4-cycle has 4 distinct colours."""
    if not edges:
        return True
    # Build adjacency with colours
    adj = {}
    for u, v, c in edges:
        adj.setdefault(u, {})[v] = c
        adj.setdefault(v, {})[u] = c
    
    # Find all 4-cycles: a-b-c-d-a
    # To avoid duplicates, we can iterate over pairs of vertices and check common neighbours
    # But for n=20, a simpler approach: iterate over all 4-tuples
    # Actually, let's be smarter. A 4-cycle is determined by 4 vertices in order.
    # We need to check that for every set of 4 vertices that form a cycle, the 4 edges have distinct colours.
    
    # Build a set of edges for quick lookup
    edge_set = set()
    for u, v, c in edges:
        edge_set.add((min(u,v), max(u,v), c))
    
    # For each 4-tuple (a,b,c,d) with a<b<c<d, check if they form a 4-cycle
    # A 4-cycle on vertices a,b,c,d can be: a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    # But we need to check all possible 4-cycles in the subgraph induced by these 4 vertices
    
    n = 20
    for a, b, c, d in combinations(range(n), 4):
        # Check all 3 ways to form a 4-cycle on these 4 vertices
        # Cycle 1: a-b-c-d-a
        # Cycle 2: a-b-d-c-a  
        # Cycle 3: a-c-b-d-a
        
        def check_cycle(v1, v2, v3, v4):
            # Check if edges v1-v2, v2-v3, v3-v4, v4-v1 all exist
            c1 = adj.get(v1, {}).get(v2)
            c2 = adj.get(v2, {}).get(v3)
            c3 = adj.get(v3, {}).get(v4)
            c4 = adj.get(v4, {}).get(v1)
            if c1 is None or c2 is None or c3 is None or c4 is None:
                return True  # Not a cycle
            return len({c1, c2, c3, c4}) == 4
        
        if not check_cycle(a, b, c, d):
            return False
        if not check_cycle(a, b, d, c):
            return False
        if not check_cycle(a, c, b, d):
            return False
    return True

def _greedy_construct():
    """Greedy construction: try to add edges one by one, choosing the edge that
    allows the most future edges."""
    n = 20
    colors = list(range(n))
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # State: for each edge, which colours are still available
    # And track which 4-cycles are affected
    
    # Let's use a different approach: build the graph incrementally
    # Track: adj[u][v] = color, or None if no edge
    adj = [[None]*n for _ in range(n)]
    edges = []
    
    # For each 4-cycle, track the set of colours used
    # A 4-cycle is (a,b,c,d) with edges (a,b),(b,c),(c,d),(d,a)
    # We need to enumerate all 4-cycles
    
    # Precompute all 4-cycles
    four_cycles = []
    for a, b, c, d in combinations(range(n), 4):
        # Three possible 4-cycles on these 4 vertices
        for cyc in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
            four_cycles.append(cyc)
    
    # For each 4-cycle, track colours used
    cycle_colors = [set() for _ in four_cycles]
    
    # For each edge, track which 4-cycles it belongs to
    edge_to_cycles = {}
    for i, cyc in enumerate(four_cycles):
        for j in range(4):
            u, v = cyc[j], cyc[(j+1)%4]
            key = (min(u,v), max(u,v))
            edge_to_cycles.setdefault(key, []).append(i)
    
    # Greedy: at each step, pick the edge-colour that maximizes flexibility
    # We'll try each possible edge and colour, pick the one that doesn't violate
    # and leaves the most options open
    
    remaining_edges = set(all_edges)
    best_edges = []
    
    for step in range(len(all_edges)):
        best_score = -1
        best_edge = None
        best_color = None
        
        for u, v in all_edges:
            if (u,v) in best_edges:
                continue
            for c in range(n):
                # Check if adding (u,v,c) violates any 4-cycle
                valid = True
                for cyc_idx in edge_to_cycles.get((min(u,v),max(u,v)), []):
                    if c in cycle_colors[cyc_idx]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Score: how many future edges are still possible
                # Simplified: count how many other edges can still be added
                # This is expensive, so use a heuristic
                # Just pick the first valid one with some randomness
                if best_score < 0 or random.random() < 0.1:
                    best_score = random.random()
                    best_edge = (u,v)
                    best_color = c
        
        if best_edge is None:
            break
        
        u, v = best_edge
        c = best_color
        best_edges.append((u,v))
        adj[u][v] = c
        adj[v][u] = c
        
        for cyc_idx in edge_to_cycles.get((min(u,v),max(u,v)), []):
            cycle_colors[cyc_idx].add(c)
    
    result = [[u, v, c] for u, v, c in best_edges]
    return result

def _local_search(edges, max_iters=5000, time_limit=100):
    """Local search: try to improve by swapping colours or adding/removing edges."""
    start_time = time.time()
    n = 20
    colors = list(range(n))
    
    # Build state
    adj = [[None]*n for _ in range(n)]
    for u, v, c in edges:
        adj[u][v] = c
        adj[v][u] = c
    
    # Precompute 4-cycles and their edge memberships
    four_cycles = []
    for a, b, c, d in combinations(range(n), 4):
        for cyc in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
            four_cycles.append(cyc)
    
    edge_to_cycles = {}
    for i, cyc in enumerate(four_cycles):
        for j in range(4):
            u, v = cyc[j], cyc[(j+1)%4]
            key = (min(u,v), max(u,v))
            edge_to_cycles.setdefault(key, []).append(i)
    
    def get_cycle_colors(cyc_idx):
        cyc = four_cycles[cyc_idx]
        cols = []
        for j in range(4):
            u, v = cyc[j], cyc[(j+1)%4]
            c = adj[u][v]
            if c is None:
                return None  # Not a complete cycle
            cols.append(c)
        return set(cols)
    
    def is_valid():
        for i in range(len(four_cycles)):
            cols = get_cycle_colors(i)
            if cols is not None and len(cols) != 4:
                return False
        return True
    
    def count_edges():
        cnt = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] is not None:
                    cnt += 1
        return cnt
    
    current_edges = count_edges()
    best_edges = current_edges
    best_adj = [row[:] for row in adj]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try random moves
        move_type = random.choice(['add', 'remove', 'recolor'])
        
        if move_type == 'add':
            # Try to add a random edge with a random colour
            u, v = random.sample(range(n), 2)
            if adj[u][v] is not None:
                continue
            c = random.randint(0, n-1)
            adj[u][v] = c
            adj[v][u] = c
            if is_valid():
                new_edges = count_edges()
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = [row[:] for row in adj]
            else:
                adj[u][v] = None
                adj[v][u] = None
        
        elif move_type == 'remove':
            u, v = random.sample(range(n), 2)
            if adj[u][v] is None:
                continue
            adj[u][v] = None
            adj[v][u] = None
            # Don't track best for removals unless it helps later additions
        
        elif move_type == 'recolor':
            u, v = random.sample(range(n), 2)
            if adj[u][v] is None:
                continue
            old_c = adj[u][v]
            new_c = random.randint(0, n-1)
            if new_c == old_c:
                continue
            adj[u][v] = new_c
            adj[v][u] = new_c
            if is_valid():
                pass  # Same edge count, but might enable future additions
            else:
                adj[u][v] = old_c
                adj[v][u] = old_c
    
    # Reconstruct best
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v] is not None:
                result.append([u, v, best_adj[u][v]])
    return result

def construct():
    """Construct a dense edge-coloured graph on 20 vertices where every 4-cycle is rainbow."""
    start_time = time.time()
    time_limit = 145  # Leave 5 seconds margin
    
    random.seed(42)
    
    # Start with greedy construction
    edges = _greedy_construct()
    
    # Verify
    if not _check_validity(edges):
        # Fallback to a known valid small construction
        # Use a star-like structure that's guaranteed valid
        edges = [[0, v, v] for v in range(1, 20)]
    
    # Local search to improve
    if time.time() - start_time < time_limit - 10:
        edges = _local_search(edges, max_iters=20000, time_limit=time_limit - 15)
    
    # Final verification
    if not _check_validity(edges):
        # Return a safe fallback
        edges = [[0, v, v] for v in range(1, 20)]
    
    return edges

# EVOLVE-BLOCK-END
