"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    n = 20
    num_colors = 20
    
    # All possible edges in complete graph
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    total_edges = len(all_edges)
    
    # Adjacency for fast lookup
    adj = [set() for _ in range(n)]
    
    # For each edge, track which 4-cycles it participates in
    # A 4-cycle is u-v-w-x-u, so edge (u,v) is in cycles where
    # u and v are adjacent in the cycle
    edge_to_cycles = defaultdict(set)
    
    # Build cycle membership for each edge
    # For edge (u,v), find all pairs (w,x) such that u-v-w-x-u is a 4-cycle
    # This means: v-w, w-x, x-u must be edges, and u,v,w,x all distinct
    for i, (u, v) in enumerate(all_edges):
        for w in range(n):
            if w == u or w == v:
                continue
            for x in range(n):
                if x == u or x == v or x == w:
                    continue
                # Check if u-v-w-x-u forms a 4-cycle
                # Need edges: (u,v), (v,w), (w,x), (x,u)
                if (v, w) in [(a,b) for a,b in all_edges] and (w,x) in [(a,b) for a,b in all_edges] and (x,u) in [(a,b) for a,b in all_edges]:
                    # Normalize the cycle representation
                    cycle = frozenset([(u,v), (v,w), (w,x), (x,u)])
                    for e in cycle:
                        edge_to_cycles[e].add(cycle)
    
    # Precompute: for each edge, the set of cycles containing it
    # Use a more efficient approach
    edge_set = set(all_edges)
    edge_to_cycles = defaultdict(set)
    
    for u in range(n):
        for v in range(u+1, n):
            if (u,v) not in edge_set:
                continue
            # Find all 4-cycles containing edge (u,v)
            # u-v-w-x-u: need v-w, w-x, x-u edges
            for w in range(n):
                if w == u or w == v:
                    continue
                if (min(v,w), max(v,w)) not in edge_set:
                    continue
                for x in range(n):
                    if x in (u, v, w):
                        continue
                    if (min(w,x), max(w,x)) not in edge_set:
                        continue
                    if (min(x,u), max(x,u)) not in edge_set:
                        continue
                    cycle_edges = frozenset([(min(u,v),max(u,v)), (min(v,w),max(v,w)), (min(w,x),max(w,x)), (min(x,u),max(x,u))])
                    edge_to_cycles[(u,v)].add(cycle_edges)
                    edge_to_cycles[(v,w)].add(cycle_edges)
                    edge_to_cycles[(w,x)].add(cycle_edges)
                    edge_to_cycles[(x,u)].add(cycle_edges)
    
    # State
    edge_color = {}  # (u,v) -> color
    edge_present = set()
    
    # For each edge, track available colors given current coloring
    # and for each cycle, track used colors
    cycle_used_colors = defaultdict(set)
    
    def is_valid_color_assignment(edge, color):
        """Check if assigning color to edge is valid (no cycle becomes non-rainbow)."""
        u, v = edge
        # Check all cycles containing this edge
        for cycle in edge_to_cycles.get(edge, set()):
            # Remove this edge's current color from cycle tracking (if present)
            # Check if adding this color would make cycle non-rainbow
            # The cycle has 4 edges; if any other edge in cycle has same color, invalid
            for e in cycle:
                if e == edge:
                    continue
                if e in edge_color and edge_color[e] == color:
                    return False
        return True
    
    def try_add_edge(edge, color):
        """Add edge with color, return True if successful."""
        u, v = edge
        if edge in edge_present:
            return False
        if not is_valid_color_assignment(edge, color):
            return False
        
        # Assign
        edge_color[edge] = color
        edge_present.add(edge)
        adj[u].add(v)
        adj[v].add(u)
        
        # Update cycle color tracking
        for cycle in edge_to_cycles.get(edge, set()):
            cycle_used_colors[cycle].add(color)
        
        return True
    
    def remove_edge(edge):
        """Remove edge and its color."""
        u, v = edge
        if edge not in edge_present:
            return
        color = edge_color[edge]
        del edge_color[edge]
        edge_present.discard(edge)
        adj[u].discard(v)
        adj[v].discard(u)
        
        # Update cycle color tracking
        for cycle in edge_to_cycles.get(edge, set()):
            cycle_used_colors[cycle].discard(color)
    
    # Greedy construction with backtracking
    # Order edges by degree of constraint (most constrained first)
    edge_order = sorted(all_edges, key=lambda e: len(edge_to_cycles.get(e, set())), reverse=True)
    
    best_edge_count = 0
    best_state = {}
    
    # Try multiple random orderings and colorings
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Reset state
        edge_color = {}
        edge_present = set()
        adj = [set() for _ in range(n)]
        cycle_used_colors = defaultdict(set)
        
        # Shuffle edge order with bias toward high-constraint edges
        shuffled = list(edge_order)
        random.shuffle(shuffled)
        
        current_count = 0
        
        for edge in shuffled:
            if time.time() > deadline - 5.0:
                break
            
            # Try colors in random order
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            for color in colors:
                if is_valid_color_assignment(edge, color):
                    # Assign
                    u, v = edge
                    edge_color[edge] = color
                    edge_present.add(edge)
                    adj[u].add(v)
                    adj[v].add(u)
                    for cycle in edge_to_cycles.get(edge, set()):
                        cycle_used_colors[cycle].add(color)
                    current_count += 1
                    break
        
        if current_count > best_edge_count:
            best_edge_count = current_count
            best_state = dict(edge_color)
    
    # If greedy didn't find anything good, try a different approach:
    # Start with a sparse base and add edges
    if best_edge_count < 50:
        # Try building from a random sparse graph
        for attempt in range(20):
            if time.time() > deadline - 3.0:
                break
            
            edge_color = {}
            edge_present = set()
            adj = [set() for _ in range(n)]
            cycle_used_colors = defaultdict(set)
            
            # Start with a random spanning tree
            perm = list(range(n))
            random.shuffle(perm)
            for i in range(1, n):
                u = perm[i]
                v = perm[0]  # connect to first
                edge = (min(u,v), max(u,v))
                if edge not in edge_present:
                    # Find a valid color
                    for color in range(num_colors):
                        if is_valid_color_assignment(edge, color):
                            edge_color[edge] = color
                            edge_present.add(edge)
                            adj[u].add(v)
                            adj[v].add(u)
                            for cycle in edge_to_cycles.get(edge, set()):
                                cycle_used_colors[cycle].add(color)
                            break
            
            # Now try to add more edges
            remaining = [e for e in all_edges if e not in edge_present]
            random.shuffle(remaining)
            
            for edge in remaining:
                if time.time() > deadline - 3.0:
                    break
                for color in range(num_colors):
                    if is_valid_color_assignment(edge, color):
                        u, v = edge
                        edge_color[edge] = color
                        edge_present.add(edge)
                        adj[u].add(v)
                        adj[v].add(u)
                        for cycle in edge_to_cycles.get(edge, set()):
                            cycle_used_colors[cycle].add(color)
                        break
            
            current_count = len(edge_present)
            if current_count > best_edge_count:
                best_edge_count = current_count
                best_state = dict(edge_color)
    
    # Convert to output format
    result = []
    for (u, v), color in best_state.items():
        result.append([u, v, color])
    
    return result
# EVOLVE-BLOCK-END
