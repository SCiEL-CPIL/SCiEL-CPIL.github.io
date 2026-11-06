"""Dense 4-cycle-rainbow edge-coloured graph on 20 vertices."""
import time
import random
from itertools import combinations

N = 20
DEADLINE_MARGIN = 5.0

def _precompute_cycles():
    """Precompute all 4-cycles and map each edge to the cycles it belongs to."""
    cycles = []
    edge_to_cycles = {e: [] for e in combinations(range(N), 2)}
    
    for a, b, c, d in combinations(range(N), 4):
        # Three distinct 4-cycles on vertices a,b,c,d
        # 1: a-b-c-d-a
        # 2: a-b-d-c-a
        # 3: a-c-b-d-a
        
        perms = [
            ((a,b), (b,c), (c,d), (d,a)),
            ((a,b), (b,d), (d,c), (c,a)),
            ((a,c), (c,b), (b,d), (d,a))
        ]
        
        for edges in perms:
            # Normalize edges (min, max)
            norm_edges = tuple(sorted((min(u,v), max(u,v)) for u,v in edges))
            # Check for duplicates? No, these are distinct cycles.
            cyc_id = len(cycles)
            cycles.append(norm_edges)
            for e in norm_edges:
                edge_to_cycles[e].append(cyc_id)
                
    return cycles, edge_to_cycles

def construct():
    t0 = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    cycles, edge_to_cycles = _precompute_cycles()
    all_edges = list(combinations(range(N), 2))
    
    best_edges_count = 0
    best_result = []
    
    # Helper to check if adding edge (u,v) with color c is valid given current graph
    def is_valid_addition(u, v, c, current_edges, current_colors):
        e = (min(u, v), max(u, v))
        if e in current_edges:
            return False
            
        # Check all 4-cycles containing e
        for cyc_id in edge_to_cycles[e]:
            cyc_edges = cycles[cyc_id]
            # Check if other 3 edges are in the graph
            others = [ed for ed in cyc_edges if ed != e]
            if all(ed in current_edges for ed in others):
                # Get colors of others
                cols = [current_colors[ed] for ed in others]
                # If c is in cols, or cols has duplicates, it's not rainbow
                if c in cols or len(set(cols)) < 3:
                    return False
        return True

    def try_greedy_build(edge_order, seed_offset):
        """
        Builds a graph by iterating through edge_order and adding edges if possible.
        Returns (edge_count, edges_set, colors_dict)
        """
        current_edges = set()
        current_colors = {}
        
        for e in edge_order:
            u, v = e
            # Try colors in a random order to break symmetry and find a valid one faster?
            # Or fixed order? Fixed is faster. Random might help escape local minima in color space?
            # Let's try random color order for each edge.
            colors_to_try = list(range(20))
            random.shuffle(colors_to_try)
            
            added = False
            for c in colors_to_try:
                if is_valid_addition(u, v, c, current_edges, current_colors):
                    current_edges.add(e)
                    current_colors[e] = c
                    added = True
                    break
                    
        return len(current_edges), current_edges, current_colors

    def local_search_improve(edges_set, colors_dict):
        """
        Attempts to improve the graph by removing an edge and refilling.
        Returns improved (edges_set, colors_dict) or None if no improvement.
        """
        current_edges = set(edges_set)
        current_colors = dict(colors_dict)
        current_count = len(current_edges)
        
        # Identify missing edges
        missing_edges = [e for e in all_edges if e not in current_edges]
        
        if not missing_edges:
            return None
            
        # Try a few random removals
        for _ in range(10): # Try 10 random removals
            if time.time() > deadline:
                return None
                
            # Pick a random edge to remove
            if not current_edges:
                break
            e_remove = random.choice(list(current_edges))
            del current_colors[e_remove]
            current_edges.remove(e_remove)
            
            # Try to add back missing edges greedily
            # We prioritize the one we just removed? No, we want to fill up.
            # Shuffle missing edges + the removed one
            candidates = missing_edges + [e_remove]
            random.shuffle(candidates)
            
            for e in candidates:
                if e in current_edges:
                    continue
                u, v = e
                # Try to add
                colors_to_try = list(range(20))
                random.shuffle(colors_to_try)
                for c in colors_to_try:
                    if is_valid_addition(u, v, c, current_edges, current_colors):
                        current_edges.add(e)
                        current_colors[e] = c
                        break
                        
        if len(current_edges) > current_count:
            return current_edges, current_colors
        else:
            return None

    # Main Loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Generate a random edge order
        random_edge_order = all_edges[:]
        random.shuffle(random_edge_order)
        
        # Greedy Build
        count, edges, colors = try_greedy_build(random_edge_order, attempts)
        
        if count > best_edges_count:
            best_edges_count = count
            best_result = [[u, v, colors[(u,v)]] for (u,v) in sorted(edges)]
            
        # Local Search Improvement
        # Only if we have time
        if time.time() < deadline - 5.0:
            improved = local_search_improve(edges, colors)
            if improved:
                new_edges, new_colors = improved
                new_count = len(new_edges)
                if new_count > best_edges_count:
                    best_edges_count = new_count
                    best_result = [[u, v, new_colors[(u,v)]] for (u,v) in sorted(new_edges)]
                    
        # If we reach a high density (e.g., > 100), we might want to focus on local search more
        # But for now, random restarts are good.
        
        # Early stop if we hit theoretical max? 
        # Max edges is 190. Unlikely to reach.
        
    return best_result if best_result else [[0, v, 0] for v in range(1, N)]

# Initial check for safety
if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} edges")
