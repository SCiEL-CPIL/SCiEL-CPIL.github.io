import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        # Three distinct 4-cycles on 4 vertices
        c1 = [(u,v), (v,w), (w,x), (x,u)]
        c2 = [(u,v), (v,x), (x,w), (w,u)]
        c3 = [(u,w), (w,v), (v,x), (x,u)]
        
        for cycle in [c1, c2, c3]:
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)
    
    # State
    edge_colors = {}
    
    def is_color_valid_for_edge(e, c, current_edge_colors):
        """Check if assigning color c to edge e keeps all 4-cycles containing e rainbow."""
        for idx in edge_to_cycles.get(e, []):
            cycle = all_cycles[idx]
            # Check if all other edges in this cycle exist
            other_edges = [oe for oe in cycle if oe != e]
            if not all(oe in current_edge_colors for oe in other_edges):
                continue
            # Check colors
            other_colors = [current_edge_colors[oe] for oe in other_edges]
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True

    def try_add_edge(e, current_edge_colors):
        """Try to add edge e with a valid color. Returns True if successful."""
        # Try colors in a random order
        colors = list(range(num_colors))
        random.shuffle(colors)
        for c in colors:
            if is_color_valid_for_edge(e, c, current_edge_colors):
                current_edge_colors[e] = c
                return True
        return False

    def recolor_attempt(edge_colors, e_to_recolor, target_color=None):
        """
        Attempt to recolor a random set of edges to free up space or satisfy constraints.
        This is a simple local move: pick a random edge, try a new color.
        Returns True if any edge was successfully recolored.
        """
        if not edge_colors:
            return False
        
        # Pick a random edge
        e = random.choice(list(edge_colors.keys()))
        old_color = edge_colors[e]
        
        # Try to find a new valid color
        colors = list(range(num_colors))
        random.shuffle(colors)
        
        for new_color in colors:
            if new_color == old_color:
                continue
            # Check validity
            # Temporarily assume it's removed? No, we just check if new_color is valid given others.
            # Wait, is_color_valid_for_edge checks if c is valid given the *other* edges in cycles.
            # It does not check if c is valid given the edge's own presence (which is implicit).
            # So we can just check directly.
            if is_color_valid_for_edge(e, new_color, edge_colors):
                edge_colors[e] = new_color
                return True
        return False

    best_edge_count = 0
    best_solution = []
    
    # Strategy: Constructive with Randomized Edge Order
    # We try multiple times with different random orderings of edges.
    
    num_restarts = 1000
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        edge_colors = {}
        
        # Generate a random order of all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        # Construct
        for e in all_edges:
            if time.time() > deadline - 2.0:
                break
            if try_add_edge(e, edge_colors):
                continue
            else:
                # Stuck. Try some recoloring moves to unblock.
                # Limit recoloring attempts to avoid spending too much time here
                for _ in range(10):
                    if time.time() > deadline - 2.0:
                        break
                    if recolor_attempt(edge_colors, None):
                        # Try adding the edge again
                        if try_add_edge(e, edge_colors):
                            break
                    else:
                        break
                # If still stuck, just skip this edge for this restart
                # We might try to add it later if we had a more complex structure, 
                # but simple greedy skip is okay for now.
                pass
        
        # Update best
        if len(edge_colors) > best_edge_count:
            best_edge_count = len(edge_colors)
            best_solution = [[u, v, c] for (u, v), c in edge_colors.items()]
            
        # Early exit if we hit a theoretical bound or high density
        # Max edges is 190. If we get close, we can stop.
        if best_edge_count >= 150: # Heuristic threshold
            break

    # If we haven't found a good solution, do a final local search on the best found
    if best_solution:
        # Convert back to dict
        edge_colors = {}
        for e in best_solution:
            u, v, c = e
            edge_colors[(u, v)] = c
            
        # Try to improve further with local search
        current_edges = set(edge_colors.keys())
        missing_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in current_edges]
        
        for _ in range(5000): # Limited local search steps
            if time.time() > deadline - 1.0:
                break
            
            improved = False
            # Try to add a missing edge
            if missing_edges:
                e_add = random.choice(missing_edges)
                if try_add_edge(e_add, edge_colors):
                    missing_edges.remove(e_add)
                    improved = True
                    if len(edge_colors) > best_edge_count:
                        best_edge_count = len(edge_colors)
                        best_solution = [[u, v, c] for (u, v), c in edge_colors.items()]
            
            # Try recoloring to enable addition
            if not improved and edge_colors:
                if recolor_attempt(edge_colors, None):
                    if missing_edges:
                        e_add = random.choice(missing_edges)
                        if try_add_edge(e_add, edge_colors):
                            missing_edges.remove(e_add)
                            improved = True
                            if len(edge_colors) > best_edge_count:
                                best_edge_count = len(edge_colors)
                                best_solution = [[u, v, c] for (u, v), c in edge_colors.items()]
            
            if not improved:
                # Random restart of local search state? No, just continue.
                # If we get stuck completely, break out of local search
                # We don't track no_improve count strictly here, just run for a fixed number of steps
                pass

    if not best_solution:
        # Fallback: return empty or a trivial valid graph
        # An empty graph is valid.
        return []
        
    return best_solution
