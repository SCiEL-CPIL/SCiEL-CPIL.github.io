import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    n = 20
    n_colors = 20
    
    # 1. Precomputation
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Map edge to index
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    # Precompute all 4-cycles
    # A 4-cycle is defined by 4 vertices {w, x, y, z} and a pairing of edges.
    # There are 3 distinct 4-cycles for any set of 4 vertices in K_n.
    all_4cycles = []
    for quad in combinations(range(n), 4):
        w, x, y, z = quad
        # The 3 cycles are:
        # 1. w-x-y-z-w
        # 2. w-x-z-y-w
        # 3. w-y-x-z-w
        cycles = [
            [(w, x), (x, y), (y, z), (z, w)],
            [(w, x), (x, z), (z, y), (y, w)],
            [(w, y), (y, x), (x, z), (z, w)],
        ]
        for cyc in cycles:
            try:
                edge_idxs = tuple(edge_to_idx[e] for e in cyc)
                all_4cycles.append(edge_idxs)
            except KeyError:
                pass

    # Map edge index to list of 4-cycle indices containing it
    edge_cycles = [[] for _ in range(num_edges)]
    for ci, cyc in enumerate(all_4cycles):
        for ei in cyc:
            edge_cycles[ei].append(ci)

    # State representation:
    # edges_set: set of edge indices present
    # colors: dict {edge_idx: color}
    
    def get_cycle_colors(cyc, edges_set, colors, override=None):
        """
        Get colors of a cycle.
        override: tuple (edge_idx, color) to simulate a change.
        Returns list of colors or None if cycle is not complete.
        """
        cs = []
        for e in cyc:
            if e not in edges_set:
                return None
            if override and e == override[0]:
                cs.append(override[1])
            else:
                cs.append(colors[e])
        return cs

    def is_rainbow(cyc, edges_set, colors, override=None):
        cs = get_cycle_colors(cyc, edges_set, colors, override)
        if cs is None:
            return True # Incomplete cycle is not a violation
        return len(set(cs)) == 4

    def find_bad_cycles(edges_set, colors):
        """Find all 4-cycles that are complete and not rainbow."""
        bad = []
        for ci, cyc in enumerate(all_4cycles):
            # Quick check: are all edges present?
            if all(e in edges_set for e in cyc):
                if not is_rainbow(cyc, edges_set, colors):
                    bad.append(ci)
        return bad

    def try_recolour(ei, new_c, edges_set, colors):
        """
        Check if changing edge ei to new_c is valid.
        Returns True if valid, False otherwise.
        """
        if ei not in edges_set:
            return False
            
        # Check all cycles containing ei
        for ci in edge_cycles[ei]:
            cyc = all_4cycles[ci]
            # If the cycle is not complete, changing ei doesn't make it bad (it remains incomplete)
            # But wait, if the cycle IS complete, we must check.
            if all(e in edges_set for e in cyc):
                # Simulate the change
                if not is_rainbow(cyc, edges_set, colors, override=(ei, new_c)):
                    return False
        return True

    def repair(edges_set, colors):
        """
        Remove edges or recolour to eliminate all bad cycles.
        """
        # We iterate until no bad cycles remain
        # To avoid infinite loops, we limit iterations or use a counter
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline:
                return edges_set, colors
            
            bad_cycles = find_bad_cycles(edges_set, colors)
            if not bad_cycles:
                return edges_set, colors
            
            # Pick a random bad cycle
            ci = random.choice(bad_cycles)
            cyc = all_4cycles[ci]
            
            # Try to fix by recolouring one of the 4 edges
            fixed = False
            edges_in_cyc = list(cyc)
            random.shuffle(edges_in_cyc)
            
            for ei in edges_in_cyc:
                # Try all colours
                possible_colors = [c for c in range(n_colors) if c != colors.get(ei, -1)]
                random.shuffle(possible_colors)
                
                for new_c in possible_colors:
                    if try_recolour(ei, new_c, edges_set, colors):
                        colors[ei] = new_c
                        fixed = True
                        break
                if fixed:
                    break
            
            if not fixed:
                # If we can't fix by recolouring, remove an edge
                # Choose the edge with the fewest other cycles? Or random?
                # Random is simpler
                ei_remove = random.choice(cyc)
                del colors[ei_remove]
                edges_set.remove(ei_remove)
                
        return edges_set, colors

    def maximize_edges(edges_set, colors):
        """
        Try to add missing edges.
        """
        missing = [ei for ei in range(num_edges) if ei not in edges_set]
        random.shuffle(missing)
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Re-shuffle to avoid bias
            random.shuffle(missing)
            
            for ei in missing:
                if ei in edges_set:
                    continue
                
                # Try all colours
                possible_colors = list(range(n_colors))
                random.shuffle(possible_colors)
                
                for c in possible_colors:
                    # Check if adding ei with color c is valid
                    # This is equivalent to try_recolour but for a new edge
                    # We need to check cycles containing ei.
                    # For a new edge, the cycle is only "complete" if the other 3 edges are present.
                    valid = True
                    for ci in edge_cycles[ei]:
                        cyc = all_4cycles[ci]
                        # Check if other 3 edges are present
                        if all(e in edges_set for e in cyc if e != ei):
                            # Cycle is complete with this new edge
                            # Check rainbow
                            cs = [colors[e] for e in cyc if e != ei] + [c]
                            if len(set(cs)) != 4:
                                valid = False
                                break
                            
                    if valid:
                        edges_set.add(ei)
                        colors[ei] = c
                        missing.remove(ei)
                        improved = True
                        break # Move to next missing edge
        
        return edges_set, colors

    def local_search(edges_set, colors):
        """
        Combine repair and maximize.
        """
        # First repair
        edges_set, colors = repair(edges_set, colors)
        # Then maximize
        edges_set, colors = maximize_edges(edges_set, colors)
        
        # Try a few more rounds of repair/maximize if time permits
        # Sometimes adding an edge creates a bad cycle if we didn't check carefully?
        # No, maximize_edges checks validity.
        # But maybe a better colouring exists?
        # We can try random recolouring of existing edges to free up capacity.
        
        if time.time() > deadline - 5.0:
            return edges_set, colors
            
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            # Pick a random edge and try to change its colour
            if not edges_set:
                break
            ei = random.choice(list(edges_set))
            old_c = colors[ei]
            possible_colors = [c for c in range(n_colors) if c != old_c]
            random.shuffle(possible_colors)
            
            for new_c in possible_colors:
                if try_recolour(ei, new_c, edges_set, colors):
                    # Accept the change if it allows adding a new edge?
                    # Or just accept if it maintains validity.
                    # Let's accept it and see if we can add more.
                    colors[ei] = new_c
                    edges_set, colors = maximize_edges(edges_set, colors)
                    break
                    
        return edges_set, colors

    best_edges_set = set()
    best_colors = {}
    best_count = 0

    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Initialize with a dense random graph
        # Probability of edge: start high, e.g., 0.8
        p = 0.8
        edges_set = set()
        colors = {}
        
        for ei in range(num_edges):
            if random.random() < p:
                edges_set.add(ei)
                colors[ei] = random.randint(0, n_colors - 1)
                
        # Repair and Maximize
        edges_set, colors = local_search(edges_set, colors)
        
        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges_set = edges_set.copy()
            best_colors = colors.copy()
            
        # If we are doing well, we can try to start denser?
        # Or just keep restarting.
        
    # Final polish on the best solution
    if time.time() < deadline - 2.0:
        best_edges_set, best_colors = local_search(best_edges_set, best_colors)
        best_count = len(best_edges_set)

    # Build result
    result = []
    for ei in best_edges_set:
        u, v = all_edges[ei]
        c = best_colors[ei]
        result.append([u, v, c])

    return result
