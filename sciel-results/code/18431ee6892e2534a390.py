import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s kill limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles and their edge sets
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    for u, v, w, x in itertools.combinations(range(n), 4):
        # Three distinct 4-cycles on 4 vertices (edge sets)
        cycles = [
            [(u, v), (v, w), (w, x), (x, u)],
            [(u, v), (v, x), (x, w), (w, u)],
            [(u, w), (w, v), (v, x), (x, u)],
        ]
        for cycle in cycles:
            norm_cycle = tuple(sorted((min(e), max(e)) for e in cycle))
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)
    
    def can_add_edge(e, c, edge_colors):
        """Check if adding edge e with color c keeps all 4-cycles containing e rainbow."""
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            # Check if all other edges in cycle are present
            all_present = True
            other_colors = []
            for oe in cycle:
                if oe == e:
                    continue
                if oe not in edge_colors:
                    all_present = False
                    break
                other_colors.append(edge_colors[oe])
            if not all_present:
                continue
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True

    def find_valid_color(e, edge_colors):
        """Find a valid color for edge e, or None."""
        # Collect colors used in cycles containing e
        used_in_cycles = set()
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            for oe in cycle:
                if oe != e and oe in edge_colors:
                    used_in_cycles.add(edge_colors[oe])
        
        # Try unused colors first, then others
        unused = [c for c in range(num_colors) if c not in used_in_cycles]
        used = [c for c in range(num_colors) if c in used_in_cycles]
        # Deterministic order for reproducibility and systematic search
        for c in unused + used:
            if can_add_edge(e, c, edge_colors):
                return c
        return None

    # Phase 1: Construct a high-quality initial solution
    edge_colors = {}
    
    # Greedy construction
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    for e in all_edges:
        if time.time() > deadline:
            break
        c = find_valid_color(e, edge_colors)
        if c is not None:
            edge_colors[e] = c

    # Phase 2: Systematic Local Search
    missing_set = set(e for e in all_edges if e not in edge_colors)
    best_count = len(edge_colors)
    best_state = dict(edge_colors)
    
    def save_if_better():
        nonlocal best_count, best_state
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_state = dict(edge_colors)

    # Iterative improvement loop
    while time.time() < deadline:
        improved = False
        
        # Move type 1: Add a missing edge (Systematic)
        for e_add in list(missing_set):
            if time.time() > deadline:
                break
            c = find_valid_color(e_add, edge_colors)
            if c is not None:
                edge_colors[e_add] = c
                missing_set.discard(e_add)
                improved = True
                save_if_better()
        
        # Move type 2: Recolor an existing edge (Systematic)
        # We try to recolor edges to potentially free up colors for missing edges
        # A simple heuristic: try all colors for each edge
        for e_recolor in list(edge_colors.keys()):
            if time.time() > deadline:
                break
            old_c = edge_colors[e_recolor]
            # Try all colors
            for new_c in range(num_colors):
                if new_c == old_c:
                    continue
                # Temporarily remove
                del edge_colors[e_recolor]
                if can_add_edge(e_recolor, new_c, edge_colors):
                    edge_colors[e_recolor] = new_c
                    # Check if this allows adding any missing edges
                    # To keep it fast, we just mark as improved if we changed color
                    # The next addition pass will exploit this
                    improved = True
                    break # Only one recolor per edge per pass
                else:
                    # Revert
                    edge_colors[e_recolor] = old_c
        
        # Move type 3: If no improvement, try a random restart of local search or swap
        if not improved:
            # Try a random swap to escape local optimum
            if edge_colors and missing_set:
                e_remove = random.choice(tuple(edge_colors.keys()))
                e_add = random.choice(tuple(missing_set))
                c_remove = edge_colors[e_remove]
                del edge_colors[e_remove]
                missing_set.add(e_remove)
                
                c_add = find_valid_color(e_add, edge_colors)
                if c_add is not None:
                    edge_colors[e_add] = c_add
                    missing_set.discard(e_add)
                    save_if_better()
                    # Try to re-add e_remove
                    c_rem = find_valid_color(e_remove, edge_colors)
                    if c_rem is not None:
                        edge_colors[e_remove] = c_rem
                        missing_set.discard(e_remove)
                        save_if_better()
                else:
                    # Revert
                    edge_colors[e_remove] = c_remove
                    missing_set.discard(e_remove)
            
            # If still stuck, do a random perturbation
            if random.random() < 0.1:
                # Remove a few random edges to allow re-optimization
                for _ in range(5):
                    if not edge_colors:
                        break
                    e = random.choice(tuple(edge_colors.keys()))
                    missing_set.add(e)
                    del edge_colors[e]

    # Use the best state found
    edge_colors = best_state
    
    # Final cleanup: ensure validity
    for _ in range(200):
        if time.time() > deadline:
            break
        # Check for conflicts
        conflict_found = False
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    # Remove an edge
                    e_to_remove = cycle[0]
                    del edge_colors[e_to_remove]
                    conflict_found = True
                    break
        if not conflict_found:
            break

    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    return result
