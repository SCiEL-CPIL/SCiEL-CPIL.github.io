import time
import random
import numpy as np
from itertools import combinations

# Precompute cycles for a given n
# Returns list of cycles, each cycle is a tuple of 6 edge indices
# Also returns mapping from edge index to list of cycle indices
def precompute_cycles(n):
    edges = []
    edge_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = len(edges)
            edges.append((i, j))
    
    E = len(edges)
    cycles = []
    # Generate all 6-cycles
    # A 6-cycle is defined by 6 vertices v0..v5 and the edges (v0,v1), (v1,v2), ..., (v5,v0)
    # We must avoid duplicates.
    # We can iterate over combinations of 6 vertices.
    # For each set of 6 vertices, there are 60 distinct cycles (6!/12).
    # 6! = 720. 12 symmetries (6 rotations * 2 directions).
    
    for verts in combinations(range(n), 6):
        # Generate all permutations of these 6 vertices
        # To avoid duplicates, we can fix v0 = min(verts) and v1 < v5?
        # Simpler: generate all permutations, check canonical form?
        # Or just generate all and use a set of frozensets of edges?
        # Number of perms 720. 720 * 38760 = 27M. Too slow in pure Python.
        
        # Better:
        # Fix v0 = verts[0].
        # Permute remaining 5. 5! = 120.
        # For each perm, check if it's a valid cycle (it is).
        # But we might generate the same cycle with different rotations/reflections.
        # We can enforce: v0 < v1 and v0 < v5?
        # Actually, just generate all 120 perms for fixed v0.
        # Then check if the cycle is "canonical".
        # A cycle is canonical if v0 is the smallest vertex, and v1 < v5.
        
        v0 = verts[0]
        rest = list(verts[1:])
        for perm in permutations(rest):
            v1, v2, v3, v4, v5 = perm
            # Check canonical: v0 < v1 and v0 < v5
            if v1 < v5:
                # Check if v0 is smallest (it is, by construction)
                # This generates each cycle exactly once?
                # Yes, for a fixed set of 6 vertices, there are 60 cycles.
                # 120 perms / 2 (v1<v5 condition) = 60.
                # So this works.
                
                e0 = edge_map[(v0, v1)]
                e1 = edge_map[(v1, v2)]
                e2 = edge_map[(v2, v3)]
                e3 = edge_map[(v3, v4)]
                e4 = edge_map[(v4, v5)]
                e5 = edge_map[(v5, v0)]
                
                cycles.append((e0, e1, e2, e3, e4, e5))
    
    # Build edge to cycles map
    edge_to_cycles = [[] for _ in range(E)]
    for idx, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(idx)
            
    return cycles, edge_to_cycles, E

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_n = 9
    best_colors = None
    # Seed solution for n=9
    n_seed = 9
    best_colors = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    
    # Try to find n=17
    for target_n in [17, 18]:
        if time.time() > deadline - 20:
            break
            
        # Precompute
        cycles, edge_to_cycles, E = precompute_cycles(target_n)
        
        # Search
        found = False
        # We do multiple restarts
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Init random coloring
            colors = np.random.randint(0, 4, E)
            
            # Init counts
            # count[c][e] = number of cycles containing e where other 5 edges are color c
            counts = np.zeros((4, E), dtype=np.int32)
            penalty = 0
            
            # Calculate initial counts and penalty
            # For each cycle:
            #   Get colors of its 6 edges
            #   If all same, penalty += 1
            #   For each edge e in cycle:
            #       Get colors of other 5 edges
            #       If all same color c, counts[c][e] += 1
            
            # Optimization: Process cycles in vectorized way?
            # M is ~2.3M. Looping in Python is slow.
            # Let's try to optimize the inner loop.
            
            # We can pre-store the edges of each cycle in a NumPy array
            # cycle_edges: shape (M, 6)
            cycle_edges_arr = np.array(cycles)
            M = len(cycles)
            
            # Get colors for all edges in all cycles
            # colors_cycle: shape (M, 6)
            colors_cycle = colors[cycle_edges_arr]
            
            # Check for monochromatic cycles
            # A cycle is mono if all 6 colors are same.
            # min == max
            is_mono = (np.min(colors_cycle, axis=1) == np.max(colors_cycle, axis=1))
            penalty = np.sum(is_mono)
            
            # Update counts
            # For each cycle, for each of the 6 positions, 
            # if the other 5 are same, increment count for that color at that edge.
            
            # This is hard to vectorize fully without memory blowup.
            # M=2.3M, 6 positions. 13.8M updates.
            # We can do it in a loop over cycles, but that's slow.
            # Let's try a hybrid:
            
            # For each position k in 0..5:
            #   other_colors = colors_cycle with column k removed
            #   is_other_mono = (min(other_colors, axis=1) == max(other_colors, axis=1))
            #   mono_color = min(other_colors, axis=1) [where is_other_mono]
            #   edges_in_pos_k = cycle_edges_arr[:, k]
            #   Use np.add.at to update counts
            
            for k in range(6):
                other_cols = np.delete(colors_cycle, k, axis=1)
                min_o = np.min(other_cols, axis=1)
                max_o = np.max(other_cols, axis=1)
                is_other_mono = (min_o == max_o)
                # Only update where is_other_mono is True
                if np.any(is_other_mono):
                    mono_colors = min_o[is_other_mono]
                    edges = cycle_edges_arr[is_other_mono, k]
                    # np.add.at(counts, (mono_colors, edges), 1)
                    # This is slow for 2M elements.
                    # Alternative: use bincount?
                    # We need to increment counts[c][e] for pairs (c, e).
                    # We can flatten (c, e) into index c*E + e.
                    flat_indices = mono_colors * E + edges
                    # Count occurrences
                    bin_counts = np.bincount(flat_indices, minlength=4*E)
                    # Add to counts
                    counts_flat = counts.flatten()
                    counts_flat += bin_counts
                    counts = counts_flat.reshape(4, E)
            
            # Local Search
            # While penalty > 0
            step_limit = 5000
            for step in range(step_limit):
                if time.time() > deadline - 5:
                    break
                if penalty == 0:
                    found = True
                    best_n = target_n
                    best_colors = colors.tolist()
                    break
                
                # Find best move
                # For each edge e, for each new color c != colors[e]:
                #   delta = counts[c][e] - counts[colors[e]][e]
                #   We want to minimize delta (most negative)
                
                # Vectorized:
                # current_colors: shape (E,)
                # counts: shape (4, E)
                # delta_matrix[c][e] = counts[c][e] - counts[current_colors[e]][e]
                
                # We can compute this for all c, e.
                # But we must exclude c == current_colors[e]
                
                # Let's just pick a random edge with high penalty contribution?
                # Or pick the edge that minimizes delta.
                
                # To save time, we can sample edges.
                # Or compute full delta matrix (4 x E = 760 elements). Fast.
                
                curr_c = colors
                # counts_curr: shape (E,) = counts[curr_c, e]
                counts_curr = np.array([counts[c, e] for c, e in zip(curr_c, range(E))])
                # This loop is slow.
                # Better:
                # counts_curr = counts[np.arange(E), curr_c]
                
                counts_curr = counts[np.arange(E), curr_c]
                
                # delta[c, e] = counts[c, e] - counts_curr[e]
                # We want min over c != curr_c[e]
                
                # Create a matrix of deltas
                # Deltas shape (4, E)
                deltas = counts - counts_curr[None, :]
                
                # Mask out c == curr_c[e]
                # mask[c, e] = (c != curr_c[e])
                mask = np.arange(4)[:, None] != curr_c[None, :]
                deltas[~mask] = np.inf
                
                # Find min delta and corresponding edge, color
                flat_idx = np.argmin(deltas)
                new_c = flat_idx // E
                e = flat_idx % E
                
                # If min delta is inf, no move possible? (Shouldn't happen)
                if deltas[flat_idx] == np.inf:
                    break
                
                # Apply move
                old_c = curr_c[e]
                if new_c == old_c:
                    # This can happen if mask logic failed, but it shouldn't.
                    # If it does, pick random other color.
                    new_c = (old_c + 1) % 4
                    # Recalc delta?
                    # Just do it.
                    pass
                
                # Update penalty
                # penalty += counts[new_c][e] - counts[old_c][e]
                penalty += counts[new_c][e] - counts[old_c][e]
                
                # Update colors
                colors[e] = new_c
                
                # Update counts
                # This is the expensive part.
                # For each cycle containing e:
                #   Check if "other 5" uniformity changed.
                
                # Get cycles containing e
                cyc_indices = edge_to_cycles[e]
                
                # For each such cycle:
                #   Get the cycle's edges
                #   Get the positions of e in the cycle
                #   Actually, we just need to update counts for the other 5 edges in that cycle.
                
                for ci in cyc_indices:
                    cyc = cycle_edges_arr[ci]
                    # Find index of e in cyc
                    # cyc is array of 6 edges.
                    pos = np.where(cyc == e)[0][0]
                    
                    # Get other 5 edges
                    other_edges = np.delete(cyc, pos)
                    
                    # Get their current colors (before update? No, e is already updated in colors array)
                    # Wait, we need to know the state BEFORE the update to decrement, and AFTER to increment.
                    # We updated colors[e] already.
                    # So "After" state is in `colors`.
                    # "Before" state: e was old_c.
                    
                    # After state:
                    # Colors of other 5 edges:
                    other_colors_after = colors[other_edges]
                    min_a = np.min(other_colors_after)
                    max_a = np.max(other_colors_after)
                    if min_a == max_a:
                        c_after = min_a
                        # Increment counts[c_after][other_edges]
                        # But we must be careful: we are incrementing for EACH other edge?
                        # No.
                        # The count `counts[c][e']` is defined for a specific edge e'.
                        # For a cycle C, it contributes to `counts[c][e']` for each e' in C
                        # IF the other 5 edges are color c.
                        # So for this cycle C, if the other 5 edges (w.r.t e') are all c,
                        # then counts[c][e'] should be incremented.
                        
                        # So for each e' in other_edges:
                        #   Check if the other 5 edges w.r.t e' are all c.
                        #   The other 5 w.r.t e' are: e and the other 4 in other_edges.
                        #   We know e is new_c.
                        #   So we need: new_c == c AND the other 4 are c.
                        
                        # This is getting complicated to vectorize per cycle.
                        # Given the time, we might skip the exact count update and just recompute?
                        # No, recompute is too slow.
                        
                        # Let's approximate:
                        # We only update counts for the edges in this cycle.
                        # For each e' in other_edges:
                        #   The "other 5" for e' are: e and other_edges except e'.
                        #   Let's call this set S.
                        #   Colors of S: colors[e] (which is new_c) and colors[other_edges except e'].
                        #   If all are same color c, then counts[c][e'] += 1.
                        #   Before the flip, colors[e] was old_c.
                        #   If all were same color c_old, then counts[c_old][e'] -= 1.
                        
                        # We can compute this for the 5 edges in this cycle.
                        
                        # After:
                        for j, e_prime in enumerate(other_edges):
                            # Other 5 for e_prime:
                            # e and other_edges except e_prime
                            other_5_edges = np.concatenate(([e], np.delete(other_edges, j)))
                            cols = colors[other_5_edges]
                            if np.min(cols) == np.max(cols):
                                c_new = cols[0]
                                counts[c_new][e_prime] += 1
                                
                        # Before:
                        # We need to revert colors[e] temporarily?
                        # Or just use old_c.
                        for j, e_prime in enumerate(other_edges):
                            other_5_edges = np.concatenate(([e], np.delete(other_edges, j)))
                            # Replace e's color with old_c in this check
                            # other_5_edges contains e.
                            # We can just check:
                            # cols_before = colors[other_5_edges]
                            # But colors[e] is new_c now.
                            # So cols_before = np.where(other_5_edges == e, old_c, colors[other_5_edges])
                            cols_b = colors[other_5_edges].copy()
                            cols_b[other_5_edges == e] = old_c
                            if np.min(cols_b) == np.max(cols_b):
                                c_old = cols_b[0]
                                counts[c_old][e_prime] -= 1

            if found:
                break

    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    else:
        # Fallback to seed
        return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}

# Need permutations import
from itertools import permutations
