import time
import random
import itertools

def construct():
    deadline = time.time() + 110  # Leave 40s safety margin
    
    # Generate all pairs in lexicographic order
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    n_edges = len(pairs)  # 120
    
    # Map edge to index for quick lookup
    edge_index = {}
    for idx, (i, j) in enumerate(pairs):
        edge_index[(i, j)] = idx

    # Precompute all K4s (sets of 4 vertices) and their 6 edges
    k4s = list(itertools.combinations(range(16), 4))
    n_k4s = len(k4s)
    k4_edges = []
    for quad in k4s:
        edges = []
        for a, b in itertools.combinations(quad, 2):
            idx = edge_index[(a, b)]
            edges.append(idx)
        k4_edges.append(edges)
        
    # Precompute for each edge, the list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(n_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    # Initial coloring: Difference coloring (XOR)
    # Colors 0..14. (i ^ j) is 1..15, so (i ^ j) - 1 is 0..14.
    colours = [0] * n_edges
    for idx, (i, j) in enumerate(pairs):
        colours[idx] = (i ^ j) - 1
        
    def count_colors_in_k4(k4_idx):
        edges = k4_edges[k4_idx]
        return len(set(colours[e] for e in edges))
        
    def total_violations():
        v = 0
        for k4_idx in range(n_k4s):
            if count_colors_in_k4(k4_idx) < 5:
                v += 1
        return v
        
    def is_valid():
        for k4_idx in range(n_k4s):
            if count_colors_in_k4(k4_idx) < 5:
                return False
        return True

    # Phase 1: Simulated Annealing to find a valid 15-coloring
    # Start with XOR, which has many violations (parallelograms)
    best_colours = colours[:]
    best_violations = total_violations()
    
    T = 1.0
    cooling = 0.999
    
    # If XOR is already valid (unlikely, but check), skip to Phase 2
    if best_violations == 0:
        pass
    else:
        iterations = 0
        max_iterations = 200000 # Safety cap, time is primary limit
        while time.time() < deadline - 30 and iterations < max_iterations:
            iterations += 1
            # Pick a random edge
            e_idx = random.randint(0, n_edges - 1)
            old_c = colours[e_idx]
            new_c = random.randint(0, 14)
            
            if old_c == new_c:
                continue
                
            # Calculate delta in violations
            # Only K4s containing e_idx can change status
            delta = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Temporarily apply change
            colours[e_idx] = new_c
            
            # Calculate new violations for affected K4s
            # We need to know if a K4 went from valid to invalid or vice versa
            # To do this efficiently, we'd need to store the count per K4.
            # For simplicity in this rewrite, we'll just recalculate the affected ones.
            # 91 K4s per move is fast.
            
            # Optimization: Pre-compute counts? No, just re-evaluate.
            
            # We need the OLD violation status of these K4s.
            # We don't store it. So we must revert and check, or store.
            # Let's store a boolean array 'valid' for each K4.
            
            # Actually, let's just calculate the delta by checking the 91 K4s.
            # This requires knowing if they were valid before.
            # So we maintain an array `k4_valid` of size n_k4s.
            
            # Re-initializing for this loop structure is messy.
            # Let's use a simpler local search: Hill Climbing with random restarts or Tabu.
            
            # Let's stick to a simple local search that fixes violations.
            # Find a violated K4.
            violated = [k4_idx for k4_idx in range(n_k4s) if not (len(set(colours[e] for e in k4_edges[k4_idx])) >= 5)]
            
            if not violated:
                break
                
            k4_idx = random.choice(violated)
            edges = k4_edges[k4_idx]
            current_colors = set(colours[e] for e in edges)
            
            # Try to change one edge to a new color not in current_colors
            # to increase the count.
            fixed = False
            for e_idx in edges:
                old_c = colours[e_idx]
                # Try colors not in current_colors
                # If we pick a color in current_colors, count stays same or decreases.
                # We need to increase to >= 5.
                # Current count is < 5.
                # If we change an edge with unique color to a new color, count stays same.
                # If we change an edge with shared color to a new color, count increases by 1.
                
                # Identify if old_c is unique in this K4
                count_old = sum(1 for e in edges if colours[e] == old_c)
                
                # Try all 15 colors
                for new_c in range(15):
                    if new_c in current_colors:
                        # If we change to an existing color, the set of colors might shrink or stay same.
                        # If old_c was unique and we change to new_c (existing), we lose old_c and keep new_c. Size same.
                        # If old_c was shared, we might lose it if it was the last one? No, we change one edge.
                        # If count_old > 1, old_c remains. Size same.
                        # If count_old == 1, old_c is removed. Size decreases by 1 (if new_c was already there).
                        # So changing to existing color NEVER increases the count.
                        continue
                    
                    # new_c is NOT in current_colors.
                    # New set = (current_colors - {old_c if count_old==1}) U {new_c}
                    # If count_old == 1: Size = |current_colors| - 1 + 1 = |current_colors|. No increase.
                    # If count_old > 1: Size = |current_colors| + 1. Increase!
                    
                    if count_old > 1:
                        # This change increases the color count for this K4.
                        # Apply it and check if it breaks other K4s?
                        # We accept it and let the loop handle others.
                        colours[e_idx] = new_c
                        fixed = True
                        break
                if fixed:
                    break
                    
            if not fixed:
                # If no single edge change works (e.g. all colors are unique in the K4, so count is 6? No, we picked violated, so count < 5.
                # If count < 5, there must be duplicates. So some color has count > 1.
                # So there must be a fix.
                # Unless we are stuck in a state where fixing one breaks another?
                # The loop continues, so it will try other K4s.
                pass

    # Phase 2: Minimize colors
    # Check if valid
    if not is_valid():
        # Fallback to 17
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    best_colours = colours[:]
    best_num_colors = len(set(colours))
    
    # Try to merge colors to reduce count
    # Current colors are 0..14.
    # We want to see if we can map some colors to others.
    
    # Heuristic: Try to merge pairs of colors that appear on few edges or are "similar".
    # Simply try all pairs? 15*14/2 = 105 pairs.
    # For each pair (c1, c2), map c2 -> c1. Check validity.
    # If valid, accept and continue.
    
    used_colors = set(colours)
    list_used = list(used_colors)
    
    # Sort by frequency?
    freq = {c: colours.count(c) for c in list_used}
    # Try to merge rare colors first?
    list_used.sort(key=lambda c: freq[c])
    
    # We want to reduce the set size.
    # Iterate until no more merges possible.
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        current_used = set(colours)
        current_list = list(current_used)
        # Sort by frequency ascending
        current_list.sort(key=lambda c: colours.count(c))
        
        for i in range(len(current_list)):
            c1 = current_list[i]
            for j in range(i+1, len(current_list)):
                c2 = current_list[j]
                # Try mapping c2 to c1
                # Save old state
                old_colours = colours[:]
                
                for e in range(n_edges):
                    if colours[e] == c2:
                        colours[e] = c1
                        
                if is_valid():
                    changed = True
                    break
                else:
                    colours = old_colours[:]
            if changed:
                break
                
    # Final result
    final_colours = colours
    num_colors = len(set(final_colours))
    
    # If we ended up with > 15 colors (shouldn't happen), fallback
    if num_colors > 15:
         return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
         
    return final_colours

# EVOLVE-BLOCK-END
