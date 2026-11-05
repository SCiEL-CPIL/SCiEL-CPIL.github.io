import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    
    # Generate all pairs in lexicographic order
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    n_edges = len(pairs)  # 120
    
    # Map edge to index for quick lookup
    edge_index = {}
    for idx, (i, j) in enumerate(pairs):
        edge_index[(i, j)] = idx
        
    # Precompute all K4s (sets of 4 vertices) and their 6 edges
    k4s = list(itertools.combinations(range(16), 4))
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
            
    n_k4s = len(k4_edges)

    def get_violation_count(colours):
        count = 0
        for k4_idx in range(n_k4s):
            edges = k4_edges[k4_idx]
            if len(set(colours[e] for e in edges)) < 5:
                count += 1
        return count

    def get_violations(colours):
        violated = set()
        for k4_idx in range(n_k4s):
            edges = k4_edges[k4_idx]
            if len(set(colours[e] for e in edges)) < 5:
                violated.add(k4_idx)
        return violated

    def local_search(colours, max_colors, deadline, max_iterations=200000):
        violated = get_violations(colours)
        if not violated:
            return True, colours
            
        steps = 0
        no_improve = 0
        min_violations = len(violated)
        
        while violated and steps < max_iterations and time.time() < deadline:
            steps += 1
            
            # Pick a random violated K4
            k4_idx = random.choice(list(violated))
            edges = k4_edges[k4_idx]
            
            # Current colors in this K4
            current_colors = set(colours[e] for e in edges)
            
            # We need to change one edge to a color NOT in current_colors
            # to increase the distinct color count.
            # If we change edge e with color c_old to c_new not in current_colors:
            # If c_old is unique in this K4, the set size stays same (lose c_old, gain c_new).
            # If c_old is not unique, the set size increases by 1 (keep c_old, gain c_new).
            # So we prefer edges with non-unique colors.
            
            color_counts = {}
            for e in edges:
                c = colours[e]
                color_counts[c] = color_counts.get(c, 0) + 1
            
            candidate_edges = []
            for e in edges:
                c = colours[e]
                if color_counts[c] > 1:
                    candidate_edges.append(e)
            
            if not candidate_edges:
                # All colors unique? Then len(current_colors) = 6. Not violated.
                # This case shouldn't happen if violated.
                # But if it does, pick any edge.
                candidate_edges = edges
                
            e = random.choice(candidate_edges)
            
            # Pick a new color not in current_colors
            available_colors = [c for c in range(max_colors) if c not in current_colors]
            if not available_colors:
                # If max_colors <= len(current_colors), we can't add a new color.
                # But we need to fix violation.
                # If max_colors is small, this might be impossible?
                # For 13 colors, len(current_colors) <= 4, so available is large.
                continue
                
            new_color = random.choice(available_colors)
            
            # Apply change
            old_color = colours[e]
            colours[e] = new_color
            
            # Update violations for all K4s containing edge e
            for k4 in edge_to_k4s[e]:
                was_violated = k4 in violated
                # Check if still violated
                edges_k4 = k4_edges[k4]
                if len(set(colours[ek] for ek in edges_k4)) < 5:
                    is_now_violated = True
                else:
                    is_now_violated = False
                
                if was_violated and not is_now_violated:
                    violated.discard(k4)
                elif not was_violated and is_now_violated:
                    violated.add(k4)
            
            # Track progress
            if len(violated) < min_violations:
                min_violations = len(violated)
                no_improve = 0
            else:
                no_improve += 1
                
            # Restart local search if stuck for too long
            if no_improve > 5000:
                break

        return len(violated) == 0, colours

    best_colours = None
    best_colors_used = 100

    # Try max_colors from 13 to 15
    for max_colors in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
            
        # Multiple trials
        for trial in range(50):
            if time.time() > deadline - 5:
                break
                
            colours = [random.randint(0, max_colors - 1) for _ in range(n_edges)]
            
            success, colours = local_search(colours, max_colors, deadline - 5)
            
            if success:
                # Verify
                if not get_violations(colours):
                    # Count unique colors used
                    unique_used = len(set(colours))
                    if unique_used < best_colors_used:
                        best_colors_used = unique_used
                        best_colours = colours
                    # If we found a 13-color solution, we can stop early?
                    # Or keep searching for better? 13 is the lower bound.
                    if max_colors == 13:
                        return colours
                    # If we found 14, we might try to improve to 13?
                    # But time is limited. Let's just return the first valid one found in the 13-15 loop.
                    # Actually, the loop goes 13, 14, 15.
                    # If we find a valid 13, we return.
                    # If we find a valid 14, we save it and try 13 again? No, we already tried 13.
                    # So if we are at 14, we should return it if 13 failed.
                    return colours
    
    # Fallback: return best found or a 17-color solution
    if best_colours:
        return best_colours
        
    # Simple 17-color solution
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
