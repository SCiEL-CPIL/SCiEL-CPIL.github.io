import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    timeout = 110.0  # Leave some margin for safety
    
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: i for i, e in enumerate(edges)}
    
    # Generate all K4 subgraphs
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # Edges in this K4
                    e_ab = edge_index[(a, b)]
                    e_ac = edge_index[(a, c)]
                    e_ad = edge_index[(a, d)]
                    e_bc = edge_index[(b, c)]
                    e_bd = edge_index[(b, d)]
                    e_cd = edge_index[(c, d)]
                    k4s.append([e_ab, e_ac, e_ad, e_bc, e_bd, e_cd])
    
    # For each edge, store indices of K4s it belongs to
    edge_to_k4s = defaultdict(list)
    for k4_idx, k4_edges in enumerate(k4s):
        for e_idx in k4_edges:
            edge_to_k4s[e_idx].append(k4_idx)
            
    # State: colors of edges
    # We want to minimize number of colors used, subject to all K4s having >= 5 distinct colors.
    
    def count_colors(colors):
        return len(set(colors))
        
    def is_valid_k4(k4_edges, colors):
        c = count_colors([colors[e] for e in k4_edges])
        return c >= 5
        
    def count_violations(colors):
        violations = 0
        for k4_edges in k4s:
            if not is_valid_k4(k4_edges, colors):
                violations += 1
        return violations

    # Initial guess: Try to find a valid coloring with C_max colors
    # Start with a random assignment using C_max colors
    # We will try C_max = 15, then 14, then 13
    
    best_result = None
    best_colors_count = 100
    
    # Helper to run local search for a given C_max
    def local_search(C_max, max_iterations=50000):
        # Initialize with random colors in [0, C_max)
        colors = [random.randint(0, C_max - 1) for _ in range(len(edges))]
        
        # Calculate initial violations
        # For efficiency, we can maintain a list of colors for each K4? 
        # Or just recompute. 1820 K4s * 6 edges = ~11k ops per full eval. 
        # 50k iterations * 11k ops = 550M ops. Might be too slow in Python.
        
        # Optimized approach:
        # Maintain for each K4 a set of colors? 
        # When an edge changes, we only need to update the K4s containing that edge.
        # There are 15 K4s per edge.
        
        # Let's maintain a dictionary: k4_idx -> set of colors
        k4_color_sets = [set() for _ in k4s]
        for k4_idx, k4_edges in enumerate(k4s):
            for e_idx in k4_edges:
                k4_color_sets[k4_idx].add(colors[e_idx])
                
        violations = sum(1 for s in k4_color_sets if len(s) < 5)
        colors_used = set(colors)
        
        # If valid, return
        if violations == 0:
            return colors, len(colors_used)
            
        current_violations = violations
        current_colors_used = len(colors_used)
        
        # Simple Hill Climbing / Random Restart
        # We accept moves that reduce violations.
        # If violations == 0, we accept moves that reduce colors_used.
        
        for _ in range(max_iterations):
            if time.time() - start_time > timeout:
                break
                
            # Pick a random edge
            e_idx = random.randint(0, len(edges) - 1)
            old_color = colors[e_idx]
            
            # Pick a new color
            # To reduce colors used, we might pick a color that is already used frequently
            # Or just random
            new_color = random.randint(0, C_max - 1)
            if new_color == old_color:
                continue
                
            # Determine affected K4s
            affected_k4s = edge_to_k4s[e_idx]
            
            # Calculate change in violations
            delta_violations = 0
            for k4_idx in affected_k4s:
                old_set = k4_color_sets[k4_idx]
                old_size = len(old_set)
                
                # Create a copy of the set to simulate
                # Actually, we can just check if new_color is in old_set
                # If new_color is in old_set, the set doesn't change size.
                # If new_color is NOT in old_set, and old_color IS in old_set:
                #   We remove old_color. If old_color was unique to this K4, size decreases by 1.
                #   If old_color was used by other edges in this K4, size stays same.
                #   And we add new_color. Size increases by 1.
                
                # Simpler: Just recompute the set for these 15 K4s?
                # 15 * (6 edges) = 90 ops. Very fast.
                
                new_set = set()
                for e in k4s[k4_idx]:
                    if e == e_idx:
                        new_set.add(new_color)
                    else:
                        new_set.add(colors[e])
                        
                new_size = len(new_set)
                
                if old_size < 5 and new_size >= 5:
                    delta_violations -= 1
                elif old_size >= 5 and new_size < 5:
                    delta_violations += 1
                    
                # Store the new set for later update if accepted
                # We'll update after decision
                k4_color_sets[k4_idx] = new_set
                
            new_violations = current_violations + delta_violations
            
            # Calculate change in colors used
            # This is trickier to update incrementally without a global count per color.
            # Let's maintain a count of colors used globally.
            # color_counts[color] = number of edges with this color
            
            # We need to update color_counts
            # Let's do it outside the loop for simplicity, or maintain it.
            # Let's maintain color_counts.
            
            # For now, let's just check violations.
            # If violations becomes 0, we then try to minimize colors.
            
            if new_violations < current_violations:
                # Accept
                colors[e_idx] = new_color
                current_violations = new_violations
                # Update colors_used
                # We need to track which colors are still used.
                # If old_color count becomes 0, remove from used set.
                # If new_color count was 0, add to used set.
                # We need color_counts.
                
                # Let's restart the inner logic with color_counts.
                # This draft is getting complex. Let's simplify.
                
                # Reset k4_color_sets if we don't update them correctly?
                # Actually, I updated k4_color_sets[k4_idx] = new_set above.
                # But I didn't revert if rejected.
                # Let's restructure.
                
                break # Restart this function with better state management
                
            elif new_violations == current_violations:
                # Accept if it reduces colors used?
                # We need to know if it reduces colors used.
                # Let's just accept random moves if violations don't increase, hoping to find a valid one.
                # This is a random walk on the space of colorings.
                # If we find a valid one, we stop.
                
                # To be more effective, let's just accept if violations don't increase.
                colors[e_idx] = new_color
                current_violations = new_violations
                
            else:
                # Revert k4_color_sets
                for k4_idx in affected_k4s:
                    old_set = set()
                    for e in k4s[k4_idx]:
                        if e == e_idx:
                            old_set.add(old_color)
                        else:
                            old_set.add(colors[e])
                    k4_color_sets[k4_idx] = old_set
                    
        # Check if we found a valid one
        # We need to recompute violations from scratch to be sure, or trust the state
        # Let's recompute to be safe.
        final_violations = sum(1 for s in k4_color_sets if len(s) < 5)
        if final_violations == 0:
            return colors, len(set(colors))
        return None, None

    # The above local_search is flawed because it doesn't properly handle the "reduce colors" phase 
    # and the state update/revert logic is messy.
    # Let's write a cleaner version.
    
    def optimize_coloring(C_max, time_limit=5.0):
        # Initialize
        colors = [random.randint(0, C_max - 1) for _ in range(len(edges))]
        
        # Maintain k4_color_sets
        k4_color_sets = [set() for _ in k4s]
        for k4_idx, k4_edges in enumerate(k4s):
            for e_idx in k4_edges:
                k4_color_sets[k4_idx].add(colors[e_idx])
                
        violations = sum(1 for s in k4_color_sets if len(s) < 5)
        
        # If valid, try to minimize colors
        # If not valid, try to reduce violations
        
        # We will use a simple heuristic:
        # 1. While violations > 0:
        #    Pick a random violating K4.
        #    Pick a random edge in it.
        #    Change its color to a random color NOT in the current set of that K4 (to increase distinct colors).
        # 2. While violations == 0:
        #    Pick a random edge.
        #    Try to change its color to a color that is already used by another edge in the same K4s? 
        #    Or just try to merge colors: pick a color c1 that is used, pick an edge with c1, change it to c2 (another used color).
        #    If violations remain 0, accept.
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            if violations > 0:
                # Find a violating K4
                # Pick random K4 until we find a violating one? 
                # Or maintain a list of violating K4s?
                # For simplicity, pick random K4.
                k4_idx = random.randint(0, len(k4s) - 1)
                if len(k4_color_sets[k4_idx]) >= 5:
                    continue
                    
                # Pick a random edge in this K4
                e_idx = random.choice(k4s[k4_idx])
                old_color = colors[e_idx]
                
                # We want to increase the number of distinct colors in this K4.
                # So we should pick a new color that is NOT in k4_color_sets[k4_idx].
                # If we pick a color that IS in the set, the size might stay same or decrease (if old_color was unique).
                # So, pick a new color from [0, C_max) that is not in the set.
                current_colors_in_k4 = k4_color_sets[k4_idx]
                available_colors = [c for c in range(C_max) if c not in current_colors_in_k4]
                if not available_colors:
                    # All colors are used in this K4. We can't increase distinct count by changing one edge?
                    # If we change old_color to new_color (which is in set), and old_color is unique, size decreases.
                    # If old_color is not unique, size stays same.
                    # So we are stuck for this K4?
                    # Maybe change an edge to a color NOT in set? But none available.
                    # This implies C_max is too small for this K4?
                    # A K4 has 6 edges. If C_max=5, max distinct colors is 5.
                    # If C_max=5, it's possible to have 5 distinct colors.
                    # If we have <5, it means we have duplicates.
                    # To increase distinct count, we need to replace a duplicate with a new color.
                    # If all colors 0..C_max-1 are present, we can't add a new one.
                    # So we must replace a duplicate with another existing color? No, that doesn't help.
                    # We must replace a duplicate with a color that makes the set larger?
                    # If set is {1, 2, 3}, and we have edges colored 1, 1, 2, 3, 4, 5 (C_max=6).
                    # Set is {1,2,3,4,5}. Size 5. Valid.
                    # If set is {1, 2, 3}, size 3. Edges: 1, 1, 2, 2, 3, 3.
                    # Change one 1 to 4. Set becomes {1, 2, 3, 4}. Size 4.
                    # So we need a color NOT in the set.
                    # If C_max is small, maybe no such color exists?
                    # If C_max=5, and set is {1,2,3,4,5}, size 5.
                    # If set is {1,2,3}, and C_max=3. No color outside set.
                    # So we can't fix it?
                    # This suggests C_max must be at least 5.
                    continue
                
                new_color = random.choice(available_colors)
                
                # Apply change
                colors[e_idx] = new_color
                
                # Update k4_color_sets for all K4s containing e_idx
                for k4_idx_2 in edge_to_k4s[e_idx]:
                    old_set = k4_color_sets[k4_idx_2]
                    # Remove old_color if no other edge in this K4 has it
                    # This is expensive to check.
                    # Let's just recompute the set for this K4.
                    new_set = set()
                    for e in k4s[k4_idx_2]:
                        new_set.add(colors[e])
                    k4_color_sets[k4_idx_2] = new_set
                    
                # Recalculate violations? 
                # We can just update the global violation count by checking the affected K4s.
                # But we need to know the old violation status of each affected K4.
                # Let's just recompute total violations. 1820 sets. Fast enough?
                # 1820 * (check len < 5) is very fast.
                violations = sum(1 for s in k4_color_sets if len(s) < 5)
                
            else:
                # Violations == 0. Try to reduce colors used.
                # Pick a random edge.
                e_idx = random.randint(0, len(edges) - 1)
                old_color = colors[e_idx]
                
                # Try to change to a color that is already used in the same K4s?
                # Or just pick a random color and see if it reduces the global color count.
                # To reduce global color count, we should change old_color to new_color 
                # such that new_color is already used, and old_color becomes unused?
                # Or just reduce the count of old_color.
                
                # Let's pick a new_color randomly from the currently used colors.
                used_colors = list(set(colors))
                if len(used_colors) <= 1:
                    break # Can't reduce further
                
                new_color = random.choice(used_colors)
                if new_color == old_color:
                    continue
                    
                # Apply change
                colors[e_idx] = new_color
                
                # Update k4_color_sets
                for k4_idx_2 in edge_to_k4s[e_idx]:
                    new_set = set()
                    for e in k4s[k4_idx_2]:
                        new_set.add(colors[e])
                    k4_color_sets[k4_idx_2] = new_set
                    
                # Check if still valid
                new_violations = sum(1 for s in k4_color_sets if len(s) < 5)
                if new_violations > 0:
                    # Revert
                    colors[e_idx] = old_color
                    for k4_idx_2 in edge_to_k4s[e_idx]:
                        new_set = set()
                        for e in k4s[k4_idx_2]:
                            new_set.add(colors[e])
                        k4_color_sets[k4_idx_2] = new_set
                    # violations remains 0
                else:
                    violations = new_violations
                    # Accepted. Colors used might have decreased.
                    
        # Final check
        final_violations = sum(1 for s in k4_color_sets if len(s) < 5)
        if final_violations == 0:
            return colors, len(set(colors))
        return None, None

    # Try C_max = 15
    # If we find a solution with 15 colors, score is 17/15 = 1.133
    # If we find a solution with 14 colors, score is 17/14 = 1.214
    # If we find a solution with 13 colors, score is 17/13 = 1.307
    
    # We will try to find a solution with 15 colors first.
    # If time permits, try 14.
    
    # To increase the chance of finding a solution, we can run multiple restarts.
    
    best_c = 17 # Seed uses 17
    best_colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Try 15
    for attempt in range(5):
        if time.time() - start_time > timeout - 10:
            break
        res_colors, res_c = optimize_coloring(15, time_limit=10.0)
        if res_colors is not None:
            if res_c < best_c:
                best_c = res_c
                best_colors = res_colors
            # If we found 15, we can try to reduce it further?
            # optimize_coloring already tries to reduce colors once valid.
            # But it might get stuck at 15.
            # We can run a second phase to try to reduce to 14.
            if res_c == 15:
                # Try to reduce to 14
                # We can just run optimize_coloring with C_max=14 starting from this state?
                # My optimize_coloring initializes randomly.
                # Let's modify it to accept an initial state.
                pass
                
    # Try 14
    for attempt in range(5):
        if time.time() - start_time > timeout - 5:
            break
        res_colors, res_c = optimize_coloring(14, time_limit=5.0)
        if res_colors is not None:
            if res_c < best_c:
                best_c = res_c
                best_colors = res_colors
                
    # Try 13
    for attempt in range(3):
        if time.time() - start_time > timeout - 2:
            break
        res_colors, res_c = optimize_coloring(13, time_limit=2.0)
        if res_colors is not None:
            if res_c < best_c:
                best_c = res_c
                best_colors = res_colors

    # If we didn't find anything better than 17, return seed
    if best_c >= 17:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    else:
        return best_colors

# Run construct to test? No, just define it.
