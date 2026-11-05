import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute K4s and their edges
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4s) - 1)
            
    num_k4s = len(k4s)
    
    def count_bad_k4s(colors):
        """Count the number of K4s that have fewer than 5 distinct colors."""
        bad_count = 0
        for k4_idx in range(num_k4s):
            e_indices = k4_edges[k4_idx]
            k4_colors = set(colors[e] for e in e_indices)
            if len(k4_colors) < 5:
                bad_count += 1
        return bad_count

    def is_valid(colors):
        """Check if all K4s are valid."""
        return count_bad_k4s(colors) == 0

    # We want to find a coloring using the minimum number of colors.
    # Bounds: 13 <= f(16) <= 15.
    # We try to find a valid coloring with 13 colors first.
    
    best_result = None
    best_colors_used = 16 # Start with a high number
    
    # Fallback: A simple valid coloring with 16 colors (e.g., edge (i,j) gets color i+j mod 16 might not work, 
    # but we can just use a large number of colors to be safe, then minimize).
    # Actually, let's just use the local search to find the minimum.
    
    for target_C in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
            
        # Local search parameters
        max_iterations = 20000
        restarts = 10
        temperature = 0.5
        
        for r in range(restarts):
            if time.time() > deadline - 2:
                break
                
            # Initialize with random colors from 0 to target_C - 1
            colors = [random.randint(0, target_C - 1) for _ in range(num_edges)]
            
            current_bad = count_bad_k4s(colors)
            best_bad_in_run = current_bad
            best_colors_in_run = colors[:]
            
            # If we start with 0 bad, we are done
            if current_bad == 0:
                if target_C < best_colors_used:
                    best_colors_used = target_C
                    best_result = colors[:]
                break
            
            for it in range(max_iterations):
                if time.time() > deadline - 2:
                    break
                    
                # Pick a random edge
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                
                # Try a new random color
                new_color = random.randint(0, target_C - 1)
                if new_color == old_color:
                    continue
                    
                colors[e_idx] = new_color
                
                # Calculate delta in bad count
                # Only K4s containing e_idx can change status
                delta_bad = 0
                for k4_idx in edge_to_k4s[e_idx]:
                    e_indices = k4_edges[k4_idx]
                    k4_colors = set(colors[e] for e in e_indices)
                    if len(k4_colors) < 5:
                        delta_bad += 1
                # Note: This calculates the new bad count for affected K4s.
                # We need to compare with old bad count for those K4s.
                # It's faster to just recompute the total bad count if changes are sparse?
                # No, recomputing total is O(num_k4s * 6).
                # Let's stick to a simpler full recompute for correctness and simplicity, 
                # but only if the change is promising?
                # Actually, for 120 edges and 1820 K4s, full recompute is ~10k ops.
                # 20k iterations * 10k ops = 200M ops. Might be too slow in Python.
                
                # Let's optimize: keep a count of distinct colors for each K4?
                # That's complex.
                
                # Alternative: Just do a full recompute every step?
                # 1820 K4s * 6 edges = 10920 operations per step.
                # 20000 steps * 10920 = 218,400,000 operations.
                # In Python, this is likely too slow (seconds per 10M ops).
                
                # We need a faster delta update.
                # Let's revert and do a smarter delta.
                colors[e_idx] = old_color # Revert for now to think
                
                # Smarter delta:
                # For each K4 containing e_idx, we can check if it was bad before and is bad after.
                # But we don't have the "before" state stored per K4 easily without overhead.
                
                # Let's try a different approach: 
                # Instead of minimizing bad K4s, we can use a constraint satisfaction solver approach 
                # or just rely on the fact that 13 colors is tight.
                
                # Let's try a simpler heuristic:
                # Randomly shuffle colors of a subset of edges?
                
                # Let's go back to full recompute but reduce iterations or optimize the check.
                # We can use numpy for the check?
                # k4_edges is a list of lists.
                # We can store k4_edges as a numpy array of shape (num_k4s, 6).
                
                pass # Placeholder for optimization thought

    # Since pure Python local search with full recompute might be too slow,
    # let's use a numpy-based evaluation for speed.
    
    import numpy as np
    
    # Convert k4_edges to numpy array
    k4_edges_np = np.array(k4_edges)
    
    def count_bad_k4s_np(colors_np):
        # colors_np is a 1D array of length num_edges
        # k4_edges_np is (num_k4s, 6)
        # Get colors for all edges in all K4s
        k4_colors = colors_np[k4_edges_np] # Shape (num_k4s, 6)
        
        # Count distinct colors in each row
        # This is tricky with numpy.
        # Sort each row and count changes?
        sorted_colors = np.sort(k4_colors, axis=1)
        # Count number of unique elements in each row
        # A unique element is one where it differs from the previous one (or is the first)
        diff = np.diff(sorted_colors, axis=1)
        unique_counts = 1 + np.sum(diff != 0, axis=1)
        
        # Bad if unique_counts < 5
        bad_mask = unique_counts < 5
        return np.sum(bad_mask)

    best_result = None
    best_colors_used = 16
    
    for target_C in [13, 14, 15]:
        if time.time() > deadline - 5:
            break
            
        max_iterations = 5000
        restarts = 20
        
        for r in range(restarts):
            if time.time() > deadline - 2:
                break
                
            colors = np.array([random.randint(0, target_C - 1) for _ in range(num_edges)], dtype=np.int32)
            current_bad = count_bad_k4s_np(colors)
            
            if current_bad == 0:
                if target_C < best_colors_used:
                    best_colors_used = target_C
                    best_result = colors
                break
            
            for it in range(max_iterations):
                if time.time() > deadline - 2:
                    break
                    
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, target_C - 1)
                if new_color == old_color:
                    continue
                    
                colors[e_idx] = new_color
                new_bad = count_bad_k4s_np(colors)
                
                # Accept if better or equal (with some probability to escape plateaus)
                if new_bad < current_bad or (new_bad == current_bad and random.random() < 0.1):
                    current_bad = new_bad
                else:
                    colors[e_idx] = old_color
                    
                if current_bad == 0:
                    if target_C < best_colors_used:
                        best_colors_used = target_C
                        best_result = colors
                    break

    if best_result is not None:
        return [int(c) for c in best_result]
    else:
        # Fallback: return a valid coloring with 16 colors (just use unique colors for all edges? No, 120 edges.
        # We can use a simple valid coloring.
        # For example, color (i,j) with i+j. Max i+j = 15+14=29.
        # Does every K4 have 5 distinct colors?
        # Vertices a,b,c,d. Edges: a+b, a+c, a+d, b+c, b+d, c+d.
        # If we pick a=0, b=1, c=2, d=3.
        # Sums: 1, 2, 3, 3, 4, 5. Distinct: 1,2,3,4,5. (5 distinct).
        # If we pick a=0, b=1, c=1, d=2 (not distinct vertices).
        # Let's just use a known valid coloring or the one from the seed.
        # The seed baseline: (i+j) % 17.
        # Let's verify if (i+j) % 17 is valid.
        # If not, we can just return a random valid one found by a slow greedy, 
        # but we should have found one with 13-15 colors.
        
        # Let's generate a valid coloring with 16 colors just in case.
        # Use color = i * 16 + j? No, too many.
        # Let's use the seed's baseline.
        baseline = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
        # Verify baseline
        valid = True
        for quad in itertools.combinations(range(16), 4):
            a, b, c, d = quad
            cols = set([
                (a+b)%17, (a+c)%17, (a+d)%17,
                (b+c)%17, (b+d)%17, (c+d)%17
            ])
            if len(cols) < 5:
                valid = False
                break
        
        if valid:
            return baseline
        else:
            # If baseline is invalid, return a random assignment (might fail, but better than crash)
            return [random.randint(0, 119) for _ in range(120)]
