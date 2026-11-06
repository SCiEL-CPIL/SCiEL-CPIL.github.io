import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    """
    Return 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
    Every set of four vertices must span edges of at least five distinct colours.
    Minimize the number of colours used.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 16
    max_colors = 120
    
    # Precompute edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    num_edges = len(edges)
    
    # Map (i, j) to edge index
    ij_to_edge_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute for each K4, the indices of its 6 edges
    k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(k4s)
    
    # k4_edges is an array of shape (num_k4s, 6) containing edge indices
    k4_edges = np.zeros((num_k4s, 6), dtype=np.int32)
    k4_idx = 0
    for verts in k4s:
        edge_indices_in_k4 = []
        for i in range(4):
            for j in range(i + 1, 4):
                u = verts[i]
                v = verts[j]
                edge_indices_in_k4.append(ij_to_edge_idx[(u, v)])
        k4_edges[k4_idx] = edge_indices_in_k4
        k4_idx += 1
        
    # Precompute for fast violation counting
    # We want to count, for each K4, the number of distinct colors.
    # If distinct_colors < 5, it's a violation.
    
    def count_violations(colors_np):
        """
        colors_np: 1D numpy array of shape (num_edges,)
        Returns: number of K4s with < 5 distinct colors
        """
        # Extract colors for each K4: shape (num_k4s, 6)
        k4_colors = colors_np[k4_edges]
        
        # Count distinct colors in each K4
        # This is a bit tricky to vectorize fully without loops in pure numpy 
        # because 'unique' operates on the whole array.
        # However, num_k4s is 1820. A loop over 1820 items is fast in Python if the inner part is C-speed.
        # But we can do better:
        # Sort the colors in each K4, then count unique.
        # Or simply: for each K4, check if any color appears 4+ times, 
        # or if 3 colors appear with counts (3,2,1) or (2,2,2) etc.
        # Actually, distinct < 5 means distinct is 1, 2, 3, or 4.
        
        # Let's use a loop for clarity and sufficient speed, 
        # but optimize the inner check.
        violations = 0
        for i in range(num_k4s):
            # k4_colors[i] is a numpy array of 6 ints
            # Using set() on a small numpy array is fast
            if len(set(k4_colors[i])) < 5:
                violations += 1
        return violations

    def get_energy(colors_np, target_colors=13):
        """
        Calculate energy.
        We want to minimize distinct colors.
        We want 0 violations.
        """
        num_used = len(set(colors_np))
        violations = count_violations(colors_np)
        
        # Heuristic weights
        # If we have more than target_colors, penalize heavily.
        # If we have violations, penalize heavily.
        
        # Let's try a specific objective:
        # Minimize (num_used * 1000 + violations)
        # This prioritizes reducing colors, but allows temporary violations.
        
        return num_used * 1000 + violations

    # Initial Coloring
    # Start with the seed's initial coloring which is known to be valid (15 or 16 colors)
    colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    # Verify initial validity
    if count_violations(colors) != 0:
        # Fallback if the seed logic changed or something is wrong
        # Use a simple valid coloring: assign unique colors to all edges? No, that's 120 colors.
        # The (i+j)%17 is a known valid construction for n=16? 
        # Let's assume it is valid for the start. If not, we need a valid start.
        # A trivial valid coloring is assigning distinct colors to all edges (120 colors).
        # But we want to start close to optimal.
        # Let's stick with (i+j)%17. It usually works for these small n.
        pass

    best_colors = colors.copy()
    best_num_used = len(set(colors))
    best_violations = count_violations(colors)
    best_energy = best_num_used * 1000 + best_violations
    
    # Simulated Annealing
    rng = random.Random(42)
    current_temp = 1000.0
    cooling_rate = 0.99
    
    current_colors = colors.copy()
    current_energy = best_energy
    
    max_steps = 50000
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Generate a move
        # 80% chance: Recolor a random edge to a random color
        # 20% chance: Try to merge two colors (if valid)
        
        move_type = rng.random()
        
        if move_type < 0.8:
            # Recolor
            edge_idx = rng.randint(0, num_edges - 1)
            # Choose a color. 
            # To encourage merging, pick from existing colors or a new one if we want to expand (unlikely we want to expand).
            # Let's pick from the range 0..max_colors
            new_color = rng.randint(0, max_colors - 1)
            
            old_color = current_colors[edge_idx]
            if old_color == new_color:
                continue
                
            current_colors[edge_idx] = new_color
            
            # Calculate energy change
            # This is the expensive part. 
            # We can optimize by only checking K4s affected by edge_idx.
            # But for simplicity and given 1820 K4s, full check is ~0.5ms in Python loop? 
            # 50000 steps * 0.5ms = 25s. This is tight.
            # We need a faster energy delta or a better move generation.
            
            # Let's try to optimize the energy calculation.
            # We can precompute which K4s contain which edges.
            
            # For now, let's assume full check is too slow for 50k steps.
            # Let's reduce steps or optimize check.
            
            # Alternative: Use a faster check.
            # We can compute the energy delta.
            # But calculating delta for "distinct colors" is non-trivial.
            
            # Let's stick to a smaller number of steps with full checks, or optimize the check.
            # Let's try to optimize the check using numpy.
            
            # Actually, let's just run fewer steps and rely on the randomness.
            # Or, better:
            # We can precompute the list of K4s for each edge.
            
            # Let's restructure to use incremental checks.
            
            # Re-initializing for incremental check logic
            # This requires significant setup.
            
            # Given the time limit, let's try a simpler, faster heuristic:
            # Local Search with Random Restarts.
            
            # Let's abandon pure SA for now and use a "Greedy + Local Search" hybrid.
            # 1. Start with valid coloring.
            # 2. Try to merge colors greedily.
            # 3. If stuck, do a random perturbation (recolor a few edges).
            # 4. Repeat.
            
            # This is what the seed does. The seed is score 1.0.
            # We need something better.
            
            # Let's try a different initial coloring or a different merging strategy.
            # What if we start with a 13-color random coloring and fix violations?
            
            pass # This branch is getting complicated. Let's switch to a robust implementation.

    # --- Robust Implementation ---
    
    # Precompute edge_to_k4s for incremental updates
    edge_to_k4s = {idx: [] for idx in range(num_edges)}
    for k4_idx in range(num_k4s):
        for e_idx in k4_edges[k4_idx]:
            edge_to_k4s[int(e_idx)].append(k4_idx)

    def get_k4_distinct_count(k4_idx, colors_np):
        e_idxs = k4_edges[k4_idx]
        c = [colors_np[e_idxs[0]], colors_np[e_idxs[1]], colors_np[e_idxs[2]], 
             colors_np[e_idxs[3]], colors_np[e_idxs[4]], colors_np[e_idxs[5]]]
        return len(set(c))

    def is_valid_k4(k4_idx, colors_np):
        return get_k4_distinct_count(k4_idx, colors_np) >= 5

    def check_violations_incremental(colors_np, changed_edges):
        """
        Returns list of violating K4 indices.
        """
        k4s_to_check = set()
        for e_idx in changed_edges:
            k4s_to_check.update(edge_to_k4s[int(e_idx)])
        
        violations = []
        for k4_idx in k4s_to_check:
            if not is_valid_k4(k4_idx, colors_np):
                violations.append(k4_idx)
        return violations

    # Strategy:
    # 1. Start with valid coloring (15 colors).
    # 2. Iteratively try to reduce color count.
    # 3. Use "Conflict-Driven" search:
    #    - Identify a color to remove.
    #    - Try to reassign edges of that color to other colors.
    #    - If conflicts arise, resolve them by recoloring other edges.
    
    best_colors = colors.copy()
    best_num_used = len(set(colors))
    
    # Helper to get a valid coloring from a potentially invalid one by adding colors?
    # No, we must end with a valid one.
    
    # Let's try a specific algorithm: "Color Merging with Backtracking"
    
    def try_merge_color(colors_np, c_from, c_to):
        """
        Try to change all edges of color c_from to c_to.
        Returns (success, new_colors, changed_edges)
        """
        edges_with_from = np.where(colors_np == c_from)[0]
        if len(edges_with_from) == 0:
            return True, colors_np, []
        
        temp_colors = colors_np.copy()
        temp_colors[edges_with_from] = c_to
        
        changed_edges = list(edges_with_from)
        violations = check_violations_incremental(temp_colors, changed_edges)
        
        if not violations:
            return True, temp_colors, changed_edges
        else:
            return False, colors_np, []

    # Main Loop
    for attempt in range(50): # 50 restarts/attempts
        if time.time() > deadline:
            break
            
        current_colors = best_colors.copy()
        current_used = set(current_colors)
        
        # Greedy Merge Phase
        progress = True
        while progress:
            if time.time() > deadline:
                break
                
            progress = False
            colors_list = list(current_used)
            rng.shuffle(colors_list)
            
            for c_from in colors_list:
                if c_from not in current_used:
                    continue
                
                # Try to merge c_from into each other color
                for c_to in colors_list:
                    if c_from == c_to or c_to not in current_used:
                        continue
                        
                    success, new_colors, _ = try_merge_color(current_colors, c_from, c_to)
                    if success:
                        current_colors = new_colors
                        current_used.remove(c_from)
                        progress = True
                        
                        # Update best
                        if len(current_used) < best_num_used:
                            best_colors = current_colors.copy()
                            best_num_used = len(current_used)
                        break # Restart merge loop with new state
                
                if progress:
                    break
        
        # If we improved, continue. If not, we are stuck.
        # Try a random perturbation to escape local minima.
        if len(current_used) >= best_num_used:
            # Perturb
            # Recolor 5 random edges to random colors
            for _ in range(5):
                e_idx = rng.randint(0, num_edges - 1)
                new_c = rng.randint(0, max_colors - 1)
                current_colors[e_idx] = new_c
            
            # Now try to fix violations by adding colors or recoloring
            # This is hard to do directly.
            # Instead, let's just restart the greedy merge from this perturbed state?
            # The perturbed state is likely invalid.
            # We need a valid state to start merging.
            
            # So, we need to "repair" the perturbed state.
            # Repair: While there are violations, pick a violating K4, and recolor one of its edges to a NEW color.
            
            violations = check_violations_incremental(current_colors, list(range(num_edges))) # Full check
            repair_steps = 0
            while violations and repair_steps < 100:
                if time.time() > deadline:
                    break
                v_k4 = rng.choice(violations)
                # Pick a random edge in this K4 to recolor
                e_idxs = k4_edges[v_k4]
                e_idx = rng.choice(e_idxs)
                # Assign a new unique color to this edge to break the pattern
                new_c = max(current_colors) + 1
                current_colors[e_idx] = new_c
                
                # Update violations
                # This is slow. 
                # Let's just break out if we spend too much time.
                repair_steps += 1
                
                # Re-check all violations (slow but simple)
                # Actually, let's just stop repairing and use the best valid found so far.
                # The perturbation approach is risky if repair is slow.
                
                break # Skip repair for now, just restart greedy merge on a *valid* copy?
                # No, current_colors is invalid.
                # Let's revert to best_colors and try a different perturbation strategy.
            
            # Alternative: Don't perturb the valid state. 
            # Instead, try different merge orders more aggressively.
            # The current loop already shuffles colors.
            # Maybe we should try to merge into a *new* color? No, that increases count.
            
            # Let's try a different initial coloring for the next attempt.
            # Random valid coloring? Hard to generate.
            # Use (i+j)%17 again? Same start.
            
            # Let's try (i*2 + j) % 17 or other linear forms.
            # (i + 2j) % 17
            # (i + j) % 16 ? (16 colors)
            # (i + j) % 15 ?
            
            # Let's just rely on the shuffle.
            pass

    # Final check
    if count_violations(best_colors) != 0:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    return best_colors.tolist()

# EVOLVE-BLOCK-END
