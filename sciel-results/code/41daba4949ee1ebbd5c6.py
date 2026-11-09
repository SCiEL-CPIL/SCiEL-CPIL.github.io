"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import math

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    n = len(boxes)
    container_vol = W * H * D
    
    # Pre-calculate volumes for sorting and scoring
    box_vols = [w*h*d for w,h,d in boxes]
    
    def compute_score(placements):
        """
        Compute score: B + 0.1 * (1 - last_bin_occupied_volume / container_volume)
        Lower is better.
        placements: list of (bin_id, x, y, z, w, h, d)
        """
        if not placements:
            return float('inf')
        
        bins_used = set(p[0] for p in placements)
        if not bins_used:
            return float('inf')
            
        B = len(bins_used)
        last_bin = max(bins_used)
        
        last_vol = 0
        for p in placements:
            if p[0] == last_bin:
                last_vol += p[4] * p[5] * p[6]
                
        if container_vol == 0:
            return float('inf')
            
        frac = last_vol / container_vol
        if frac > 1.0: frac = 1.0
        if frac < 0.0: frac = 0.0
        
        score = B + 0.1 * (1 - frac)
        return score

    def is_valid_placement(x, y, z, w, h, d, bin_boxes):
        """Check if a box fits in the bin without overlapping existing boxes."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
            
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # AABB overlap check
            if not (x + w <= bx or bx + bw <= x or
                    y + h <= by or by + bh <= y or
                    z + d <= bz or bz + bd <= z):
                return False
        return True

    def generate_candidate_positions(bin_boxes, w, h, d):
        """
        Generate candidate positions for a new box in a bin.
        Uses the 'skyline' or 'corner' heuristic: positions are defined by
        the max coordinates of existing boxes.
        """
        candidates = set([(0, 0, 0)])
        
        # Collect all relevant x, y, z coordinates
        xs = {0, W}
        ys = {0, H}
        zs = {0, D}
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            xs.add(bx)
            xs.add(bx + bw)
            ys.add(by)
            ys.add(by + bh)
            zs.add(bz)
            zs.add(bz + bd)
            
        # We don't need a full cartesian product which is O(N^3), 
        # but for N=40, N^3 is 64k which is okay, but we can be smarter.
        # A common efficient heuristic is to only consider positions 
        # where x is from an existing box's right face, etc.
        
        # Let's stick to the specific corner combinations of existing boxes 
        # plus the origin, as done in the seed, but optimized.
        # Actually, the seed's method of checking all 8 corners of every box is 
        # O(N) candidates per box. This is fast.
        
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz + bd))
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
            
        # Also add positions aligned with the "front" of existing boxes
        # to allow filling gaps in the middle? 
        # For simplicity and speed, the corner method is standard for BFD.
        
        unique_candidates = list(candidates)
        # Sort by distance to origin to prioritize compact packing
        unique_candidates.sort(key=lambda p: (p[0] + p[1] + p[2], p[0], p[1], p[2]))
        
        return unique_candidates

    def greedy_pack(sort_key_func, randomize_positions=False):
        """
        Runs a greedy Best-Fit Decreasing pack.
        Returns a list of placements in original index order.
        """
        indexed_boxes = list(enumerate(boxes))
        indexed_boxes.sort(key=lambda t: sort_key_func(t[1]))
        
        bins = [] # List of lists of (x, y, z, w, h, d)
        placement_map = {}
        
        for idx, (w, h, d) in indexed_boxes:
            placed = False
            
            if bins:
                # Sort bins by volume descending (fill fuller bins first)
                # This is a heuristic to keep bins "closed" as much as possible
                bin_indices = list(range(len(bins)))
                bin_indices.sort(key=lambda bi: -sum(bx[3]*bx[4]*bx[5] for bx in bins[bi]))
                
                for bi in bin_indices:
                    bin_boxes = bins[bi]
                    
                    # Generate candidates
                    candidates = generate_candidate_positions(bin_boxes, w, h, d)
                    
                    # If randomizing, shuffle candidates slightly? 
                    # No, for greedy base, we pick the best (lowest score) valid pos.
                    # The "randomization" in Multi-Start usually comes from 
                    # randomizing the sort order or tie-breaking.
                    
                    found_pos = None
                    best_pos_score = float('inf')
                    
                    for (cx, cy, cz) in candidates:
                        if is_valid_placement(cx, cy, cz, w, h, d, bin_boxes):
                            # Score: prefer positions closer to origin
                            pos_score = cx + cy + cz
                            if pos_score < best_pos_score:
                                best_pos_score = pos_score
                                found_pos = (cx, cy, cz)
                    
                    if found_pos:
                        cx, cy, cz = found_pos
                        bin_boxes.append((cx, cy, cz, w, h, d))
                        placement_map[idx] = (bi, cx, cy, cz)
                        placed = True
                        break
            
            if not placed:
                # Open new bin
                bi = len(bins)
                bins.append([])
                bins[bi].append((0, 0, 0, w, h, d))
                placement_map[idx] = (bi, 0, 0, 0)
        
        # Construct final placement list
        final_placements = []
        for i in range(n):
            if i in placement_map:
                bi, x, y, z = placement_map[i]
                w, h, d = boxes[i]
                final_placements.append((bi, x, y, z, w, h, d))
            else:
                final_placements.append((0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
        
        return final_placements

    def simulated_annealing(initial_placements, time_limit):
        """
        Improves an initial packing using Simulated Annealing.
        Moves: 
        1. Move a box to a different valid position in the same bin.
        2. Move a box to a different bin (if it fits).
        3. Swap two boxes (if both fit in each other's spots).
        
        For simplicity and speed with N=40, we will implement:
        - Randomly pick a box.
        - Try to move it to a random valid position in its current bin.
        - Try to move it to a random valid position in another bin.
        """
        current_placements = [p[:] for p in initial_placements]
        current_score = compute_score(current_placements)
        
        # Build internal state: bin_id -> list of (idx, x, y, z, w, h, d)
        def build_state(placements):
            bins = {}
            for i, (b, x, y, z, w, h, d) in enumerate(placements):
                if b not in bins:
                    bins[b] = []
                bins[b].append((i, x, y, z, w, h, d))
            return bins

        state = build_state(current_placements)
        bins_list = list(state.keys())
        
        T = 1.0 # Temperature
        T_min = 0.01
        alpha = 0.95 # Cooling rate
        
        best_placements = current_placements
        best_score = current_score
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                break
                
            # Pick a random box
            idx = random.randint(0, n - 1)
            cur_bin, cx, cy, cz, cw, ch, cd = current_placements[idx]
            
            # 1. Try to move within the same bin
            # Generate random candidate positions in the same bin
            bin_boxes = state[cur_bin]
            # Filter out the box itself for position generation? 
            # The candidate generator uses existing boxes. 
            # If we move it, we need to check against others.
            
            # Get other boxes in this bin
            other_boxes_in_bin = [b for b in bin_boxes if b[0] != idx]
            # Extract just coords for generate_candidate_positions
            coords_only = [(b[1], b[2], b[3], b[4], b[5], b[6]) for b in other_boxes_in_bin]
            
            candidates = generate_candidate_positions(coords_only, cw, ch, cd)
            
            # Randomly pick a candidate
            if candidates:
                # Bias towards positions closer to current one? Or just random?
                # Random helps escape local minima.
                nx, ny, nz = random.choice(candidates)
                
                # Check validity in same bin
                if is_valid_placement(nx, ny, nz, cw, ch, cd, coords_only):
                    # Calculate new score
                    # We need to update the score efficiently.
                    # Since we only moved one box, B (bin count) doesn't change 
                    # unless we moved it to/from the last bin.
                    # And the last bin volume changes.
                    
                    # To be safe and simple, just compute full score. 
                    # N=40, so it's fast.
                    
                    # Create new placement list
                    new_placements = [p[:] for p in current_placements]
                    new_placements[idx] = (cur_bin, nx, ny, nz, cw, ch, cd)
                    
                    new_score = compute_score(new_placements)
                    
                    delta = new_score - current_score
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        # Accept
                        current_placements = new_placements
                        current_score = new_score
                        
                        # Update internal state
                        # Remove old, add new
                        # For efficiency in this loop, we might just rebuild state 
                        # if accepted, but that's O(N).
                        # Given N=40, rebuilding is fine.
                        state = build_state(current_placements)
                        
                        if current_score < best_score:
                            best_score = current_score
                            best_placements = [p[:] for p in current_placements]
            
            # 2. Try to move to a different bin (if any other bins exist)
            if len(bins_list) > 1:
                # Pick a random other bin
                other_bins = [b for b in bins_list if b != cur_bin]
                target_bin = random.choice(other_bins)
                
                target_boxes = state[target_bin]
                target_coords = [(b[1], b[2], b[3], b[4], b[5], b[6]) for b in target_boxes]
                
                candidates = generate_candidate_positions(target_coords, cw, ch, cd)
                
                if candidates:
                    nx, ny, nz = random.choice(candidates)
                    
                    if is_valid_placement(nx, ny, nz, cw, ch, cd, target_coords):
                        new_placements = [p[:] for p in current_placements]
                        new_placements[idx] = (target_bin, nx, ny, nz, cw, ch, cd)
                        
                        new_score = compute_score(new_placements)
                        
                        delta = new_score - current_score
                        if delta < 0 or random.random() < math.exp(-delta / T):
                            # Accept
                            current_placements = new_placements
                            current_score = new_score
                            
                            # Update state
                            state = build_state(current_placements)
                            # bins_list might have changed if target_bin was empty 
                            # and we moved the only box out? No, we moved INTO it.
                            # If cur_bin becomes empty, we should remove it from bins_list?
                            # The score function handles empty bins by just ignoring them 
                            # in the set of used bins. But our state dict might have empty lists.
                            # Let's prune empty bins in state.
                            state = {k: v for k, v in state.items() if v}
                            bins_list = list(state.keys())
                            
                            if current_score < best_score:
                                best_score = current_score
                                best_placements = [p[:] for p in current_placements]
            
            # Cool down
            T *= alpha
            
        return best_placements

    # --- Main Logic ---
    
    start_time = time.time()
    deadline = start_time + 140.0 # Safe margin before 150s limit
    
    # Define sorting strategies
    def sort_by_vol(box):
        return -(box[0] * box[1] * box[2])
    
    def sort_by_area(box):
        w, h, d = box
        return -(2*(w*h + h*d + w*d))
        
    def sort_by_maxdim(box):
        return -max(box)
        
    def sort_by_vol_asc(box):
        return box[0] * box[1] * box[2]

    def sort_by_sum(box):
        return -(box[0] + box[1] + box[2])

    def sort_by_vol_tie_maxdim(box):
        return -(box[0] * box[1] * box[2] + 0.001 * max(box))

    strategies = [
        ("vol_desc", sort_by_vol),
        ("area_desc", sort_by_area),
        ("maxdim_desc", sort_by_maxdim),
        ("vol_asc", sort_by_vol_asc),
        ("sum_desc", sort_by_sum),
        ("vol_tie_maxdim", sort_by_vol_tie_maxdim)
    ]
    
    best_placements = None
    best_score = float('inf')
    
    # Phase 1: Generate initial solutions using different sort keys
    initial_solutions = []
    
    for name, key_func in strategies:
        if time.time() > deadline - 20.0: # Reserve time for SA
            break
            
        try:
            placements = greedy_pack(key_func)
            score = compute_score(placements)
            initial_solutions.append((score, placements, name))
            
            if score < best_score:
                best_score = score
                best_placements = placements
        except Exception as e:
            pass

    # Phase 2: Simulated Annealing on the best initial solution (and maybe others if time permits)
    # We will run SA on the top 2 distinct initial solutions if time allows.
    
    # Sort initial solutions by score
    initial_solutions.sort(key=lambda x: x[0])
    
    # Determine time budget for SA
    time_elapsed = time.time() - start_time
    time_remaining = deadline - time_elapsed
    
    # If we have very little time, just return the best greedy result
    if time_remaining < 5.0:
        if best_placements:
            return best_placements
        # Fallback
        return greedy_pack(sort_by_vol)

    # Run SA on the best initial solution
    if initial_solutions:
        best_initial_score, best_initial_placements, _ = initial_solutions[0]
        
        # Allocate 60% of remaining time to the best solution
        sa_time_budget = time_remaining * 0.6
        
        improved_placements = simulated_annealing(best_initial_placements, sa_time_budget)
        improved_score = compute_score(improved_placements)
        
        if improved_score < best_score:
            best_score = improved_score
            best_placements = improved_placements
            
        # If there's a second distinct solution and time left, try it
        if len(initial_solutions) > 1:
            second_score, second_placements, second_name = initial_solutions[1]
            
            # Only if it's significantly different or if we have lots of time
            time_elapsed = time.time() - start_time
            time_remaining = deadline - time_elapsed
            
            if time_remaining > 10.0 and second_score < best_score + 0.5: # Heuristic: if it's close
                sa_time_budget_2 = time_remaining * 0.4
                improved_placements_2 = simulated_annealing(second_placements, sa_time_budget_2)
                improved_score_2 = compute_score(improved_placements_2)
                
                if improved_score_2 < best_score:
                    best_score = improved_score_2
                    best_placements = improved_placements_2

    # Final check
    if best_placements is None:
        # Fallback
        best_placements = greedy_pack(sort_by_vol)

    return best_placements

# EVOLVE-BLOCK-END
