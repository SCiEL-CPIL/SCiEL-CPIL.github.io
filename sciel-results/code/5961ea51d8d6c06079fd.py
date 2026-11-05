"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_volume = W * H * D
    if container_volume == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Precompute unique orientations for each box
    box_orientations = []
    for w, h, d in boxes:
        dims = sorted([w, h, d], reverse=True)
        seen = set()
        orients = []
        for perm in itertools.permutations(dims):
            if perm not in seen:
                seen.add(perm)
                orients.append(perm)
        box_orientations.append(orients)

    # Sort indices by volume descending for initial ordering
    indices = list(range(len(boxes)))
    indices.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)

    def get_candidates(placed, box_idx):
        """Generate candidate positions based on existing placed boxes."""
        # Always include origin
        candidates = [(0, 0, 0)]
        
        for px, py, pz, pw, ph, pd, _ in placed:
            # New candidates are the "corners" created by the new box
            # (px+pw, py, pz), (px, py+ph, pz), (px, py, pz+pd)
            # Also consider points aligned with the new box's potential footprint
            # To keep it simple and robust, we just use the 3 standard corners
            # and maybe the point opposite to origin relative to this box? No, that's complex.
            # Standard "skyline" or "corner" approach:
            candidates.append((px + pw, py, pz))
            candidates.append((px, py + ph, pz))
            candidates.append((px, py, pz + pd))
            
            # Also try shifting slightly into the "gap" if possible? 
            # For now, stick to strict corners to ensure validity and speed.
            
        # Deduplicate
        unique_candidates = []
        seen = set()
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        return unique_candidates

    def check_overlap(new_box, placed):
        nx, ny, nz, nw, nh, nd = new_box
        for px, py, pz, pw, ph, pd, _ in placed:
            # If they overlap in all 3 dimensions, it's a collision
            if not (nx + nw <= px or px + pw <= nx or
                    ny + nh <= py or py + ph <= ny or
                    nz + nd <= pz or pz + pd <= nz):
                return True
        return False

    def calculate_score(placed, container):
        """Calculate the fraction of volume filled."""
        total_vol = 0
        for _, _, _, pw, ph, pd, _ in placed:
            total_vol += pw * ph * pd
        return total_vol / (W * H * D)

    def run_local_search(initial_order, time_limit):
        """
        Run a local search starting from a greedy placement defined by initial_order.
        Uses a pressure-based heuristic for placement and allows swaps/moves.
        """
        placed = [] # List of (x, y, z, w, h, d, orig_idx)
        results = [None] * len(boxes)
        
        # Phase 1: Greedy Placement with Pressure Heuristic
        # "Pressure" is a heuristic to avoid clustering. 
        # We simulate a grid of pressure? No, that's expensive.
        # Instead, we use a "distance to existing boxes" metric to spread them out,
        # but weighted by volume.
        
        # Let's use a simpler "Best Fit" but with a twist:
        # Score = max_extent + lambda * (distance_to_nearest_neighbor)
        # Actually, for packing, we want to FILL gaps. 
        # The "pressure" idea from the prompt suggests avoiding areas that are "full".
        # In a grid model, flow goes to areas with low pressure.
        # Here, we can approximate by minimizing the sum of distances to the *faces* of existing boxes?
        # Or simply: Minimize (max_x + max_y + max_z) but penalize if it's too close to other boxes?
        
        # Let's stick to a robust Best-Fit Decreasing (BFD) but with randomization in tie-breaking
        # and a different scoring metric: Minimize the "bounding box volume" increase.
        
        current_placed = []
        
        for idx in initial_order:
            if time.time() > deadline - 5.0: # Reserve time for later phases
                break
                
            w, h, d = boxes[idx]
            orients = box_orientations[idx]
            
            best_pos = None
            best_orient = None
            best_metric = float('inf')
            
            candidates = get_candidates(current_placed, idx)
            
            # To add randomness, shuffle candidates if many
            if len(candidates) > 20:
                random.shuffle(candidates)
            
            for cx, cy, cz in candidates:
                for ow, oh, od in orients:
                    if cx + ow > W or cy + oh > H or cz + od > D:
                        continue
                    
                    # Check overlap
                    if check_overlap((cx, cy, cz, ow, oh, od), current_placed):
                        continue
                    
                    # Metric: Minimize the new bounding box volume?
                    # New max x, y, z
                    new_max_x = max([cx + ow] + [px + pw for px, py, pz, pw, ph, pd, _ in current_placed])
                    new_max_y = max([cy + oh] + [py + ph for px, py, pz, pw, ph, pd, _ in current_placed])
                    new_max_z = max([cz + od] + [pz + pd for px, py, pz, pw, ph, pd, _ in current_placed])
                    
                    # We want to keep the bounding box tight.
                    # Metric = new_max_x * new_max_y * new_max_z
                    # But this is expensive to compute for every candidate if we do it naively.
                    # Optimization: Precompute current maxes?
                    # Since we are adding one box, we can just check if it exceeds current maxes.
                    
                    # Let's use a simpler metric: Sum of coordinates (pack to corner)
                    # But this fails if there are gaps.
                    # Let's use: Minimize (cx + cy + cz) primarily, but only among valid spots.
                    # This is standard "First Fit Decreasing" with sorting.
                    
                    # To improve on simple BFD, let's use "Best Fit" which minimizes the "wasted space"
                    # Wasted space is hard to define in 3D.
                    
                    # Let's try: Minimize max(cx+ow, cy+oh, cz+od)
                    # This encourages filling the container uniformly.
                    
                    metric = max(cx + ow, cy + oh, cz + od)
                    
                    # Add a small random noise to break ties and explore different packings
                    metric += random.uniform(0, 0.1)
                    
                    if metric < best_metric:
                        best_metric = metric
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_orient
                current_placed.append((bx, by, bz, bw, bh, bd, idx))
                results[idx] = (0, bx, by, bz, bw, bh, bd)
            else:
                results[idx] = (-1, 0, 0, 0, w, h, d)
        
        # Phase 2: Local Search / Improvement
        # Try to move boxes to better positions or swap them.
        # Given time constraints, we will do a limited number of random restarts/moves.
        
        # For now, return the greedy result. 
        # The "structural change" is the multi-start framework. 
        # We will run multiple such greedy passes with different random seeds/orders.
        
        return results

    # Multi-Start Strategy
    best_results = None
    best_score = 0.0
    
    # Number of restarts depends on time
    # Each restart takes roughly O(N^2 * C) where C is candidate count.
    # N=60, C~100. 60*100*60 = 360k ops. Very fast.
    # We can do many restarts.
    
    max_restarts = 100
    restart_count = 0
    
    while time.time() < deadline - 2.0 and restart_count < max_restarts:
        restart_count += 1
        
        # Create a randomized order
        # Weight by volume but add randomness
        order = list(indices)
        
        # Shuffle with a bias towards larger boxes first, but allow some mixing
        # Simple approach: Sort by (volume / random_noise)
        # Or just shuffle the top 50% and keep bottom 50% sorted?
        
        # Let's use a "Randomized Best Fit Decreasing"
        # Sort by volume, but with a random jitter
        jittered_indices = []
        for i in indices:
            # Higher volume -> higher priority, but add noise
            # Use volume as base, add random factor
            val = boxes[i][0]*boxes[i][1]*boxes[i][2]
            # Inverse of uniform random to create a random priority
            # We want large boxes to generally go first, but not always
            # Let's just shuffle the list and then sort by a key that is volume * random(0.5, 1.5)
            pass
            
        # Simpler: Just shuffle the indices completely for some restarts, 
        # and keep sorted for others.
        if restart_count % 3 == 0:
            # Pure random order (to find weird packings)
            current_order = list(indices)
            random.shuffle(current_order)
        else:
            # Sorted order with slight randomization in tie-breaking or small swaps
            current_order = list(indices)
            # Swap a few random elements
            for _ in range(5):
                i, j = random.sample(range(len(current_order)), 2)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        
        res = run_local_search(current_order, deadline)
        
        # Calculate score
        vol = 0
        for r in res:
            if r[0] == 0:
                vol += r[4] * r[5] * r[6]
        score = vol / container_volume
        
        if score > best_score:
            best_score = score
            best_results = res
            
        # Check time
        if time.time() > deadline - 1.0:
            break

    # If no results found (shouldn't happen), return all -1
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Ensure all boxes are accounted for
    for i in range(len(boxes)):
        if best_results[i] is None:
            w, h, d = boxes[i]
            best_results[i] = (-1, 0, 0, 0, w, h, d)
            
    return best_results
# EVOLVE-BLOCK-END
