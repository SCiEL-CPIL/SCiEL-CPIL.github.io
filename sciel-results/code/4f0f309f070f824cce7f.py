"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    # 120s limit, keep 10s margin for safety/overhead
    deadline = start_time + 110.0
    
    # Preprocess: store box dimensions and original indices
    # We will store unique rotations for each box to speed up SA moves
    box_dims = []
    for i in range(n):
        w, h, d = boxes[i]
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        box_dims.append(list(rots))
        
    def get_overlap_volume(pos1, dim1, pos2, dim2):
        """Calculate the volume of overlap between two boxes."""
        # pos: (x, y, z), dim: (w, h, d)
        # Check if they overlap in all 3 axes
        # Overlap in x: max(0, min(x1+w1, x2+w2) - max(x1, x2))
        ox = max(0.0, min(pos1[0] + dim1[0], pos2[0] + dim2[0]) - max(pos1[0], pos2[0]))
        oy = max(0.0, min(pos1[1] + dim1[1], pos2[1] + dim2[1]) - max(pos1[1], pos2[1]))
        oz = max(0.0, min(pos1[2] + dim1[2], pos2[2] + dim2[2]) - max(pos1[2], pos2[2]))
        return ox * oy * oz

    def calculate_energy(placements, dims):
        """
        Calculate the total penalty energy.
        Energy = Sum(overlap_volumes) + Sum(out_of_bounds_volumes)
        Lower is better. 0 is a valid packing.
        """
        total_overlap = 0.0
        out_of_bounds = 0.0
        
        # Check out of bounds
        for i in range(len(placements)):
            x, y, z = placements[i]
            w, h, d = dims[i]
            # If box is outside, add the volume of the part that is outside? 
            # Or just a large penalty? 
            # Standard approach: Penalty = Volume of intersection with outside space.
            # Simpler: If any coordinate is < 0 or > container, add a penalty proportional to volume.
            # Let's use a hard penalty for any part sticking out.
            if x < 0 or y < 0 or z < 0 or x + w > W or y + h > H or z + d > D:
                # Calculate volume of the box that is outside the container
                # This is complex to calculate exactly. 
                # Approximation: If it's outside, add its full volume as penalty? 
                # Or better: Calculate the volume of the intersection with the container, 
                # and penalty = Total Volume - Intersection Volume.
                # Intersection x: max(0, min(W, x+w)) - max(0, x)
                ix = max(0.0, min(W, x + w) - max(0.0, x))
                iy = max(0.0, min(H, y + h) - max(0.0, y))
                iz = max(0.0, min(D, z + d) - max(0.0, z))
                inside_vol = ix * iy * iz
                total_vol = w * h * d
                out_of_bounds += (total_vol - inside_vol)
        
        # Check overlaps
        # O(N^2) check
        for i in range(len(placements)):
            for j in range(i + 1, len(placements)):
                ov = get_overlap_volume(placements[i], dims[i], placements[j], dims[j])
                if ov > 0:
                    total_overlap += ov

        return total_overlap + out_of_bounds

    def extract_solution(placements, dims):
        """
        Convert a low-energy (ideally zero-energy) placement to the final format.
        If a box is still overlapping or outside, we must decide whether to include it.
        The task requires "no overlap". 
        Strategy: If energy is 0, all boxes are valid.
        If energy > 0, we need to remove boxes to make the set valid.
        Greedy removal: Sort by volume (large to small). 
        Iterate. If box is valid (no overlap with already accepted, inside bounds), keep it.
        Else, mark as left out.
        """
        # First, identify which boxes are "mostly" inside and not overlapping?
        # Actually, if energy > 0, the configuration is invalid.
        # We need to find a valid subset.
        # Let's try to keep as many high-volume boxes as possible.
        
        # To be safe and fast, if energy is very small (numerical error), snap to grid?
        # No, let's just filter.
        
        # Re-evaluate validity of each box against the container and each other.
        # This is expensive if done naively for all N^2 again.
        # But N=60, so 1800 checks is fine.
        
        valid_indices = []
        # Sort indices by volume descending
        indices_sorted = sorted(range(n), key=lambda i: dims[i][0]*dims[i][1]*dims[i][2], reverse=True)
        
        accepted = [] # list of (x,y,z,w,h,d)
        
        for i in indices_sorted:
            x, y, z = placements[i]
            w, h, d = dims[i]
            
            # Check bounds
            if x < -1e-9 or y < -1e-9 or z < -1e-9 or x + w > W + 1e-9 or y + h > H + 1e-9 or z + d > D + 1e-9:
                continue
                
            # Check overlap with accepted
            overlap = False
            for (ax, ay, az, aw, ah, ad) in accepted:
                if not (x + w <= ax or ax + aw <= x or 
                        y + h <= ay or ay + ah <= y or 
                        z + d <= az or az + ad <= z):
                    overlap = True
                    break
            
            if not overlap:
                accepted.append((x, y, z, w, h, d))
                valid_indices.append(i)
                
        # Create result
        result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        for i in valid_indices:
            # Find the placement in 'accepted' that corresponds to index i?
            # We lost the mapping in 'accepted' list directly to index i because we sorted.
            # Let's store (index, x, y, z, w, h, d) in accepted.
            pass
        
        # Redo extraction properly
        result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        accepted = []
        
        for i in indices_sorted:
            x, y, z = placements[i]
            w, h, d = dims[i]
            
            # Snap to small epsilon if close to 0?
            # For now, strict check.
            if x < -1e-9 or y < -1e-9 or z < -1e-9 or x + w > W + 1e-9 or y + h > H + 1e-9 or z + d > D + 1e-9:
                continue
                
            overlap = False
            for (idx, ax, ay, az, aw, ah, ad) in accepted:
                if not (x + w <= ax or ax + aw <= x or 
                        y + h <= ay or ay + ah <= y or 
                        z + d <= az or az + ad <= z):
                    overlap = True
                    break
            
            if not overlap:
                accepted.append((i, x, y, z, w, h, d))
                result[i] = (0, x, y, z, w, h, d)
                
        return result

    def run_sa_restart(boxes, container, box_dims, deadline, seed):
        """
        Run one Simulated Annealing restart.
        Returns the best solution found in this restart.
        """
        random.seed(seed)
        W, H, D = container
        n = len(boxes)
        
        # Initial state: Random valid placements? Or just random inside container?
        # Random inside container is likely to have high overlap.
        # Better: Start with a greedy solution to give SA a good starting point?
        # Or just random. Let's try random first, it's simpler and explores differently.
        # Actually, a mix: Start with random positions, but ensure they are inside bounds.
        
        placements = []
        dims = []
        for i in range(n):
            rots = box_dims[i]
            # Pick a random rotation
            r_idx = random.randint(0, len(rots)-1)
            w, h, d = rots[r_idx]
            dims.append((w, h, d))
            
            # Random position inside container
            # If box is larger than container in any dim, it can't be placed. 
            # But we assume boxes fit in at least one rotation? 
            # The task implies boxes might not fit. 
            # If w > W, we can't place it inside. 
            # For SA, if it doesn't fit, we can still place it "outside" and let the penalty handle it,
            # or we can force it to a corner and let the penalty be huge.
            # Let's just clamp the random position to [0, W-w] if possible, else [0, W].
            if w <= W:
                x = random.uniform(0, W - w)
            else:
                x = 0.0 # It will be penalized for being out of bounds/overlapping
            if h <= H:
                y = random.uniform(0, H - h)
            else:
                y = 0.0
            if d <= D:
                z = random.uniform(0, D - d)
            else:
                z = 0.0
                
            placements.append((x, y, z))
            
        # Calculate initial energy
        current_energy = calculate_energy(placements, dims)
        best_energy = current_energy
        best_placements = [p[:] for p in placements]
        best_dims = [d[:] for d in dims]
        
        # SA Parameters
        # Temperature decay
        T = 1000.0  # Initial temp (scale with volume? Container volume is ~1000-10000?)
        # Let's scale T by total volume
        total_vol = sum(w*h*d for w,h,d in dims)
        T = max(1.0, total_vol * 0.1)
        cooling_rate = 0.99
        min_temp = 0.01
        
        steps = 0
        max_steps = 5000 # Per restart? Adjust based on time.
        
        while T > min_temp:
            if time.time() > deadline:
                break
                
            # Pick a random box to move
            i = random.randint(0, n - 1)
            
            # Pick a move type: Move position or Change rotation
            if random.random() < 0.8:
                # Move position
                # Small random step
                step_size = T / 10.0 # Adaptive step?
                # Or fixed small step
                dx = random.uniform(-step_size, step_size)
                dy = random.uniform(-step_size, step_size)
                dz = random.uniform(-step_size, step_size)
                
                old_x, old_y, old_z = placements[i]
                new_x = old_x + dx
                new_y = old_y + dy
                new_z = old_z + dz
                
                # Optional: Clamp to container? 
                # No, let it go outside, penalty will push it back.
                # But if it goes way outside, energy explodes.
                # Let's clamp to reasonable bounds: [-w, W] etc.
                w, h, d = dims[i]
                new_x = max(-w, min(W, new_x))
                new_y = max(-h, min(H, new_y))
                new_z = max(-d, min(D, new_z))
                
                placements[i] = (new_x, new_y, new_z)
                
            else:
                # Change rotation
                rots = box_dims[i]
                if len(rots) > 1:
                    new_rot_idx = random.randint(0, len(rots)-1)
                    # Avoid picking the same rotation if possible?
                    # Current rotation is dims[i]
                    # Find index of current rotation in rots
                    curr_w, curr_h, curr_d = dims[i]
                    # It might not be in rots if we modified it? No, we only pick from rots.
                    # But floating point? Dimensions are integers usually.
                    # Let's just pick a new one.
                    new_w, new_h, new_d = rots[new_rot_idx]
                    dims[i] = (new_w, new_h, new_d)
                    # Adjust position to keep center? Or just keep origin?
                    # Keeping origin is fine, penalty will handle it.
            
            new_energy = calculate_energy(placements, dims)
            
            # Acceptance criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_placements = [p[:] for p in placements]
                    best_dims = [d[:] for d in dims]
            else:
                # Revert
                if random.random() < 0.8:
                    # We moved position, revert it
                    # We need the old position. We didn't store it explicitly in a variable accessible here easily without refactoring.
                    # Let's refactor to store old state.
                    pass # This is a bug in the logic above. I need to store old state.
            
            T *= cooling_rate
            steps += 1
            
        return extract_solution(best_placements, best_dims)

    # Fix the SA logic above. The "revert" part was broken.
    # Let's rewrite run_sa_restart properly.
    
    def run_sa_restart_proper(boxes, container, box_dims, deadline, seed):
        random.seed(seed)
        W, H, D = container
        n = len(boxes)
        
        placements = []
        dims = []
        for i in range(n):
            rots = box_dims[i]
            r_idx = random.randint(0, len(rots)-1)
            w, h, d = rots[r_idx]
            dims.append((w, h, d))
            
            if w <= W:
                x = random.uniform(0, W - w)
            else:
                x = 0.0
            if h <= H:
                y = random.uniform(0, H - h)
            else:
                y = 0.0
            if d <= D:
                z = random.uniform(0, D - d)
            else:
                z = 0.0
                
            placements.append((x, y, z))
            
        current_energy = calculate_energy(placements, dims)
        best_energy = current_energy
        best_placements = [p[:] for p in placements]
        best_dims = [d[:] for d in dims]
        
        T = 1000.0
        cooling_rate = 0.995
        min_temp = 1.0
        
        while T > min_temp:
            if time.time() > deadline:
                break
                
            i = random.randint(0, n - 1)
            old_pl = placements[i]
            old_dim = dims[i]
            
            if random.random() < 0.8:
                # Move
                step = T * 0.1
                dx = random.uniform(-step, step)
                dy = random.uniform(-step, step)
                dz = random.uniform(-step, step)
                new_x = old_pl[0] + dx
                new_y = old_pl[1] + dy
                new_z = old_pl[2] + dz
                
                w, h, d = old_dim
                new_x = max(-w, min(W, new_x))
                new_y = max(-h, min(H, new_y))
                new_z = max(-d, min(D, new_z))
                
                placements[i] = (new_x, new_y, new_z)
                
            else:
                # Rotate
                rots = box_dims[i]
                if len(rots) > 1:
                    new_rot_idx = random.randint(0, len(rots)-1)
                    new_w, new_h, new_d = rots[new_rot_idx]
                    dims[i] = (new_w, new_h, new_d)
                    # Keep position, but maybe shift to center? No, keep it.
            
            new_energy = calculate_energy(placements, dims)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_placements = [p[:] for p in placements]
                    best_dims = [d[:] for d in dims]
            else:
                # Revert
                placements[i] = old_pl
                dims[i] = old_dim
                
            T *= cooling_rate
            
        return extract_solution(best_placements, best_dims)

    # Multi-restart SA
    # We have 110s. 
    # Estimate time per restart.
    # N=60. Energy calc is O(N^2) = 3600 ops. 
    # Python loop overhead is high. 
    # Let's run a quick test or just assume we can do ~50-100 restarts.
    # Each restart ~500-1000 steps?
    
    best_result = None
    best_score = -1.0 # Maximize volume filled
    
    # To maximize volume filled, we want to maximize sum of volumes of placed boxes.
    # The score is mean fraction of container volume filled.
    
    def score_solution(sol):
        vol = 0.0
        for p in sol:
            if p[0] == 0:
                # p is (bin, x, y, z, w, h, d)
                vol += p[4] * p[5] * p[6]
        total_vol = W * H * D
        if total_vol == 0:
            return 0.0
        return vol / total_vol

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 5.0: # Keep 5s for final processing
        seed = num_restarts
        sol = run_sa_restart_proper(boxes, container, box_dims, deadline, seed)
        score = score_solution(sol)
        
        if score > best_score:
            best_score = score
            best_result = sol
            
        num_restarts += 1
        
        # If we found a perfect packing (1.0), we can stop? 
        # Unlikely to be exactly 1.0, but if it's very high, maybe.
        if best_score > 0.99:
            break

    if best_result is None:
        # Fallback to empty or greedy if SA failed?
        # Just return all -1
        best_result = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        
    return best_result

# EVOLVE-BLOCK-END
