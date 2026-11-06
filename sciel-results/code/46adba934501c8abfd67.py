"""3D Container Loading with multi-strategy skyline + best-fit + SA polish."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import product

def load(boxes, container):
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Preprocess
    box_data = []
    for i, (w, h, d) in enumerate(boxes):
        box_data.append((i, float(w), float(h), float(d)))
    
    # Initial result: all boxes left out
    final_placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
    
    def get_rotations(w, h, d):
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        # Sort by volume (all same) but prefer smaller height first for stacking
        return sorted(list(rots), key=lambda r: (r[1], r[0], r[2]))
    
    def can_place(placed, x, y, z, bw, bh, bd):
        for (px, py, pz, pw, ph, pd) in placed:
            if not (x + bw <= px or px + pw <= x or 
                    y + bh <= py or py + ph <= y or 
                    z + bd <= pz or pz + pd <= z):
                return False
        return True
    
    def find_best_fit(placed, bw, bh, bd):
        if bw > W or bh > H or bd > D:
            return None
        
        x_candidates = set([0])
        y_candidates = set([0])
        z_candidates = set([0])
        
        for (px, py, pz, pw, ph, pd) in placed:
            x_candidates.add(px + pw)
            y_candidates.add(py + ph)
            z_candidates.add(pz + pd)
            
        x_candidates = [x for x in x_candidates if x + bw <= W]
        y_candidates = [y for y in y_candidates if y + bh <= H]
        z_candidates = [z for z in z_candidates if z + bd <= D]
        
        if not x_candidates or not y_candidates or not z_candidates:
            return None

        best_score = None
        best_coord = None
        
        for z in sorted(z_candidates):
            for y in sorted(y_candidates):
                for x in sorted(x_candidates):
                    if can_place(placed, x, y, z, bw, bh, bd):
                        # Minimize z first (bottom), then x, then y
                        score = (z, x, y)
                        if best_score is None or score < best_score:
                            best_score = score
                            best_coord = (x, y, z)
                            return best_coord  # First valid in sorted order is best
                            
        return None

    def skyline_packing(boxes_list, deadline):
        """Skyline-based packing strategy."""
        placed = []
        placements = {}
        
        # Sort by volume descending
        sorted_boxes = sorted(boxes_list, key=lambda b: b[1]*b[2]*b[3], reverse=True)
        
        for idx, w, h, d in sorted_boxes:
            if time.time() > deadline:
                break
            
            rots = get_rotations(w, h, d)
            placed_this = False
            
            for rw, rh, rd in rots:
                pos = find_best_fit(placed, rw, rh, rd)
                if pos is not None:
                    x, y, z = pos
                    placed.append((x, y, z, rw, rh, rd))
                    placements[idx] = (0, x, y, z, rw, rh, rd)
                    placed_this = True
                    break
        
        return placed, placements
    
    def multi_pass_packing(boxes_list, deadline):
        """Multi-pass best-fit decreasing."""
        placed = []
        placements = {}
        unplaced = list(boxes_list)
        
        max_passes = 8
        for p in range(max_passes):
            if time.time() > deadline:
                break
            
            if not unplaced:
                break
            
            # Sort by volume descending
            unplaced.sort(key=lambda b: b[1]*b[2]*b[3], reverse=True)
            
            newly_placed = 0
            still_unplaced = []
            
            for idx, w, h, d in unplaced:
                if time.time() > deadline:
                    still_unplaced.append((idx, w, h, d))
                    continue
                
                rots = get_rotations(w, h, d)
                placed_this = False
                
                for rw, rh, rd in rots:
                    pos = find_best_fit(placed, rw, rh, rd)
                    if pos is not None:
                        x, y, z = pos
                        placed.append((x, y, z, rw, rh, rd))
                        placements[idx] = (0, x, y, z, rw, rh, rd)
                        placed_this = True
                        newly_placed += 1
                        break
                        
                if not placed_this:
                    still_unplaced.append((idx, w, h, d))
                    
            if newly_placed == 0:
                break
            unplaced = still_unplaced
        
        return placed, placements
    
    def simulated_annealing_packing(boxes_list, deadline):
        """SA-based packing optimizing box order and rotations."""
        # Initial solution using multi-pass
        placed, placements = multi_pass_packing(boxes_list, min(deadline, start_time + 30))
        
        if not placed:
            return placed, placements
        
        # Compute initial volume
        def compute_volume(placements_dict):
            vol = 0.0
            for idx, (bin_, x, y, z, rw, rh, rd) in placements_dict.items():
                if bin_ == 0:
                    vol += rw * rh * rd
            return vol
        
        best_placements = dict(placements)
        best_volume = compute_volume(best_placements)
        current_volume = best_volume
        
        # SA parameters
        T = 1000.0
        cooling_rate = 0.995
        min_T = 1.0
        max_iterations = 5000
        
        iteration = 0
        while T > min_T and time.time() < deadline:
            iteration += 1
            
            # Choose a random placed box to potentially move
            placed_indices = list(placements.keys())
            if not placed_indices:
                break
            
            idx = random.choice(placed_indices)
            bin_, x, y, z, rw, rh, rd = placements[idx]
            
            # Get box dimensions
            box_w, box_h, box_d = boxes[idx]
            rots = get_rotations(box_w, box_h, box_d)
            
            # Try a different rotation
            new_rot = random.choice(rots)
            new_rw, new_rh, new_rd = new_rot
            
            # Remove from placed
            placed_copy = [p for p in placed if not (p[0] == x and p[1] == y and p[2] == z and 
                                                     p[3] == rw and p[4] == rh and p[5] == rd)]
            
            # Try to place at new rotation
            pos = find_best_fit(placed_copy, new_rw, new_rh, new_rd)
            if pos is None:
                # Can't place, revert
                placed = placed_copy + [(x, y, z, rw, rh, rd)]
                continue
            
            new_x, new_y, new_z = pos
            new_placed = placed_copy + [(new_x, new_y, new_z, new_rw, new_rh, new_rd)]
            
            # Compute volume change (same box, so volume is same, but packing efficiency may differ)
            # Actually volume is the same, so we need a different objective: minimize "wasted space"
            # Use a proxy: sum of z-coordinates (lower is better for compactness)
            old_z_sum = sum(p[2] for p in placed)
            new_z_sum = sum(p[2] for p in new_placed)
            
            delta_obj = new_z_sum - old_z_sum
            
            if delta_obj <= 0 or random.random() < math.exp(-delta_obj / T):
                placed = new_placed
                placements[idx] = (0, new_x, new_y, new_z, new_rw, new_rh, new_rd)
                if new_z_sum < best_volume:
                    best_volume = new_z_sum
                    best_placements = dict(placements)
            
            T *= cooling_rate
        
        return placed, best_placements
    
    # --- Main Strategy: Try multiple strategies and keep the best ---
    
    best_result = None
    best_volume = 0.0
    
    def evaluate_result(placed, placements):
        vol = 0.0
        for idx, (bin_, x, y, z, rw, rh, rd) in placements.items():
            if bin_ == 0:
                vol += rw * rh * rd
        return vol
    
    # Strategy 1: Skyline
    if time.time() < deadline:
        placed1, placements1 = skyline_packing(box_data, min(deadline, start_time + 40))
        vol1 = evaluate_result(placed1, placements1)
        if vol1 > best_volume:
            best_volume = vol1
            best_result = placements1
    
    # Strategy 2: Multi-pass
    if time.time() < deadline:
        placed2, placements2 = multi_pass_packing(box_data, min(deadline, start_time + 60))
        vol2 = evaluate_result(placed2, placements2)
        if vol2 > best_volume:
            best_volume = vol2
            best_result = placements2
    
    # Strategy 3: SA polish on the best so far
    if time.time() < deadline and best_result:
        # Use SA to polish
        sa_deadline = min(deadline, start_time + 105)
        # Reconstruct placed list from best_result
        placed_sa = []
        for idx, (bin_, x, y, z, rw, rh, rd) in best_result.items():
            if bin_ == 0:
                placed_sa.append((x, y, z, rw, rh, rd))
        
        # Run a lightweight SA-like local search
        T = 500.0
        cooling_rate = 0.99
        min_T = 10.0
        max_iter = 3000
        
        current_placements = dict(best_result)
        current_placed = list(placed_sa)
        
        for it in range(max_iter):
            if time.time() > sa_deadline or T < min_T:
                break
            
            # Pick a random placed box
            placed_idxs = [i for i, v in current_placements.items() if v[0] == 0]
            if not placed_idxs:
                break
            
            idx = random.choice(placed_idxs)
            bin_, x, y, z, rw, rh, rd = current_placements[idx]
            
            box_w, box_h, box_d = boxes[idx]
            rots = get_rotations(box_w, box_h, box_d)
            
            # Try removing and re-placing with a random rotation
            new_rot = random.choice(rots)
            new_rw, new_rh, new_rd = new_rot
            
            # Remove from placed
            new_placed = [p for p in current_placed if not (p[0] == x and p[1] == y and p[2] == z and 
                                                           p[3] == rw and p[4] == rh and p[5] == rd)]
            
            pos = find_best_fit(new_placed, new_rw, new_rh, new_rd)
            if pos is not None:
                new_x, new_y, new_z = pos
                candidate_placed = new_placed + [(new_x, new_y, new_z, new_rw, new_rh, new_rd)]
                
                # Objective: minimize sum of z (compactness)
                old_obj = sum(p[2] for p in current_placed)
                new_obj = sum(p[2] for p in candidate_placed)
                
                delta = new_obj - old_obj
                if delta <= 0 or random.random() < math.exp(-delta / T):
                    current_placed = candidate_placed
                    current_placements[idx] = (0, new_x, new_y, new_z, new_rw, new_rh, new_rd)
            
            T *= cooling_rate
        
        # Evaluate final
        vol_sa = evaluate_result(current_placed, current_placements)
        if vol_sa > best_volume:
            best_volume = vol_sa
            best_result = current_placements
    
    # Build final output
    if best_result:
        for idx in range(n):
            if idx in best_result:
                final_placements[idx] = best_result[idx]
            else:
                final_placements[idx] = (-1, 0, 0, 0, boxes[idx][0], boxes[idx][1], boxes[idx][2])
    
    return final_placements
# EVOLVE-BLOCK-END
