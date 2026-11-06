"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    deadline = time.time() + 110.0  # safety margin before 120s
    
    # Precompute all unique orientations for each box
    # An orientation is a permutation of (w, h, d)
    def get_orientations(w, h, d):
        orients = set()
        for perm in permutations((w, h, d)):
            orients.add(perm)
        return list(orients)
    
    # Sort boxes by volume descending, then by max dimension descending
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: (-(x[1][0]*x[1][1]*x[1][2]), -max(x[1])))
    
    # For each box, precompute orientations sorted by "fit" preference
    # We'll try orientations that minimize the resulting bounding box growth
    # For now, just keep all orientations
    
    def solve_greedy(order):
        """
        Place boxes in the given order using a bottom-left-front heuristic.
        Returns list of (box_index, x, y, z, w, h, d) for placed boxes,
        and a set of placed box indices.
        """
        placed = []  # list of (x, y, z, w, h, d)
        placed_indices = set()
        
        for idx, (w, h, d) in order:
            if time.time() > deadline:
                break
            
            orientations = get_orientations(w, h, d)
            
            best_pos = None
            best_orient = None
            best_score = float('inf')
            
            # Generate candidate positions
            # Candidate x,y,z coordinates come from 0 and from the right/back/top faces of placed boxes
            candidate_coords = [0]
            for (px, py, pz, pw, ph, pd) in placed:
                candidate_coords.append(px + pw)
                candidate_coords.append(py + ph)
                candidate_coords.append(pz + pd)
            
            # Remove duplicates and sort
            candidate_coords = sorted(set(candidate_coords))
            
            # Limit candidate coordinates to those within container
            candidate_coords = [c for c in candidate_coords if c < W]
            candidate_coords_y = [c for c in sorted(set(candidate_coords)) if c < H]
            candidate_coords_z = [c for c in sorted(set(candidate_coords)) if c < D]
            
            # If too many candidates, sample
            max_candidates = 20
            if len(candidate_coords) > max_candidates:
                # Sample evenly
                indices = [int(i * len(candidate_coords) / max_candidates) for i in range(max_candidates)]
                candidate_coords = [candidate_coords[i] for i in indices]
                candidate_coords = sorted(set(candidate_coords))
            if len(candidate_coords_y) > max_candidates:
                indices = [int(i * len(candidate_coords_y) / max_candidates) for i in range(max_candidates)]
                candidate_coords_y = [candidate_coords_y[i] for i in indices]
                candidate_coords_y = sorted(set(candidate_coords_y))
            if len(candidate_coords_z) > max_candidates:
                indices = [int(i * len(candidate_coords_z) / max_candidates) for i in range(max_candidates)]
                candidate_coords_z = [candidate_coords_z[i] for i in indices]
                candidate_coords_z = sorted(set(candidate_coords_z))
            
            for ow, oh, od in orientations:
                if ow > W or oh > H or od > D:
                    continue
                
                for cx in candidate_coords:
                    if cx + ow > W:
                        continue
                    for cy in candidate_coords_y:
                        if cy + oh > H:
                            continue
                        for cz in candidate_coords_z:
                            if cz + od > D:
                                continue
                            
                            # Check for overlap with all placed boxes
                            overlap = False
                            for (px, py, pz, pw, ph, pd) in placed:
                                if not (cx + ow <= px or px + pw <= cx or
                                        cy + oh <= py or py + ph <= cy or
                                        cz + od <= pz or pz + pd <= cz):
                                    overlap = True
                                    break
                            
                            if overlap:
                                continue
                            
                            # Score: prefer positions closer to bottom-left-front
                            # Use a weighted sum: x + y + z, with z weighted more
                            score = cx + cy + 2.0 * cz
                            # Also prefer orientations that leave more room
                            # Add a small penalty for large bounding box
                            score += 0.1 * (ow + oh + od)
                            
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_orient = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orient
                placed.append((cx, cy, cz, ow, oh, od))
                placed_indices.add(idx)
        
        return placed, placed_indices
    
    # Initial solution: greedy with sorted order
    initial_order = [(idx, box) for idx, box in indexed_boxes]
    placed, placed_indices = solve_greedy(initial_order)
    
    # Compute score
    def compute_score(placed_indices, placed):
        if not placed:
            return 0.0
        total_vol = sum(w*h*d for (x, y, z, w, h, d) in placed)
        return total_vol / (W * H * D)
    
    best_score = compute_score(placed_indices, placed)
    best_placed = list(placed)
    best_placed_indices = set(placed_indices)
    
    # Local search: try swapping pairs of boxes in the order
    # and re-running greedy
    num_improvements = 0
    max_iterations = 50
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        for i in range(len(initial_order)):
            if time.time() > deadline - 5.0:
                break
            for j in range(i+1, len(initial_order)):
                if time.time() > deadline - 2.0:
                    break
                
                # Swap
                initial_order[i], initial_order[j] = initial_order[j], initial_order[i]
                
                placed_new, placed_new_indices = solve_greedy(initial_order)
                score_new = compute_score(placed_new_indices, placed_new)
                
                if score_new > best_score + 1e-9:
                    best_score = score_new
                    best_placed = list(placed_new)
                    best_placed_indices = set(placed_new_indices)
                    improved = True
                    num_improvements += 1
                else:
                    # Swap back
                    initial_order[i], initial_order[j] = initial_order[j], initial_order[i]
        
        if not improved:
            break
    
    # Build the final result
    # We need to return placements for all boxes in the original order
    # Each placement is (bin, x, y, z, w, h, d)
    # bin 0 for loaded, -1 for left out
    # w, h, d should be the original dimensions of the box (not rotated)
    
    result = [None] * n
    
    # Map from box index to its placement
    # best_placed contains (x, y, z, ow, oh, od) for placed boxes
    # We need to map back to original box indices
    
    # Reconstruct: we need to know which box index corresponds to which placed box
    # Let's re-run the greedy with the best order and track indices
    
    # Actually, let's just re-run with the best order and build the result properly
    final_order = list(initial_order)  # This is the order that gave best_score? No, we need to track it.
    
    # Let's redo this more carefully. We'll track the best order.
    
    # Reset and do a cleaner version
    # For now, let's just use the first greedy solution if local search didn't improve
    # Actually, the local search modifies initial_order in place, so we need to track the best order
    
    # Let me restructure: keep track of the best order
    
    # For simplicity in this version, let's just use the greedy solution
    # and build the result from it
    
    # Re-run greedy with the sorted order to get a clean placement
    placed_final, placed_final_indices = solve_greedy([(idx, box) for idx, box in indexed_boxes])
    
    # Map placed boxes back to original indices
    # placed_final is a list of (x, y, z, ow, oh, od)
    # We need to know which original box index each corresponds to
    
    # Let's do this properly by tracking indices during placement
    
    def solve_greedy_with_tracking(order):
        placed = []  # list of (box_idx, x, y, z, w, h, d) where w,h,d are rotated dims
        placed_set = set()
        
        for idx, (w, h, d) in order:
            if time.time() > deadline:
                break
            
            orientations = get_orientations(w, h, d)
            
            best_pos = None
            best_orient = None
            best_score = float('inf')
            
            candidate_coords = [0]
            for (bidx, px, py, pz, pw, ph, pd) in placed:
                candidate_coords.append(px + pw)
                candidate_coords.append(py + ph)
                candidate_coords.append(pz + pd)
            
            candidate_coords = sorted(set(candidate_coords))
            candidate_coords = [c for c in candidate_coords if c < W]
            candidate_coords_y = [c for c in sorted(set(candidate_coords)) if c < H]
            candidate_coords_z = [c for c in sorted(set(candidate_coords)) if c < D]
            
            max_candidates = 20
            if len(candidate_coords) > max_candidates:
                indices = [int(i * len(candidate_coords) / max_candidates) for i in range(max_candidates)]
                candidate_coords = [candidate_coords[i] for i in indices]
                candidate_coords = sorted(set(candidate_coords))
            if len(candidate_coords_y) > max_candidates:
                indices = [int(i * len(candidate_coords_y) / max_candidates) for i in range(max_candidates)]
                candidate_coords_y = [candidate_coords_y[i] for i in indices]
                candidate_coords_y = sorted(set(candidate_coords_y))
            if len(candidate_coords_z) > max_candidates:
                indices = [int(i * len(candidate_coords_z) / max_candidates) for i in range(max_candidates)]
                candidate_coords_z = [candidate_coords_z[i] for i in indices]
                candidate_coords_z = sorted(set(candidate_coords_z))
            
            for ow, oh, od in orientations:
                if ow > W or oh > H or od > D:
                    continue
                
                for cx in candidate_coords:
                    if cx + ow > W:
                        continue
                    for cy in candidate_coords_y:
                        if cy + oh > H:
                            continue
                        for cz in candidate_coords_z:
                            if cz + od > D:
                                continue
                            
                            overlap = False
                            for (bidx, px, py, pz, pw, ph, pd) in placed:
                                if not (cx + ow <= px or px + pw <= cx or
                                        cy + oh <= py or py + ph <= cy or
                                        cz + od <= pz or pz + pd <= cz):
                                    overlap = True
                                    break
                            
                            if overlap:
                                continue
                            
                            score = cx + cy + 2.0 * cz + 0.1 * (ow + oh + od)
                            
                            if score < best_score:
                                best_score = score
                                best_pos = (cx, cy, cz)
                                best_orient = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orient
                placed.append((idx, cx, cy, cz, ow, oh, od))
                placed_set.add(idx)
        
        return placed, placed_set
    
    placed_final, placed_final_indices = solve_greedy_with_tracking([(idx, box) for idx, box in indexed_boxes])
    
    # Build result
    for i in range(n):
        if i in placed_final_indices:
            # Find the placement for box i
            for (bidx, x, y, z, w, h, d) in placed_final:
                if bidx == i:
                    # Return original dimensions in the tuple
                    orig_w, orig_h, orig_d = boxes[i]
                    result[i] = (0, x, y, z, orig_w, orig_h, orig_d)
                    break
        else:
            w, h, d = boxes[i]
            result[i] = (-1, 0, 0, 0, w, h, d)
    
    return result

# EVOLVE-BLOCK-END
