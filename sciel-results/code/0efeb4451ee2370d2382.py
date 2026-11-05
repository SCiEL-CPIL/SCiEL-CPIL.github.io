"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
import itertools

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # 1. Sort boxes by volume descending (Best Fit Decreasing)
    indexed_boxes = list(enumerate(boxes))
    indexed_boxes.sort(key=lambda x: x[1][0]*x[1][1]*x[1][2], reverse=True)
    
    placed = [] # List of (x, y, z, w, h, d, orig_index)
    results = [None] * len(boxes)
    
    # 2. Initialize candidate positions with (0,0,0)
    # We use a simple list for candidates, but we will generate all potential spots
    # and evaluate them to find the "best" one (minimizing max extent).
    # To avoid explosion of candidates, we maintain a set of unique corner points.
    # For N=60, the number of corners is manageable (at most 3*N + 1).
    
    candidate_points = set([(0, 0, 0)])
    
    for orig_idx, (w, h, d) in indexed_boxes:
        if time.time() > deadline:
            break
            
        # Generate unique orientations
        box_dims = sorted([w, h, d], reverse=True)
        seen_orientations = set()
        orientations = []
        for perm in itertools.permutations(box_dims):
            if perm not in seen_orientations:
                seen_orientations.add(perm)
                orientations.append(perm)
        
        best_pos = None
        best_score = float('inf')
        best_orient = None
        found_placement = False
        
        # Iterate through all current candidate points
        # We must check all of them to find the "Best Fit" (min max extent)
        for cx, cy, cz in candidate_points:
            if time.time() > deadline:
                break
                
            for ow, oh, od in orientations:
                # Check bounds
                if cx + ow > W or cy + oh > H or cz + od > D:
                    continue
                
                # Check overlap with all placed boxes
                overlap = False
                for px, py, pz, pw, ph, pd, _ in placed:
                    # Separation check:
                    # If any of these is true, they do NOT overlap
                    if not (cx + ow <= px or px + pw <= cx or 
                            cy + oh <= py or py + ph <= cy or 
                            cz + od <= pz or pz + pd <= cz):
                        overlap = True
                        break
                
                if not overlap:
                    # Calculate score: Max extent of the box in this position
                    # We want to minimize this to keep the bounding box tight
                    # Using max(x+w, y+h, z+d) is a common heuristic, but 
                    # a weighted sum or just the max of the new extents relative to container 
                    # might be better. Let's use max(cx+ow, cy+oh, cz+od)
                    # Actually, a better metric for "Best Fit" in 3D is often 
                    # minimizing the sum of distances to the origin or max coordinate.
                    # Let's use max(cx+ow, cy+oh, cz+od) as the primary sort key.
                    score = max(cx + ow, cy + oh, cz + od)
                    
                    # Tie-breaker: minimize volume wasted? Or just lexicographic order of coords
                    # To ensure deterministic and compact results, add secondary keys
                    # We want smaller x, then y, then z
                    # But the primary goal is minimizing the max extent.
                    # Let's combine them: (score, cx, cy, cz)
                    if score < best_score or (score == best_score and (cx, cy, cz) < best_pos):
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_orient = (ow, oh, od)
                        found_placement = True
        
        if found_placement:
            bx, by, bz = best_pos
            bw, bh, bd = best_orient
            
            # Place the box
            placed.append((bx, by, bz, bw, bh, bd, orig_idx))
            results[orig_idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Generate new candidate positions
            # The new box creates new "corners" that can be used to place future boxes.
            # 1. (bx + bw, by, bz)  -> Next to it in X
            # 2. (bx, by + bh, bz)  -> Next to it in Y
            # 3. (bx, by, bz + bd)  -> On top of it in Z
            
            new_candidates = [
                (bx + bw, by, bz),
                (bx, by + bh, bz),
                (bx, by, bz + bd)
            ]
            
            # Add to candidate set if within container bounds and not already present
            for nc in new_candidates:
                if 0 <= nc[0] <= W and 0 <= nc[1] <= H and 0 <= nc[2] <= D:
                    candidate_points.add(nc)
        else:
            results[orig_idx] = (-1, 0, 0, 0, w, h, d)
            
    # Fill any remaining results (if deadline hit mid-loop) as left out
    for i in range(len(boxes)):
        if results[i] is None:
            w, h, d = boxes[i]
            results[i] = (-1, 0, 0, 0, w, h, d)
            
    return results
# EVOLVE-BLOCK-END
