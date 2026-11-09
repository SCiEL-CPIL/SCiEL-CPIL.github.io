import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0
    
    W, H, D = container
    container_vol = W * H * D
    
    # Store original dimensions and index
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    # List of placed boxes: (x, y, z, w, h, d)
    placed = []
    
    # Set of candidate positions (x, y, z)
    candidates = set([(0, 0, 0)])
    
    # Initialize results with "left out" status (-1)
    results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Sort by volume descending to place large boxes first
    indexed_boxes.sort(key=lambda x: x[1] * x[2] * x[3], reverse=True)
    
    for idx, w, h, d in indexed_boxes:
        if time.time() > deadline:
            break
        
        best_pos = None
        best_rot = None
        best_score = float('inf')
        
        # Get unique rotations
        rotations = list(set(itertools.permutations((w, h, d))))
        
        for rw, rh, rd in rotations:
            for cx, cy, cz in candidates:
                # Check if box fits in container bounds
                if cx + rw > W or cy + rh > H or cz + rd > D:
                    continue
                
                # Check for overlap with all placed boxes
                valid = True
                for px, py, pz, pw, ph, pd in placed:
                    if (cx < px + pw and px < cx + rw and
                        cy < py + ph and py < cy + rh and
                        cz < pz + pd and pz < cz + rd):
                        valid = False
                        break
                
                if valid:
                    # Heuristic Score:
                    # Minimize Z (height) first, then Y (depth/back), then X (width/left).
                    # This creates a "Bottom-Left-Back" packing.
                    # Using large multipliers ensures lexicographic ordering.
                    # We also add a small penalty for the volume to prefer tighter fits if positions are similar?
                    # No, lexicographic is safer for structural integrity.
                    score = cz * 1000000 + cy * 1000 + cx
                    
                    if score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
        
        if best_pos is not None:
            bx, by, bz = best_pos
            bw, bh, bd = best_rot
            
            # Record placement
            placed.append((bx, by, bz, bw, bh, bd))
            results[idx] = (0, bx, by, bz, bw, bh, bd)
            
            # Generate new candidate points
            # 1. On top of the placed box
            candidates.add((bx, by, bz + bd))
            # 2. To the right of the placed box
            candidates.add((bx + bw, by, bz))
            # 3. In front of the placed box
            candidates.add((bx, by + bh, bz))
            # 4. Corner (right and front)
            candidates.add((bx + bw, by + bh, bz))
            # 5. Top-Right
            candidates.add((bx + bw, by, bz + bd))
            # 6. Top-Front
            candidates.add((bx, by + bh, bz + bd))
            # 7. Top-Right-Front
            candidates.add((bx + bw, by + bh, bz + bd))
            
            # Optional: Add more diverse points?
            # The "corners" of the placed box relative to the container are often useful.
            # But the standard 3 "faces" plus origin usually suffice for the "extreme point" method.
            # Let's stick to the 7 points above which cover the 8 corners of the new box.
            
    return results
