import time
import itertools
import random

def load(boxes, container):
    start_time = time.time()
    # Safety deadline: 115 seconds to ensure we return before the 120s hard limit
    deadline = start_time + 115.0
    
    W, H, D = container
    # Store original dimensions and index
    indexed_boxes = [(i, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    
    def blb_solve(ordered_indices):
        """
        Runs the Bottom-Left-Back heuristic on a specific ordering of box indices.
        Returns the best placement found for this ordering.
        """
        placed = []
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(len(boxes))]
        candidates = set([(0, 0, 0)])
        
        for idx in ordered_indices:
            w, h, d = boxes[idx]
            
            # Check time limit inside the loop
            if time.time() > deadline:
                return results, False # False indicates timeout
            
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
                        # Check intersection in X, Y, Z
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
                        # We can tweak these weights. Z is most important to keep stack low.
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
                # 1. On top
                candidates.add((bx, by, bz + bd))
                # 2. To the right
                candidates.add((bx + bw, by, bz))
                # 3. In front
                candidates.add((bx, by + bh, bz))
                # 4. Corner (right and front)
                candidates.add((bx + bw, by + bh, bz))
                # 5. Top-Right
                candidates.add((bx + bw, by, bz + bd))
                # 6. Top-Front
                candidates.add((bx, by + bh, bz + bd))
                # 7. Top-Right-Front
                candidates.add((bx + bw, by + bh, bz + bd))
                
        return results, True # True indicates completion

    def calculate_score(results):
        total_vol = 0
        for bin_id, x, y, z, w, h, d in results:
            if bin_id == 0:
                total_vol += w * h * d
        return total_vol

    best_results = None
    best_score = 0
    
    # Define different sorting strategies
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], # Volume
        lambda i: max(boxes[i]),                            # Max Dimension
        lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2], # Surface Area
        lambda i: boxes[i][0] * boxes[i][1],                # Base Area (assuming z is height, but rotations allowed so it's just one face)
        lambda i: -min(boxes[i]),                           # Min Dimension (ascending)
    ]
    
    indices = list(range(len(boxes)))
    
    # Try each strategy
    for sort_func in strategies:
        if time.time() > deadline:
            break
            
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        results, completed = blb_solve(sorted_indices)
        
        if completed:
            score = calculate_score(results)
            if score > best_score:
                best_score = score
                best_results = results
    
    # If we have time left, try some random restarts
    # This helps escape local optima of the deterministic sorts
    if best_results is not None and time.time() < deadline - 10.0:
        attempts = 0
        # We can afford a few random shuffles if the base cases were fast
        # Let's say we try up to 20 random restarts if time permits
        while time.time() < deadline - 5.0 and attempts < 20:
            # Shuffle indices, but maybe keep large boxes somewhat prioritized?
            # Pure random is a good escape mechanism.
            random.shuffle(indices)
            results, completed = blb_solve(indices)
            attempts += 1
            
            if completed:
                score = calculate_score(results)
                if score > best_score:
                    best_score = score
                    best_results = results

    # Fallback if no boxes were placed or best_results is None
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results
