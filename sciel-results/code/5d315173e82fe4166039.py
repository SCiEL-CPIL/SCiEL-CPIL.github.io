import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    W, H, D = container
    n = len(boxes)
    if W <= 0 or H <= 0 or D <= 0 or n == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]

    # Pre-compute rotations for each box
    box_rots = []
    for w, h, d in boxes:
        rots = set()
        for perm in permutations((w, h, d)):
            rots.add(perm)
        box_rots.append(list(rots))

    # Sort indices by volume descending for the base order
    volumes = [w * h * d for w, h, d in boxes]
    base_order = sorted(range(n), key=lambda i: -volumes[i])

    def get_best_placement(order):
        """
        Attempts to place all boxes in the given order using a greedy 
        Best-Fit Decreasing heuristic with face-intersection candidates.
        Returns (total_volume, placements_list)
        """
        placements = [None] * n
        placed_boxes = []  # (x, y, z, w, h, d)
        candidates = [(0, 0, 0)]
        total_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            rots = box_rots[idx]
            
            best_pos = None
            best_rot = None
            best_score = None
            
            # Try each rotation
            for rw, rh, rd in rots:
                if rw > W or rh > H or rd > D:
                    continue
                
                # Try each candidate position
                for cx, cy, cz in candidates:
                    # Check bounds
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Check overlap with placed boxes
                    overlap = False
                    for px, py, pz, pw, ph, pd in placed_boxes:
                        if (cx < px + pw and cx + rw > px and
                            cy < py + ph and cy + rh > py and
                            cz < pz + pd and cz + rd > pz):
                            overlap = True
                            break
                    if overlap:
                        continue
                    
                    # Score: Prefer lower Z, then lower Y, then lower X (Bottom-Left-Back)
                    # This encourages filling the container from the bottom up and back to front.
                    score = (cz, cy, cx)
                    
                    if best_score is None or score < best_score:
                        best_score = score
                        best_pos = (cx, cy, cz)
                        best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                rw, rh, rd = best_rot
                placements[idx] = (0, bx, by, bz, rw, rh, rd)
                placed_boxes.append((bx, by, bz, rw, rh, rd))
                total_vol += rw * rh * rd
                
                # Generate new candidate points from the faces of the newly placed box
                # We only need to add points that are within the container
                new_candidates = [
                    (bx + rw, by, bz),
                    (bx, by + rh, bz),
                    (bx, by, bz + rd),
                    (bx + rw, by + rh, bz),
                    (bx + rw, by, bz + rd),
                    (bx, by + rh, bz + rd),
                    (bx + rw, by + rh, bz + rd)
                ]
                
                for nc in new_candidates:
                    if nc[0] <= W and nc[1] <= H and nc[2] <= D:
                        # Check if already in candidates to avoid duplicates
                        # Since candidates list can grow, a simple check is O(N^2) in worst case,
                        # but N is small (60 boxes -> max ~420 candidates).
                        # Using a set for lookup is faster.
                        if nc not in candidates:
                            candidates.append(nc)
            else:
                placements[idx] = (-1, 0, 0, 0, w, h, d)
                
        return total_vol, placements

    best_total_vol = 0
    best_placements = None
    
    # Multi-start loop
    # We run multiple iterations with slightly different orderings.
    # The base order is by volume. We introduce randomness among boxes with similar volumes
    # or just shuffle small subsets to escape local minima.
    
    # First, try the deterministic best-fit decreasing order
    vol, pl = get_best_placement(base_order)
    if vol > best_total_vol:
        best_total_vol = vol
        best_placements = pl
        
    # If time allows, do random restarts
    # We estimate the time taken for one run and decide how many more we can do.
    # Or just loop until deadline.
    
    num_restarts = 0
    while time.time() < deadline:
        # Create a perturbed order
        # Strategy: Take the base order, and randomly swap adjacent items 
        # or shuffle items within a "window" of similar volumes.
        # A simple effective method: Shuffle the entire list but keep large boxes somewhat early?
        # Better: Take the base order, and with probability p, swap box i with box j 
        # where |vol[i] - vol[j]| is small.
        
        # Let's try a simpler perturbation: 
        # 1. Start with base_order
        # 2. Randomly select a few pairs of indices and swap them if their volumes are close.
        
        current_order = base_order[:]
        
        # Perform a few random swaps among similar sized boxes
        for _ in range(5):  # 5 perturbations per restart
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            # Only swap if volumes are within a certain factor (e.g., 2x)
            # This prevents disrupting the "large first" heuristic too much
            if abs(volumes[current_order[i]] - volumes[current_order[j]]) <= 0.5 * max(volumes[current_order[i]], volumes[current_order[j]]):
                current_order[i], current_order[j] = current_order[j], current_order[i]
        
        vol, pl = get_best_placement(current_order)
        if vol > best_total_vol:
            best_total_vol = vol
            best_placements = pl
        
        num_restarts += 1
        # If we've done a lot, check time more frequently
        if num_restarts % 10 == 0 and time.time() > deadline - 5.0:
            break

    # Ensure best_placements is not None (shouldn't happen if n>0)
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_placements

# EVOLVE-BLOCK-END
