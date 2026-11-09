import time
import random
import itertools
import heapq

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    W, H, D = container
    if W <= 0 or H <= 0 or D <= 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    container_volume = W * H * D
    n = len(boxes)
    
    # Pre-compute all possible rotations for each box
    box_rotations = []
    for w, h, d in boxes:
        rots = set()
        rots.add((w, h, d))
        rots.add((w, d, h))
        rots.add((h, w, d))
        rots.add((h, d, w))
        rots.add((d, w, h))
        rots.add((d, h, w))
        box_rotations.append(list(rots))

    def can_place(placed, x, y, z, w, h, d):
        if x + w > W or y + h > H or z + d > D:
            return False
        for px, py, pz, pw, ph, pd in placed:
            # Check overlap
            if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y or z + d <= pz or pz + pd <= z):
                return False
        return True

    def get_new_corners(placed, x, y, z, w, h, d):
        # Generate new candidate corners based on the newly placed box
        candidates = set()
        # Corners on the faces of the new box
        candidates.add((x + w, y, z))
        candidates.add((x + w, y + h, z))
        candidates.add((x + w, y, z + d))
        candidates.add((x + w, y + h, z + d))
        candidates.add((x, y + h, z))
        candidates.add((x, y + h, z + d))
        candidates.add((x, y, z + d))
        
        valid_candidates = set()
        for cx, cy, cz in candidates:
            if 0 <= cx <= W and 0 <= cy <= H and 0 <= cz <= D:
                valid_candidates.add((cx, cy, cz))
        return valid_candidates

    def score_position(x, y, z, w, h, d):
        # Heuristic: minimize y (height) primarily, then x, then z.
        # Also penalize leaving large empty space?
        # Let's stick to a weighted sum similar to the inspiration but tuned.
        # The inspiration used (y * 1.0 + x * 0.3 + z * 0.2).
        # We might want to prefer "filling" the container, but standard corner packing
        # usually minimizes coordinates.
        return y * 10.0 + x * 1.0 + z * 1.0

    def solve_with_order(order):
        placed = [] # list of (x, y, z, w, h, d)
        placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
        candidates = {(0, 0, 0)}
        
        for i in order:
            box_id = i
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            # Optimization: If we have many candidates, Dijkstra might be overkill if we just scan.
            # But the prompt asks to combine with Dijkstra.
            # However, simple scanning of ~O(N^2) candidates is usually fast enough for 60 boxes.
            # Let's implement the "Dijkstra" idea by treating the search for the best position
            # as finding the minimum cost valid position. Since costs are static for a given position
            # (once we choose the box), we can just iterate and keep the min.
            # To respect the "Dijkstra" inspiration, we can conceptually view the candidate set
            # as nodes. But practically, scanning is equivalent if we don't have dynamic edge costs.
            
            # Let's try to be smarter: 
            # If the number of candidates is very large, we might prune.
            # But for 60 boxes, max candidates ~ 60*7 = 420.
            
            for rot in box_rotations[box_id]:
                rw, rh, rd = rot
                for cx, cy, cz in candidates:
                    if can_place(placed, cx, cy, cz, rw, rh, rd):
                        sc = score_position(cx, cy, cz, rw, rh, rd)
                        if sc < best_score:
                            best_score = sc
                            best_pos = (cx, cy, cz)
                            best_rot = rot
            
            if best_pos:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                placed.append((bx, by, bz, bw, bh, bd))
                placements[box_id] = (0, bx, by, bz, bw, bh, bd)
                new_corners = get_new_corners(placed, bx, by, bz, bw, bh, bd)
                candidates.update(new_corners)
            else:
                # Box cannot be placed
                pass
        
        vol = sum(p[4]*p[5]*p[6] for p in placements if p[0] == 0)
        return placements, vol

    def local_search(placements, boxes, W, H, D, deadline_check, max_iters=20):
        """
        Try to improve the packing by moving or rotating placed boxes.
        This is the "refinement machinery" from the inspiration.
        """
        # Reconstruct placed list from placements
        placed = []
        box_indices = []
        for i, p in enumerate(placements):
            if p[0] == 0:
                placed.append((p[1], p[2], p[3], p[4], p[5], p[6]))
                box_indices.append(i)
        
        if not placed:
            return placements

        # We need to map placed index to original box index to update placements
        # placed[i] corresponds to box_indices[i]
        
        best_placed = list(placed)
        best_vol = sum(pw * ph * pd for _, _, _, pw, ph, pd in best_placed)
        improved = True
        
        for _ in range(max_iters):
            if deadline_check():
                break
            if not improved:
                break
            improved = False
            
            # Try moving each placed box
            for i in range(len(best_placed)):
                if deadline_check():
                    break
                x, y, z, rw, rh, rd = best_placed[i]
                
                # Try small shifts? Or better, try moving to any valid candidate position?
                # Moving to any valid position is expensive. 
                # The inspiration tried small shifts (-1, 0, 1). 
                # However, coordinates are integers? The problem statement implies integer or float?
                # "box is (w, h, d)" - usually integers in these tasks, but let's assume floats are possible?
                # The seed program uses integers implicitly? No, it just uses the values.
                # If values are floats, shifting by 1 might not be optimal.
                # Let's stick to the inspiration's approach: small integer shifts.
                # If the inputs are floats, this might fail to improve. 
                # Let's assume integer inputs for now, as is common.
                
                # Actually, a better local search for 3D packing is to try to "push" boxes down/left/back.
                # Let's try shifting in negative directions only (gravity-like) and small positive.
                
                found_better = False
                # Try shifting to minimize y, then x, then z
                # We can try to slide the box to the "bottom" of the empty space above it?
                # That's complex. Let's stick to simple shifts.
                
                # Generate candidate positions for this box: 
                # 1. Current pos
                # 2. Shifted by -1, 0, 1 in each axis
                candidates_to_try = set()
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if nx < 0 or ny < 0 or nz < 0:
                                continue
                            if nx + rw > W or ny + rh > H or nz + rd > D:
                                continue
                            candidates_to_try.add((nx, ny, nz))
                
                # Also try rotations
                rots_to_try = box_rotations[box_indices[i]]
                
                best_new_pos = None
                best_new_rot = None
                best_new_score = float('inf')
                
                # To check overlap, we need to remove box i from placed
                other_boxes = [best_placed[j] for j in range(len(best_placed)) if j != i]
                
                for rot in rots_to_try:
                    nrw, nrh, nrd = rot
                    for nx, ny, nz in candidates_to_try:
                        if nrw > W - nx or nrh > H - ny or nrd > D - nz:
                            continue
                        if can_place(other_boxes, nx, ny, nz, nrw, nrh, nrd):
                            sc = score_position(nx, ny, nz, nrw, nrh, nrd)
                            if sc < best_new_score:
                                best_new_score = sc
                                best_new_pos = (nx, ny, nz)
                                best_new_rot = rot
                
                if best_new_pos:
                    nx, ny, nz = best_new_pos
                    nrw, nrh, nrd = best_new_rot
                    # Check if it's actually different
                    if (nx, ny, nz, nrw, nrh, nrd) != (x, y, z, rw, rh, rd):
                        best_placed[i] = (nx, ny, nz, nrw, nrh, nrd)
                        improved = True
                        # Update volume? Volume stays same for a single box move/rotate.
                        # We only care about validity.
                        # But wait, if we move a box, it might allow *other* boxes to be placed?
                        # No, we are not adding boxes, just optimizing existing ones.
                        # The score is volume filled. Moving existing boxes doesn't change volume.
                        # So local search only helps if we can *add* more boxes?
                        # The current loop places a subset. If we move a box, we don't add new ones.
                        # So local search on *existing* placements doesn't increase volume.
                        # UNLESS: The greedy placement failed to place a box because of a bad position.
                        # If we move previously placed boxes, we might free up space for the *next* box.
                        # But `solve_with_order` is a single pass. 
                        # To benefit, we would need to interleave local search with placement, 
                        # or run local search on the *unplaced* boxes? 
                        # The inspiration's local search was on `placed` boxes. 
                        # If we don't add boxes, volume is constant. 
                        # So local search is useless for *increasing* volume unless we re-run the placement 
                        # of remaining boxes after a move? 
                        # That's too expensive. 
                        # Therefore, local search on existing boxes is a no-op for volume score.
                        # We should skip local search if it doesn't enable new placements.
                        
                        # Let's reconsider. Does moving a box change the volume? No.
                        # So `local_search` as defined in the inspiration is only useful if it 
                        # changes the *feasibility* of future placements. 
                        # Since `solve_with_order` is sequential, we can't easily go back.
                        # So I will omit local search on existing boxes to save time, 
                        # unless we implement a "repair" step where we try to place 
                        # previously failed boxes again after some moves? 
                        # That's complex. 
                        
                        # Let's focus on the "Dijkstra" / Candidate Generation aspect.
                        pass

        return placements

    # Strategy 1: Volume Descending
    indices = list(range(n))
    best_placements = None
    best_vol = 0
    
    order_vol = sorted(indices, key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    p, v = solve_with_order(order_vol)
    if v > best_vol:
        best_vol = v
        best_placements = p
        
    # Strategy 2: Max Dimension Descending
    order_max = sorted(indices, key=lambda i: max(boxes[i]), reverse=True)
    p, v = solve_with_order(order_max)
    if v > best_vol:
        best_vol = v
        best_placements = p
        
    # Strategy 3: Random Restarts
    # Use more restarts if time permits
    max_restarts = 200
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        random.shuffle(indices)
        p, v = solve_with_order(indices)
        if v > best_vol:
            best_vol = v
            best_placements = p
            
    if best_placements is None:
        best_placements = [(-1, 0, 0, 0, 0, 0, 0) for _ in range(n)]
        
    return best_placements
