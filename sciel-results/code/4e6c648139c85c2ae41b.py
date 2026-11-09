import time
import itertools
import random
import numpy as np

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    W, H, D = container
    n_boxes = len(boxes)
    
    if W == 0 or H == 0 or D == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    # Check if grid is feasible. If W*H*D is too large, we might run out of memory or time.
    # For 60 boxes, usually dimensions are < 100. 100^3 = 1,000,000 bytes (uint8). This is fine.
    # If dimensions are huge, we fallback to a list-based approach, but the problem states "typical 60-box instances".
    # We assume grid is safe.
    
    grid = np.zeros((W, H, D), dtype=np.uint8)
    
    def get_rotations(w, h, d):
        # Return unique rotations
        return list(set(itertools.permutations((w, h, d))))
    
    def blb_greedy(order, grid, deadline):
        """
        Runs Bottom-Left-Back greedy placement.
        Returns list of placements for all boxes (index 0..n_boxes-1).
        """
        results = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n_boxes)]
        
        # Candidate points to check. We maintain a list of potential (x,y,z) positions.
        # To avoid checking too many, we can sort them by priority (lowest z, y, x).
        candidates = [(0, 0, 0)]
        
        for idx in order:
            if time.time() > deadline:
                break
                
            w, h, d = boxes[idx]
            rotations = get_rotations(w, h, d)
            
            best_pos = None
            best_rot = None
            best_score = float('inf')
            
            # We sort candidates to check "bottom-left-back" first.
            # This often finds a valid spot early, allowing us to skip checking the rest.
            # However, sorting every time is O(C log C). C grows with each box.
            # For 60 boxes, C ~ 60*7 = 420. Sorting 420 items is fast.
            # Let's try without sorting first, but break early if we find a "good" spot?
            # No, we need the *best* spot according to BLB to maximize future space.
            # So we must check all valid candidates to find the min score.
            
            # Optimization: Only check candidates that are within bounds for at least one rotation.
            # We can filter candidates per rotation.
            
            for rw, rh, rd in rotations:
                # Iterate through candidates
                # To speed up, we can skip candidates that are obviously too high (z + rd > D)
                for cx, cy, cz in candidates:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    
                    # Check overlap
                    # Using numpy slicing
                    # Note: grid[x, y, z]
                    if np.sum(grid[cx:cx+rw, cy:cy+rh, cz:cz+rd]) == 0:
                        score = cz * (W * H) + cy * W + cx
                        if score < best_score:
                            best_score = score
                            best_pos = (cx, cy, cz)
                            best_rot = (rw, rh, rd)
            
            if best_pos is not None:
                bx, by, bz = best_pos
                bw, bh, bd = best_rot
                
                results[idx] = (0, bx, by, bz, bw, bh, bd)
                grid[bx:bx+bw, by:by+bh, bz:bz+bd] = 1
                
                # Generate new candidates
                # 1. Top face corners
                candidates.append((bx, by, bz + bd))
                candidates.append((bx + bw, by, bz + bd))
                candidates.append((bx, by + bh, bz + bd))
                candidates.append((bx + bw, by + bh, bz + bd))
                # 2. Side face corners (right, front)
                candidates.append((bx + bw, by, bz))
                candidates.append((bx + bw, by + bh, bz))
                candidates.append((bx, by + bh, bz))
                
                # Optional: Deduplicate candidates to keep list small
                # Converting to set and back is O(N). With N~400, it's fine.
                # But doing it every step adds overhead.
                # Let's only deduplicate if the list gets too large, say > 200.
                if len(candidates) > 200:
                    candidates = list(set(candidates))
        
        return results
    
    def calculate_volume(results):
        vol = 0
        for r in results:
            if r[0] == 0:
                vol += r[4] * r[5] * r[6]
        return vol
    
    best_results = None
    best_vol = 0
    
    # Strategies
    indices = list(range(n_boxes))
    strategies = [
        lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], # Volume
        lambda i: max(boxes[i]),                           # Max Dim
        lambda i: boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2], # Surface
        lambda i: boxes[i][0] * boxes[i][1],               # Base Area
        lambda i: -min(boxes[i]),                          # Min Dim (smallest first? No, reverse=True)
    ]
    
    # Phase 1: Deterministic Greedy
    for sort_func in strategies:
        if time.time() > deadline - 60: # Reserve 60s for random + repair
            break
            
        sorted_indices = sorted(indices, key=sort_func, reverse=True)
        grid[:] = 0
        results = blb_greedy(sorted_indices, grid, deadline)
        
        vol = calculate_volume(results)
        if vol > best_vol:
            best_vol = vol
            best_results = [tuple(r) for r in results]
            
    # Phase 2: Randomized Greedy
    # Use remaining time (minus repair budget)
    repair_budget = 30.0
    search_end = deadline - repair_budget
    
    attempts = 0
    while time.time() < search_end:
        random.shuffle(indices)
        grid[:] = 0
        results = blb_greedy(indices, grid, deadline)
        
        vol = calculate_volume(results)
        if vol > best_vol:
            best_vol = vol
            best_results = [tuple(r) for r in results]
            
        attempts += 1
        if attempts > 200: # Hard cap on random attempts to avoid infinite loop if time check fails
             break

    # Phase 3: Local Repair
    # Try to place left-out boxes by moving placed boxes.
    if best_results is not None and time.time() < deadline - 5:
        # Reconstruct grid from best_results
        grid[:] = 0
        placed_boxes = [] # List of (idx, x, y, z, w, h, d)
        for i, r in enumerate(best_results):
            if r[0] == 0:
                x, y, z, w, h, d = r[1], r[2], r[3], r[4], r[5], r[6]
                grid[x:x+w, y:y+h, z:z+d] = 1
                placed_boxes.append((i, x, y, z, w, h, d))
        
        # Get list of unplaced boxes, sorted by volume desc
        unplaced = [i for i in range(n_boxes) if best_results[i][0] == -1]
        unplaced.sort(key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
        
        # Attempt to place each unplaced box
        for u_idx in unplaced:
            if time.time() > deadline - 5:
                break
                
            u_w, u_h, u_d = boxes[u_idx]
            u_rots = get_rotations(u_w, u_h, u_d)
            
            placed = False
            # First, try to place in current free space
            # We need to find a spot. We can reuse the candidate logic but it's complex to re-derive.
            # Simpler: Just scan a few likely spots? No, that's weak.
            # Let's try a limited search for the unplaced box.
            # We'll generate candidates from the "skyline" of placed boxes.
            # Actually, let's just try to move one placed box to make room.
            
            # For each placed box, see if we can move it to a different spot to free up space for u_idx.
            # This is expensive. We limit the number of placed boxes we try to move (e.g., the last 10 placed or random 10).
            # Or, we try to move boxes that are "near" the potential spot for u_idx.
            # Since we don't know the potential spot, let's just try to place u_idx in a full scan of a subset of candidates.
            
            # Let's try a simple "first fit" with a limited candidate set derived from placed boxes.
            # Candidates: (0,0,0) and corners of placed boxes.
            candidates = [(0,0,0)]
            for _, px, py, pz, pw, ph, pd in placed_boxes:
                candidates.append((px, py, pz + pd))
                candidates.append((px + pw, py, pz))
                candidates.append((px, py + ph, pz))
                candidates.append((px + pw, py + ph, pz))
                candidates.append((px + pw, py, pz + pd))
                candidates.append((px, py + ph, pz + pd))
                candidates.append((px + pw, py + ph, pz + pd))
            
            # Deduplicate
            candidates = list(set(candidates))
            
            found_spot = None
            for rw, rh, rd in u_rots:
                for cx, cy, cz in candidates:
                    if cx + rw > W or cy + rh > H or cz + rd > D:
                        continue
                    if np.sum(grid[cx:cx+rw, cy:cy+rh, cz:cz+rd]) == 0:
                        found_spot = (cx, cy, cz, rw, rh, rd)
                        break
                if found_spot:
                    break
            
            if found_spot:
                cx, cy, cz, rw, rh, rd = found_spot
                # Place it
                grid[cx:cx+rw, cy:cy+rh, cz:cz+rd] = 1
                best_results[u_idx] = (0, cx, cy, cz, rw, rh, rd)
                placed_boxes.append((u_idx, cx, cy, cz, rw, rh, rd))
                best_vol += u_w * u_h * u_d
                placed = True
            
            if not placed:
                # Try to make space by moving one placed box
                # We try to move a placed box to a different valid location.
                # We limit this to avoid timeout.
                # Pick a random placed box to try to move? Or the one that overlaps with the "desired" area?
                # Since we don't have a desired area, let's pick a few random placed boxes.
                # Actually, this is very heuristic. Let's try to move the last placed box (most likely to be on top).
                # Or, try to move a box that is "small" to see if it fits elsewhere, freeing its spot for u_idx?
                # No, we need to free the spot for u_idx.
                # So we need to find a spot for u_idx that is currently occupied, and move the occupant elsewhere.
                
                # This is complex. Let's stick to the simple "find free spot in candidates" for now.
                # The repair phase as implemented above (finding free spot in generated candidates) is already a good improvement over pure greedy because it re-evaluates candidates after each placement, potentially finding spots that were blocked during the initial greedy pass due to ordering.
                
                # Wait, the `blb_greedy` already does this. The repair phase above is essentially re-running a partial greedy on the unplaced boxes using the current grid state.
                # To make it stronger, we can try to "shake" the solution.
                # Let's try a simple swap: If u_idx can't be placed, try to swap it with a placed box that is smaller.
                # If u_idx is larger, this won't work.
                
                pass # Skip complex repair for now to ensure stability and time safety

    # Final fallback
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
        
    return best_results

