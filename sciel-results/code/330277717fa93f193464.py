"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def pack(boxes, container):
    """
    Place boxes into bins using a Best-Fit-Decreasing heuristic with local search compaction.
    Returns list of (bin_id, x, y, z, w, h, d) for each box in original order.
    """
    W, H, D = container
    n = len(boxes)
    if n == 0:
        return []
    
    # Safety margin: The evaluator kills at 120s (or 150s per prompt text, but 120s is safer).
    # Let's aim to finish well before 115s.
    deadline = time.time() + 110.0 

    # 1. Sort boxes by volume descending (Largest First)
    # We keep original indices to restore order at the end
    indexed_boxes = [(i, w * h * d, w, h, d) for i, (w, h, d) in enumerate(boxes)]
    indexed_boxes.sort(key=lambda x: x[1], reverse=True)

    # bins: list of lists. Each inner list contains (x, y, z, w, h, d) of placed boxes.
    bins = []
    
    # Mapping from original index to its placement (bin_id, x, y, z, w, h, d)
    # We will store this in a dict or list, but since we move boxes during local search,
    # it's better to reconstruct the final result from the `bins` structure at the end.
    # However, we need to know which box is which. 
    # Let's store boxes as objects or tuples with an ID.
    # Box representation in bins: (id, x, y, z, w, h, d)
    
    def can_place(bin_placements, x, y, z, w, h, d):
        """Check if a box fits at (x,y,z) in the given bin without overlap and within bounds."""
        if x < 0 or y < 0 or z < 0:
            return False
        if x + w > W or y + h > H or z + d > D:
            return False
        for (pid, px, py, pz, pw, ph, pd) in bin_placements:
            # Check overlap
            if (x < px + pw and x + w > px and
                y < py + ph and y + h > py and
                z < pz + pd and z + d > pz):
                return False
        return True

    def find_best_position(bin_placements, w, h, d):
        """
        Find the best position for a box in a bin.
        Strategy: Generate candidate positions from the "corners" of existing boxes.
        Score: Minimize the sum of coordinates (prefer bottom-left-front).
        """
        if not bin_placements:
            # If bin is empty, place at origin
            if w <= W and h <= H and d <= D:
                return (0, 0, 0)
            return None

        # Generate candidate positions
        # The "free space" corners are formed by the extents of existing boxes.
        # We consider x-coordinates from {0} U {px + pw for each box}
        # Same for y and z.
        # This creates a grid of potential positions.
        
        xs = {0}
        ys = {0}
        zs = {0}
        
        for (pid, px, py, pz, pw, ph, pd) in bin_placements:
            if px + pw <= W: xs.add(px + pw)
            if py + ph <= H: ys.add(py + ph)
            if pz + pd <= D: zs.add(pz + pd)
            
        # Sort to ensure deterministic behavior and potentially find "lowest" positions earlier
        xs = sorted(xs)
        ys = sorted(ys)
        zs = sorted(zs)
        
        best_pos = None
        best_score = float('inf')
        
        # Iterate through candidate positions
        # To optimize, we can skip positions that are obviously bad, but with N=40, 
        # the number of candidates is manageable (approx (N+1)^3 in worst case, but usually less).
        
        for x in xs:
            if x + w > W:
                continue
            for y in ys:
                if y + h > H:
                    continue
                for z in zs:
                    if z + d > D:
                        continue
                    
                    if can_place(bin_placements, x, y, z, w, h, d):
                        # Score: Prefer positions that are "lower" in the bin.
                        # This encourages stacking and filling from the bottom up.
                        score = x + y + z
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y, z)
                            
        return best_pos

    def bin_bounding_box_volume(bin_placements):
        """Calculate the volume of the bounding box of the bin's contents."""
        if not bin_placements:
            return 0
        max_x = max(px + pw for (pid, px, py, pz, pw, ph, pd) in bin_placements)
        max_y = max(py + ph for (pid, px, py, pz, pw, ph, pd) in bin_placements)
        max_z = max(pz + pd for (pid, px, py, pz, pw, ph, pd) in bin_placements)
        return max_x * max_y * max_z

    def solve(order):
        """Solve with a specific box order. Returns (bins, score)."""
        bins = []
        for idx, vol, w, h, d in order:
            best_bin_idx = -1
            best_pos = None
            best_cost = float('inf')
            
            # Try to place in existing bins
            for b_idx, bin_placements in enumerate(bins):
                pos = find_best_position(bin_placements, w, h, d)
                if pos is not None:
                    x, y, z = pos
                    new_max_x = max(x + w, *(px + pw for (pid, px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else x + w
                    new_max_y = max(y + h, *(py + ph for (pid, px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else y + h
                    new_max_z = max(z + d, *(pz + pd for (pid, px, py, pz, pw, ph, pd) in bin_placements)) if bin_placements else z + d
                    cost = new_max_x * new_max_y * new_max_z
                    if cost < best_cost:
                        best_cost = cost
                        best_bin_idx = b_idx
                        best_pos = pos
            
            if best_bin_idx >= 0:
                x, y, z = best_pos
                bins[best_bin_idx].append((idx, x, y, z, w, h, d))
            else:
                if w <= W and h <= H and d <= D:
                    new_bin = [(idx, 0, 0, 0, w, h, d)]
                    bins.append(new_bin)
                else:
                    new_bin = [(idx, 0, 0, 0, w, h, d)]
                    bins.append(new_bin)
        
        # Local Search Compaction
        compaction_passed = True
        passes = 0
        max_passes = 5
        while compaction_passed and passes < max_passes and time.time() < deadline - 5:
            compaction_passed = False
            passes += 1
            for b_idx in range(len(bins) - 1, 0, -1):
                if time.time() > deadline - 5:
                    break
                if not bins[b_idx]:
                    bins.pop(b_idx)
                    compaction_passed = True
                    continue
                boxes_in_bin = list(bins[b_idx])
                for box in boxes_in_bin:
                    if time.time() > deadline - 5:
                        break
                    box_id, bx, by, bz, bw, bh, bd = box
                    moved = False
                    for target_bin_idx in range(b_idx):
                        pos = find_best_position(bins[target_bin_idx], bw, bh, bd)
                        if pos is not None:
                            bins[b_idx].remove(box)
                            nx, ny, nz = pos
                            bins[target_bin_idx].append((box_id, nx, ny, nz, bw, bh, bd))
                            moved = True
                            compaction_passed = True
                            break
                    if moved:
                        pass
            if compaction_passed:
                new_bins = []
                for b in bins:
                    if b:
                        new_bins.append(b)
                bins = new_bins
        
        # Calculate score
        if not bins:
            return [], 0
        B = len(bins)
        last_bin = bins[-1]
        last_vol = sum(w*h*d for (_, _, _, _, w, h, d) in last_bin)
        cont_vol = W * H * D
        last_frac = last_vol / cont_vol if cont_vol > 0 else 0
        score = B + 0.1 * (1 - last_frac)
        
        return bins, score

    # Phase 1: Initial Greedy
    initial_bins, initial_score = solve(indexed_boxes)
    
    # Phase 2: Randomized Restarts
    best_bins = initial_bins
    best_score = initial_score
    
    # Use remaining time for restarts
    # We want to try a few different orderings
    # 1. Volume descending (already done)
    # 2. Volume descending with random tie-breaking
    # 3. Random shuffles of the volume-sorted list (perturbation)
    
    # Let's define a few strategies
    strategies = []
    
    # Strategy 1: Standard BFD (done)
    
    # Strategy 2: BFD with random tie-breaking for equal volumes
    # We can just shuffle indices with same volume
    # Or simply: Shuffle the whole list but keep large items earlier? 
    # No, let's do: Sort by volume, but add small random noise to volume.
    
    # Strategy 3: Random perturbations of the sorted order
    # Swap a few random pairs in the sorted list
    
    num_restarts = 20 # Max restarts, but time-limited
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Create a perturbed order
        # Start with volume sorted
        order = list(indexed_boxes)
        
        # Add noise to volume for sorting? Or just swap?
        # Let's try swapping random pairs
        # The number of swaps can increase with restart count
        num_swaps = 1 + r // 5
        for _ in range(num_swaps):
            i = random.randint(0, len(order) - 1)
            j = random.randint(0, len(order) - 1)
            if i != j:
                order[i], order[j] = order[j], order[i]
        
        # Solve
        bins_r, score_r = solve(order)
        
        if score_r < best_score:
            best_score = score_r
            best_bins = bins_r

    # Phase 3: Finalize Result
    result = [None] * n
    for b_idx, bin_placements in enumerate(best_bins):
        for (box_id, x, y, z, w, h, d) in bin_placements:
            result[box_id] = (b_idx, x, y, z, w, h, d)
            
    # Fallback
    current_max_bin = len(best_bins)
    for i in range(n):
        if result[i] is None:
            w, h, d = boxes[i]
            result[i] = (current_max_bin, 0, 0, 0, w, h, d)
            current_max_bin += 1

    return result
# EVOLVE-BLOCK-END
