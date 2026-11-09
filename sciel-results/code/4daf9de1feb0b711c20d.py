"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    n = len(boxes)
    container_vol = W * H * D
    if container_vol == 0:
        # Degenerate case
        return [(0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]

    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    # 1. Initial Greedy Pack (Best-Fit Decreasing style)
    # We use a simple greedy approach to get a valid starting state.
    # State: list of (bin_id, x, y, z, w, h, d)
    
    # Sort indices by volume descending
    indices = list(range(n))
    indices.sort(key=lambda i: -(boxes[i][0]*boxes[i][1]*boxes[i][2]))
    
    # bins_data: list of lists of placed boxes (x, y, z, w, h, d)
    bins_data = []
    # placement: index -> (bin_id, x, y, z)
    placement = {}

    def get_bins():
        return bins_data

    def try_place_greedy(idx, w, h, d):
        # Try existing bins
        # Sort bins by "fullness" (volume used) descending to prefer filling up
        # But for speed in initialization, just iterate.
        # Actually, Best-Fit: find the bin where the box fits in the tightest space?
        # Simpler: First Fit Decreasing with some heuristics.
        
        # Heuristic: Try to place in bin with most used space first? 
        # Or bin with least remaining space in some dimension?
        # Let's stick to a robust simple greedy:
        # For each bin, check if it fits. If multiple, pick the one that minimizes 
        # the sum of distances to walls (compaction) for that specific placement.
        
        best_bin = -1
        best_pos = None
        best_compactness = float('inf')
        
        for bi, bin_boxes in enumerate(bins_data):
            # Check if box fits in this bin at all? 
            # Simple check: volume used + box volume <= container volume (necessary not sufficient)
            # Better: just try to find a spot.
            
            # Generate candidate spots: (0,0,0) and corners of existing boxes
            # For initialization speed, we can be a bit lazy or use a grid.
            # Given N=40, O(N^2) per box is fine.
            
            candidates = [(0,0,0)]
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                candidates.append((bx + bw, by, bz))
                candidates.append((bx, by + bh, bz))
                candidates.append((bx, by, bz + bd))
                candidates.append((bx + bw, by + bh, bz))
                candidates.append((bx + bw, by, bz + bd))
                candidates.append((bx, by + bh, bz + bd))
                candidates.append((bx + bw, by + bh, bz + bd))
                
            # Remove duplicates
            candidates = list(set(candidates))
            
            for (cx, cy, cz) in candidates:
                if cx + w > W or cy + h > H or cz + d > D:
                    continue
                
                # Check overlap
                overlap = False
                for (bx, by, bz, bw, bh, bd) in bin_boxes:
                    if not (cx + w <= bx or bx + bw <= cx or
                            cy + h <= by or by + bh <= cy or
                            cz + d <= bz or bz + bd <= cz):
                        overlap = True
                        break
                
                if not overlap:
                    # Calculate compactness score (sum of distances to max walls)
                    # We want to minimize gaps: (W - (cx+w)) + (H - (cy+h)) + (D - (cz+d))
                    # Actually, standard "bottom-left-back" is cx+cy+cz.
                    # Let's use cx+cy+cz to keep it simple and aligned with "compactness"
                    score = cx + cy + cz
                    if score < best_compactness:
                        best_compactness = score
                        best_bin = bi
                        best_pos = (cx, cy, cz)
        
        if best_bin != -1:
            bins_data[best_bin].append((best_pos[0], best_pos[1], best_pos[2], w, h, d))
            placement[idx] = (best_bin, best_pos[0], best_pos[1], best_pos[2])
            return True
        
        # Open new bin
        bi = len(bins_data)
        bins_data.append([])
        bins_data[bi].append((0, 0, 0, w, h, d))
        placement[idx] = (bi, 0, 0, 0)
        return True

    for idx in indices:
        w, h, d = boxes[idx]
        if w > W or h > H or d > D:
            # Invalid instance assumption: just place at 0,0,0 in bin 0 to avoid crash, 
            # though this will result in overlap. The problem states valid instances.
            # For safety, we still need to return something.
            placement[idx] = (0, 0, 0, 0)
            if not bins_data:
                bins_data.append([])
            bins_data[0].append((0,0,0,w,h,d))
            continue
        try_place_greedy(idx, w, h, d)
    
    # 2. Local Search (Simulated Annealing)
    # Objective: Minimize Total Waste
    # Waste = Sum over bins of (Container Vol - Used Vol)
    # Note: This effectively minimizes B * Container Vol - Total Box Vol
    # Since Total Box Vol is constant, minimizing Waste is equivalent to minimizing B.
    # However, SA can also improve the "last bin filled" metric if we tweak the objective,
    # but primarily we want to reduce B.
    
    # To make SA effective, we need to define moves.
    # Move 1: Move a box from Bin A to Bin B.
    # Move 2: Move a box to a different position within the same Bin.
    
    # For speed and simplicity, we will implement a "Shake" strategy:
    # Randomly select a box, remove it, and try to re-insert it into any bin 
    # (including its original) at the "best" compactness position.
    # If the new total waste is lower (or accepted by SA), keep it.
    
    # We need a fast way to calculate waste and check validity.
    # Since we are doing many iterations, we need efficiency.
    
    # State representation for SA:
    # We maintain `bins_data` and `placement`.
    
    def calc_total_waste():
        total_waste = 0
        for bin_boxes in bins_data:
            used_vol = sum(b[3]*b[4]*b[5] for b in bin_boxes)
            total_waste += (container_vol - used_vol)
        return total_waste

    current_waste = calc_total_waste()
    
    # SA Parameters
    T = 1000.0
    cooling_rate = 0.99
    min_T = 1.0
    
    # Pre-calculate box volumes for speed
    box_vols = [w*h*d for w,h,d in boxes]
    
    # Helper to find best position for a box in a specific bin
    # Returns (x, y, z) or None
    def find_best_pos_in_bin(bin_id, w, h, d, exclude_idx=None):
        bin_boxes = bins_data[bin_id]
        candidates = [(0,0,0)]
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            candidates.append((bx + bw, by, bz))
            candidates.append((bx, by + bh, bz))
            candidates.append((bx, by, bz + bd))
            candidates.append((bx + bw, by + bh, bz))
            candidates.append((bx + bw, by, bz + bd))
            candidates.append((bx, by + bh, bz + bd))
            candidates.append((bx + bw, by + bh, bz + bd))
        
        candidates = list(set(candidates))
        
        best_pos = None
        best_score = float('inf')
        
        for (cx, cy, cz) in candidates:
            if cx + w > W or cy + h > H or cz + d > D:
                continue
            
            overlap = False
            for (bx, by, bz, bw, bh, bd) in bin_boxes:
                if not (cx + w <= bx or bx + bw <= cx or
                        cy + h <= by or by + bh <= cy or
                        cz + d <= bz or bz + bd <= cz):
                    overlap = True
                    break
            
            if not overlap:
                score = cx + cy + cz
                if score < best_score:
                    best_score = score
                    best_pos = (cx, cy, cz)
        
        return best_pos

    # SA Loop
    iterations = 0
    while time.time() < deadline and T > min_T:
        # Select a random box to move
        idx = random.randint(0, n-1)
        w, h, d = boxes[idx]
        
        # Get current position
        cur_bin, cx, cy, cz = placement[idx]
        
        # Remove from current bin
        # Find the exact box in the list
        # We need to match (cx, cy, cz, w, h, d)
        # Note: Multiple boxes could have same dims and pos? Unlikely if valid.
        # We'll find the first match.
        removed = False
        for i, b in enumerate(bins_data[cur_bin]):
            if b[0]==cx and b[1]==cy and b[2]==cz and b[3]==w and b[4]==h and b[5]==d:
                bins_data[cur_bin].pop(i)
                removed = True
                break
        
        if not removed:
            # Should not happen
            continue
            
        # Try to re-insert into any bin
        best_new_waste = float('inf')
        best_new_bin = -1
        best_new_pos = None
        
        # We can optimize by only trying bins that have enough volume
        # But for now, try all.
        for bi in range(len(bins_data)):
            pos = find_best_pos_in_bin(bi, w, h, d)
            if pos:
                # Calculate waste change
                # Waste change = (New Waste of Bin bi) - (Old Waste of Bin bi) 
                #               + (New Waste of Bin cur_bin) - (Old Waste of Bin cur_bin)
                
                # Old Waste of Bin bi:
                old_vol_bi = sum(b[3]*b[4]*b[5] for b in bins_data[bi])
                old_waste_bi = container_vol - old_vol_bi
                
                # New Waste of Bin bi:
                new_vol_bi = old_vol_bi + box_vols[idx]
                new_waste_bi = container_vol - new_vol_bi
                
                # Old Waste of Bin cur_bin (after removal):
                # We already removed it. So current bins_data[cur_bin] is the "after" state.
                # We need to compare against the state BEFORE removal.
                # Let's restructure: 
                # 1. Remove box.
                # 2. Calc waste of all bins.
                # 3. For each candidate bin, simulate adding box.
                
                # This is getting expensive. Let's simplify.
                # We just calculate the total waste if we place it in bin `bi`.
                # Total Waste = Sum(ContVol - UsedVol) for all bins.
                # = B * ContVol - Sum(UsedVol).
                # Since Sum(UsedVol) is constant (sum of all box vols), minimizing Waste 
                # is equivalent to minimizing B (number of bins).
                # Wait, if we move a box from Bin A to Bin B, and Bin A becomes empty, B decreases by 1.
                # If Bin A is not empty, B stays same.
                # So the objective function should just be B (number of non-empty bins).
                # But "last bin occupied volume fraction" is also in the score.
                # Score = B + 0.1 * (1 - last_frac).
                # Minimizing B is the dominant term.
                
                # So, let's define the objective for SA as:
                # Objective = B * 1000 + (1 - last_frac) 
                # This prioritizes reducing B.
                
                # However, calculating B and last_frac every step is O(N).
                # Let's stick to minimizing Total Waste as a proxy, which correlates with B.
                # Actually, Total Waste = B * ContVol - TotalBoxVol.
                # So minimizing Total Waste is EXACTLY minimizing B.
                
                # So, if we move a box from Bin A to Bin B:
                # If Bin A becomes empty: Waste decreases by ContVol. (Huge gain)
                # If Bin A is not empty: Waste change is 0 (box moves from one bin to another, total used vol same, B same).
                # Wait.
                # Waste = Sum(C - V_i) for i in 1..B.
                # If we move box from A to B, and A is not empty:
                # V_A decreases by v, V_B increases by v.
                # Waste_A increases by v, Waste_B decreases by v. Total Waste unchanged.
                # If A becomes empty:
                # Waste_A was C - V_A. Now A is empty, so it's not counted in B?
                # If we define B as number of non-empty bins, then yes.
                # But our `bins_data` list might keep empty bins.
                # We should clean up empty bins or handle B count carefully.
                
                # Let's define a function to get the "True Score" based on current state.
                def get_true_score():
                    # Count non-empty bins
                    active_bins = [b for b in bins_data if len(b) > 0]
                    B = len(active_bins)
                    if B == 0:
                        return 0.0
                    # Find the last bin (max index in original labeling? 
                    # The problem says "largest label defines the last bin".
                    # Our bins_data indices are 0..N-1.
                    # We should ensure we use the max index that has boxes.
                    max_bin_idx = max(i for i, b in enumerate(bins_data) if len(b) > 0)
                    last_vol = sum(b[3]*b[4]*b[5] for b in bins_data[max_bin_idx])
                    frac = last_vol / container_vol
                    return B + 0.1 * (1 - frac)
                
                # For SA, we need a fast proxy.
                # Proxy: Number of non-empty bins.
                # If we can reduce non-empty bins, great.
                # If not, we can try to improve the last bin's fill.
                
                # Let's just use `get_true_score` if it's fast enough?
                # N=40. B <= 40.
                # get_true_score is O(N) to sum volumes.
                # SA iterations can be ~10,000-100,000.
                # 100,000 * 40 = 4,000,000 ops. This is very fast in Python (< 1s).
                # So we CAN use the true score!
                
                # Re-evaluating the move:
                # We removed box from cur_bin.
                # We are considering placing it in bi.
                
                # Let's just simulate the placement and calculate the score.
                # To avoid modifying state repeatedly, we can calculate the score delta.
                
                # Delta Score = Score(New State) - Score(Old State)
                
                # This is complex to derive analytically.
                # Given the low N, let's just do the full check for the best candidate.
                # But we need to find the BEST candidate first.
                # We can use a simple heuristic for candidate selection:
                # 1. Try to place in a bin that would become "more full" or "complete".
                # 2. Or just try all bins, calculate the score for each, and pick the best.
                
                pass # Placeholder, we'll implement the loop below

        # Actually, let's simplify the SA step:
        # 1. Remove box.
        # 2. For each bin, try to place.
        # 3. Calculate the score of the resulting state (temporarily placing).
        # 4. Pick the best one.
        # 5. Apply SA acceptance.
        
        best_score_candidate = float('inf')
        best_candidate_bin = -1
        best_candidate_pos = None
        
        for bi in range(len(bins_data)):
            pos = find_best_pos_in_bin(bi, w, h, d)
            if pos:
                # Temporarily place
                bins_data[bi].append((pos[0], pos[1], pos[2], w, h, d))
                placement[idx] = (bi, pos[0], pos[1], pos[2])
                
                score = get_true_score()
                
                if score < best_score_candidate:
                    best_score_candidate = score
                    best_candidate_bin = bi
                    best_candidate_pos = pos
                    best_candidate_score = score
                else:
                    # Undo temporary placement
                    bins_data[bi].pop()
                    # Note: placement[idx] is updated, but we will overwrite it if we accept
                    # or revert if we reject.
        
        if best_candidate_bin == -1:
            # Could not place anywhere? Should not happen if it was placed before.
            # Revert to original position
            bins_data[cur_bin].append((cx, cy, cz, w, h, d))
            placement[idx] = (cur_bin, cx, cy, cz)
        else:
            # We have the best candidate.
            # If we didn't keep it in the state (because we popped it in the else block of the loop),
            # we need to place it.
            # Wait, the logic above is flawed.
            # If we found a better score in bin X, we left it there.
            # If we found a worse score in bin Y, we popped it.
            # But what if the FIRST bin tried was the best? We left it there.
            # What if a LATER bin was better? We popped the previous one and left the new one.
            # So at the end of the loop, the box is in `best_candidate_bin` IF we accepted it as best so far.
            # BUT, we need to compare `best_candidate_score` with `current_score` (before removal).
            
            # We need the score BEFORE removal to calculate delta.
            # Let's restructure:
            # 1. Calc score_before.
            # 2. Remove box.
            # 3. Loop bins, find best placement and score_after_best.
            # 4. Accept if score_after_best < score_before or random < exp(-delta/T).
            
            pass # We will rewrite the loop properly below.

    # Let's rewrite the SA loop cleanly.
    
    # Restore state to initial greedy pack
    # (We messed with bins_data in the placeholder logic above? 
    # No, the placeholder logic was `pass`. So state is still initial greedy.)
    
    # Recalculate current score
    current_score = get_true_score()
    
    T = 1000.0
    cooling_rate = 0.995
    min_T = 1.0
    
    # Pre-compute box volumes
    box_vols = [w*h*d for w,h,d in boxes]
    
    while time.time() < deadline and T > min_T:
        idx = random.randint(0, n-1)
        w, h, d = boxes[idx]
        
        # Get current position
        cur_bin, cx, cy, cz = placement[idx]
        
        # 1. Calculate Score Before
        score_before = current_score
        
        # 2. Remove box from cur_bin
        removed = False
        for i, b in enumerate(bins_data[cur_bin]):
            if b[0]==cx and b[1]==cy and b[2]==cz and b[3]==w and b[4]==h and b[5]==d:
                bins_data[cur_bin].pop(i)
                removed = True
                break
        
        if not removed:
            T *= cooling_rate
            continue
            
        # 3. Try to re-insert into all bins, find the one that minimizes score
        best_new_score = float('inf')
        best_new_bin = -1
        best_new_pos = None
        
        for bi in range(len(bins_data)):
            pos = find_best_pos_in_bin(bi, w, h, d)
            if pos:
                # Temporarily place
                bins_data[bi].append((pos[0], pos[1], pos[2], w, h, d))
                placement[idx] = (bi, pos[0], pos[1], pos[2])
                
                score_candidate = get_true_score()
                
                if score_candidate < best_new_score:
                    best_new_score = score_candidate
                    best_new_bin = bi
                    best_new_pos = pos
                    # Keep it placed for now
                else:
                    # Undo
                    bins_data[bi].pop()
                    # placement[idx] will be overwritten if we find a better one later,
                    # or reverted if we reject.
        
        if best_new_bin == -1:
            # Revert to original
            bins_data[cur_bin].append((cx, cy, cz, w, h, d))
            placement[idx] = (cur_bin, cx, cy, cz)
        else:
            # We have the best candidate in `best_new_bin`.
            # The box is currently placed in `best_new_bin` (from the loop above).
            # We need to decide whether to accept this move.
            
            delta = best_new_score - score_before
            
            if delta < 0:
                # Accept
                current_score = best_new_score
                # State is already updated to best_new_bin
            else:
                # Reject with probability exp(-delta/T)
                if random.random() < pow(2.718281828, -delta / T):
                    # Accept
                    current_score = best_new_score
                    # State is already updated
                else:
                    # Reject: Revert to original position
                    # Remove from best_new_bin
                    # Find the box we just added
                    # It's the last one we added? 
                    # Not necessarily, if we added to bin A, then bin B was better, we popped A.
                    # So the box is in best_new_bin.
                    # We need to remove it from best_new_bin and put it back in cur_bin.
                    
                    # Remove from best_new_bin
                    # Find match
                    bx, by, bz = best_new_pos
                    for i, b in enumerate(bins_data[best_new_bin]):
                        if b[0]==bx and b[1]==by and b[2]==bz and b[3]==w and b[4]==h and b[5]==d:
                            bins_data[best_new_bin].pop(i)
                            break
                    
                    # Put back in cur_bin
                    bins_data[cur_bin].append((cx, cy, cz, w, h, d))
                    placement[idx] = (cur_bin, cx, cy, cz)
                    # current_score remains score_before
        
        T *= cooling_rate

    # 3. Finalize
    # Clean up empty bins? The score function handles it, but for output, 
    # we should map to consecutive bin IDs? 
    # "Bin labels are nonnegative integers and need not be consecutive"
    # So we can keep the original indices.
    
    final_placements = []
    for i in range(n):
        bi, x, y, z = placement[i]
        w, h, d = boxes[i]
        final_placements.append((bi, x, y, z, w, h, d))
        
    return final_placements

# EVOLVE-BLOCK-END
