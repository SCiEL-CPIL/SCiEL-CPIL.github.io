"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def pack(boxes, container):
    W, H, D = container
    if not boxes:
        return []
    
    n = len(boxes)
    container_vol = W * H * D
    if container_vol == 0:
        # Degenerate case, return empty placements or zeros
        return [(0, 0, 0, 0, 0, 0) for _ in boxes]

    start_time = time.time()
    DEADLINE = start_time + 145.0

    # Pre-calculate box volumes
    box_vols = [w * h * d for w, h, d in boxes]

    def check_validity(placement, bins):
        """
        Verify that placement is valid.
        placement: list of (bin_idx, x, y, z, w, h, d) corresponding to input boxes order.
        bins: dict bin_idx -> list of (x,y,z,w,h,d)
        """
        if len(placement) != n:
            return False
        
        # Check overlaps within each bin
        bin_boxes = {}
        for i, (bi, x, y, z, w, h, d) in enumerate(placement):
            # Check bounds
            if x < 0 or y < 0 or z < 0:
                return False
            if x + w > W or y + h > H or z + d > D:
                return False
            
            # Check dimensions match
            if (w, h, d) != boxes[i]:
                return False
            
            if bi not in bin_boxes:
                bin_boxes[bi] = []
            bin_boxes[bi].append((x, y, z, w, h, d))
        
        # Check overlaps
        for bi, boxes_in_bin in bin_boxes.items():
            m = len(boxes_in_bin)
            for i in range(m):
                x1, y1, z1, w1, h1, d1 = boxes_in_bin[i]
                for j in range(i + 1, m):
                    x2, y2, z2, w2, h2, d2 = boxes_in_bin[j]
                    # Overlap if not separated in any dimension
                    if not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                            y1 + h1 <= y2 or y2 + h2 <= y1 or
                            z1 + d1 <= z2 or z2 + d2 <= z1):
                        return False
        return True

    def calculate_score(bins):
        """
        Calculate score for a packing represented as dict bin_idx -> list of boxes.
        Returns (score, is_valid, last_bin_vol_frac)
        """
        if not bins:
            return 0.0, True, 1.0 # Empty is valid? Score 0?
            
        # Check validity
        # We assume bins are valid if constructed by our placer, but let's double check bounds/overlap for safety in local eval?
        # For speed in search, we trust the placer. But for final result, we should ensure validity.
        
        occupied_bins = set(bins.keys())
        B = len(occupied_bins)
        
        if B == 0:
            return 0.0, True, 1.0
            
        last_bin_idx = max(occupied_bins)
        
        last_bin_vol = sum(w*h*d for (x,y,z,w,h,d) in bins[last_bin_idx])
        
        if container_vol == 0:
            frac = 1.0
        else:
            frac = last_bin_vol / container_vol
            if frac > 1.0: frac = 1.0
            if frac < 0.0: frac = 0.0
        
        score = B + 0.1 * (1 - frac)
        return score, True, frac

    def get_candidates(bin_boxes, w, h, d):
        """
        Generate candidate positions for placing a box of size (w,h,d) in a bin.
        Returns a list of (score, x, y, z)
        """
        candidates = set()
        
        # Origin
        candidates.add((0, 0, 0))
        
        # Faces of existing boxes
        for (bx, by, bz, bw, bh, bd) in bin_boxes:
            # X-axis faces
            candidates.add((bx + bw, by, bz))
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
            
            # Y-axis faces
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx + bw, by + bh, bz + bd))
            
            # Z-axis faces
            candidates.add((bx, by, bz + bd))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz + bd))
            
            # Combinations (corners)
            candidates.add((bx + bw, by + bh, bz + bd))
            candidates.add((bx + bw, by, bz + bd))
            candidates.add((bx, by + bh, bz + bd))
            candidates.add((bx + bw, by + bh, bz))
            candidates.add((bx, by, bz + bd))
            candidates.add((bx + bw, by, bz))
            candidates.add((bx, by + bh, bz))
            candidates.add((bx, by, bz)) # redundant with origin if empty, but safe

        # Filter and score
        scored = []
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
                # Heuristic: Prefer positions close to (0,0,0) to keep packing compact.
                # Also prefer filling 'gaps' if possible?
                # Simple sum of coordinates works well for "bottom-left-front"
                score = cx + cy + cz
                # Add a small penalty for large gaps?
                # Let's stick to simple coordinate sum for now.
                scored.append((score, cx, cy, cz))
        
        # Sort by score (ascending)
        scored.sort(key=lambda x: x[0])
        return scored

    def try_pack(order, bin_priority_func):
        """
        Attempt to pack boxes in the given order.
        bin_priority_func: function(bins, current_bin_indices) -> sorted list of bin indices to try
        """
        bins = {} # bin_idx -> list of (x,y,z,w,h,d)
        next_bin_idx = 0
        placement = [None] * n # (bin_idx, x, y, z)
        
        for i in order:
            w, h, d = boxes[i]
            
            placed = False
            
            # If no bins exist, create one
            if not bins:
                bins[0] = []
                next_bin_idx = 1
                cx, cy, cz = 0, 0, 0
                bins[0].append((cx, cy, cz, w, h, d))
                placement[i] = (0, cx, cy, cz)
                continue
            
            # Get candidate positions for existing bins
            # We need to decide which bins to try.
            # Strategy: Try all existing bins, but ordered by some heuristic.
            # Common heuristics:
            # 1. Best Fit: Bin where the box fits with least leftover space? Hard to calc quickly.
            # 2. Worst Fit: Bin with most free space?
            # 3. First Fit: Try bins in order 0, 1, 2...
            # 4. Most Full: Try bins with highest current volume first.
            
            existing_bins = list(bins.keys())
            # Sort bins by volume (descending) to try to fill full bins? 
            # Or ascending to spread? 
            # For minimizing B, filling full bins is good.
            # Let's try sorting by current volume descending.
            bin_vols = {bi: sum(bw*bh*bd for (bx,by,bz,bw,bh,bd) in bins[bi]) for bi in existing_bins}
            existing_bins.sort(key=lambda bi: -bin_vols[bi])
            
            # For each bin, find best position
            # To save time, we can limit the number of candidate positions we check per bin?
            # For N=40, checking all candidates is likely fast enough.
            
            best_candidate_overall = None
            best_candidate_score = float('inf')
            
            for bi in existing_bins:
                cands = get_candidates(bins[bi], w, h, d)
                if not cands:
                    continue
                
                # Pick the best candidate in this bin (first one in sorted list)
                # Or should we pick the best across ALL bins?
                # Yes, global best fit is generally better.
                cand_score, cx, cy, cz = cands[0]
                
                # Refine score: We want to minimize B first.
                # If we place in an existing bin, B doesn't increase.
                # If we place in a new bin, B increases.
                # So placing in ANY existing bin is better than a new bin.
                # Among existing bins, we want the most compact placement.
                
                # Add a small bias to prefer bins that are already "fuller" or "more used"?
                # Or just use the coordinate sum.
                # Let's use the coordinate sum.
                
                if cand_score < best_candidate_score:
                    best_candidate_score = cand_score
                    best_candidate_overall = (bi, cx, cy, cz)
            
            if best_candidate_overall:
                bi, cx, cy, cz = best_candidate_overall
                bins[bi].append((cx, cy, cz, w, h, d))
                placement[i] = (bi, cx, cy, cz)
                placed = True
            else:
                # Cannot place in any existing bin, open a new one
                bi = next_bin_idx
                bins[bi] = []
                next_bin_idx += 1
                cx, cy, cz = 0, 0, 0
                bins[bi].append((cx, cy, cz, w, h, d))
                placement[i] = (bi, cx, cy, cz)
        
        # Post-processing: Re-index bins to maximize last-bin volume
        # The score depends on the bin with the MAX index.
        # We can swap bin indices arbitrarily.
        # We want the bin with the largest volume to have the largest index.
        
        if not bins:
            return []
        
        # Calculate volume of each bin
        bin_volumes = []
        for bi, boxes_in_bin in bins.items():
            vol = sum(bw*bh*bd for (bx,by,bz,bw,bh,bd) in boxes_in_bin)
            bin_volumes.append((vol, bi))
        
        # Sort by volume descending
        bin_volumes.sort(key=lambda x: -x[0])
        
        # Assign new indices:
        # Largest volume -> highest index
        # Smallest volume -> lowest index
        
        # We need to map old_bi to new_bi
        old_to_new = {}
        # We want indices 0..K-1
        # The bin with max vol should be K-1
        # The bin with min vol should be 0
        
        # So, if we sort descending:
        # vol[0] (max) -> index K-1
        # vol[K-1] (min) -> index 0
        
        K = len(bin_volumes)
        for i, (vol, old_bi) in enumerate(bin_volumes):
            # i=0 is max vol. New index should be K-1
            # i=K-1 is min vol. New index should be 0
            new_bi = K - 1 - i
            old_to_new[old_bi] = new_bi
        
        # Construct final placements
        final_placements = []
        for i in range(n):
            if placement[i] is None:
                # Should not happen
                final_placements.append((0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]))
            else:
                old_bi, x, y, z = placement[i]
                new_bi = old_to_new[old_bi]
                w, h, d = boxes[i]
                final_placements.append((new_bi, x, y, z, w, h, d))
                
        return final_placements

    def evaluate_placement(placement):
        """
        Evaluate a list of placements.
        Returns score.
        """
        if not placement:
            return 0.0
        
        bins = {}
        for i, (bi, x, y, z, w, h, d) in enumerate(placement):
            if bi not in bins:
                bins[bi] = []
            bins[bi].append((x, y, z, w, h, d))
            
        # Check validity
        if not check_validity(placement, bins):
            return float('inf')
            
        score, _, _ = calculate_score(bins)
        return score

    # --- Search Strategies ---
    
    # Generate different orderings
    # 1. Volume Desc
    # 2. Max Dim Desc
    # 3. Volume Asc (sometimes helps with small gaps?)
    # 4. Random
    # 5. Area Desc
    
    def get_ordering(strategy, rng=None):
        indices = list(range(n))
        if strategy == "vol_desc":
            indices.sort(key=lambda i: -box_vols[i])
        elif strategy == "vol_asc":
            indices.sort(key=lambda i: box_vols[i])
        elif strategy == "maxdim_desc":
            indices.sort(key=lambda i: -max(boxes[i]))
        elif strategy == "maxdim_asc":
            indices.sort(key=lambda i: max(boxes[i]))
        elif strategy == "area_desc":
            indices.sort(key=lambda i: -(2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
        elif strategy == "area_asc":
            indices.sort(key=lambda i: (2*(boxes[i][0]*boxes[i][1] + boxes[i][1]*boxes[i][2] + boxes[i][0]*boxes[i][2])))
        elif strategy == "random":
            if rng:
                rng.shuffle(indices)
            else:
                random.shuffle(indices)
        else:
            # Default
            indices.sort(key=lambda i: -box_vols[i])
        return indices

    best_placements = None
    best_score = float('inf')
    
    # Define a set of deterministic strategies to run first
    deterministic_strategies = ["vol_desc", "maxdim_desc", "area_desc", "vol_asc", "maxdim_asc", "area_asc"]
    
    # Run deterministic strategies
    for strat in deterministic_strategies:
        if time.time() > DEADLINE - 10:
            break
        order = get_ordering(strat)
        placements = try_pack(order, None)
        score = evaluate_placement(placements)
        if score < best_score:
            best_score = score
            best_placements = placements

    # If time remains, run randomized restarts
    rng = random.Random(42)
    
    # We can also vary the bin ordering heuristic in future, but for now we just vary box order.
    # And maybe we can try "Worst Fit" bin selection? 
    # The current try_pack uses "Best Fit" (lowest coord sum) across bins.
    
    # Let's add a flag to try_pack for bin selection strategy?
    # For now, just more random orders.
    
    restarts = 0
    while time.time() < DEADLINE - 5:
        strat = "random"
        # Occasionally try a mix: Sort by volume, then randomize ties?
        # Or just full random.
        order = get_ordering(strat, rng)
        
        placements = try_pack(order, None)
        score = evaluate_placement(placements)
        
        if score < best_score:
            best_score = score
            best_placements = placements
            # If we get a very good score (e.g. 1 bin), we can stop?
            # Score 1.0 + 0.1*(1-frac). If 1 bin, score is 1.0 + 0.1*(1-frac).
            # If we fill the bin completely, score is 1.0.
            # If we use 2 bins, score is at least 2.0.
            # So < 2.0 is a good result.
            if best_score < 1.1: # Very close to optimal 1.0
                break
                
        restarts += 1
        # Limit restarts if we are spending too much time per restart
        if restarts % 10 == 0:
             if time.time() > DEADLINE - 10:
                 break

    if best_placements is None:
        # Fallback: Simple sequential packing
        # Try vol_desc one more time just in case
        order = get_ordering("vol_desc")
        best_placements = try_pack(order, None)
        
        if not best_placements:
             # Absolute fallback
             best_placements = []
             bins = [[]]
             for i, (w, h, d) in enumerate(boxes):
                 # Just try to place at 0,0,0 in bin 0, if fail, new bin?
                 # This is dangerous. Let's use the first successful deterministic run.
                 # If even that failed, something is wrong.
                 pass

    return best_placements

# EVOLVE-BLOCK-END
