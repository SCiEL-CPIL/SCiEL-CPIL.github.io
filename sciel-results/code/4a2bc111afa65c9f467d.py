"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import permutations

def _rotations(w, h, d):
    """Return unique rotations of a box as (w, h, d) tuples, sorted by volume (same) then by orientation."""
    seen = set()
    result = []
    for perm in permutations((w, h, d)):
        if perm not in seen:
            seen.add(perm)
            result.append(perm)
    # Sort rotations: prefer orientations that are "flatter" (smaller height) to allow stacking
    # Actually, we want to try orientations that fit well. Let's just return unique ones.
    return result

def load(boxes, container):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    W, H, D = container
    container_volume = W * H * D
    if container_volume == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    if n == 0:
        return []
    
    # Pre-compute all unique rotations for each box
    box_rots = []
    for w, h, d in boxes:
        box_rots.append(_rotations(w, h, d))
    
    # Sort boxes by volume descending (larger boxes first tend to pack better)
    # But we need to keep track of original indices for the output
    indices = list(range(n))
    # Sort by volume descending
    indices.sort(key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2], reverse=True)
    
    best_placements = None
    best_volume = 0
    
    # We will try a few different strategies/heuristics
    # Strategy 1: Greedy with skyline (simplified)
    # Strategy 2: Greedy with corner placement
    
    # Let's implement a robust greedy approach with multiple restarts using different orderings
    
    # To respect the time budget, we run a fixed number of attempts if time permits.
    # Each attempt: sort boxes in a specific way, then place greedily.
    
    # We'll use a simple "extreme point" or "skyline" style placement.
    # For efficiency, we maintain a list of placed boxes and check for overlaps.
    # With 60 boxes, O(N^2) per placement is 3600 checks, which is fast.
    
    def try_order(order):
        """
        Try to place boxes in the given order (list of original indices).
        Returns (placements, total_volume)
        """
        placed = []  # List of (x, y, z, w, h, d)
        total_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            rots = box_rots[idx]
            
            # Try to find a position for this box
            # We try candidate positions:
            # 1. (0,0,0)
            # 2. On top of existing boxes
            # 3. Beside existing boxes
            
            # Generate candidate positions
            candidates = [(0, 0, 0)]
            
            # For each placed box, generate positions adjacent to it
            for (px, py, pz, pw, ph, pd) in placed:
                # Right
                candidates.append((px + pw, py, pz))
                # Front
                candidates.append((px, py + ph, pz))
                # Top
                candidates.append((px, py, pz + pd))
                # Combinations of adjacent faces might help, but let's start simple
                # Actually, checking all pairs of boxes for "corner" positions is better but slower.
                # Let's add positions formed by the max x, max y, max z of subsets?
                # A simpler heuristic: try positions where x, y, z are either 0 or the end of some placed box.
                # This can explode. Let's stick to simple neighbors + origin.
            
            # Also, try positions aligned with the "skyline"
            # For each unique x, y in placed boxes + 0, find the lowest z where it fits.
            
            # Let's implement a more systematic search for the best position for this box.
            # We will iterate over all possible x, y positions from the set of "interesting" coordinates.
            # Interesting x: 0, and px+pw for each placed box.
            # Interesting y: 0, and py+ph for each placed box.
            # For each (x, y), find the lowest z such that the box fits.
            
            interesting_x = set([0])
            interesting_y = set([0])
            for (px, py, pz, pw, ph, pd) in placed:
                interesting_x.add(px + pw)
                interesting_y.add(py + ph)
                # Also maybe px, py? No, placing inside is not allowed.
                # But we might need to place in a gap.
                # Gaps are hard to find with this simple approach.
                # Let's also add px, py to interesting coordinates to allow filling gaps?
                # If we add px, we might place a box starting at px, which could overlap if there's a box there.
                # The overlap check will handle it.
                interesting_x.add(px)
                interesting_y.add(py)
            
            best_pos = None
            best_rot = None
            best_score = -1  # We want to minimize z, then x, then y to keep it compact
            
            for bw, bh, bd in rots:
                for x in interesting_x:
                    for y in interesting_y:
                        # Check bounds
                        if x + bw > W or y + bh > H:
                            continue
                        
                        # Find lowest z
                        # z candidates: 0, and pz+pd for each placed box
                        interesting_z = set([0])
                        for (px, py, pz, pw, ph, pd) in placed:
                            interesting_z.add(pz + pd)
                            interesting_z.add(pz)
                        
                        # We want the lowest z that fits
                        # Sort z candidates
                        z_candidates = sorted(interesting_z)
                        
                        found = False
                        for z in z_candidates:
                            if z + bd > D:
                                continue
                            
                            # Check overlap with all placed boxes
                            overlap = False
                            for (px, py, pz, pw, ph, pd) in placed:
                                if not (x + bw <= px or px + pw <= x):
                                    if not (y + bh <= py or py + ph <= y):
                                        if not (z + bd <= pz or pz + pd <= z):
                                            overlap = True
                                            break
                            
                            if not overlap:
                                # This position is valid
                                # Score: lower z is better, then lower x, then lower y
                                score = (z, x, y)
                                if best_pos is None or score < best_score:
                                    best_pos = (x, y, z)
                                    best_rot = (bw, bh, bd)
                                    best_score = score
                                found = True
                                break  # Found the lowest z for this x,y,rot, so move to next x,y,rot? 
                                       # Actually, we want the global best (lowest z, then x, then y).
                                       # So we shouldn't break here unless we are sure this is the best for this x,y.
                                       # But since we sorted z, this is the lowest z for this x,y.
                                       # We still need to check other x,y to see if they have even lower z.
                        
                        # If we found a valid z, we don't need to check higher z for this x,y,rot.
            
            if best_pos is not None:
                x, y, z = best_pos
                bw, bh, bd = best_rot
                placed.append((x, y, z, bw, bh, bd))
                total_vol += bw * bh * bd
                # Update interesting coordinates? 
                # The next iteration will recompute them, which is fine for N=60.
            else:
                # Box cannot be placed
                pass
        
        # Construct placements list in original order
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        for (x, y, z, bw, bh, bd) in placed:
            # We need to map back to original index.
            # This is tricky because we lost the index in the 'placed' list.
            # Let's modify try_order to keep track of indices.
            pass
        
        return placements, total_vol
    
    # The above try_order is flawed because it doesn't track which box was placed where.
    # Let's rewrite it properly.
    
    def attempt(order):
        placed_info = []  # List of (idx, x, y, z, bw, bh, bd)
        total_vol = 0
        
        for idx in order:
            w, h, d = boxes[idx]
            rots = box_rots[idx]
            
            interesting_x = set([0])
            interesting_y = set([0])
            for (_, px, py, pz, pw, ph, pd) in placed_info:
                interesting_x.add(px + pw)
                interesting_x.add(px)
                interesting_y.add(py + ph)
                interesting_y.add(py)
            
            best_pos = None
            best_rot = None
            best_score = None
            
            for bw, bh, bd in rots:
                for x in interesting_x:
                    if x + bw > W:
                        continue
                    for y in interesting_y:
                        if y + bh > H:
                            continue
                        
                        interesting_z = set([0])
                        for (_, px, py, pz, pw, ph, pd) in placed_info:
                            interesting_z.add(pz + pd)
                            interesting_z.add(pz)
                        
                        z_candidates = sorted(interesting_z)
                        
                        for z in z_candidates:
                            if z + bd > D:
                                continue
                            
                            overlap = False
                            for (_, px, py, pz, pw, ph, pd) in placed_info:
                                if not (x + bw <= px or px + pw <= x):
                                    if not (y + bh <= py or py + ph <= y):
                                        if not (z + bd <= pz or pz + pd <= z):
                                            overlap = True
                                            break
                            
                            if not overlap:
                                score = (z, x, y)
                                if best_score is None or score < best_score:
                                    best_pos = (x, y, z)
                                    best_rot = (bw, bh, bd)
                                    best_score = score
                                break  # Lowest z found for this x,y,rot
            
            if best_pos is not None:
                x, y, z = best_pos
                bw, bh, bd = best_rot
                placed_info.append((idx, x, y, z, bw, bh, bd))
                total_vol += bw * bh * bd
        
        placements = [(-1, 0, 0, 0, boxes[i][0], boxes[i][1], boxes[i][2]) for i in range(n)]
        for (idx, x, y, z, bw, bh, bd) in placed_info:
            placements[idx] = (0, x, y, z, bw, bh, bd)
        
        return placements, total_vol
    
    # Run a few attempts with different orderings
    # 1. Volume descending
    # 2. Volume ascending
    # 3. Random shuffles
    
    orders = []
    orders.append(indices)  # Volume desc
    
    indices_asc = sorted(range(n), key=lambda i: boxes[i][0] * boxes[i][1] * boxes[i][2])
    orders.append(indices_asc)  # Volume asc
    
    # Add some random shuffles
    random.seed(42)
    for _ in range(5):
        shuffled = list(range(n))
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
        placements, vol = attempt(order)
        if vol > best_volume:
            best_volume = vol
            best_placements = placements
    
    if best_placements is None:
        # Fallback to the seed algorithm if something went wrong
        placements, x, y, z, row_h, layer_d = [], 0, 0, 0, 0, 0
        for w, h, d in boxes:
            if x + w > W:
                x, y, row_h = 0, y + row_h, 0
            if y + h > H:
                y, z, layer_d, x, row_h = 0, z + layer_d, 0, 0, 0
            if z + d > D:
                placements.append((-1, 0, 0, 0, w, h, d))
                continue
            placements.append((0, x, y, z, w, h, d))
            x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
        return placements
    
    return best_placements
# EVOLVE-BLOCK-END
