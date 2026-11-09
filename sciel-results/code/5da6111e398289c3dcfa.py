"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin for 120s limit
    
    W, H, D = container
    container_vol = W * H * D
    if container_vol == 0:
        return [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    n = len(boxes)
    
    # Generate all 6 orientations for a box
    def get_orientations(w, h, d):
        dims = list(itertools.permutations((w, h, d)))
        # Remove duplicates
        seen = set()
        unique = []
        for dim in dims:
            if dim not in seen:
                seen.add(dim)
                unique.append(dim)
        return unique
    
    # Pre-compute orientations
    box_orientations = []
    for w, h, d in boxes:
        box_orientations.append(get_orientations(w, h, d))
    
    # Sort boxes by volume descending (larger first)
    box_indices = list(range(n))
    box_volumes = [boxes[i][0] * boxes[i][1] * boxes[i][2] for i in range(n)]
    box_indices.sort(key=lambda i: box_volumes[i], reverse=True)
    
    def can_place(placed, x, y, z, bw, bh, bd):
        """Check if a box of size (bw, bh, bd) at position (x, y, z) overlaps with any placed box."""
        for (px, py, pz, pw, ph, pd) in placed:
            # Check overlap on all three axes
            if (x < px + pw and x + bw > px and
                y < py + ph and y + bh > py and
                z < pz + pd and z + bd > pz):
                return False
        # Check bounds
        if x + bw > W or y + bh > H or z + bd > D:
            return False
        if x < 0 or y < 0 or z < 0:
            return False
        return True
    
    def find_position(placed, bw, bh, bd):
        """Find the best position for a box using corner-based heuristic."""
        # Candidate positions: bottom-left-front corner and positions adjacent to existing boxes
        candidates = [(0, 0, 0)]
        
        # Add positions next to existing boxes
        for (px, py, pz, pw, ph, pd) in placed:
            # Right of existing box
            candidates.append((px + pw, py, pz))
            # Above existing box
            candidates.append((px, py + ph, pz))
            # In front of existing box
            candidates.append((px, py, pz + pd))
            # Diagonal corners (extended corners)
            candidates.append((px + pw, py + ph, pz))
            candidates.append((px + pw, py, pz + pd))
            candidates.append((px, py + ph, pz + pd))
            candidates.append((px + pw, py + ph, pz + pd))
            # Also try positions at the far edges of existing boxes
            candidates.append((px, py + ph, pz))
            candidates.append((px + pw, py, pz))
        
        # Remove duplicate candidates
        candidates = list(set(candidates))
        
        best_pos = None
        best_score = float('inf')
        
        for (cx, cy, cz) in candidates:
            if cx < 0 or cy < 0 or cz < 0:
                continue
            if cx + bw > W or cy + bh > H or cz + bd > D:
                continue
            if not can_place(placed, cx, cy, cz, bw, bh, bd):
                continue
            
            # Score: prefer positions closer to origin (tighter packing)
            # Encourage tight packing by preferring lower z, then y, then x
            # Use larger weights to strongly prefer bottom layers
            score = cz * 1000 + cy * 100 + cx
            
            if score < best_score:
                best_score = score
                best_pos = (cx, cy, cz)
        
        return best_pos
    
    def run_pack(order):
        """Try to pack boxes in the given order."""
        placed = []  # List of (x, y, z, w, h, d)
        results = [None] * n  # Will hold (bin, x, y, z, w, h, d) for each box
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            for (ow, oh, od) in orientations:
                pos = find_position(placed, ow, oh, od)
                if pos is not None:
                    cx, cy, cz = pos
                    # Score: prefer positions closer to origin
                    score = cz * 100 + cy * 10 + cx
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orientation = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                placed.append((cx, cy, cz, ow, oh, od))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)
        
        return results
    
    def try_reinsert(placed, results, left_out_indices):
        """Try to re-insert left-out boxes by removing some placed boxes and repacking."""
        if not left_out_indices:
            return results, placed
        
        # Try to insert each left-out box by temporarily removing some placed boxes
        for li_idx in left_out_indices:
            bw, bh, bd = boxes[li_idx]
            orientations = box_orientations[li_idx]
            
            # Try each orientation
            for (ow, oh, od) in orientations:
                # Find a position for this box
                pos = find_position(placed, ow, oh, od)
                if pos is not None:
                    cx, cy, cz = pos
                    placed.append((cx, cy, cz, ow, oh, od))
                    results[li_idx] = (0, cx, cy, cz, ow, oh, od)
                    return results, placed
        
        return results, placed
    
    def run_pack_with_reinsert(order):
        """Try to pack boxes in the given order, then attempt to re-insert left-out boxes."""
        placed = []  # List of (x, y, z, w, h, d)
        results = [None] * n  # Will hold (bin, x, y, z, w, h, d) for each box
        
        for idx in order:
            bw, bh, bd = boxes[idx]
            orientations = box_orientations[idx]
            
            best_pos = None
            best_orientation = None
            best_score = float('inf')
            
            for (ow, oh, od) in orientations:
                pos = find_position(placed, ow, oh, od)
                if pos is not None:
                    cx, cy, cz = pos
                    # Score: prefer positions closer to origin
                    score = cz * 100 + cy * 10 + cx
                    if score < best_score:
                        best_score = score
                        best_pos = pos
                        best_orientation = (ow, oh, od)
            
            if best_pos is not None:
                cx, cy, cz = best_pos
                ow, oh, od = best_orientation
                placed.append((cx, cy, cz, ow, oh, od))
                results[idx] = (0, cx, cy, cz, ow, oh, od)
            else:
                results[idx] = (-1, 0, 0, 0, bw, bh, bd)
        
        # Identify left-out boxes
        left_out = [i for i in range(n) if results[i][0] == -1]
        
        if left_out:
            # Try to re-insert left-out boxes
            results, placed = try_reinsert(placed, results, left_out)
        
        return results
    
    # Try multiple packing orders
    best_results = None
    best_volume = -1
    
    # Order 1: Original order
    order1 = list(range(n))
    results1 = run_pack(order1)
    vol1 = sum(w*h*d for (b, x, y, z, w, h, d) in results1 if b == 0)
    if vol1 > best_volume:
        best_volume = vol1
        best_results = results1
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 2: Sorted by volume descending
    order2 = box_indices
    results2 = run_pack(order2)
    vol2 = sum(w*h*d for (b, x, y, z, w, h, d) in results2 if b == 0)
    if vol2 > best_volume:
        best_volume = vol2
        best_results = results2
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 3: Sorted by max dimension descending
    box_max_dim = [max(w, h, d) for w, h, d in boxes]
    order3 = sorted(range(n), key=lambda i: box_max_dim[i], reverse=True)
    results3 = run_pack(order3)
    vol3 = sum(w*h*d for (b, x, y, z, w, h, d) in results3 if b == 0)
    if vol3 > best_volume:
        best_volume = vol3
        best_results = results3
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Order 4: Sorted by aspect ratio (more cubic first)
    box_aspect = [max(w, h, d) / min(w, h, d) if min(w, h, d) > 0 else float('inf') for w, h, d in boxes]
    order4 = sorted(range(n), key=lambda i: box_aspect[i])
    results4 = run_pack(order4)
    vol4 = sum(w*h*d for (b, x, y, z, w, h, d) in results4 if b == 0)
    if vol4 > best_volume:
        best_volume = vol4
        best_results = results4
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        return best_results if best_results else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Try many random orders if time permits
    random.seed(42)
    for trial in range(200):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = list(range(n))
        random.shuffle(order)
        results = run_pack(order)
        vol = sum(w*h*d for (b, x, y, z, w, h, d) in results if b == 0)
        if vol > best_volume:
            best_volume = vol
            best_results = results
    
    # Try local improvement: perturbed orders based on best
    if best_results is not None and time.time() - start_time < DEADLINE - 10:
        # Try re-packing with slightly perturbed orders
        for trial in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Start from volume-sorted order and make small random swaps
            order = list(box_indices)
            for _ in range(random.randint(1, 8)):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                order[i], order[j] = order[j], order[i]
            
            results = run_pack(order)
            vol = sum(w*h*d for (b, x, y, z, w, h, d) in results if b == 0)
            if vol > best_volume:
                best_volume = vol
                best_results = results
    
    # Try reinsertion-based improvement if time permits
    if best_results is not None and time.time() - start_time < DEADLINE - 10:
        # For the best result, try to improve by reinserting left-out boxes
        left_out = [i for i in range(n) if best_results[i][0] == -1]
        if left_out:
            # Build placed list from best_results
            placed = [(x, y, z, w, h, d) for (b, x, y, z, w, h, d) in best_results if b == 0]
            
            # Try to insert each left-out box
            for li_idx in left_out:
                if time.time() - start_time > DEADLINE - 5:
                    break
                bw, bh, bd = boxes[li_idx]
                orientations = box_orientations[li_idx]
                
                for (ow, oh, od) in orientations:
                    pos = find_position(placed, ow, oh, od)
                    if pos is not None:
                        cx, cy, cz = pos
                        placed.append((cx, cy, cz, ow, oh, od))
                        best_results[li_idx] = (0, cx, cy, cz, ow, oh, od)
                        break
    
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results
# EVOLVE-BLOCK-END
