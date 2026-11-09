"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

# EVOLVE-BLOCK-START
def load(boxes, container):
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    
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
        
        # Also try positions aligned with existing box edges
        for (px, py, pz, pw, ph, pd) in placed:
            candidates.append((px, py, pz + pd))
            candidates.append((px + pw, py, pz))
        
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
            # Use a weighted sum that encourages filling gaps near walls
            score = cx + cy + cz
            # Penalize positions that leave large gaps
            # Encourage tight packing by preferring lower z, then y, then x
            score = cz * 100 + cy * 10 + cx
            
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
    
    # Try a few random orders if time permits
    import random
    random.seed(42)
    for trial in range(10):
        if time.time() - start_time > DEADLINE - 5:
            break
        order = list(range(n))
        random.shuffle(order)
        results = run_pack(order)
        vol = sum(w*h*d for (b, x, y, z, w, h, d) in results if b == 0)
        if vol > best_volume:
            best_volume = vol
            best_results = results
    
    if best_results is None:
        best_results = [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    return best_results
# EVOLVE-BLOCK-END
