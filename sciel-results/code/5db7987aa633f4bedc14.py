import time
import random
import numpy as np
from itertools import permutations

def load(boxes, container):
    W, H, D = container
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s
    
    if not boxes:
        return []
    
    n = len(boxes)
    box_volumes = [w * h * d for w, h, d in boxes]
    container_volume = W * H * D
    
    # Get all unique rotations for each box
    def get_rotations(w, h, d):
        rots = set()
        for perm in permutations([w, h, d]):
            rots.add(perm)
        return list(rots)
    
    rotations = [get_rotations(w, h, d) for w, h, d in boxes]
    
    # Efficient 3D packing using a 3D array for occupancy
    def try_pack(order, rot_choices):
        """Try to pack boxes in given order with given rotation choices."""
        # Use a 3D numpy array for occupancy
        occ = np.zeros((W, H, D), dtype=bool)
        placements = [None] * n
        total_vol = 0
        
        for i, box_idx in enumerate(order):
            w, h, d = rot_choices[box_idx]
            
            # Find the best position using a sliding window approach
            best_pos = None
            best_score = float('inf')
            
            # Scan from bottom to top, front to back, left to right
            for z in range(D - d + 1):
                for y in range(H - h + 1):
                    for x in range(W - w + 1):
                        # Check if the box fits at (x, y, z)
                        if not occ[x:x+w, y:y+h, z:z+d].any():
                            score = z * (W * H) + y * W + x
                            if score < best_score:
                                best_score = score
                                best_pos = (x, y, z)
                            break
                    if best_pos is not None:
                        break
                if best_pos is not None:
                    break
            
            if best_pos is not None:
                x, y, z = best_pos
                occ[x:x+w, y:y+h, z:z+d] = True
                placements[box_idx] = (0, x, y, z, w, h, d)
                total_vol += w * h * d
            else:
                placements[box_idx] = (-1, 0, 0, 0, boxes[box_idx][0], boxes[box_idx][1], boxes[box_idx][2])
        
        return placements, total_vol
    
    best_placements = None
    best_volume = 0
    
    # Strategy 1: Sort by volume descending
    order1 = sorted(range(n), key=lambda i: -box_volumes[i])
    rot_choices1 = [rotations[i][0] for i in range(n)]
    placements1, vol1 = try_pack(order1, rot_choices1)
    if vol1 > best_volume:
        best_volume = vol1
        best_placements = placements1
    
    if time.time() - start_time > deadline:
        return best_placements if best_placements else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Strategy 2: Sort by volume ascending
    order2 = sorted(range(n), key=lambda i: box_volumes[i])
    rot_choices2 = [rotations[i][0] for i in range(n)]
    placements2, vol2 = try_pack(order2, rot_choices2)
    if vol2 > best_volume:
        best_volume = vol2
        best_placements = placements2
    
    if time.time() - start_time > deadline:
        return best_placements if best_placements else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Strategy 3: Sort by max dimension descending
    order3 = sorted(range(n), key=lambda i: -(max(boxes[i])))
    rot_choices3 = [rotations[i][0] for i in range(n)]
    placements3, vol3 = try_pack(order3, rot_choices3)
    if vol3 > best_volume:
        best_volume = vol3
        best_placements = placements3
    
    if time.time() - start_time > deadline:
        return best_placements if best_placements else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
    
    # Strategy 4: Try different rotation choices for each box
    for trial in range(5):
        if time.time() - start_time > deadline:
            break
        
        order4 = list(range(n))
        random.shuffle(order4)
        
        # Choose rotations randomly
        rot_choices4 = [random.choice(rotations[i]) for i in range(n)]
        
        placements4, vol4 = try_pack(order4, rot_choices4)
        if vol4 > best_volume:
            best_volume = vol4
            best_placements = placements4
    
    return best_placements if best_placements else [(-1, 0, 0, 0, w, h, d) for w, h, d in boxes]
