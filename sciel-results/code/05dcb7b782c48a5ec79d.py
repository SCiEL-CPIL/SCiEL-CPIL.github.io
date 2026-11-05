"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def load(boxes, container):
    W, H, D = container
    
    # Grid resolution
    # 30 cells per dimension balances accuracy and speed for 60 boxes.
    target_dim = 30
    step_x = max(1, int(np.ceil(W / target_dim)))
    step_y = max(1, int(np.ceil(H / target_dim)))
    step_z = max(1, int(np.ceil(D / target_dim)))
    
    nx = int(np.ceil(W / step_x))
    ny = int(np.ceil(H / step_y))
    nz = int(np.ceil(D / step_z))
    
    # Occupancy grid: 1 if occupied, 0 if empty
    occ = np.zeros((nx, ny, nz), dtype=np.uint8)
    
    results = [None] * len(boxes)
    
    # Sort boxes by volume descending (Best Fit Decreasing)
    box_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][0]*boxes[i][1]*boxes[i][2], reverse=True)
    
    for idx in box_indices:
        w, h, d = boxes[idx]
        
        best_pos = None
        best_orient = None
        
        # Generate unique orientations
        orientations = [
            (w, h, d), (w, d, h), (h, w, d),
            (h, d, w), (d, w, h), (d, h, w)
        ]
        unique_orients = []
        seen = set()
        for o in orientations:
            if o not in seen:
                unique_orients.append(o)
                seen.add(o)
        
        # Try to place the box with the best orientation
        # Heuristic: First Fit with Corner Preference (min z, then min y, then min x)
        
        for ow, oh, od in unique_orients:
            gw = int(np.ceil(ow / step_x))
            gh = int(np.ceil(oh / step_y))
            gd = int(np.ceil(od / step_z))
            
            if gw > nx or gh > ny or gd > nz:
                continue
            
            # Search for the first valid position
            # Order: z (height) from bottom up, y (depth) from front to back, x (width) from left to right
            
            found = False
            for z in range(nz - gd + 1):
                if found:
                    break
                for y in range(ny - gh + 1):
                    if found:
                        break
                    for x in range(nx - gw + 1):
                        # Check if the box fits at (x, y, z)
                        # Using np.any on the slice
                        if not np.any(occ[x:x+gw, y:y+gh, z:z+gd]):
                            # Place the box
                            occ[x:x+gw, y:y+gh, z:z+gd] = 1
                            
                            real_x = x * step_x
                            real_y = y * step_y
                            real_z = z * step_z
                            
                            results[idx] = (0, real_x, real_y, real_z, ow, oh, od)
                            best_pos = (x, y, z)
                            best_orient = (ow, oh, od)
                            found = True
                            break
            
            if found:
                break
        
        if best_pos is None:
            # Box could not be placed
            results[idx] = (-1, 0, 0, 0, w, h, d)
            
    return results

# EVOLVE-BLOCK-END
