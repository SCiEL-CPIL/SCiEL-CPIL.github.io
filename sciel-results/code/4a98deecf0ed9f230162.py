"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def pack(boxes, container):
    W, H, D = container
    
    # Sort boxes by volume descending
    sorted_indices = sorted(range(len(boxes)), key=lambda i: -boxes[i][0]*boxes[i][1]*boxes[i][2])
    
    # Use a simple shelf algorithm with better sorting
    placements = []
    b, x, y, z, row_h, layer_d = 0, 0, 0, 0, 0, 0
    
    start_time = time.time()
    
    for idx in sorted_indices:
        # Check time
        if time.time() - start_time > 140:
            break
        
        w, h, d = boxes[idx]
        if x + w > W:
            x, y, row_h = 0, y + row_h, 0
        if y + h > H:
            y, z, layer_d = 0, z + layer_d, 0
            x, row_h = 0, 0
        if z + d > D:
            b, x, y, z, row_h, layer_d = b + 1, 0, 0, 0, 0, 0
        placements.append((b, x, y, z, w, h, d))
        x, row_h, layer_d = x + w, max(row_h, h), max(layer_d, d)
    
    return placements

# EVOLVE-BLOCK-END
