if not (x + w <= px or px + pw <= x): # X Overlap?
    if not (y + h <= py or py + ph <= y): # Y Overlap?
        if not (z + d <= pz or pz + pd <= z): # Z Overlap?
            return False # All 3 overlap -> Collision
