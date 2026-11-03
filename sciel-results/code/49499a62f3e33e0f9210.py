def construct():
    start_time = time.time()
    
    inner = []
    for row in range(3):
        for col in range(4):
            x = col * 2.0
            y = row * np.sqrt(3)
            if row % 2 == 1:
                x += 1.0
            x -= 3.0
            y -= np.sqrt(3)
            inner.append([x, y, 0.0])
            
    side, cx, cy, rot = compute_min_outer_side(inner)
    
    return inner, (cx, cy), side, rot
