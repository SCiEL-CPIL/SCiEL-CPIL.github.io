def _pivot_move(chain, i, angle, sequence):
    N = len(chain)
    if i < 1 or i >= N - 1:
        return None
    
    pivot_point = chain[i - 1]
    px, py = pivot_point
    
    new_tail = []
    
    # Pre-define rotation functions or use if/else
    if angle == 90:
        for k in range(i, N):
            x, y = chain[k]
            vx = x - px
            vy = y - py
            rx = -vy
            ry = vx
            new_tail.append((px + rx, py + ry))
    elif angle == 180:
        for k in range(i, N):
            x, y = chain[k]
            vx = x - px
            vy = y - py
            rx = -vx
            ry = -vy
            new_tail.append((px + rx, py + ry))
    else: # 270
        for k in range(i, N):
            x, y = chain[k]
            vx = x - px
            vy = y - py
            rx = vy
            ry = -vx
            new_tail.append((px + rx, py + ry))
            
    head_set = set(chain[0:i-1])
    
    for pos in new_tail:
        if pos in head_set:
            return None
            
    return chain[0:i] + new_tail
