def generate_spiral(n):
    points = [(0,0)]
    if n == 1: return points
    q, r = 0, 0
    # Directions: East, SE, SW, West, NW, NE
    dirs = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    step = 1
    dir_idx = 0
    while len(points) < n:
        for _ in range(2):
            for _ in range(step):
                q += dirs[dir_idx][0]
                r += dirs[dir_idx][1]
                if (q,r) not in points:
                    points.append((q,r))
                    if len(points) == n:
                        return points
            dir_idx = (dir_idx + 1) % 6
        step += 1
    return points
