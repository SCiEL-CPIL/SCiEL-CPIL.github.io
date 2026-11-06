import time

def construct():
    start_time = time.time()
    deadline = 140.0

    n_points = 32
    points = []
    
    points.append((0, 0))
    
    if n_points == 1:
        return [[0, 0]]
        
    # Directions for the 6 sides of the hexagon perimeter
    # Starting from (k, 0)
    side_dirs = [
        (0, -1),  # Side 1
        (-1, 0),  # Side 2
        (-1, 1),  # Side 3
        (0, 1),   # Side 4
        (1, 0),   # Side 5
        (1, -1)   # Side 6
    ]
    
    for k in range(1, 10):
        if len(points) >= n_points:
            break
            
        # Start at (k, 0)
        current = (k, 0)
        if len(points) < n_points:
            points.append(current)
            
        for dir_vec in side_dirs:
            if len(points) >= n_points:
                break
            # Walk k steps
            for _ in range(k):
                current = (current[0] + dir_vec[0], current[1] + dir_vec[1])
                if len(points) < n_points:
                    points.append(current)
                if len(points) >= n_points:
                    break

    return [[a, b] for a, b in points]
