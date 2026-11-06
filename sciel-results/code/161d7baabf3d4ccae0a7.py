<<<<<<< SEARCH
def _neighbor(points, step_size):
    """
    Generates a neighbor by perturbing one point's coordinates.
    """
    new_points = [p[:] for p in points]
    idx = random.randint(0, 15)
    # Perturb x or y
    if random.random() < 0.5:
        new_points[idx][0] += random.randint(-step_size, step_size)
    else:
        new_points[idx][1] += random.randint(-step_size, step_size)
        
    # Clamp to bounds to ensure validity check is fast (though score handles it)
    new_points[idx][0] = max(-1000, min(1000, new_points[idx][0]))
    new_points[idx][1] = max(-1000, min(1000, new_points[idx][1]))
    
    return new_points
=======
def _neighbor(points, step_size):
    """
    Generates a neighbor by perturbing one point's coordinates.
    """
    new_points = [p[:] for p in points]
    idx = random.randint(0, 15)
    # Perturb x or y
    if random.random() < 0.5:
        new_points[idx][0] += random.randint(-step_size, step_size)
    else:
        new_points[idx][1] += random.randint(-step_size, step_size)
        
    # Clamp to bounds to ensure validity check is fast (though score handles it)
    new_points[idx][0] = max(-1000, min(1000, new_points[idx][0]))
    new_points[idx][1] = max(-1000, min(1000, new_points[idx][1]))
    
    return new_points

def _kick(points):
    """
    Generates a significantly perturbed version of points to escape local optima.
    """
    new_points = [p[:] for p in points]
    # Perturb 2-4 points with large steps
    num_kicks = random.randint(2, 4)
    for _ in range(num_kicks):
        idx = random.randint(0, 15)
        new_points[idx][0] += random.randint(-50, 50)
        new_points[idx][1] += random.randint(-50, 50)
        new_points[idx][0] = max(-1000, min(1000, new_points[idx][0]))
        new_points[idx][1] = max(-1000, min(1000, new_points[idx][1]))
    return new_points
>>>>>>> REPLACE

