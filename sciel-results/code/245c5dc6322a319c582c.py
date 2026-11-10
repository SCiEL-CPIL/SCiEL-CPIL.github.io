import random

N = 64

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a set of points in [0, N)^2 with no three points forming an isosceles triangle.
    Uses a randomized greedy approach with multiple restarts.
    Avoids both non-degenerate and degenerate isosceles triangles.
    """
    best_set = []
    points = [(x, y) for x in range(N) for y in range(N)]
    
    # Number of restarts to find a good configuration
    num_restarts = 100
    
    for _ in range(num_restarts):
        random.shuffle(points)
        current_set = []
        
        for p in points:
            px, py = p
            valid = True
            
            # Check against all pairs in current_set
            for i in range(len(current_set)):
                ax, ay = current_set[i]
                d_pa_sq = (px - ax)**2 + (py - ay)**2
                
                for j in range(i + 1, len(current_set)):
                    bx, by = current_set[j]
                    
                    # Distance squared P-B
                    d_pb_sq = (px - bx)**2 + (py - by)**2
                    # Distance squared A-B
                    d_ab_sq = (bx - ax)**2 + (by - ay)**2
                    
                    # Check isosceles conditions
                    # This catches degenerate (collinear) isosceles triangles as well
                    if d_pa_sq == d_pb_sq or d_pa_sq == d_ab_sq or d_pb_sq == d_ab_sq:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_set.append(p)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            if len(best_set) > 30:
                break
                
    return best_set
# EVOLVE-BLOCK-END
