"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    
    # Generate hexagon of radius 3
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a-b)) <= 3:
                points.append((a, b))
    
    # points should have 37 elements
    # Calculate degrees within this set
    point_set = set(points)
    neighbors = {p: [] for p in points}
    for p in points:
        a, b = p
        for da, db in [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]:
            q = (a+da, b+db)
            if q in point_set:
                neighbors[p].append(q)
    
    # We need to remove 5 points to get 32 points.
    # We want to minimize the number of edges lost.
    # Edges lost = sum(degrees of removed) - 2 * (edges between removed)
    
    # Greedy removal:
    # Start with all points.
    current_points = set(points)
    removed = []
    
    for _ in range(5):
        best_point = None
        best_loss = float('inf')
        
        for p in current_points:
            # Calculate loss if we remove p
            # Loss = degree of p in current_points
            # (Edges to removed points are already "lost" in the sense that they don't exist in current graph)
            # Wait, if we remove p, we lose edges from p to its neighbors in current_points.
            # So loss = len(neighbors_in_current)
            
            deg = 0
            for q in neighbors[p]:
                if q in current_points:
                    deg += 1
            
            # Tie-breaking: prefer points with more neighbors that are also low-degree?
            # Or just pick the one with min deg.
            if deg < best_loss:
                best_loss = deg
                best_point = p
            elif deg == best_loss and best_point is not None:
                # Optional: check if p is adjacent to best_point?
                # If we remove adjacent points, we save edges.
                # But we are removing one by one.
                # Let's just pick the first one.
                pass
        
        if best_point is not None:
            current_points.remove(best_point)
            removed.append(best_point)
    
    # Convert back to list of lists
    result = [[a, b] for a, b in current_points]
    return result
# EVOLVE-BLOCK-END
