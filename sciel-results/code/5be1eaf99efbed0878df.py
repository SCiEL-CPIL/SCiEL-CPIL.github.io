import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the centers of the 12 unit hexagons in a 4x3 grid
    # Using flat-top hexagon orientation
    # Distance between centers horizontally: sqrt(3)
    # Distance between centers vertically: 1.5
    # Horizontal shift for alternating rows: sqrt(3)/2
    
    centers = []
    rows = 4
    cols = 3
    
    # Calculate raw coordinates
    for r in range(rows):
        for c in range(cols):
            x = c * math.sqrt(3)
            # Shift odd rows to the right by half the horizontal distance
            if r % 2 == 1:
                x += math.sqrt(3) / 2
            y = r * 1.5
            centers.append([x, y, 0.0])
            
    # Center the cluster at (0,0)
    # Find bounding box to determine center offset
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    
    # Translate all centers
    inner = []
    for c in centers:
        inner.append([c[0] - cx, c[1] - cy, c[2]])
        
    # Calculate required outer side length
    # The bounding box of the cluster is W x H
    # We need to fit this into a regular hexagon of side S
    # Width constraint: 2*S >= W_cluster + margin?
    # Actually, we calculated S based on the extreme vertices.
    
    # Recalculate extreme vertices of the cluster
    # For flat-top hex, vertices are at angles 0, 60, 120, 180, 240, 300
    # Relative to center
    # Max x extent of a single hex is 1
    # Max y extent of a single hex is sqrt(3)/2
    
    # We already centered the centers.
    # The max x coordinate of any vertex is max(center_x) + 1
    # The min x coordinate of any vertex is min(center_x) - 1
    # The max y coordinate of any vertex is max(center_y) + sqrt(3)/2
    # The min y coordinate of any vertex is min(center_y) - sqrt(3)/2
    
    max_x_vertex = max(c[0] for c in inner) + 1
    min_x_vertex = min(c[0] for c in inner) - 1
    max_y_vertex = max(c[1] for c in inner) + math.sqrt(3) / 2
    min_y_vertex = min(c[1] for c in inner) - math.sqrt(3) / 2
    
    width = max_x_vertex - min_x_vertex
    height = max_y_vertex - min_y_vertex
    
    # For a regular hexagon of side S (flat-top):
    # Width (vertex to vertex) = 2*S
    # Height (side to side) = S * sqrt(3)
    
    S_width = width / 2
    S_height = height / math.sqrt(3)
    
    side = max(S_width, S_height)
    
    # Add a small epsilon to ensure no floating point errors cause intersection
    side += 1e-6
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
