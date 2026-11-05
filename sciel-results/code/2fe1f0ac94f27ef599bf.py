import math

def construct():
    """
    Returns a list of 32 distinct integer pairs [a, b] representing points on the triangular lattice
    that maximize the number of unit-distance pairs (edges).
    The optimal number of edges is 76.
    """
    
    def get_points_up_to_layer(layer):
        points = []
        for q in range(-layer, layer + 1):
            for r in range(-layer, layer + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= layer:
                    dist = (abs(q) + abs(r) + abs(s)) // 2
                    points.append((q, r, dist))
        return points

    all_points = get_points_up_to_layer(3)
    
    layer_0_2 = [p for p in all_points if p[2] <= 2]
    layer_3 = [p for p in all_points if p[2] == 3]
    
    def get_angle(q, r):
        real = q + r / 2.0
        imag = r * math.sqrt(3) / 2.0
        return math.atan2(imag, real)
    
    layer_3_sorted = sorted(layer_3, key=lambda p: get_angle(p[0], p[1]))
    
    # We determined that starting at index 1 (skipping the first corner)
    # gives a block with 4 corners and 9 edges, yielding 76 total edges.
    # Let's verify the first point is indeed a corner.
    # If the sorting starts exactly at a corner, index 0 is a corner.
    # If not, we might need to adjust.
    # However, the set of 13 points starting at index 1 is robust.
    
    # To be safe, we can calculate the "degree to interior" for each point in layer 3
    # and select the 13 contiguous points that maximize the sum of degrees.
    
    # Build a set of interior points for fast lookup
    interior_set = set((p[0], p[1]) for p in layer_0_2)
    
    def count_interior_neighbors(q, r):
        count = 0
        neighbors = [
            (q+1, r), (q-1, r),
            (q, r+1), (q, r-1),
            (q+1, r-1), (q-1, r+1)
        ]
        for nq, nr in neighbors:
            if (nq, nr) in interior_set:
                count += 1
        return count

    # Calculate degrees for each point in layer_3_sorted
    degrees = [count_interior_neighbors(p[0], p[1]) for p in layer_3_sorted]
    
    # Find the best contiguous block of 13 in the circular list of 18
    # Since it's a cycle, we duplicate the list to handle wrap-around
    extended_degrees = degrees + degrees
    max_sum = -1
    best_start = 0
    for i in range(18):
        s = sum(extended_degrees[i:i+13])
        if s > max_sum:
            max_sum = s
            best_start = i
            
    # Select the points
    selected_indices = [(best_start + j) % 18 for j in range(13)]
    selected_layer_3 = [layer_3_sorted[i] for i in selected_indices]
    
    final_points = [(p[0], p[1]) for p in layer_0_2] + [(p[0], p[1]) for p in selected_layer_3]
    
    return [[q, r] for q, r in final_points]
