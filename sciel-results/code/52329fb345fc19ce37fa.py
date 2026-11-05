import math

def construct():
    # Target number of points
    n = 32
    
    # Helper to get Cartesian coordinates from (a, b)
    # z = a + b * exp(i * pi / 3)
    # x = a + b * 0.5
    # y = b * sqrt(3) / 2
    def to_cartesian(a, b):
        return (a + 0.5 * b, b * math.sqrt(3) / 2)
    
    # Helper to get hex distance
    def hex_dist(a, b):
        return (abs(a) + abs(b) + abs(a + b)) // 2
        
    # Generate points in rings
    # We need up to ring 3 or 4
    # Ring 0: 1
    # Ring 1: 6
    # Ring 2: 12
    # Ring 3: 18
    # Total 37. We need 32.
    
    points_by_ring = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    # Scan a box large enough to cover ring 4
    # Max coordinate for ring k is k
    max_k = 4
    for a in range(-max_k, max_k + 1):
        for b in range(-max_k, max_k + 1):
            d = hex_dist(a, b)
            if d <= max_k:
                points_by_ring[d].append((a, b))
                
    # Sort points in each ring by angle to form a spiral
    # Angle = atan2(y, x)
    for k in points_by_ring:
        points_by_ring[k].sort(key=lambda p: math.atan2(to_cartesian(p[0], p[1])[1], to_cartesian(p[0], p[1])[0]))
        
    # Collect points in order of ring, then angle
    selected_points = []
    for k in range(max_k + 1):
        for p in points_by_ring[k]:
            selected_points.append(p)
            if len(selected_points) == n:
                break
        if len(selected_points) == n:
            break
            
    # Convert to list of lists [a, b]
    result = [[a, b] for a, b in selected_points]
    
    # Verification (optional, but good for debugging locally)
    # def count_edges(pts):
    #     count = 0
    #     for i in range(len(pts)):
    #         for j in range(i+1, len(pts)):
    #             a1, b1 = pts[i]
    #             a2, b2 = pts[j]
    #             da = a1 - a2
    #             db = b1 - b2
    #             # Check if distance is 1
    #             # Neighbors: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    #             if (da == 1 and db == 0) or (da == -1 and db == 0) or \
    #                (da == 0 and db == 1) or (da == 0 and db == -1) or \
    #                (da == 1 and db == -1) or (da == -1 and db == 1):
    #                 count += 1
    #     return count
    # print(count_edges(result))
    
    return result
