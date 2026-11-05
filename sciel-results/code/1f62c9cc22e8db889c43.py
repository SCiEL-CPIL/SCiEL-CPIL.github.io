# EVOLVE-BLOCK-START
import time

def construct():
    """
    Returns a list of 32 integer pairs [a, b] representing points on a triangular lattice
    a + b*omega (where omega = exp(i*pi/3)) that form an optimal Harborth configuration.
    This configuration has exactly 76 pairs at unit distance, which is the theoretical maximum.
    
    Strategy:
    1. Generate the full hexagon of radius 3 (37 points).
    2. Remove a specific set of 5 boundary points to minimize edge loss.
    3. The removed points form a contiguous path on the boundary, preserving internal connectivity.
    4. The resulting 32 points achieve the maximum of 76 edges.
    """
    
    # Deadline guard: Ensure we return within time limits.
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit

    # Generate all points in the hexagon of radius 3
    # Condition for hexagon of radius R=3: |a| <= 3, |b| <= 3, |a+b| <= 3
    full_hex_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if abs(a) <= 3 and abs(b) <= 3 and abs(a + b) <= 3:
                full_hex_points.append([a, b])
    
    # Verify we have 37 points
    assert len(full_hex_points) == 37, f"Expected 37 points in hexagon, got {len(full_hex_points)}"

    # Points to remove to achieve 76 edges
    # These 5 points form a contiguous path on the boundary of the hexagon:
    # (3,0) -> (2,1) -> (1,2) -> (0,3) -> (-1,3)
    # Removing this path minimizes the loss of unit-distance edges compared to other removal patterns.
    remove_set = {
        (3, 0), 
        (2, 1), 
        (1, 2), 
        (0, 3), 
        (-1, 3)
    }
    
    # Filter points to get the final 32-point set
    result = []
    for p in full_hex_points:
        if (p[0], p[1]) not in remove_set:
            result.append(p)
            
    # Verify count
    assert len(result) == 32, f"Expected 32 points, got {len(result)}"
    
    # Check if we hit deadline (should not happen, but good practice)
    if time.time() > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
