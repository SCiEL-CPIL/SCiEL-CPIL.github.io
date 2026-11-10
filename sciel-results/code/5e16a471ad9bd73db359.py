# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    Uses a 3-direction grid construction (Vertical, Horizontal, Diagonal).
    This allows for up to 3 incidences per point if the point is an intersection
    of one line from each direction.
    
    Construction:
    - 9 vertical lines: x = u, u in {1..9}
    - 9 horizontal lines: y = v, v in {1..9}
    - 9 diagonal lines: x + y = w, w in {6..14}
    
    Points are selected from the 61 intersections (u,v) where u+v is in {6..14}.
    We pick 54 such points. Each has exactly 3 incidences.
    Total incidences = 54 * 3 = 162.
    """
    # Deadline guard (though this is deterministic and fast)
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Define the sets of intercepts
    U = list(range(1, 10))  # x-coordinates
    V = list(range(1, 10))  # y-coordinates
    
    # Calculate sums u+v and their frequencies
    sum_freq = {}
    for u in U:
        for v in V:
            s = u + v
            sum_freq[s] = sum_freq.get(s, 0) + 1
            
    # Sort sums by frequency (descending) to pick the best 9
    # The frequencies are symmetric around 10.
    # We want the 9 sums with highest counts.
    sorted_sums = sorted(sum_freq.keys(), key=lambda s: sum_freq[s], reverse=True)
    # In case of ties, the order doesn't strictly matter for the count, 
    # but we should ensure we pick 9 distinct sums.
    # The top 9 frequencies are 9, 8, 8, 7, 7, 6, 6, 5, 5.
    # Sums: 10 (9), 9 (8), 11 (8), 8 (7), 12 (7), 7 (6), 13 (6), 6 (5), 14 (5).
    W = sorted(sorted_sums[:9])
    
    # Generate all valid points (u, v) such that u+v in W
    valid_points = []
    for u in U:
        for v in V:
            if u + v in W:
                valid_points.append((u, v))
                
    # We need exactly 54 points. We have 61 valid ones.
    # Take the first 54.
    points = np.array(valid_points[:54], dtype=np.int64)
    
    # Construct lines
    lines = []
    # Vertical lines: x - u = 0 => (1, 0, -u)
    for u in U:
        lines.append([1, 0, -u])
        
    # Horizontal lines: y - v = 0 => (0, 1, -v)
    for v in V:
        lines.append([0, 1, -v])
        
    # Diagonal lines: x + y - w = 0 => (1, 1, -w)
    for w in W:
        lines.append([1, 1, -w])
        
    lines = np.array(lines, dtype=np.int64)
    
    # Sanity check: ensure shapes and types
    assert points.shape == (54, 2)
    assert lines.shape == (27, 3)
    assert points.dtype == np.int64
    assert lines.dtype == np.int64
    
    # Check for duplicate points
    unique_points = np.unique(points, axis=0)
    if len(unique_points) != 54:
        raise ValueError("Duplicate points found")
        
    # Check for duplicate lines (normalized)
    normalized_lines = []
    for l in lines:
        A, B, C = l
        # GCD of A, B, C
        g = np.gcd(np.gcd(abs(A), abs(B)), abs(C))
        if g != 0:
            A, B, C = A // g, B // g, C // g
        # Make first nonzero positive
        if A < 0:
            A, B, C = -A, -B, -C
        elif A == 0 and B < 0:
            A, B, C = -A, -B, -C
        normalized_lines.append((A, B, C))
        
    unique_lines = set(normalized_lines)
    if len(unique_lines) != 27:
        raise ValueError("Duplicate lines found")

    return points, lines

# EVOLVE-BLOCK-END
