import math
from collections import deque

N = 11

def construct():
    # BFS to find N closest centers in hexagonal lattice
    centers = []
    visited = set()
    queue = deque()
    queue.append((0, 0))
    visited.add((0, 0))
    
    # Directions for hex grid neighbors (axial coordinates)
    # (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
    directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    while len(centers) < N and queue:
        q, r = queue.popleft()
        centers.append((q, r))
        
        for dq, dr in directions:
            nq, nr = q + dq, r + dr
            if (nq, nr) not in visited:
                visited.add((nq, nr))
                queue.append((nq, nr))
                
    # Convert to Cartesian coordinates
    # x = 2q - r
    # y = sqrt(3) * r
    cart_centers = []
    for q, r in centers:
        x = 2.0 * q - r
        y = math.sqrt(3) * r
        cart_centers.append((x, y))
        
    # Calculate minimum side length S of the container
    # The container must contain the unit hexagons.
    # This is equivalent to the centers being inside a container of side S-1.
    # Let S_prime = S - 1.
    # Constraints for centers to be inside a hexagon of side S_prime (vertices at distance S_prime):
    # 1. |x| <= S_prime
    # 2. |sqrt(3)*x + y| <= sqrt(3)*S_prime
    # 3. |sqrt(3)*x - y| <= sqrt(3)*S_prime
    
    max_abs_x = max(abs(x) for x, y in cart_centers)
    max_val_1 = max(abs(math.sqrt(3)*x + y) for x, y in cart_centers)
    max_val_2 = max(abs(math.sqrt(3)*x - y) for x, y in cart_centers)
    
    s_prime = max(max_abs_x, max_val_1 / math.sqrt(3), max_val_2 / math.sqrt(3))
    
    # S = S_prime + 1
    S = s_prime + 1.0
    
    inner = [[x, y, 0.0] for x, y in cart_centers]
    return inner, (0.0, 0.0), S, 0.0
