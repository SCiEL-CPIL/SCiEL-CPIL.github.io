# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy expansion of the complement set S.
    
    Strategy:
    1. Start with a small valid S (e.g., a single line).
    2. Greedily add points to S that maintain the property: for every x in S and every
       direction v, there exists t != 0 with x + t*v in S.
    3. Use a priority queue to select the next point to add, prioritizing points that
       are "easy" to add (i.e., they have many neighbors already in S across all directions).
    4. Return N = F_49^2 \\ S.
    
    The goal is to maximize |S| (minimize |N|) while maintaining the Nikodym property.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    p = 7
    q = p * p  # 49
    k_const = 3  # alpha^2 = 3, non-residue mod 7
    
    # Precompute field operations
    # Element (a, b) represents a + b*alpha
    # Addition: (a1, b1) + (a2, b2) = (a1+a2 mod p, b1+b2 mod p)
    # Multiplication: (a1, b1) * (a2, b2) = (a1*a2 + b1*b2*k, a1*b2 + a2*b1) mod p
    
    def add_field(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
    
    def mul_field(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return ((a1 * a2 + b1 * b2 * k_const) % p, (a1 * b2 + a2 * b1) % p)
    
    def scalar_mul(t, v):
        # t is a field element, v is a point (x, y) where x, y are field elements
        return (mul_field(t, v[0]), mul_field(t, v[1]))
    
    def add_point(p1, p2):
        return (add_field(p1[0], p2[0]), add_field(p1[1], p2[1]))
    
    # Generate all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    field_index = {e: i for i, e in enumerate(field_elements)}
    
    # Generate all points in F_49^2
    all_points = []
    point_index = {}
    for x in field_elements:
        for y in field_elements:
            pt = (x, y)
            point_index[pt] = len(all_points)
            all_points.append(pt)
    
    total_points = len(all_points)  # 2401
    
    # Directions: v = (1, m) for m in F_4, and v = (0, 1)
    directions = []
    for m in field_elements:
        directions.append(((1, 0), m))  # (1, m)
    directions.append(((0, 1), (0, 0)))  # (0, 1)
    
    num_directions = len(directions)  # 50
    
    # Precompute lines for each point and direction
    # lines[point_idx][dir_idx] = list of (t, point_idx) for t in F_q
    lines = [[[] for _ in range(num_directions)] for _ in range(total_points)]
    
    for pi, pt in enumerate(all_points):
        for di, (vx, vy) in enumerate(directions):
            for t in field_elements:
                if t == (0, 0):
                    continue
                offset = scalar_mul(t, (vx, vy))
                new_pt = add_point(pt, offset)
                lines[pi][di].append((t, point_index[new_pt]))
    
    # Greedy construction of S
    # Start with a single line in one direction
    # Pick direction 0: v = (1, (0,0))
    # Start point: (0,0)
    start_pt = (0, 0)
    start_pt_idx = point_index[start_pt]
    
    S = set()
    S.add(start_pt_idx)
    
    # Add all other points on the line through start_pt in direction 0
    for t, pi in lines[start_pt_idx][0]:
        S.add(pi)
    
    # Priority queue: (score, point_idx)
    # Score: number of directions where the point already has at least one neighbor in S
    # Higher score = easier to add
    
    # Compute initial scores
    def compute_score(pi):
        """Compute how many directions have at least one neighbor in S"""
        count = 0
        for di in range(num_directions):
            has_neighbor = False
            for t, ni in lines[pi][di]:
                if ni in S:
                    has_neighbor = True
                    break
            if has_neighbor:
                count += 1
        return count
    
    # Build initial priority queue
    pq = []
    for pi in range(total_points):
        if pi not in S:
            score = compute_score(pi)
            if score > 0:
                heapq.heappush(pq, (-score, pi))  # Min-heap, so negate for max
    
    # Greedily add points
    while pq and time.time() < deadline - 10:
        neg_score, pi = heapq.heappop(pq)
        score = -neg_score
        
        if pi in S:
            continue
        
        # Check if adding pi maintains the property
        # For every direction, pi must have at least one neighbor in S (after adding)
        valid = True
        for di in range(num_directions):
            has_neighbor = False
            for t, ni in lines[pi][di]:
                if ni in S:
                    has_neighbor = True
                    break
            if not has_neighbor:
                valid = False
                break
        
        if not valid:
            continue
        
        # Add pi to S
        S.add(pi)
        
        # Update scores of neighbors
        for di in range(num_directions):
            for t, ni in lines[pi][di]:
                if ni in S or ni == pi:
                    continue
                # Recompute score for ni
                new_score = compute_score(ni)
                if new_score > 0:
                    heapq.heappush(pq, (-new_score, ni))
    
    # Build N = complement of S
    N = [all_points[i] for i in range(total_points) if i not in S]
    
    if len(N) == 0:
        # Edge case: if S = all points, we need at least one point in N
        # This shouldn't happen, but handle it
        N = [all_points[0]]
    
    # Convert to required format: (k, 2, 2) array
    result = np.zeros((len(N), 2, 2), dtype=np.int64)
    for i, (x, y) in enumerate(N):
        result[i, 0, 0] = x[0]
        result[i, 0, 1] = x[1]
        result[i, 1, 0] = y[0]
        result[i, 1, 1] = y[1]
    
    return result

# EVOLVE-BLOCK-END
