import time
import random
import itertools
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Leave 15s margin for safety to avoid timeout
    
    n = 30
    triples = list(itertools.combinations(range(n), 3))
    triple_count = len(triples)
    
    # Precompute pairs for each point
    pairs_for_point = [[] for _ in range(n)]
    for i in range(n):
        pairs_for_point[i] = [(j, k) for j in range(n) for k in range(j+1, n) if j != i and k != i]
    
    def calc_areas(pts):
        """Calculate doubled areas for all triples."""
        areas = [0] * triple_count
        for idx, (i, j, k) in enumerate(triples):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            areas[idx] = det
        return areas

    def get_max_freq(areas):
        counts = Counter(areas)
        if 0 in counts:
            del counts[0]
        if not counts:
            return 0
        return max(counts.values())

    def init_counts(pts):
        counts = Counter()
        for idx, (i, j, k) in enumerate(triples):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            x3, y3 = pts[k]
            det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            if det > 0:
                counts[det] += 1
        return counts

    def generate_lattice_points(b1, b2, num_points=30):
        """Generate points as linear combinations of two basis vectors.
        
        Points are of form a*b1 + b*b2 where a,b are integers.
        We choose a, b to fit within [-30, 30]^2 and get distinct points.
        """
        points = []
        used = set()
        
        # Try different combinations of coefficients
        max_coord = 30
        min_coord = -30
        
        # Generate candidate points
        candidates = []
        for a in range(-10, 11):
            for b in range(-10, 11):
                x = a * b1[0] + b * b2[0]
                y = a * b1[1] + b * b2[1]
                if min_coord <= x <= max_coord and min_coord <= y <= max_coord:
                    if (x, y) not in used:
                        candidates.append((x, y))
                        used.add((x, y))
        
        # Shuffle and take first num_points
        random.shuffle(candidates)
        if len(candidates) < num_points:
            # If not enough, fill with random points
            while len(candidates) < num_points:
                x = random.randint(min_coord, max_coord)
                y = random.randint(min_coord, max_coord)
                if (x, y) not in used:
                    candidates.append((x, y))
                    used.add((x, y))
        
        return candidates[:num_points]

    # Try multiple lattice constructions
    best_points = None
    best_max_freq = 0
    
    # Basis vectors to try
    basis_vectors = [
        ((1, 0), (0, 1)),      # Standard grid
        ((1, 1), (1, -1)),     # Diamond
        ((2, 1), (1, 2)),      # Skewed
        ((1, 2), (2, 1)),      # Another skew
        ((3, 1), (1, 3)),      # Steeper
        ((1, 0), (1, 1)),      # Sheared
        ((2, 0), (0, 2)),      # Scaled grid
        ((1, 1), (2, -1)),     # More skewed
    ]
    
    for b1, b2 in basis_vectors:
        if time.time() > deadline:
            break
        
        pts = generate_lattice_points(b1, b2, n)
        
        # Validate
        if len(set(pts)) != n:
            continue
        if any(not (-30 <= p[0] <= 30 and -30 <= p[1] <= 30) for p in pts):
            continue
        
        global_counts = init_counts(pts)
        if not global_counts:
            continue
            
        current_max_freq = max(global_counts.values())
        current_target_area = [a for a, c in global_counts.items() if c == current_max_freq][0]
        
        if current_max_freq > best_max_freq:
            best_max_freq = current_max_freq
            best_points = pts[:]
    
    # If no valid lattice points found, use default grid
    if best_points is None:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_max_freq = 0
    
    # Local search to improve
    pts = best_points[:]
    global_counts = init_counts(pts)
    if not global_counts:
        return pts
        
    current_max_freq = max(global_counts.values())
    current_target_area = [a for a, c in global_counts.items() if c == current_max_freq][0]
    
    improved = True
    no_improve_count = 0
    max_no_improve = 10
    
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            # Try several random moves for point i
            for _ in range(8):
                nx = random.randint(-30, 30)
                ny = random.randint(-30, 30)
                new_pos = (nx, ny)
                
                if new_pos in pts:
                    continue
                
                old_pos = pts[i]
                
                # Decrement old contributions
                for j, k in pairs_for_point[i]:
                    x1, y1 = old_pos
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        global_counts[det] -= 1
                        if global_counts[det] == 0:
                            del global_counts[det]
                            
                pts[i] = new_pos
                
                # Increment new contributions
                new_target_count = global_counts.get(current_target_area, 0)
                for j, k in pairs_for_point[i]:
                    x1, y1 = new_pos
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                    if det > 0:
                        global_counts[det] += 1
                        if det == current_target_area:
                            new_target_count += 1

                # Check if we increased the target area count
                if new_target_count > current_max_freq:
                    current_max_freq = new_target_count
                    improved = True
                    no_improve_count = 0
                    if current_max_freq > best_max_freq:
                        best_max_freq = current_max_freq
                        best_points = pts[:]
                    break # Move to next point
                else:
                    # Revert
                    pts[i] = old_pos
                    # Re-increment old
                    for j, k in pairs_for_point[i]:
                        x1, y1 = old_pos
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                        if det > 0:
                            global_counts[det] += 1
                    # Re-decrement new
                    for j, k in pairs_for_point[i]:
                        x1, y1 = new_pos
                        x2, y2 = pts[j]
                        x3, y3 = pts[k]
                        det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                        if det > 0:
                            global_counts[det] -= 1
                            if global_counts[det] == 0:
                                del global_counts[det]
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break

    return best_points

# EVOLVE-BLOCK-END
