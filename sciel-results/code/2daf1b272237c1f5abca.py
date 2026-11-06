"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate 36-point hexagon
    points = set()
    for a in range(-3, 4):
        for b in range(-3, 4):
            if abs(a) <= 3 and abs(b) <= 3 and abs(a + b) <= 3:
                points.add((a, b))
    
    if len(points) != 36:
        return [[a, b] for a in range(16) for b in range(2)]

    def get_neighbors(p):
        a, b = p
        return [(a+1, b), (a-1, b), (a, b+1), (a, b-1), (a+1, b-1), (a-1, b+1)]

    # 2. Greedy reduction to 32 points
    current_points = set(points)
    for _ in range(4):
        min_deg = float('inf')
        min_point = None
        for p in current_points:
            deg = sum(1 for n in get_neighbors(p) if n in current_points)
            if deg < min_deg:
                min_deg = deg
                min_point = p
        current_points.remove(min_point)
        
    best_points = set(current_points)
    best_edges = sum(1 for p in best_points for n in get_neighbors(p) if n in best_points) // 2

    # 3. Local Search
    # Candidate pool: Points in a larger hexagon (side 4) to allow expansion
    candidate_pool = set()
    for a in range(-4, 5):
        for b in range(-4, 5):
            if abs(a) <= 4 and abs(b) <= 4 and abs(a + b) <= 4:
                candidate_pool.add((a, b))
                
    # Remove current points from pool for "out" candidates
    out_candidates = list(candidate_pool - best_points)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to break symmetry
        random.shuffle(out_candidates)
        for p_out in out_candidates:
            if time.time() > deadline:
                break
            # Calculate gain
            # Edges gained by p_out
            gain = sum(1 for n in get_neighbors(p_out) if n in best_points)
            if gain <= 0:
                continue
            # Try swapping each p_in in best_points
            # Optimization: Only swap if gain > min_degree_in_best?
            # We need gain - loss > 0 => loss < gain
            # loss = degree of p_in in best_points
            # So we only need to check p_in with degree < gain
            for p_in in list(best_points):
                if time.time() > deadline:
                    break
                loss = sum(1 for n in get_neighbors(p_in) if n in best_points)
                if loss < gain:
                    # Perform swap
                    best_points.remove(p_in)
                    best_points.add(p_out)
                    improved = True
                    break
            if improved:
                break

    return [list(p) for p in best_points]
# EVOLVE-BLOCK-END
