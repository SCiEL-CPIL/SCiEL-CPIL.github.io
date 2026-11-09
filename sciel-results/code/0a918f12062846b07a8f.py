import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Precompute collinearity for all triples in [-100, 100]^2? No, too big (201^3).
    # But we only need it for points we generate.
    # For efficiency, we compute on the fly but cache within a candidate if possible?
    # Actually, for N=20, O(N^3) is 8000 ops, very fast.

    def are_collinear(p1, p2, p3):
        return (p1[0] - p2[0]) * (p2[1] - p3[1]) == (p2[0] - p3[0]) * (p1[1] - p2[1])

    def has_four_collinear(points):
        n = len(points)
        # Check all pairs
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if are_collinear(p1, p2, points[k]):
                        count += 1
                        if count >= 4:
                            return True
        return False

    def calculate_score(points):
        """
        Exact branch and bound for the largest subset with no 3 collinear points.
        Uses bitmask representation for efficiency.
        """
        n = len(points)
        
        # Precompute for each pair (i, j) the set of k such that i, j, k are collinear.
        # pair_conflicts[i][j] will be a bitmask of such k's.
        # This allows fast pruning: if we have selected i and j, we cannot select any k in pair_conflicts[i][j].
        
        # For n=20, we can use Python integers as bitmasks.
        # pair_conflicts is a 20x20 matrix of bitmasks.
        pair_conflicts = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                pair_conflicts[i][j] = mask
                pair_conflicts[j][i] = mask

        best_score = 0
        full_mask = (1 << n) - 1

        def backtrack(available, current_size, selected_mask):
            nonlocal best_score
            
            # Pruning 1: Upper bound on size
            if current_size + bin(available).count('1') <= best_score:
                return
            
            if available == 0:
                if current_size > best_score:
                    best_score = current_size
                return

            # Heuristic: Pick a point from available.
            # To improve pruning, it's often good to pick a point that is "constrained"
            # or simply the first one for simplicity, but let's try to pick one that
            # reduces the search space effectively.
            # A common heuristic for MIS is to pick the vertex with the highest degree
            # in the remaining graph. Here, the "degree" of a point x in the remaining
            # hypergraph is the number of points y in available such that x and y
            # form a conflict with some z in selected_mask? No, that's not quite right.
            
            # Let's stick to a simple deterministic choice first: lowest index in available.
            # We can improve this by sorting points initially by some heuristic if needed.
            
            # Get the lowest set bit
            i_bit = available & -available
            i = i_bit.bit_length() - 1
            
            # Option 1: Skip i
            backtrack(available ^ i_bit, current_size, selected_mask)
            
            # Option 2: Include i
            new_selected = selected_mask | i_bit
            new_available = available ^ i_bit
            
            # Prune new_available:
            # If we include i, then for any j in new_available, 
            # if there exists k in selected_mask such that i, j, k are collinear,
            # then j cannot be selected.
            # Also, if there exists k in new_available such that i, j, k are collinear,
            # we don't remove j yet, but we know that if we later select j, k will be removed.
            # The immediate pruning is: remove j from available if j conflicts with i given current selected_mask.
            # Conflict condition: j is collinear with i and some k in selected_mask.
            # i.e., j is in pair_conflicts[i][k] for some k in selected_mask.
            
            # Let's compute the forbidden mask for i given selected_mask.
            forbidden = 0
            temp_sel = selected_mask
            while temp_sel:
                k_bit = temp_sel & -temp_sel
                k = k_bit.bit_length() - 1
                forbidden |= pair_conflicts[i][k]
                temp_sel ^= k_bit
            
            new_available &= ~forbidden
            
            backtrack(new_available, current_size + 1, new_selected)

        backtrack(full_mask, 0, 0)
        return best_score

    def generate_random_points():
        points = []
        seen = set()
        while len(points) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])
        return points

    def generate_parabola_points():
        # Parabola y = x^2 has no 4 collinear, but score is 20 (no 3 collinear).
        # We want to minimize score, so this is a bad start, but valid.
        return [[x, x * x] for x in range(-10, 10)]

    best_points = None
    best_score = 21  # Max possible is 20
    
    # Strategy:
    # 1. Try to find a configuration with low score (many collinear triples) via random search + local search.
    # 2. The score is the size of the largest "no 3 collinear" subset.
    #    Minimizing this means maximizing the number of collinear triples such that they cover the points densely.
    #    Ideally, we want a structure where every point is part of many triples.
    #    A projective plane of order 4 has 21 points, lines of size 5.
    #    If we could embed a subset, the score would be low.
    #    However, we are in Z^2.
    
    # Let's start with random points and improve via local search.
    
    # Initial random restarts
    num_restarts = 20
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        points = generate_random_points()
        if not has_four_collinear(points):
            score = calculate_score(points)
            if score < best_score:
                best_score = score
                best_points = [p[:] for p in points]
    
    if best_points is None:
        best_points = generate_parabola_points()
        best_score = 20

    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle order of points to try to avoid getting stuck in a specific local minimum
        indices = list(range(20))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            old_point = best_points[i][:]
            
            # Try to move point i to a new position that reduces the score
            # We try random positions, but also positions on lines defined by other points
            # to increase collinearity.
            
            # 1. Random moves
            for _ in range(5):
                new_point = [random.randint(-100, 100), random.randint(-100, 100)]
                if any(new_point == p for p in best_points if best_points.index(p) != i):
                    continue
                
                test_points = best_points[:]
                test_points[i] = new_point
                
                if has_four_collinear(test_points):
                    continue
                    
                score = calculate_score(test_points)
                if score < best_score:
                    best_score = score
                    best_points = test_points
                    improved = True
                    break
                elif score == best_score and random.random() < 0.05:
                    # Accept equal score with low probability to escape local minima
                    best_points = test_points
                    improved = True
                    break
            
            # 2. Structured moves: place on a line defined by two other points
            # Pick two other points j, k
            other_indices = [idx for idx in range(20) if idx != i]
            random.shuffle(other_indices)
            for _ in range(5):
                if len(other_indices) < 2:
                    break
                j = other_indices[0]
                k = other_indices[1]
                
                p1 = best_points[j]
                p2 = best_points[k]
                
                # Generate points on the line passing through p1 and p2
                # Direction vector
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # Try a few integer points on this line
                # Parametric: p1 + t * (dx, dy)
                # We want integer coordinates, so t should be such that p1 + t*dx, p1 + t*dy are integers.
                # Since p1, p2 are integers, dx, dy are integers.
                # Any integer t gives integer coordinates.
                
                found_move = False
                for t in range(-10, 11):
                    if t == 0:
                        # This is p1, already used
                        continue
                    # Check if this point is p2
                    if t == 1:
                        continue
                        
                    nx = p1[0] + t * dx
                    ny = p1[1] + t * dy
                    
                    if abs(nx) > 100 or abs(ny) > 100:
                        continue
                        
                    new_point = [nx, ny]
                    
                    # Check distinctness
                    is_distinct = True
                    for idx_p, p in enumerate(best_points):
                        if idx_p != i and p == new_point:
                            is_distinct = False
                            break
                    if not is_distinct:
                        continue
                        
                    test_points = best_points[:]
                    test_points[i] = new_point
                    
                    if has_four_collinear(test_points):
                        continue
                        
                    score = calculate_score(test_points)
                    if score < best_score:
                        best_score = score
                        best_points = test_points
                        improved = True
                        found_move = True
                        break
                    elif score == best_score and random.random() < 0.05:
                        best_points = test_points
                        improved = True
                        found_move = True
                        break
                        
                if found_move:
                    break

    return best_points
