import time
import random
import sys
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    N = 20
    COORD_MIN = -100
    COORD_MAX = 100

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy)) if (dx != 0 or dy != 0) else 1
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        c = dy * x1 - dx * y1
        return (dx, dy, c)

    def build_line_map(points):
        n = len(points)
        line_map = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                key = line_key(points[i], points[j])
                line_map[key].add(i)
                line_map[key].add(j)
        return line_map

    def is_valid(points):
        line_map = build_line_map(points)
        for line_pts in line_map.values():
            if len(line_pts) >= 4:
                return False
        return True

    def get_collinear_triples(points):
        n = len(points)
        line_map = build_line_map(points)
        triples = []
        for line_pts in line_map.values():
            if len(line_pts) >= 3:
                pts = sorted(line_pts)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        for k in range(j + 1, len(pts)):
                            triples.append((pts[i], pts[j], pts[k]))
        return triples

    def heuristic_score(points):
        """
        Heuristic: Number of collinear triples.
        Higher is better for minimizing the max no-3-collinear subset.
        """
        return len(get_collinear_triples(points))

    def calculate_exact_score(points):
        """
        Calculate the size of the largest subset with no three collinear.
        Uses branch and bound on the 3-uniform hypergraph of collinear triples.
        """
        n = len(points)
        triples = get_collinear_triples(points)
        
        if not triples:
            return n

        # Build conflict graph: two vertices conflict if they are in a triple
        # We want the largest subset with NO three collinear.
        # This is equivalent to finding the maximum independent set in the 3-uniform hypergraph?
        # No, "no three collinear" means we cannot pick a set of 3 points that are collinear.
        # This is a 3-uniform hypergraph independence problem.
        # However, we can model this as a standard graph problem if we consider the "forbidden" triples.
        # Actually, the standard formulation is: Find max subset S such that no triple in S is collinear.
        
        # Branch and bound on the hypergraph is hard.
        # But note: if we pick 2 points, we forbid the 3rd point of any collinear triple containing them.
        # Let's use a recursive backtracking with pruning.
        
        # Precompute for each pair (i,j), the set of points k such that (i,j,k) is collinear.
        # If we include i and j, we cannot include any k in conflict[i][j].
        
        conflict_pairs = [[set() for _ in range(n)] for _ in range(n)]
        for i, j, k in triples:
            conflict_pairs[i][j].add(k)
            conflict_pairs[j][i].add(k)
            conflict_pairs[i][k].add(j)
            conflict_pairs[k][i].add(j)
            conflict_pairs[j][k].add(i)
            conflict_pairs[k][j].add(i)

        best = [0]
        
        # We will use a bitmask for the available points.
        # We branch on including or excluding a point.
        # To prune, we can use the fact that if we have a set of points S, 
        # the maximum additional points we can add is bounded by...
        
        # A simpler bound: The number of remaining points.
        # Better bound: If we have selected points S, any new point v must not form a triple with any pair in S.
        # Let's stick to a simple DFS with memoization or strong pruning.
        
        # Since N=20, we can use bitmasks.
        # State: (available_mask, selected_mask)
        # This state space is huge.
        
        # Alternative: Branch on the first available point.
        # If we include it, we remove it and all points that form a triple with it and ANY other selected point?
        # No, if we include v, we just remove v. But we must ensure that no triple is formed.
        # A triple is formed if we have u, w in selected and v is collinear with u, w.
        # So, if we include v, we must check if v forms a triple with any pair in selected.
        # If it does, we cannot include v.
        
        # Let's reformulate:
        # We are building a set S.
        # If we add v to S, we must ensure that for all pairs (u, w) in S, v is NOT collinear with u, w.
        # This is equivalent to: v is not in conflict_pairs[u][w] for any u, w in S.
        
        # This check is O(|S|^2) per node.
        
        def backtrack(available_mask, selected_mask, current_size):
            if current_size > best[0]:
                best[0] = current_size
            
            if available_mask == 0:
                return
            
            # Pruning
            if current_size + available_mask.bit_count() <= best[0]:
                return
            
            # Pick a vertex to branch on.
            # Heuristic: pick the vertex with the most conflicts within the available set?
            # Or just the first one.
            v_bit = available_mask & -available_mask
            v = v_bit.bit_length() - 1
            
            # Option 1: Exclude v
            backtrack(available_mask ^ v_bit, selected_mask, current_size)
            
            # Option 2: Include v
            # Check if v can be added
            can_add = True
            # For all pairs (u, w) in selected_mask, check if v is collinear with u, w
            # i.e., v in conflict_pairs[u][w]
            sm = selected_mask
            # Iterate through pairs in selected_mask
            # Since N=20, we can just loop
            for u in range(n):
                if not (sm & (1 << u)):
                    continue
                for w in range(u + 1, n):
                    if not (sm & (1 << w)):
                        continue
                    if v in conflict_pairs[u][w]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                backtrack(available_mask ^ v_bit, selected_mask | v_bit, current_size + 1)
                
        backtrack((1 << n) - 1, 0, 0)
        return best[0]

    def generate_random_points():
        pts = []
        while len(pts) < N:
            x = random.randint(COORD_MIN, COORD_MAX)
            y = random.randint(COORD_MIN, COORD_MAX)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        return pts

    # Initial Configuration
    best_config = None
    best_score = 20
    best_heuristic = -1
    
    # Try to find a valid starting point
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        pts = generate_random_points()
        if is_valid(pts):
            best_config = pts
            break
            
    if best_config is None:
        # Fallback: Parabola (Score 20)
        return [[x, x * x] for x in range(-10, 10)]

    # Local Search
    current_pts = [p[:] for p in best_config]
    current_heuristic = heuristic_score(current_pts)
    
    # Simulated Annealing / Random Walk
    # We want to maximize heuristic_score (more triples)
    # But we care about exact_score.
    
    temp = 1.0
    cooling = 0.99
    
    while time.time() < deadline - 5:
        # Pick a point to move
        idx = random.randint(0, N - 1)
        old_x, old_y = current_pts[idx]
        
        # New random position
        new_x = random.randint(COORD_MIN, COORD_MAX)
        new_y = random.randint(COORD_MIN, COORD_MAX)
        
        if new_x == old_x and new_y == old_y:
            continue
            
        # Check distinctness
        if any(current_pts[i] == [new_x, new_y] for i in range(N) if i != idx):
            continue
            
        # Apply move
        current_pts[idx] = [new_x, new_y]
        
        # Check validity (no 4 collinear)
        # Optimization: Only check lines involving the moved point
        valid = True
        for i in range(N):
            if i == idx:
                continue
            # Line through idx and i
            # Check if any other point j is on this line
            # And if so, if a 4th point k is on this line
            # Actually, just count points on line (idx, i)
            count = 2 # idx and i
            for j in range(N):
                if j != idx and j != i:
                    if are_collinear(current_pts[idx], current_pts[i], current_pts[j]):
                        count += 1
            if count >= 4:
                valid = False
                break
                
        if not valid:
            # Revert
            current_pts[idx] = [old_x, old_y]
            continue
            
        # Calculate new heuristic
        new_heuristic = heuristic_score(current_pts)
        
        # Acceptance criteria
        if new_heuristic > current_heuristic:
            current_heuristic = new_heuristic
        else:
            # Random acceptance to escape local minima
            # Delta = new - current (negative or zero)
            delta = new_heuristic - current_heuristic
            if random.random() < pow(2.718, delta / temp):
                current_heuristic = new_heuristic
            else:
                # Revert
                current_pts[idx] = [old_x, old_y]
                continue
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
        # Periodically check exact score
        # We can check every 500 steps or if heuristic improves significantly
        # For now, let's check if we have found a new heuristic record
        # Or just check every 1000 iterations?
        # Let's check if time allows.
        if current_heuristic > best_heuristic:
            best_heuristic = current_heuristic
            # Calculate exact score
            exact = calculate_exact_score(current_pts)
            if exact < best_score:
                best_score = exact
                best_config = [p[:] for p in current_pts]
                
        # Also check exact score periodically even if heuristic didn't improve?
        # Maybe every 2000 steps.
        # Let's add a counter.
        
    # Final check on best_config
    if best_config:
        final_score = calculate_exact_score(best_config)
        if final_score < best_score:
            best_score = final_score
            
    return best_config

# Test
if __name__ == "__main__":
    res = construct()
    print(res)
