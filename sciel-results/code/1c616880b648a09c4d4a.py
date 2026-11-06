import time
import random
import sys

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

    def is_valid(points):
        """Check if no 4 points are collinear."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                            if count >= 4:
                                return False
        return True

    def get_collinear_triples(points):
        """Returns a list of tuples (i, j, k) representing collinear triples."""
        n = len(points)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        triples.append((i, j, k))
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

        # Build adjacency: for each vertex, list of edges (triples) it belongs to
        # And for each edge, list of vertices.
        # Actually, for branch and bound, we need to know which vertices to remove.
        
        # Precompute for each vertex v, the set of vertices that cannot be in the independent set
        # if v is included.
        # If v is in a triple (v, a, b), then a and b cannot be in the set.
        # So, if we include v, we must exclude all 'partners' of v in any triple.
        
        conflict_mask = [0] * n
        for i, j, k in triples:
            conflict_mask[i] |= (1 << j) | (1 << k)
            conflict_mask[j] |= (1 << i) | (1 << k)
            conflict_mask[k] |= (1 << i) | (1 << j)

        best = [0]
        
        # Order vertices by degree (number of conflicts) to prune better?
        # Or just use index order.
        # Let's sort vertices by degree descending.
        degrees = [bin(conflict_mask[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        # Map original indices to new order?
        # It's easier to just work with the original indices but pick the "best" candidate.
        
        def backtrack(available_mask, current_size):
            # Pruning: even if we take all remaining, can we beat best?
            if current_size + bin(available_mask).count('1') <= best[0]:
                return
            
            if available_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Pick a vertex. Heuristic: pick the one with highest degree in the remaining subgraph?
            # Or just the first one.
            # Let's pick the vertex with the most conflicts within the available set.
            # This is expensive to compute every step.
            # Simple heuristic: pick the lowest set bit.
            # Better: pick a vertex that is in many triples within the available set.
            
            # For simplicity and speed, let's just pick the first available vertex.
            # But we can do a simple greedy bound.
            
            v_bit = available_mask & -available_mask
            v = v_bit.bit_length() - 1
            
            # Option 1: Exclude v
            backtrack(available_mask ^ v_bit, current_size)
            
            # Option 2: Include v
            # Remove v and all its conflicts
            new_available = available_mask ^ v_bit
            new_available &= ~conflict_mask[v]
            
            backtrack(new_available, current_size + 1)
            
        backtrack((1 << n) - 1, 0)
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
