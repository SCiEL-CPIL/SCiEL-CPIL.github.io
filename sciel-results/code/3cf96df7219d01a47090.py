import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # 120s limit, use 100s for search
    deadline = start_time + 100
    
    def are_collinear(p1, p2, p3):
        return (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]) == 0)

    def has_four_collinear(points):
        for combo in combinations(points, 4):
            p1, p2, p3, p4 = combo
            if are_collinear(p1, p2, p3) and are_collinear(p1, p2, p4):
                return True
        return False

    def count_triples(points):
        count = 0
        for combo in combinations(points, 3):
            if are_collinear(*combo):
                count += 1
        return count

    def random_point():
        return [random.randint(-100, 100), random.randint(-100, 100)]

    def generate_valid_set():
        points = []
        attempts = 0
        while len(points) < 20:
            p = random_point()
            if p not in points:
                is_valid = True
                # Check if p forms a 4-collinear set with any 3 existing points
                for combo in combinations(points, 3):
                    if are_collinear(p, *combo):
                        is_valid = False
                        break
                if is_valid:
                    points.append(p)
                else:
                    attempts += 1
                    if attempts > 5000:
                        return [] # Failed
        return points

    best_points = None
    best_triples = -1

    # Run multiple restarts to find a good basin
    restarts = 0
    while time.time() < deadline and restarts < 10:
        current_points = generate_valid_set()
        if not current_points:
            restarts += 1
            continue
            
        current_triples = count_triples(current_points)
        
        # Local search
        no_improve_count = 0
        while time.time() < deadline:
            idx = random.randint(0, 19)
            old_point = current_points[idx]
            
            improved = False
            # Try 10 random moves
            for _ in range(10):
                new_point = random_point()
                if new_point == old_point or new_point in current_points:
                    continue
                
                new_points = current_points[:]
                new_points[idx] = new_point
                
                if has_four_collinear(new_points):
                    continue
                
                new_triples = count_triples(new_points)
                
                if new_triples > current_triples:
                    current_points = new_points
                    current_triples = new_triples
                    improved = True
                    break
                elif new_triples == current_triples and random.random() < 0.05:
                    current_points = new_points
                    improved = True
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 100:
                    break
            else:
                no_improve_count = 0
                
        if current_triples > best_triples:
            best_triples = current_triples
            best_points = current_points[:]
            
        restarts += 1

    if best_points:
        return best_points
    else:
        # Fallback
        return [[x, x * x] for x in range(-10, 10)]
# EVOLVE-BLOCK-END
