import random
import math
import time
from collections import defaultdict

def construct():
    """
    Constructs a set of 16 integer points maximizing the number of 4-point lines.
    Uses a Simulated Annealing approach initialized with points derived from
    the Finite Projective Plane of order 3 (PG(3,3)).
    """
    
    # 1. Generate the 16 points of PG(3,3) as vectors in F_3^4
    # We represent points as lines through the origin in 3^4.
    # We pick a representative for each 1D subspace.
    # A simple way is to iterate all non-zero vectors and keep the first one seen
    # for each line (scalar multiple).
    
    pg33_vectors = []
    seen_lines = set()
    for x0 in range(3):
        for x1 in range(3):
            for x2 in range(3):
                for x3 in range(3):
                    if x0 == 0 and x1 == 0 and x2 == 0 and x3 == 0:
                        continue
                    v = (x0, x1, x2, x3)
                    # Normalize: make the first non-zero element 1
                    # In F_3, non-zero elements are 1, 2.
                    # If first non-zero is 2, multiply by 2 (which is -1) to make it 1.
                    # 2 * 2 = 4 = 1 mod 3.
                    
                    # Find first non-zero
                    idx = -1
                    for i in range(4):
                        if v[i] != 0:
                            idx = i
                            break
                            
                    if v[idx] == 2:
                        v = ((v[0]*2)%3, (v[1]*2)%3, (v[2]*2)%3, (v[3]*2)%3)
                    
                    if v not in seen_lines:
                        seen_lines.add(v)
                        pg33_vectors.append(v)
    
    # There should be 16 vectors
    # assert len(pg33_vectors) == 16
    
    # 2. Map these 16 points to 2D integer coordinates
    # We use a random linear projection from 4D to 2D, then scale.
    
    def initialize_points():
        # Random projection matrix 2x4
        M = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(2)]
        
        mapped_points = []
        for v in pg33_vectors:
            x = M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2] + M[0][3]*v[3]
            y = M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2] + M[1][3]*v[3]
            mapped_points.append([x, y])
            
        # Scale to fit in [-1000, 1000]
        # Find max abs coordinate
        max_c = 1e-9
        for p in mapped_points:
            max_c = max(max_c, abs(p[0]), abs(p[1]))
            
        scale = 500.0 / max_c  # Scale to roughly [-500, 500] to leave room for moves
        
        int_points = []
        for p in mapped_points:
            ix = int(round(p[0] * scale))
            iy = int(round(p[1] * scale))
            # Ensure bounds
            ix = max(-1000, min(1000, ix))
            iy = max(-1000, min(1000, iy))
            int_points.append([ix, iy])
            
        # Ensure distinctness? 
        # If collisions happen, add small random jitter.
        # With random projection, collisions are unlikely unless points are close.
        # We will rely on the optimizer to fix duplicates if they score poorly (invalid).
        # But distinct points are required.
        # Let's add a small random offset if duplicates exist.
        for i in range(16):
            for j in range(i+1, 16):
                if int_points[i] == int_points[j]:
                    int_points[j][0] += random.randint(1, 3)
                    int_points[j][1] += random.randint(1, 3)
                    
        return int_points

    # 3. Evaluation Function
    def evaluate(points):
        n = 16
        line_counts = defaultdict(int)
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                if A == 0 and B == 0 and C == 0:
                    # Duplicate points, invalid
                    return -1000, False
                
                g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
                if g != 0:
                    A //= g
                    B //= g
                    C //= g
                
                # Normalize sign
                if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                    A = -A
                    B = -B
                    C = -C
                    
                line_counts[(A, B, C)] += 1
                
        score = 0
        for key, pairs in line_counts.items():
            # pairs = k * (k-1) / 2
            # k=4 -> 6 pairs
            # k=5 -> 10 pairs
            if pairs == 6:
                score += 1
            elif pairs > 6:
                return -1000, False # Invalid: 5+ collinear
                
        return score, True

    # 4. Simulated Annealing
    def simulate_annealing(initial_points, time_limit=100):
        current_points = initial_points[:]
        current_score, _ = evaluate(current_points)
        
        best_points = current_points[:]
        best_score = current_score
        
        start_time = time.time()
        T = 50.0  # Initial temperature
        cooling_rate = 0.9999
        
        while time.time() - start_time < time_limit:
            if T < 0.1:
                break
                
            idx = random.randint(0, 15)
            old_x, old_y = current_points[idx]
            
            # Move strategy: 
            # Small moves (radius 1-2) and occasional large jumps (radius 50-100)
            if random.random() < 0.1:
                # Large jump
                dx = random.randint(-100, 100)
                dy = random.randint(-100, 100)
            else:
                # Small move
                dx = random.randint(-2, 2)
                dy = random.randint(-2, 2)
                
            if dx == 0 and dy == 0:
                continue
                
            new_x = old_x + dx
            new_y = old_y + dy
            
            if abs(new_x) > 1000 or abs(new_y) > 1000:
                continue
                
            # Check duplicate
            is_dup = False
            for i in range(16):
                if i != idx and current_points[i][0] == new_x and current_points[i][1] == new_y:
                    is_dup = True
                    break
            if is_dup:
                continue
                
            # Apply move
            current_points[idx] = [new_x, new_y]
            new_score, new_valid = evaluate(current_points)
            
            if new_valid:
                delta = new_score - current_score
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best_points = current_points[:]
                else:
                    # Revert
                    current_points[idx] = [old_x, old_y]
            else:
                # Invalid, revert
                current_points[idx] = [old_x, old_y]
                
            T *= cooling_rate
            
        return best_points

    # Run a few iterations with different initializations to find a good basin
    best_overall = None
    best_overall_score = -1
    
    start_global = time.time()
    while time.time() - start_global < 110:
        init_pts = initialize_points()
        res_pts = simulate_annealing(init_pts, time_limit=20)
        res_score, _ = evaluate(res_pts)
        
        if res_score > best_overall_score:
            best_overall_score = res_score
            best_overall = res_pts

    return best_overall

if __name__ == "__main__":
    pts = construct()
    # Quick check
    print(pts)
