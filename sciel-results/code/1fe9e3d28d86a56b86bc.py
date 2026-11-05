import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = gcd(gcd(abs(A), abs(B)), abs(C))
        if g > 0:
            A //= g
            B //= g
            C //= g
        # Sign convention: first non-zero positive
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def evaluate(points):
        """
        Returns (penalty, triples_count, max_line_size)
        penalty > 0 if any line has >= 4 points.
        triples_count is the number of lines with exactly 3 points.
        max_line_size is the maximum number of points on any line.
        """
        n = len(points)
        line_counts = {}
        
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                key = get_line_key(points[i], points[j])
                if key in line_counts:
                    line_counts[key] += 1
                else:
                    line_counts[key] = 1
        
        penalty = 0
        triples = 0
        max_line_size = 0
        
        for key, count in line_counts.items():
            # count is number of pairs on this line
            # 1 pair -> 2 points
            # 3 pairs -> 3 points (C(3,2)=3)
            # 6 pairs -> 4 points (C(4,2)=6)
            # 10 pairs -> 5 points
            if count == 1:
                size = 2
            elif count == 3:
                size = 3
                triples += 1
            elif count == 6:
                size = 4
                penalty += 1
            elif count == 10:
                size = 5
                penalty += 1
            else:
                # For safety, determine size
                if count == 0: continue
                # Approximate size
                size = int((1 + (1 + 8*count)**0.5) / 2)
                if size >= 4:
                    penalty += 1
            
            if size > max_line_size:
                max_line_size = size

        return penalty, triples, max_line_size

    def get_cost(points):
        p, t, m = evaluate(points)
        # We want to minimize penalty, then maximize triples.
        # Cost = penalty * 10000 - triples
        return p * 10000 - t

    # Initial state: Random points
    def random_points():
        pts = []
        seen = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append([x, y])
        return pts

    best_pts = None
    best_cost = float('inf')
    best_score = 20  # Worst case score is 20 (no 3 collinear)
    
    # Run Simulated Annealing
    for run in range(100): # Multiple restarts
        if time.time() > deadline - 15:
            break
            
        current_pts = random_points()
        current_cost = get_cost(current_pts)
        
        # SA parameters
        T = 50.0
        T_min = 0.1
        cooling_rate = 0.9995
        
        # Steps per run
        steps = 2000
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Perturb
            new_pts = [p[:] for p in current_pts]
            idx = random.randint(0, 19)
            # Small random move
            dx = random.randint(-3, 3)
            dy = random.randint(-3, 3)
            if dx == 0 and dy == 0:
                dx = random.choice([-1, 1])
                
            new_pts[idx][0] += dx
            new_pts[idx][1] += dy
            
            # Clamp
            if new_pts[idx][0] < -100: new_pts[idx][0] = -100
            if new_pts[idx][0] > 100: new_pts[idx][0] = 100
            if new_pts[idx][1] < -100: new_pts[idx][1] = -100
            if new_pts[idx][1] > 100: new_pts[idx][1] = 100
            
            # Check distinctness quickly
            is_unique = True
            for k in range(20):
                if k != idx and new_pts[k] == new_pts[idx]:
                    is_unique = False
                    break
            if not is_unique:
                continue
            
            new_cost = get_cost(new_pts)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_pts = new_pts
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pts = [p[:] for p in current_pts]
                    
                    # If we have a valid configuration (penalty 0), compute score
                    p, t, m = evaluate(best_pts)
                    if p == 0:
                        # Compute the score: largest subset with no three collinear
                        # This is the complement of the minimum hitting set of triples?
                        # No, it's the maximum size of an independent set in the 3-uniform hypergraph of collinear triples.
                        # This is NP-hard. We need a fast approximation or exact for small n.
                        # n=20 is small. We can use a simple branch and bound or greedy.
                        # However, doing this every time is expensive.
                        # Let's only do it if we found a new best valid config.
                        
                        # For now, let's estimate.
                        # If we have many triples, the max subset is smaller.
                        # If we have 0 triples, score is 20.
                        # If we have k triples, score is roughly 20 - something.
                        
                        # Let's do a quick greedy for the score
                        # Maximize subset with no 3 collinear.
                        # Equivalent to: remove minimum points to break all triples.
                        # This is Set Cover / Hitting Set.
                        
                        # Let's just track the number of triples for now as a proxy, 
                        # and do exact score calculation at the end or periodically.
                        # Actually, the task says "score is the largest subset with no three collinear".
                        # So we need to return points, and the evaluator computes the score.
                        # So we just need to minimize the score.
                        # Minimizing the score is equivalent to maximizing the density of collinear triples?
                        # Not exactly.
                        # If we have a configuration with many triples, it's hard to pick a large subset without hitting a triple.
                        # So yes, maximizing triples (while keeping penalty 0) should minimize the score.
                        
                        pass
            
            T *= cooling_rate
            if T < T_min:
                T = T_min

    # If we found a valid best_pts, return it.
    if best_pts is not None:
        p, t, m = evaluate(best_pts)
        if p == 0:
            return best_pts

    # Fallback: return a valid random configuration if SA failed to find a valid one
    # This is unlikely but possible.
    while True:
        pts = random_points()
        p, t, m = evaluate(pts)
        if p == 0:
            return pts
        if time.time() > deadline - 2:
            # Return best invalid? No, must be valid.
            # Try harder?
            # Just return the last one, hoping it's valid?
            # No, we must return a valid one.
            # Let's do a quick repair.
            # For each point involved in a 4-collinear set, move it randomly.
            # This is messy.
            # Given the high probability of finding a valid config in SA, this is a rare edge case.
            # We'll just loop until we find one, assuming we have time.
            pass

# Test/Debug
if __name__ == "__main__":
    pts = construct()
    print(pts)
    p, t, m = evaluate(pts)
    print(f"Penalty: {p}, Triples: {t}, MaxLine: {m}")
