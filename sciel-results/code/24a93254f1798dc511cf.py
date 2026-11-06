import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        g = math.gcd(A, math.gcd(B, C))
        if g == 0:
            return None
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)

    def has_4_collinear(pts):
        n = len(pts)
        if n < 4:
            return False
        seen_lines = set()
        for i in range(n):
            for j in range(i + 1, n):
                key = get_line_key(pts[i], pts[j])
                if key is None or key in seen_lines:
                    continue
                seen_lines.add(key)
                A, B, C = key
                count = 0
                for k in range(n):
                    x3, y3 = pts[k]
                    if A * x3 + B * y3 + C == 0:
                        count += 1
                if count >= 4:
                    return True
        return False

    def count_triples(pts):
        """Count the number of collinear triples in pts."""
        n = len(pts)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        count += 1
        return count

    def calculate_score(pts):
        n = len(pts)
        if n < 3:
            return n
        
        triples_with_idx = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                for k in range(j + 1, n):
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        triples_with_idx[i].append((j, k))
                        triples_with_idx[j].append((i, k))
                        triples_with_idx[k].append((i, j))
                        
        best_score = 0
        
        order = sorted(range(n), key=lambda i: len(triples_with_idx[i]), reverse=True)
        
        def bbb(step, current_mask, current_size):
            nonlocal best_score
            if elapsed() > deadline - 5:
                return
            if current_size > best_score:
                best_score = current_size
            if step == n:
                return
            if current_size + (n - step) <= best_score:
                return
                
            idx = order[step]
            
            bbb(step + 1, current_mask, current_size)
            
            valid = True
            for a, b in triples_with_idx[idx]:
                if (current_mask & (1 << a)) and (current_mask & (1 << b)):
                    valid = False
                    break
            
            if valid:
                bbb(step + 1, current_mask | (1 << idx), current_size + 1)
                
        bbb(0, 0, 0)
        return best_score

    def generate_random_valid_set():
        """Generate a random set of 20 points with no 4 collinear."""
        pts = []
        while len(pts) < 20:
            p = (random.randint(-100, 100), random.randint(-100, 100))
            if p in pts:
                continue
            
            valid = True
            for i in range(len(pts)):
                a = pts[i]
                for j in range(i + 1, len(pts)):
                    b = pts[j]
                    if (b[0] - a[0]) * (p[1] - a[1]) != (b[1] - a[1]) * (p[0] - a[0]):
                        continue
                    count = 2
                    for k in range(len(pts)):
                        if k == i or k == j:
                            continue
                        c = pts[k]
                        if (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0]):
                            count += 1
                    if count >= 3:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                pts.append(p)
        
        if len(pts) == 20:
            return pts
        return None

    best_points = None
    best_triple_count = -1
    best_exact_score = float('inf')
    
    candidates = []
    
    for _ in range(20):
        if elapsed() > deadline - 30:
            break
        pts = generate_random_valid_set()
        if pts:
            t = count_triples(pts)
            if t > best_triple_count:
                best_triple_count = t
                best_points = pts
            candidates.append(pts)
            
    for pts in candidates:
        if elapsed() > deadline - 20:
            break
            
        current_pts = [tuple(p) for p in pts]
        current_triples = count_triples(current_pts)
        
        for step in range(200):
            if elapsed() > deadline - 10:
                break
                
            improved = False
            for i in range(20):
                if elapsed() > deadline - 10:
                    break
                    
                old_p = current_pts[i]
                for _ in range(5):
                    if random.random() < 0.5:
                        new_p = (old_p[0] + random.randint(-5, 5), old_p[1] + random.randint(-5, 5))
                    else:
                        new_p = (random.randint(-100, 100), random.randint(-100, 100))
                        
                    if new_p in current_pts:
                        continue
                    if not (-100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100):
                        continue
                        
                    new_pts = current_pts[:]
                    new_pts[i] = new_p
                    
                    if has_4_collinear(new_pts):
                        continue
                        
                    new_triples = count_triples(new_pts)
                    
                    if new_triples > current_triples:
                        current_pts = new_pts
                        current_triples = new_triples
                        improved = True
                        break
                    
            if not improved:
                if random.random() < 0.1:
                    i = random.randint(0, 19)
                    new_p = (random.randint(-100, 100), random.randint(-100, 100))
                    if new_p not in current_pts and -100 <= new_p[0] <= 100 and -100 <= new_p[1] <= 100:
                         new_pts = current_pts[:]
                         new_pts[i] = new_p
                         if not has_4_collinear(new_pts):
                             current_pts = new_pts
                             current_triples = count_triples(current_pts)
                else:
                    break
        
        if current_triples > best_triple_count:
            best_triple_count = current_triples
            best_points = current_pts

    if best_points is None:
        while True:
            pts = generate_random_valid_set()
            if pts:
                best_points = pts
                break
    else:
        exact_score = calculate_score(best_points)
        best_exact_score = exact_score
        
        for pts in candidates:
            if elapsed() > deadline - 5:
                break
            if pts == best_points:
                continue
            t = count_triples(pts)
            if t >= best_triple_count - 2:
                s = calculate_score(pts)
                if s < best_exact_score:
                    best_exact_score = s
                    best_points = pts

    return [[x, y] for x, y in best_points]

if __name__ == "__main__":
    pts = construct()
    print(pts)
    print("Score:", calculate_score(pts))
    print("Has 4 collinear:", has_4_collinear(pts))
