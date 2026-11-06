import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    N = 20
    
    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)
    
    def has_four_collinear_fast(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                count = 0
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if (x3 - x1) * dy == (y3 - y1) * dx:
                        count += 1
                        if count >= 2:
                            return True
        return False

    def score(points):
        mask_collinear = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j: continue
                mask = 0
                for k in range(N):
                    if k == i or k == j: continue
                    if are_collinear(points[i], points[j], points[k]):
                        mask |= (1 << k)
                mask_collinear[i][j] = mask
        
        best = 0
        import sys
        sys.setrecursionlimit(10000)
        
        def dfs(current, available, size):
            nonlocal best
            if size + bin(available).count('1') <= best:
                return
            if available == 0:
                if size > best:
                    best = size
                return
            
            i = (available & -available).bit_length() - 1
            bit = 1 << i
            
            dfs(current, available & ~bit, size)
            
            valid = True
            for j in range(N):
                if current & (1 << j):
                    if mask_collinear[i][j] & current:
                        valid = False
                        break
            
            if valid:
                new_available = available & ~bit
                for j in range(N):
                    if current & (1 << j):
                        new_available &= ~mask_collinear[i][j]
                dfs(current | bit, new_available, size + 1)
        
        dfs(0, (1<<N)-1, 0)
        return best

    def generate_random_valid():
        while True:
            points = []
            used = set()
            while len(points) < N:
                x = random.randint(-50, 50)
                y = random.randint(-50, 50)
                p = (x, y)
                if p in used:
                    continue
                used.add(p)
                points.append(p)
            
            if not has_four_collinear_fast(points):
                return points

    best_points = None
    best_score = 21
    
    for _ in range(50):
        if time.time() > deadline: break
        pts = generate_random_valid()
        s = score(pts)
        if s < best_score:
            best_score = s
            best_points = pts
    
    if best_points is None:
        return [[x, x*x] for x in range(-10, 10)]

    current_points = best_points
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        # 1-swap
        for _ in range(20): # Try 20 random swaps per iteration
            if time.time() > deadline: break
            i = random.randint(0, N-1)
            new_p = (random.randint(-100, 100), random.randint(-100, 100))
            if new_p in current_points:
                continue
            
            new_points = current_points.copy()
            new_points[i] = new_p
            
            if has_four_collinear_fast(new_points):
                continue
            
            s = score(new_points)
            if s < current_score:
                current_points = new_points
                current_score = s
                improved = True
                break
        
        if not improved:
            # 2-swap
            improved = False
            for _ in range(10):
                if time.time() > deadline: break
                i = random.randint(0, N-1)
                j = random.randint(0, N-1)
                if i == j: continue
                
                new_p1 = (random.randint(-100, 100), random.randint(-100, 100))
                new_p2 = (random.randint(-100, 100), random.randint(-100, 100))
                
                if new_p1 in current_points or new_p2 in current_points or new_p1 == new_p2:
                    continue
                
                new_points = current_points.copy()
                new_points[i] = new_p1
                new_points[j] = new_p2
                
                if has_four_collinear_fast(new_points):
                    continue
                
                s = score(new_points)
                if s < current_score:
                    current_points = new_points
                    current_score = s
                    improved = True
                    break
            
            if not improved:
                break

    return [list(p) for p in current_points]
# EVOLVE-BLOCK-END
