import math
import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0
    
    def evaluate(points):
        """Fast evaluation: count lines with exactly 4 points, reject if any line has 5+."""
        n = len(points)
        # Check bounds and duplicates
        seen = set()
        for x, y in points:
            if abs(x) > 1000 or abs(y) > 1000:
                return False, 0
            t = (x, y)
            if t in seen:
                return False, 0
            seen.add(t)
        
        lines = {}
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                A = dy // g
                B = -(dx // g)
                C = -(A * x1 + B * y1)
                g2 = math.gcd(abs(A), abs(B))
                if g2 > 0:
                    A //= g2
                    B //= g2
                    C //= g2
                if A < 0 or (A == 0 and B < 0):
                    A, B, C = -A, -B, -C
                key = (A, B, C)
                if key not in lines:
                    lines[key] = set()
                lines[key].add(i)
                lines[key].add(j)
        
        score = 0
        for key, pts_set in lines.items():
            c = len(pts_set)
            if c >= 5:
                return False, 0
            if c == 4:
                score += 1
        return True, score

    # --- Precompute the affine plane of order 4 line structure ---
    # 16 points indexed 0..15, 20 lines of 4 points each.
    # We represent each line as a tuple of 4 indices.
    # The affine plane AG(2,4) can be constructed as follows:
    # Points: (x, y) where x, y in GF(4)
    # GF(4) = {0, 1, alpha, alpha+1} with alpha^2 + alpha + 1 = 0
    # Lines: for each slope m in GF(4) ∪ {∞}, and each intercept b in GF(4):
    #   - m = ∞: x = b (4 vertical lines)
    #   - m finite: y = m*x + b (4 lines for each slope, 4 slopes total = 16 lines)
    # Total: 4 + 16 = 20 lines.
    
    # Map GF(4) elements to 0,1,2,3 for indexing
    # Let 0->0, 1->1, alpha->2, alpha+1->3
    # Addition table for GF(4):
    #   0+0=0, 0+1=1, 0+a=a, 0+(a+1)=a+1
    #   1+1=0, 1+a=a+1, 1+(a+1)=a
    #   a+a=0, a+(a+1)=1
    #   (a+1)+(a+1)=0
    # Multiplication table:
    #   0*anything=0
    #   1*anything=anything
    #   a*a=a+1, a*(a+1)=1, (a+1)*(a+1)=a
    
    gf4_add = {
        (0,0):0, (0,1):1, (0,2):2, (0,3):3,
        (1,0):1, (1,1):0, (1,2):3, (1,3):2,
        (2,0):2, (2,1):3, (2,2):0, (2,3):1,
        (3,0):3, (3,1):2, (3,2):1, (3,3):0,
    }
    gf4_mul = {
        (0,0):0, (0,1):0, (0,2):0, (0,3):0,
        (1,0):0, (1,1):1, (1,2):2, (1,3):3,
        (2,0):0, (2,1):2, (2,2):3, (2,3):1,
        (3,0):0, (3,1):3, (3,2):1, (3,3):2,
    }
    
    def gf4_add_fn(a, b):
        return gf4_add[(a, b)]
    
    def gf4_mul_fn(a, b):
        return gf4_mul[(a, b)]
    
    # Build lines of AG(2,4)
    # Point index: x*4 + y where x,y in {0,1,2,3}
    affine_lines = []
    
    # Vertical lines: x = c for c in {0,1,2,3}
    for c in range(4):
        line = tuple(c * 4 + y for y in range(4))
        affine_lines.append(line)
    
    # Lines with slope m in {0,1,2,3} and intercept b in {0,1,2,3}
    for m in range(4):
        for b in range(4):
            line = []
            for x in range(4):
                y = gf4_add_fn(gf4_mul_fn(m, x), b)
                line.append(x * 4 + y)
            affine_lines.append(tuple(line))
    
    # Verify: 20 lines, each of size 4, each pair of points in exactly one line
    assert len(affine_lines) == 20
    for line in affine_lines:
        assert len(line) == 4
        assert len(set(line)) == 4
    # Check each pair is in exactly one line
    pair_count = {}
    for line in affine_lines:
        for p, q in combinations(line, 2):
            pair = (min(p,q), max(p,q))
            pair_count[pair] = pair_count.get(pair, 0) + 1
    assert all(v == 1 for v in pair_count.values())
    assert len(pair_count) == len(list(combinations(range(16), 2)))  # 120 pairs
    
    # --- Search strategy ---
    # The affine plane structure has 20 lines of 4 points. If we can map the 16 abstract
    # points to integer coordinates such that all 20 lines correspond to collinear 4-point
    # lines in Z^2, we'd get 20 four-point lines. This is the theoretical maximum for
    # the affine plane. In practice, we may not achieve all 20, but we can try to get close.
    # 
    # Approach: Use the affine plane line structure as a constraint template.
    # Assign integer coordinates to 16 abstract points, then check how many of the 20
    # affine lines are realized as collinear 4-point lines.
    #
    # We'll use a hybrid approach:
    # 1. Start with known good configurations (4x4 grid, etc.)
    # 2. Use local search with moves guided by the affine plane structure
    # 3. Try to "snap" to affine-plane-like configurations
    
    best_score = -1
    best_points = None
    
    # Initial configurations
    initial_configs = []
    
    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])
    
    # Various grids
    for sx in [1, 2, 3]:
        for sy in [1, 2, 3]:
            initial_configs.append([[x * sx, y * sy] for x in range(4) for y in range(4)])
    
    # Affine plane-inspired: use the structure to guide coordinate assignment
    # Try to place points so that lines of the affine plane correspond to real lines
    
    # Random configurations
    for _ in range(30):
        initial_configs.append([[random.randint(-100, 100), random.randint(-100, 100)] for _ in range(16)])
    
    # Evaluate all initial configs
    for init_pts in initial_configs:
        if time.time() > deadline - 5:
            break
        valid, score = evaluate(init_pts)
        if valid and score > best_score:
            best_score = score
            best_points = [p[:] for p in init_pts]
    
    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(4)]
        best_score = evaluate(best_points)[1]
    
    # --- Local search with affine-plane-guided moves ---
    current_points = [p[:] for p in best_points]
    current_valid, current_score = evaluate(current_points)
    
    no_improve = 0
    max_no_improve = 80
    
    while time.time() < deadline - 2:
        if not current_valid:
            # Random restart
            current_points = [[random.randint(-200, 200), random.randint(-200, 200)] for _ in range(16)]
            current_valid, current_score = evaluate(current_points)
            no_improve = 0
            continue
        
        occupied = set((p[0], p[1]) for p in current_points)
        
        # Generate candidate moves:
        # 1. Extend existing 3-point lines
        # 2. Move points to create new 4-point lines
        # 3. Random moves
        
        candidates = []
        
        # Find all 3-point lines and try extending them
        pts_arr = np.array(current_points, dtype=np.int64)
        line_dict = {}
        for i in range(16):
            for j in range(i+1, 16):
                dx = pts_arr[j,0] - pts_arr[i,0]
                dy = pts_arr[j,1] - pts_arr[i,1]
                g = math.gcd(abs(dx), abs(dy))
                if g == 0:
                    continue
                A = dy // g
                B = -(dx // g)
                C = -(A * pts_arr[i,0] + B * pts_arr[i,1])
                g2 = math.gcd(abs(A), abs(B))
                if g2 > 0:
                    A //= g2; B //= g2; C //= g2
                if A < 0 or (A == 0 and B < 0):
                    A, B, C = -A, -B, -C
                key = (A, B, C)
                if key not in line_dict:
                    line_dict[key] = set()
                line_dict[key].add(i)
                line_dict[key].add(j)
        
        for key, pts_set in line_dict.items():
            if len(pts_set) == 3:
                A, B, C = key
                # Get the three points
                p_list = list(pts_set)
                p1 = current_points[p_list[0]]
                p2 = current_points[p_list[1]]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = math.gcd(abs(dx), abs(dy))
                if g > 0:
                    dx //= g; dy //= g
                    # Try extending
                    for t in range(-5, 8):
                        if t in (0, 1):
                            continue
                        nx = p1[0] + t * dx
                        ny = p1[1] + t * dy
                        if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                            if (nx, ny) not in occupied:
                                candidates.append((nx, ny))
        
        # Add random candidates
        for _ in range(10):
            rx = random.randint(-300, 300)
            ry = random.randint(-300, 300)
            if (rx, ry) not in occupied:
                candidates.append((rx, ry))
        
        if not candidates:
            idx = random.randint(0, 15)
            old = current_points[idx][:]
            current_points[idx] = [random.randint(-300, 300), random.randint(-300, 300)]
            valid, score = evaluate(current_points)
            if valid and score > current_score:
                current_score = score
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[idx] = old
            no_improve += 1
            if no_improve > max_no_improve:
                for _ in range(4):
                    k = random.randint(0, 15)
                    current_points[k] = [random.randint(-300, 300), random.randint(-300, 300)]
                current_valid, current_score = evaluate(current_points)
                no_improve = 0
            continue
        
        # Try moves
        tried = False
        sample_size = min(len(candidates), 15)
        for new_pt in random.sample(candidates, sample_size):
            if time.time() > deadline - 2:
                break
            
            idx = random.randint(0, 15)
            old = current_points[idx][:]
            if old[0] == new_pt[0] and old[1] == new_pt[1]:
                continue
            
            current_points[idx] = [new_pt[0], new_pt[1]]
            valid, score = evaluate(current_points)
            
            if valid and score > current_score:
                current_score = score
                no_improve = 0
                tried = True
                if current_score > best_score:
                    best_score = score
                    best_points = [p[:] for p in current_points]
            else:
                current_points[idx] = old
        
        if not tried:
            no_improve += 1
            if no_improve > max_no_improve:
                # Random kick
                for _ in range(4):
                    k = random.randint(0, 15)
                    current_points[k] = [random.randint(-300, 300), random.randint(-300, 300)]
                current_valid, current_score = evaluate(current_points)
                no_improve = 0
    
    # Final: try to improve best_points with a focused search
    if best_points and time.time() < deadline - 5:
        cp = [p[:] for p in best_points]
        cv, cs = evaluate(cp)
        if cv:
            no_imp2 = 0
            while time.time() < deadline - 2 and no_imp2 < 100:
                occ = set((p[0], p[1]) for p in cp)
                cands = []
                # Extend 3-lines
                pts_arr = np.array(cp, dtype=np.int64)
                ld = {}
                for i in range(16):
                    for j in range(i+1, 16):
                        dx = pts_arr[j,0] - pts_arr[i,0]
                        dy = pts_arr[j,1] - pts_arr[i,1]
                        g = math.gcd(abs(dx), abs(dy))
                        if g == 0: continue
                        A = dy // g; B = -(dx // g)
                        C = -(A * pts_arr[i,0] + B * pts_arr[i,1])
                        g2 = math.gcd(abs(A), abs(B))
                        if g2 > 0: A //= g2; B //= g2; C //= g2
                        if A < 0 or (A == 0 and B < 0): A, B, C = -A, -B, -C
                        k = (A, B, C)
                        if k not in ld: ld[k] = set()
                        ld[k].add(i); ld[k].add(j)
                for k, s in ld.items():
                    if len(s) == 3:
                        pl = list(s)
                        p1 = cp[pl[0]]; p2 = cp[pl[1]]
                        dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
                        g = math.gcd(abs(dx), abs(dy))
                        if g > 0:
                            dx //= g; dy //= g
                            for t in range(-6, 9):
                                if t in (0,1): continue
                                nx = p1[0] + t*dx; ny = p1[1] + t*dy
                                if -1000 <= nx <= 1000 and -1000 <= ny <= 1000:
                                    if (nx, ny) not in occ:
                                        cands.append((nx, ny))
                for _ in range(8):
                    rx = random.randint(-400, 400); ry = random.randint(-400, 400)
                    if (rx, ry) not in occ:
                        cands.append((rx, ry))
                
                improved = False
                for np_ in random.sample(cands, min(len(cands), 20)):
                    if time.time() > deadline - 2: break
                    idx = random.randint(0, 15)
                    old = cp[idx][:]
                    if old == list(np_): continue
                    cp[idx] = list(np_)
                    v, sc = evaluate(cp)
                    if v and sc > cs:
                        cs = sc; improved = True
                        if cs > best_score:
                            best_score = cs
                            best_points = [p[:] for p in cp]
                    else:
                        cp[idx] = old
                if not improved:
                    no_imp2 += 1
                else:
                    no_imp2 = 0
    
    return best_points
