import time
import random
import math
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 135.0

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def check_no4(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i+1, n):
                cnt = 0
                for k in range(n):
                    if k != i and k != j and cross(pts[i], pts[j], pts[k]) == 0:
                        cnt += 1
                if cnt >= 2:
                    return False
        return True

    def count_triples(pts):
        n = len(pts)
        tris = 0
        for i in range(n):
            for j in range(i+1, n):
                cnt = 0
                for k in range(n):
                    if k != i and k != j and cross(pts[i], pts[j], pts[k]) == 0:
                        cnt += 1
                tris += cnt * (cnt + 1) // 2
        return tris

    def line_key(p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        g = math.gcd(abs(dx), abs(dy))
        if g == 0:
            g = 1
        dx //= g
        dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        x0 = p[0]
        y0 = p[1]
        return (dx, dy, x0*dy - y0*dx)

    def build_line_map(pts):
        n = len(pts)
        lines = defaultdict(list)
        for i in range(n):
            for j in range(i+1, n):
                key = line_key(pts[i], pts[j])
                lines[key].append((i, j))
        return lines

    def triples_from_lines(lines, n):
        tris = 0
        for key, pairs in lines.items():
            cnt = len(pairs)
            if cnt >= 1:
                # cnt pairs on a line means cnt+1 points? No.
                # Each pair is on the line. Number of points on line = ?
                # Actually pairs are (i,j) both on line. Unique points:
                pts_on_line = set()
                for (i,j) in pairs:
                    pts_on_line.add(i)
                    pts_on_line.add(j)
                m = len(pts_on_line)
                if m >= 3:
                    tris += m * (m-1) * (m-2) // 6
        return tris

    def exact_max_independent(pts):
        """Find largest subset with no 3 collinear via branch and bound."""
        n = len(pts)
        # Build all collinear triples
        triples = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if cross(pts[i], pts[j], pts[k]) == 0:
                        triples.append((i, j, k))
        
        if not triples:
            return n
        
        # Build point -> triples incidence
        pt_tris = defaultdict(set)
        for idx, (a,b,c) in enumerate(triples):
            pt_tris[a].add(idx)
            pt_tris[b].add(idx)
            pt_tris[c].add(idx)
        
        best = [0]
        triples_set = set(range(len(triples)))
        
        def solve(remaining, chosen, forbidden):
            if not remaining:
                best[0] = max(best[0], len(chosen))
                return
            # Lower bound on remaining we must exclude
            # Simple: if len(remaining) > len(chosen) + best[0] - len(chosen):
            #   prune if chosen + (n - len(chosen) - len(forbidden)) <= best[0]
            potential = len(chosen) + (n - len(chosen) - len(forbidden))
            if potential <= best[0]:
                return
            
            # Pick a triple
            t_idx = next(iter(remaining))
            a, b, c = triples[t_idx]
            
            # Option 1: include a (then b,c forbidden)
            new_forbidden = forbidden | {b, c}
            new_remaining = remaining - pt_tris[a]
            if a not in forbidden:
                solve(new_remaining, chosen | {a}, new_forbidden)
            
            # Option 2: include b
            new_forbidden = forbidden | {a, c}
            new_remaining = remaining - pt_tris[b]
            if b not in forbidden:
                solve(new_remaining, chosen | {b}, new_forbidden)
            
            # Option 3: include c
            new_forbidden = forbidden | {a, b}
            new_remaining = remaining - pt_tris[c]
            if c not in forbidden:
                solve(new_remaining, chosen | {c}, new_forbidden)
        
        solve(triples_set, set(), set())
        return best[0]

    def fast_score(pts):
        """Return (valid, independent_set_size_estimate)."""
        if not check_no4(pts):
            return False, 0
        # Use line map for quick triple count
        lines = build_line_map(pts)
        tris = triples_from_lines(lines, len(pts))
        # For exact score, we need MVC, but for search we use triple count as proxy
        # Higher triples -> smaller independent set
        return True, tris

    def simulated_annealing(pts, t_end):
        """SA to maximize collinear triples while maintaining no-4-collinear."""
        curr = [p[:] for p in pts]
        valid, curr_score = fast_score(curr)
        if not valid:
            return curr, 0
        
        best_pts = [p[:] for p in curr]
        best_score = curr_score
        
        temp = 50.0
        cooling = 0.98
        
        while time.time() < t_end:
            i = random.randint(0, 19)
            old = curr[i][:]
            # Try a few random moves
            for _ in range(5):
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                nx = old[0] + dx
                ny = old[1] + dy
                if abs(nx) > 100 or abs(ny) > 100:
                    continue
                # Check collision
                collide = False
                for j, p in enumerate(curr):
                    if j != i and p[0] == nx and p[1] == ny:
                        collide = True
                        break
                if collide:
                    continue
                
                curr[i] = [nx, ny]
                valid, new_score = fast_score(curr)
                
                if not valid:
                    curr[i] = old
                    continue
                
                delta = new_score - curr_score
                if delta > 0 or random.random() < math.exp(delta / temp):
                    curr_score = new_score
                    if curr_score > best_score:
                        best_score = curr_score
                        best_pts = [p[:] for p in curr]
                    break
                else:
                    curr[i] = old
            
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
        
        return best_pts, best_score

    def two_opt_move(pts, t_end):
        """Try moving pairs of points to find better configurations."""
        curr = [p[:] for p in pts]
        valid, curr_score = fast_score(curr)
        if not valid:
            return curr, 0
        
        best_pts = [p[:] for p in curr]
        best_score = curr_score
        
        while time.time() < t_end:
            i, j = random.sample(range(20), 2)
            old_i = curr[i][:]
            old_j = curr[j][:]
            
            for _ in range(3):
                dx1 = random.randint(-4, 4)
                dy1 = random.randint(-4, 4)
                dx2 = random.randint(-4, 4)
                dy2 = random.randint(-4, 4)
                if dx1 == 0 and dy1 == 0 and dx2 == 0 and dy2 == 0:
                    continue
                
                nx1 = old_i[0] + dx1
                ny1 = old_i[1] + dy1
                nx2 = old_j[0] + dx2
                ny2 = old_j[1] + dy2
                
                if abs(nx1) > 100 or abs(ny1) > 100 or abs(nx2) > 100 or abs(ny2) > 100:
                    continue
                
                # Check collisions
                collide = False
                for k, p in enumerate(curr):
                    if k != i and k != j:
                        if (p[0] == nx1 and p[1] == ny1) or (p[0] == nx2 and p[1] == ny2):
                            collide = True
                            break
                    elif k == i:
                        if nx2 == old_i[0] and ny2 == old_i[1] and (nx1 != nx2 or ny1 != ny2):
                            pass
                    elif k == j:
                        if nx1 == old_j[0] and ny1 == old_j[1] and (nx2 != nx1 or ny2 != ny1):
                            pass
                if (nx1 == nx2 and ny1 == ny2):
                    collide = True
                if collide:
                    continue
                
                curr[i] = [nx1, ny1]
                curr[j] = [nx2, ny2]
                valid, new_score = fast_score(curr)
                
                if not valid:
                    curr[i] = old_i
                    curr[j] = old_j
                    continue
                
                if new_score > curr_score:
                    curr_score = new_score
                    if curr_score > best_score:
                        best_score = curr_score
                        best_pts = [p[:] for p in curr]
                    break
                else:
                    curr[i] = old_i
                    curr[j] = old_j
        
        return best_pts, best_score

    def algebraic_seed():
        """Construct a good initial seed using algebraic geometry ideas.
        Use points on a parabola and add points on lines through pairs."""
        pts = []
        used = set()
        
        # Start with parabola y = x^2 for x in -4..4 (9 points)
        for x in range(-4, 5):
            y = x * x
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        
        # Add points on y = x^2 + 50 for some x
        for x in range(-3, 4):
            y = x * x + 50
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        
        # Add points to reach 20, choosing positions that create collinearities
        while len(pts) < 20:
            # Try to add point that creates a triple with two existing points
            best_add = None
            best_gain = -1
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    # Find points on line through pts[i], pts[j]
                    dx = pts[j][0] - pts[i][0]
                    dy = pts[j][1] - pts[i][1]
                    g = math.gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    sx = dx // g
                    sy = dy // g
                    # Parametric: (x0 + t*sx, y0 + t*sy)
                    for t in range(-20, 21):
                        if t == 0:
                            continue
                        nx = pts[i][0] + t * sx
                        ny = pts[i][1] + t * sy
                        if abs(nx) > 100 or abs(ny) > 100:
                            continue
                        if (nx, ny) in used:
                            continue
                        # Count how many existing points are on this line
                        cnt = 0
                        for k, p in enumerate(pts):
                            if k != i and k != j and cross(pts[i], pts[j], p) == 0:
                                cnt += 1
                        gain = cnt  # Adding this point creates cnt new triples with existing collinear points
                        if gain > best_gain:
                            best_gain = gain
                            best_add = [nx, ny]
            if best_add is not None and best_gain >= 1:
                pts.append(best_add)
                used.add((best_add[0], best_add[1]))
            else:
                # Random fill
                while True:
                    x = random.randint(-100, 100)
                    y = random.randint(-100, 100)
                    if (x, y) not in used:
                        pts.append([x, y])
                        used.add((x, y))
                        break
        
        return pts[:20]

    def random_seed(seed_val):
        random.seed(seed_val)
        pts = []
        used = set()
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        return pts

    # Generate multiple diverse seeds
    seeds = []
    
    # Algebraic seed
    alg = algebraic_seed()
    if check_no4(alg):
        seeds.append(alg)
    
    # More algebraic variants with different offsets
    for offset in [30, 60, 80]:
        pts = []
        used = set()
        for x in range(-4, 5):
            y = x * x
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        for x in range(-3, 4):
            y = x * x + offset
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        while len(pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in used:
                pts.append([x, y])
                used.add((x, y))
        if check_no4(pts[:20]):
            seeds.append(pts[:20])
    
    # Random seeds
    for s in range(10):
        r = random_seed(42 + s * 7)
        if check_no4(r):
            seeds.append(r)
    
    if not seeds:
        # Fallback
        seeds = [random_seed(0)]
    
    # Optimize each seed with SA + 2-opt
    best_pts = seeds[0]
    best_score = -1
    
    for seed_pts in seeds:
        if time.time() > deadline:
            break
        
        t_sa = min(time.time() + 8, deadline - 5)
        sa_pts, sa_score = simulated_annealing(seed_pts, t_sa)
        
        if time.time() > deadline - 5:
            break
        
        t_2opt = min(time.time() + 10, deadline - 2)
        opt_pts, opt_score = two_opt_move(sa_pts, t_2opt)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
    
    # Final verification
    if not check_no4(best_pts):
        # Fallback to a known valid configuration
        best_pts = [[x, x*x] for x in range(-4, 5)]
        best_pts += [[x, x*x + 50] for x in range(-3, 4)]
        while len(best_pts) < 20:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if all(p != [x,y] for p in best_pts):
                best_pts.append([x, y])
        if not check_no4(best_pts):
            # Last resort: very spread out
            best_pts = [[x, y] for x in range(-4, 6) for y in [x*10, x*10+1]]
            best_pts = best_pts[:20]
    
    return best_pts
