import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

    def has_4_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                count = 2
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            count += 1
                            if count >= 4:
                                return True
        return False

    def count_3_collinear(points):
        n = len(points)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        count += 1
        return count

    def max_no_3_collinear(points):
        """Exact branch and bound for largest subset with no three collinear."""
        n = len(points)
        
        # Precompute collinearity masks
        # coll_pair_mask[i][j] = bitmask of k such that i, j, k are collinear (k != i, j)
        coll_pair_mask = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                mask = 0
                for k in range(n):
                    if k != i and k != j:
                        if are_collinear(points[i], points[j], points[k]):
                            mask |= (1 << k)
                coll_pair_mask[i][j] = mask
                coll_pair_mask[j][i] = mask
        
        best = [0]
        
        # Precompute a heuristic ordering: points with more collinear partners first
        deg = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if coll_pair_mask[i][j]:
                    deg[i] += 1
                    deg[j] += 1
        
        # We'll use a bitmask-based branch and bound
        # State: A (selected set as bitmask), C (candidate set as bitmask)
        
        def backtrack(A, C, depth):
            # Pruning: even if we take all remaining, we can't beat best
            remaining = bin(C).count('1')
            if depth + remaining <= best[0]:
                return
            
            if C == 0:
                if depth > best[0]:
                    best[0] = depth
                return
            
            # Heuristic: pick the point in C with highest degree (most conflicts)
            max_deg = -1
            p = -1
            temp_C = C
            while temp_C:
                bit = temp_C & -temp_C
                idx = bit.bit_length() - 1
                if deg[idx] > max_deg:
                    max_deg = deg[idx]
                    p = idx
                temp_C ^= bit
            
            if p == -1:
                return
            
            p_bit = 1 << p
            
            # Option 1: Exclude p
            backtrack(A, C ^ p_bit, depth)
            
            # Option 2: Include p (if compatible)
            # Check compatibility: for all a in A, (coll_pair_mask[p][a] & A) == 0
            compatible = True
            temp_A = A
            while temp_A:
                a_bit = temp_A & -temp_A
                a = a_bit.bit_length() - 1
                if coll_pair_mask[p][a] & A:
                    compatible = False
                    break
                temp_A ^= a_bit
            
            if compatible:
                new_C = C ^ p_bit
                # Remove from C any q that would form a triple with p and some a in A
                temp_A = A
                while temp_A:
                    a_bit = temp_A & -temp_A
                    a = a_bit.bit_length() - 1
                    new_C &= ~coll_pair_mask[p][a]
                    temp_A ^= a_bit
                
                backtrack(A | p_bit, new_C, depth + 1)
        
        backtrack(0, (1 << n) - 1, 0)
        return best[0]

    best_score = 20
    best_config = None
    best_triples = 0

    def try_config(pts):
        nonlocal best_score, best_config, best_triples
        if len(set(map(tuple, pts))) < 20:
            return False
        if has_4_collinear(pts):
            return False
        triples = count_3_collinear(pts)
        score = max_no_3_collinear(pts)
        if score < best_score or (score == best_score and triples > best_triples):
            best_score = score
            best_config = [p[:] for p in pts]
            best_triples = triples
            if best_score <= 10:
                return True
        return False

    # Phase 1: Generate many random candidates
    for attempt in range(2000):
        if time.time() > deadline - 30:
            break
        
        # Random dense configurations
        if attempt % 3 == 0:
            # 5x4 grid with small perturbations
            pts = []
            for i in range(5):
                for j in range(4):
                    di = random.choice([-1, 0, 0, 1])
                    dj = random.choice([-1, 0, 0, 1])
                    pts.append([i + di, j + dj])
        elif attempt % 3 == 1:
            # 4x5 grid
            pts = []
            for i in range(4):
                for j in range(5):
                    di = random.choice([-1, 0, 0, 1])
                    dj = random.choice([-1, 0, 0, 1])
                    pts.append([i + di, j + dj])
        else:
            # Random in small range
            pts = []
            for _ in range(20):
                x = random.randint(-6, 6)
                y = random.randint(-6, 6)
                pts.append([x, y])
        
        if try_config(pts):
            break

    # Phase 2: Local search / hill climbing
    if best_config is not None and time.time() < deadline - 10:
        current = [p[:] for p in best_config]
        current_score = best_score
        current_triples = best_triples
        no_improve_count = 0
        
        while time.time() < deadline - 5 and no_improve_count < 100:
            improved = False
            
            # Try moving each point to nearby positions
            for i in range(20):
                if time.time() > deadline - 5:
                    break
                
                orig = current[i]
                # Try all positions in a small neighborhood
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = orig[0] + dx, orig[1] + dy
                        if not (-100 <= nx <= 100 and -100 <= ny <= 100):
                            continue
                        
                        new_pts = [p[:] for p in current]
                        new_pts[i] = [nx, ny]
                        
                        # Quick validity check
                        if len(set(map(tuple, new_pts))) < 20:
                            continue
                        if has_4_collinear(new_pts):
                            continue
                        
                        new_triples = count_3_collinear(new_pts)
                        # Only compute exact score if triples improved or score might improve
                        if new_triples > current_triples or new_triples >= current_triples:
                            new_score = max_no_3_collinear(new_pts)
                            if new_score < current_score or (new_score == current_score and new_triples > current_triples):
                                current = new_pts
                                current_score = new_score
                                current_triples = new_triples
                                best_score = current_score
                                best_config = [p[:] for p in current]
                                best_triples = current_triples
                                improved = True
                                no_improve_count = 0
                                if current_score <= 10:
                                    return current
                                break
                        else:
                            # Even if triples didn't improve, score might have changed
                            new_score = max_no_3_collinear(new_pts)
                            if new_score < current_score:
                                current = new_pts
                                current_score = new_score
                                current_triples = new_triples
                                best_score = current_score
                                best_config = [p[:] for p in current]
                                best_triples = current_triples
                                improved = True
                                no_improve_count = 0
                                if current_score <= 10:
                                    return current
                                break
                    if improved:
                        break
                
                if improved:
                    continue
                no_improve_count += 1
            
            if not improved:
                no_improve_count += 1
                # Random restart from current best
                if no_improve_count > 20 and time.time() < deadline - 15:
                    # Perturb current best
                    perturbed = [p[:] for p in current]
                    for _ in range(random.randint(1, 3)):
                        idx = random.randint(0, 19)
                        perturbed[idx] = [random.randint(-8, 8), random.randint(-8, 8)]
                    if try_config(perturbed):
                        current = [p[:] for p in best_config]
                        current_score = best_score
                        current_triples = best_triples
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

    if best_config is None:
        # Fallback
        return [[i, j] for i in range(5) for j in range(4)]
    
    return best_config
