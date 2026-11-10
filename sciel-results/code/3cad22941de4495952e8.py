# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import gcd

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    x = points[:, 0][:, np.newaxis]
    y = points[:, 1][:, np.newaxis]
    A = lines[:, 0][np.newaxis, :]
    B = lines[:, 1][np.newaxis, :]
    C = lines[:, 2][np.newaxis, :]
    vals = x * A + y * B + C
    return np.sum(vals == 0)

def _line_from_two_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = -(A * x1 + B * y1)
    if A == 0 and B == 0:
        return None
    return (A, B, C)

def _best_line_through_points(points, existing_lines, exclude_line_idx=None):
    """Find the line that maximizes incidences, not already in existing_lines."""
    best_score = -1
    best_line = None
    best_norm = None
    
    # Try all pairs of points
    n = len(points)
    seen_norms = set()
    if existing_lines is not None:
        for i, l in enumerate(existing_lines):
            if i != exclude_line_idx:
                seen_norms.add(_normalize_line(*l))
    
    for i in range(n):
        for j in range(i+1, n):
            l = _line_from_two_points(points[i], points[j])
            if l is None:
                continue
            norm = _normalize_line(*l)
            if norm in seen_norms:
                continue
            # Count incidences
            A, B, C = l
            count = np.sum(A * points[:, 0] + B * points[:, 1] + C == 0)
            if count > best_score:
                best_score = count
                best_line = l
                best_norm = norm
                seen_norms.add(norm)
    
    return best_line, best_norm, best_score

def _count_point_incidence(p, lines):
    x, y = p
    return np.sum(lines[:, 0] * x + lines[:, 1] * y + lines[:, 2] == 0)

def _count_line_incidence(l, points):
    A, B, C = l
    return np.sum(A * points[:, 0] + B * points[:, 1] + C == 0)

def _generate_initial_config():
    """Generate a concentrated initial configuration."""
    N = 54
    M = 27
    
    # Strategy: Place points on a few lines, then choose lines to maximize incidences
    # Try multiple random concentrated configurations
    
    best_points = None
    best_lines = None
    best_score = -1
    
    for attempt in range(200):
        # Randomly choose 3-5 base lines
        num_base_lines = random.randint(3, 5)
        base_lines = []
        
        for _ in range(num_base_lines):
            # Random line through origin with small integer coefficients
            A = random.randint(-3, 3)
            B = random.randint(-3, 3)
            if A == 0 and B == 0:
                A = 1
            C = random.randint(-5, 5)
            norm = _normalize_line(A, B, C)
            if norm not in [tuple(l) for l in base_lines]:
                base_lines.append(norm)
        
        if len(base_lines) < num_base_lines:
            continue
        
        # Place points on these lines
        points = []
        used_positions = set()
        
        # Distribute points among base lines
        points_per_line = [N // num_base_lines] * num_base_lines
        for i in range(N % num_base_lines):
            points_per_line[i] += 1
        
        for line_idx, (A, B, C) in enumerate(base_lines):
            count = points_per_line[line_idx]
            # Find integer points on this line
            candidates = []
            for x in range(-20, 20):
                for y in range(-20, 20):
                    if A * x + B * y + C == 0:
                        if (x, y) not in used_positions:
                            candidates.append((x, y))
            
            random.shuffle(candidates)
            for p in candidates[:count]:
                points.append(p)
                used_positions.add(p)
        
        if len(points) < N:
            # Fill remaining with random points
            while len(points) < N:
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                if (x, y) not in used_positions:
                    points.append((x, y))
                    used_positions.add((x, y))
        
        points = np.array(points, dtype=np.int64)
        
        # Now choose M lines to maximize incidences
        lines = []
        remaining = M
        
        # First, add the base lines
        for l in base_lines:
            if len(lines) < M:
                lines.append(l)
                remaining -= 1
        
        # Add best lines from point pairs
        while remaining > 0:
            best_l, best_norm, best_score_line = _best_line_through_points(
                points, lines, None
            )
            if best_l is None:
                # Add a random line
                A = random.randint(-5, 5)
                B = random.randint(-5, 5)
                if A == 0 and B == 0:
                    A = 1
                C = random.randint(-10, 10)
                norm = _normalize_line(A, B, C)
                if norm not in [tuple(l) for l in lines]:
                    lines.append(norm)
                    remaining -= 1
                else:
                    continue
            else:
                lines.append(best_l)
                remaining -= 1
        
        lines = np.array(lines, dtype=np.int64)
        
        # Check for duplicate lines
        norm_set = set()
        valid = True
        for l in lines:
            norm = _normalize_line(*l)
            if norm in norm_set:
                valid = False
                break
            norm_set.add(norm)
        
        if not valid:
            continue
        
        # Check for duplicate points
        point_set = set()
        for p in points:
            if (p[0], p[1]) in point_set:
                valid = False
                break
            point_set.add((p[0], p[1]))
        
        if not valid:
            continue
        
        score = _count_incidents(points, lines)
        if score > best_score:
            best_score = score
            best_points = points
            best_lines = lines
    
    return best_points, best_lines, best_score

def _optimize(points, lines, deadline):
    """Simulated annealing optimization."""
    N = len(points)
    M = len(lines)
    
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = _count_incidents(current_points, current_lines)
    
    best_points = current_points.copy()
    best_lines = current_lines.copy()
    best_score = current_score
    
    current_norm_lines = [_normalize_line(*l) for l in current_lines]
    
    T = 3.0
    T_min = 0.01
    cooling_rate = 0.99995
    
    # Precompute for speed
    point_set = set(map(tuple, current_points))
    
    while time.time() < deadline:
        if random.random() < 0.5:
            # Move a point
            i = random.randint(0, N - 1)
            old_p = tuple(current_points[i])
            
            best_delta = -1000
            best_new_p = None
            
            # Try multiple candidate positions
            for _ in range(40):
                dx = random.randint(-4, 4)
                dy = random.randint(-4, 4)
                if dx == 0 and dy == 0:
                    continue
                new_x = old_p[0] + dx
                new_y = old_p[1] + dy
                
                if (new_x, new_y) in point_set:
                    continue
                
                new_inc = _count_point_incidence((new_x, new_y), current_lines)
                old_inc = _count_point_incidence(old_p, current_lines)
                delta = new_inc - old_inc
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_p = (new_x, new_y)
            
            if best_new_p is None:
                continue
            
            if best_delta > 0 or random.random() < np.exp(best_delta / T):
                # Remove old point from set
                point_set.discard(old_p)
                point_set.add(best_new_p)
                current_points[i] = best_new_p
                current_score += best_delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
        else:
            # Move a line
            j = random.randint(0, M - 1)
            old_l = tuple(current_lines[j])
            
            best_delta = -1000
            best_new_l = None
            best_new_norm = None
            
            # Try lines through pairs of points
            for _ in range(40):
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a == b:
                    continue
                
                l = _line_from_two_points(current_points[a], current_points[b])
                if l is None:
                    continue
                
                norm = _normalize_line(*l)
                
                # Check duplicate
                is_dup = False
                for k in range(M):
                    if k != j and current_norm_lines[k] == norm:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                new_inc = _count_line_incidence(l, current_points)
                old_inc = _count_line_incidence(old_l, current_points)
                delta = new_inc - old_inc
                
                if delta > best_delta:
                    best_delta = delta
                    best_new_l = l
                    best_new_norm = norm
            
            if best_new_l is None:
                continue
            
            if best_delta > 0 or random.random() < np.exp(best_delta / T):
                current_lines[j] = best_new_l
                current_norm_lines[j] = best_new_norm
                current_score += best_delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = current_lines.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 595.0
    
    # Generate initial configuration
    points, lines, init_score = _generate_initial_config()
    
    # Optimize
    points, lines, final_score = _optimize(points, lines, deadline)
    
    return points, lines

# EVOLVE-BLOCK-END
