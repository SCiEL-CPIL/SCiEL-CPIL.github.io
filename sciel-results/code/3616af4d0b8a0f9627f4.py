import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for 150s limit

    def line_hash(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return None # Should not happen if points are distinct
            
        g = math.gcd(abs(dx), abs(dy))
        a = dy // g
        b = -dx // g
        
        # c = -(a*x1 + b*y1)
        c = -(a * x1 + b * y1)
        
        # Normalize (a, b, c)
        g2 = math.gcd(abs(a), abs(b), abs(c))
        if g2 > 0:
            a //= g2
            b //= g2
            c //= g2
            
        # Sign convention: a > 0, or a==0 and b > 0, or a==0 and b==0 and c > 0
        if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
            a = -a
            b = -b
            c = -c
            
        return (a, b, c)

    def evaluate(points):
        n = len(points)
        if n != 16:
            return -1, []
        
        # Check distinctness
        if len(set(map(tuple, points))) != 16:
            return -1, []
            
        line_counts = defaultdict(list)
        for i in range(n):
            for j in range(i + 1, n):
                key = line_hash(points[i], points[j])
                if key is None:
                    return -1, []
                line_counts[key].append(i)
                
        score = 0
        three_lines = []
        
        for key, indices in line_counts.items():
            # Deduplicate indices (should be unique in the loop, but just in case)
            unique_indices = list(set(indices))
            count = len(unique_indices)
            
            if count >= 5:
                return -1, []
            if count == 4:
                score += 1
            elif count == 3:
                three_lines.append(tuple(unique_indices))
                
        return score, three_lines

    def get_completion_move(points, three_lines, current_idx_to_move):
        """
        Try to find a position for points[current_idx_to_move] that completes a 3-line to a 4-line.
        Returns new position [x, y] or None.
        """
        if not three_lines:
            return None
            
        # Pick a random 3-line
        line_idx = random.randint(0, len(three_lines) - 1)
        p_indices = three_lines[line_idx]
        
        # The line is defined by these 3 points.
        # We want to move points[current_idx_to_move] to a 4th position on this line.
        # But points[current_idx_to_move] might be one of the points on the line?
        # If it is, we can't move it to complete the line (it's already there).
        # We need to move a point NOT on the line to a spot on the line.
        # So, if current_idx_to_move is in p_indices, this move is invalid for this line.
        if current_idx_to_move in p_indices:
            return None
            
        p1 = points[p_indices[0]]
        p2 = points[p_indices[1]]
        
        # Vector
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # We need to find an integer point Q = p1 + k * (dx, dy)
        # such that Q is not in points, and Q is within bounds.
        # The existing 3 points are p1, p2, and p3.
        # Let's find the k values for the existing points.
        # p1 is k=0.
        # p2 is k=1 (by definition of vector).
        # p3 = points[p_indices[2]].
        # p3 = p1 + k3 * (dx, dy) => k3 = (p3[0]-p1[0])/dx if dx!=0 else (p3[1]-p1[1])/dy
        
        occupied_k = set()
        occupied_k.add(0)
        occupied_k.add(1)
        
        p3 = points[p_indices[2]]
        if dx != 0:
            if (p3[0] - p1[0]) % dx != 0:
                return None # Should not happen
            k3 = (p3[0] - p1[0]) // dx
        else:
            if (p3[1] - p1[1]) % dy != 0:
                return None
            k3 = (p3[1] - p1[1]) // dy
        occupied_k.add(k3)
        
        # Try k values near the occupied ones
        # Sort occupied k's
        ks = sorted(occupied_k)
        # Try k = ks[0]-1, ks[0]-2, ..., ks[-1]+1, ks[-1]+2, ...
        candidates_k = []
        for k in ks:
            candidates_k.append(k - 1)
            candidates_k.append(k + 1)
            candidates_k.append(k - 2)
            candidates_k.append(k + 2)
        
        random.shuffle(candidates_k)
        
        for k in candidates_k:
            if k in occupied_k:
                continue
                
            new_x = p1[0] + k * dx
            new_y = p1[1] + k * dy
            
            # Bounds check
            if not (-1000 <= new_x <= 1000 and -1000 <= new_y <= 1000):
                continue
                
            new_pt = [new_x, new_y]
            
            # Check if new_pt is already in points
            # We are moving current_idx_to_move, so we should check against all OTHER points
            is_distinct = True
            for i, p in enumerate(points):
                if i == current_idx_to_move:
                    continue
                if p == new_pt:
                    is_distinct = False
                    break
            
            if is_distinct:
                return new_pt
                
        return None

    # Initial configuration: 4x4 grid
    best_points = [[i, j] for i in range(4) for j in range(4)]
    best_score, _ = evaluate(best_points)
    
    current_points = [p[:] for p in best_points]
    current_score, current_three_lines = evaluate(current_points)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate move
        move_type = random.random()
        
        new_points = [p[:] for p in current_points]
        idx = random.randint(0, 15)
        old_pt = current_points[idx][:]
        
        if move_type < 0.3 and current_three_lines:
            # Try line completion
            new_pt = get_completion_move(current_points, current_three_lines, idx)
            if new_pt is not None:
                new_points[idx] = new_pt
            else:
                # Fallback to random move if completion failed
                # Small random move
                step = random.randint(1, 3)
                new_x = old_pt[0] + random.randint(-step, step)
                new_y = old_pt[1] + random.randint(-step, step)
                if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                    new_points[idx] = [new_x, new_y]
                else:
                    continue # Skip move if out of bounds
        else:
            # Random move
            if random.random() < 0.1:
                # Large jump
                new_x = random.randint(-50, 50)
                new_y = random.randint(-50, 50)
            else:
                # Small move
                step = random.randint(1, 3)
                new_x = old_pt[0] + random.randint(-step, step)
                new_y = old_pt[1] + random.randint(-step, step)
            
            if -1000 <= new_x <= 1000 and -1000 <= new_y <= 1000:
                new_points[idx] = [new_x, new_y]
            else:
                continue # Skip move if out of bounds
        
        # Check distinctness
        if len(set(map(tuple, new_points))) != 16:
            continue
            
        new_score, new_three_lines = evaluate(new_points)
        
        # Acceptance
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            current_three_lines = new_three_lines
            
            if new_score > best_score:
                best_score = new_score
                best_points = [p[:] for p in current_points]
                
            if best_score >= 15:
                break # Found optimum (theoretical max for integer points is likely 15)
        else:
            delta = new_score - current_score
            if delta < 0 and random.random() < math.exp(delta / T):
                current_points = new_points
                current_score = new_score
                current_three_lines = new_three_lines
            # else: reject (current_points unchanged)
            
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Occasional restart to escape local minima
        if random.random() < 0.001:
            # Reset to best or random?
            # Let's try to reset to a perturbed best
            current_points = [p[:] for p in best_points]
            for i in range(16):
                if random.random() < 0.2:
                    current_points[i][0] += random.randint(-5, 5)
                    current_points[i][1] += random.randint(-5, 5)
            # Ensure valid
            if len(set(map(tuple, current_points))) != 16:
                current_points = [p[:] for p in best_points]
            current_score, current_three_lines = evaluate(current_points)
            if current_score < 0:
                current_points = [p[:] for p in best_points]
                current_score, current_three_lines = evaluate(best_points)

    return best_points

if __name__ == "__main__":
    # Simple test
    pts = construct()
    score, _ = evaluate(pts)
    print(f"Score: {score}")
    print(pts)
