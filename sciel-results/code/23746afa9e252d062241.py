import random
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    N = 20
    # Initial random configuration
    # Try to start with a somewhat structured random set
    # Or just random
    points = []
    seen = set()
    while len(points) < N:
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        if (x, y) not in seen:
            seen.add((x, y))
            points.append([x, y])
            
    # Check if initial is valid (no 4 collinear)
    # If not, regenerate until valid (should be rare)
    while not _is_valid(points):
        points = []
        seen = set()
        while len(points) < N:
            x = random.randint(-100, 100)
            y = random.randint(-100, 100)
            if (x, y) not in seen:
                seen.add((x, y))
                points.append([x, y])

    # State
    best_points = [p[:] for p in points]
    best_triples = _count_triples(points)
    
    # Precompute line counts for current state
    line_counts = _get_line_counts(points)
    current_triples = _count_triples_from_counts(line_counts)
    
    # Local Search
    while time.time() < deadline:
        # Pick a point to move
        idx = random.randint(0, N - 1)
        old_p = points[idx]
        
        # Generate new point
        new_p = [random.randint(-100, 100), random.randint(-100, 100)]
        if new_p == old_p:
            continue
        # Ensure distinctness
        if new_p in points:
            continue

        # Remove old_p from state
        # Update line_counts
        for j in range(N):
            if j == idx: continue
            q = points[j]
            l = _get_line(old_p, q)
            line_counts[l] -= 1
            if line_counts[l] == 0:
                del line_counts[l]
        
        # Add new_p to state (tentatively)
        # Check validity and update counts
        valid = True
        added_lines = []
        for j in range(N):
            if j == idx: continue
            q = points[j]
            l = _get_line(new_p, q)
            if l not in line_counts:
                line_counts[l] = 0
            line_counts[l] += 1
            added_lines.append(l)
            if line_counts[l] >= 4:
                valid = False
                break
        
        if not valid:
            # Rollback
            for l in added_lines:
                line_counts[l] -= 1
                if line_counts[l] == 0:
                    del line_counts[l]
            # Restore old_p
            for j in range(N):
                if j == idx: continue
                q = points[j]
                l = _get_line(old_p, q)
                if l not in line_counts:
                    line_counts[l] = 0
                line_counts[l] += 1
            continue
            
        # Calculate new triple count
        # We need to know how many lines have count exactly 3
        # We can maintain a count of such lines
        # But for simplicity, just recount from line_counts? 
        # N=20, lines ~ 190. Recounting is fast.
        new_triples = _count_triples_from_counts(line_counts)
        
        if new_triples > best_triples:
            best_triples = new_triples
            best_points = [p[:] for p in points]
            points[idx] = new_p
            # Accept move
        elif new_triples == best_triples and random.random() < 0.1:
            # Accept equal to explore
            points[idx] = new_p
        else:
            # Rollback
            for l in added_lines:
                line_counts[l] -= 1
                if line_counts[l] == 0:
                    del line_counts[l]
            # Restore old_p
            for j in range(N):
                if j == idx: continue
                q = points[j]
                l = _get_line(old_p, q)
                if l not in line_counts:
                    line_counts[l] = 0
                line_counts[l] += 1
            
    return best_points

def _get_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None # Should not happen
    g = math.gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line representation: (dx, dy, c) where x*dx + y*dy = c ? No.
    # Standard form: A x + B y = C
    # A = dy, B = -dx
    # C = A*x1 + B*y1
    A = dy
    B = -dx
    C = A * x1 + B * y1
    # Normalize A, B, C
    g = math.gcd(abs(A), abs(B))
    if g != 0:
        A //= g
        B //= g
        C //= g
    # Ensure A > 0 or (A == 0 and B > 0)
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def _is_valid(points):
    lines = {}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            l = _get_line(points[i], points[j])
            lines[l] = lines.get(l, 0) + 1
            if lines[l] >= 4:
                return False
    return True

def _count_triples(points):
    lines = {}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            l = _get_line(points[i], points[j])
            lines[l] = lines.get(l, 0) + 1
    count = 0
    for l, c in lines.items():
        if c == 3:
            count += 1
    return count

def _get_line_counts(points):
    lines = {}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            l = _get_line(points[i], points[j])
            lines[l] = lines.get(l, 0) + 1
    return lines

def _count_triples_from_counts(line_counts):
    count = 0
    for l, c in line_counts.items():
        if c == 3:
            count += 1
    return count
# EVOLVE-BLOCK-END
