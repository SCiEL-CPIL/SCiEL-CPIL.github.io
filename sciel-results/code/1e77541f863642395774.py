"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations

_start_time = time.time()
_DEADLINE = 110  # seconds, safety margin for 120s kill

def _gcd(a, b):
    """Compute gcd of a and b."""
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(i, j, pts):
    """Get canonical line key for pair (i,j) in points pts."""
    x1, y1 = pts[i]
    x2, y2 = pts[j]
    dx = x2 - x1
    dy = y2 - y1
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    g = _gcd(dx, dy)
    if g > 1:
        dx //= g
        dy //= g
    # Line: (x - x1)*dy == (y - y1)*dx => dy*x - dx*y = dy*x1 - dx*y1
    # Normalize the constant term
    c = dy * x1 - dx * y1
    # Make (dx, dy, c) canonical: ensure (dx, dy) is in canonical form
    # Then c should be consistent
    return (dx, dy, c)

def _count_four_lines_full(points):
    """Count lines containing exactly 4 of the given points. Full recompute."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(i, j, points)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    count = 0
    for key, idxs in lines.items():
        if len(idxs) == 4:
            count += 1
    return count

def _build_line_index(points):
    """Build a line index: dict from line_key to set of point indices."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            key = _line_key(i, j, points)
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
    return lines

def _count_four_lines_from_index(lines):
    """Count lines with exactly 4 points from a pre-built index."""
    count = 0
    for key, idxs in lines.items():
        if len(idxs) == 4:
            count += 1
    return count

def _is_valid_full(points):
    """Check validity: distinct, bounds, no five collinear."""
    if len(points) != 16:
        return False
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    lines = _build_line_index(points)
    for key, idxs in lines.items():
        if len(idxs) >= 5:
            return False
    return True

def _evaluate_move(points, idx, new_pos, lines):
    """
    Evaluate moving point idx to new_pos.
    Returns (new_points, new_lines, new_score, is_valid) or None if invalid.
    Uses incremental update: only recompute lines involving point idx.
    """
    x, y = new_pos
    if abs(x) > 1000 or abs(y) > 1000:
        return None
    # Check distinct
    for j in range(16):
        if j != idx and points[j] == [x, y]:
            return None
    
    new_pts = [list(p) for p in points]
    new_pts[idx] = [x, y]
    
    # Collect all lines that involve point idx (old position)
    old_pos = points[idx]
    affected_old = set()
    for key, idxs in lines.items():
        if idx in idxs:
            affected_old.add(key)
    
    # Build new line index by removing old lines involving idx and adding new ones
    new_lines = {}
    for key, idxs in lines.items():
        if idx in idxs:
            remaining = idxs - {idx}
            if len(remaining) >= 2:
                new_lines[key] = remaining
        else:
            new_lines[key] = idxs
    
    # Add new lines involving idx at new position
    for j in range(16):
        if j == idx:
            continue
        key = _line_key(idx, j, new_pts)
        if key not in new_lines:
            new_lines[key] = set()
        new_lines[key].add(idx)
        new_lines[key].add(j)
    
    # Check no five collinear
    for key, idxs in new_lines.items():
        if len(idxs) >= 5:
            return None
    
    # Count four-point lines
    score = 0
    for key, idxs in new_lines.items():
        if len(idxs) == 4:
            score += 1
    
    return (new_pts, new_lines, score, True)

def _initial_grid():
    """4x4 grid: gives 20 four-point lines (affine plane AG(2,4) structure)."""
    return [[x, y] for x in range(4) for y in range(4)]

def _initial_affine_like():
    """Try to create an affine-plane-like configuration."""
    # The 4x4 grid already has this structure
    # Points: (0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),...
    # Lines: 4 horizontal, 4 vertical, 12 diagonal (slopes 1, -1, 1/2, -1/2, 2, -2, etc.)
    # Total: 20 lines with exactly 4 points
    return [[x, y] for x in range(4) for y in range(4)]

def _local_search_incremental(points, lines, score, max_iter=20000, deadline=None):
    """Local search with incremental line evaluation."""
    current = [list(p) for p in points]
    current_lines = {k: set(v) for k, v in lines.items()}
    current_score = score
    best = [list(p) for p in current]
    best_lines = {k: set(v) for k, v in current_lines.items()}
    best_score = current_score
    no_improve = 0
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        idx = random.randint(0, 15)
        improved = False
        
        # Try moves: small random, then targeted
        moves_to_try = []
        # Small random moves
        for _ in range(8):
            dx = random.randint(-4, 4)
            dy = random.randint(-4, 4)
            if dx == 0 and dy == 0:
                continue
            moves_to_try.append((dx, dy))
        
        # Targeted: try to align with existing lines
        for key, idxs in current_lines.items():
            if len(idxs) == 3 and idx not in idxs:
                # This line has 3 points, maybe we can move idx to join it
                # Get the line equation
                dx, dy, c = key
                # Find a point on this line that's close to current idx
                # Line: dy*x - dx*y = c
                # Try nearby integer points
                cx, cy = current[idx]
                # Project onto line and find nearby integer points
                for t in range(-3, 4):
                    # Parametric: start from one of the 3 points, go along direction
                    pass  # Simplified: just try small offsets
        
        for dx, dy in moves_to_try:
            new_pos = [current[idx][0] + dx, current[idx][1] + dy]
            result = _evaluate_move(current, idx, new_pos, current_lines)
            if result is None:
                continue
            new_pts, new_lines, new_score, _ = result
            if new_score > current_score:
                current = new_pts
                current_lines = new_lines
                current_score = new_score
                if current_score > best_score:
                    best = [list(p) for p in current]
                    best_lines = {k: set(v) for k, v in current_lines.items()}
                    best_score = current_score
                no_improve = 0
                improved = True
                break
        
        if not improved:
            no_improve += 1
            if no_improve > 100:
                break
    
    return best, best_lines, best_score

def construct():
    global _start_time
    random.seed(42)
    
    deadline = _start_time + _DEADLINE
    
    # Start with 4x4 grid (affine plane structure, 20 four-point lines)
    best_pts = _initial_grid()
    best_lines = _build_line_index(best_pts)
    best_score = _count_four_lines_from_index(best_lines)
    
    # Phase 1: Local search from grid with incremental evaluation
    if time.time() < deadline - 20:
        best_pts, best_lines, best_score = _local_search_incremental(
            best_pts, best_lines, best_score, max_iter=50000, deadline=deadline
        )
    
    # Phase 2: Try different initial configurations with perturbations
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Create a perturbed grid
        pts = [[x, y] for x in range(4) for y in range(4)]
        # Randomly perturb 1-3 points
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, 15)
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            new_x = pts[idx][0] + dx
            new_y = pts[idx][1] + dy
            if abs(new_x) <= 1000 and abs(new_y) <= 1000:
                pts[idx] = [new_x, new_y]
        
        # Check validity
        if not _is_valid_full(pts):
            continue
        
        lines = _build_line_index(pts)
        score = _count_four_lines_from_index(lines)
        
        if score > best_score:
            # Local search from this configuration
            if time.time() < deadline - 10:
                pts, lines, score = _local_search_incremental(
                    pts, lines, score, max_iter=20000, deadline=deadline
                )
            if score > best_score:
                best_pts = pts
                best_lines = lines
                best_score = score
    
    # Phase 3: Final polish with larger steps
    if time.time() < deadline - 5:
        current = [list(p) for p in best_pts]
        current_lines = {k: set(v) for k, v in best_lines.items()}
        current_score = best_score
        
        for _ in range(5000):
            if time.time() > deadline - 3:
                break
            idx = random.randint(0, 15)
            improved = False
            # Try a range of moves
            for dx in range(-10, 11):
                if improved:
                    break
                for dy in range(-10, 11):
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = [current[idx][0] + dx, current[idx][1] + dy]
                    result = _evaluate_move(current, idx, new_pos, current_lines)
                    if result is None:
                        continue
                    new_pts, new_lines, new_score, _ = result
                    if new_score > current_score:
                        current = new_pts
                        current_lines = new_lines
                        current_score = new_score
                        if current_score > best_score:
                            best_pts = [list(p) for p in current]
                            best_lines = {k: set(v) for k, v in current_lines.items()}
                            best_score = current_score
                        improved = True
                        break
            
            if not improved and random.random() < 0.1:
                # Occasionally try a bigger jump
                dx = random.randint(-20, 20)
                dy = random.randint(-20, 20)
                new_pos = [current[idx][0] + dx, current[idx][1] + dy]
                result = _evaluate_move(current, idx, new_pos, current_lines)
                if result is not None:
                    new_pts, new_lines, new_score, _ = result
                    if new_score > current_score:
                        current = new_pts
                        current_lines = new_lines
                        current_score = new_score
                        if current_score > best_score:
                            best_pts = [list(p) for p in current]
                            best_lines = {k: set(v) for k, v in current_lines.items()}
                            best_score = current_score
    
    return [[int(x), int(y)] for x, y in best_pts]

# EVOLVE-BLOCK-END
