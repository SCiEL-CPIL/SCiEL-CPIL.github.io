"""No-isosceles set construction via simulated annealing with conflict counting."""
import random
import math
import time

N = 64
DEADLINE = 320.0  # safety margin before 330s kill

def dist2(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def count_conflicts(points):
    """Count number of isosceles triangles in the set."""
    n = len(points)
    if n < 3:
        return 0
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = dist2(points[i], points[j])
            # For each pair (i,j), check if any k forms isosceles with i,j
            # d_ik == d_ij or d_jk == d_ij or d_ik == d_jk
            for k in range(n):
                if k == i or k == j:
                    continue
                d_ik = dist2(points[i], points[k])
                d_jk = dist2(points[j], points[k])
                if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                    conflicts += 1
    return conflicts

def incremental_conflict_change(points, add_point, remove_point=None):
    """
    Compute change in conflict count when adding add_point and optionally removing remove_point.
    Returns (delta_add, delta_remove) or combined delta.
    For simplicity, we compute the full conflict change for add and remove separately.
    """
    n = len(points)
    
    def conflicts_involving(p, pts):
        """Count isosceles triangles involving p and pairs from pts."""
        c = 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d_ij = dist2(pts[i], pts[j])
                d_pi = dist2(p, pts[i])
                d_pj = dist2(p, pts[j])
                if d_pi == d_ij or d_pj == d_ij or d_pi == d_pj:
                    c += 1
        return c

    # Conflicts added by add_point with existing points
    # We need to count triangles (add, a, b) where a,b in points
    delta_add = conflicts_involving(add_point, points)
    
    # Conflicts removed by remove_point with remaining points (excluding add_point interaction)
    delta_remove = 0
    if remove_point is not None:
        # Remove remove_point from consideration
        remaining = [p for p in points if p != remove_point]
        # But also need to account for triangles involving both add and remove
        # This gets complex; for SA, let's just do full recompute for small sets
        delta_remove = conflicts_involving(remove_point, remaining)
    
    return delta_add, delta_remove

def full_conflicts(points):
    """Full O(n^3) conflict count - use for verification."""
    n = len(points)
    if n < 3:
        return 0
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = dist2(points[i], points[j])
            for k in range(j + 1, n):
                d_ik = dist2(points[i], points[k])
                d_jk = dist2(points[j], points[k])
                if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                    conflicts += 1
    return conflicts

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Start with a small random seed
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    # Initial solution: pick a few random points
    points = list(all_points[:5])
    
    best_points = list(points)
    best_conflicts = full_conflicts(best_points)
    best_size = len(best_points)
    
    # Simulated annealing
    T = 50.0
    T_min = 0.01
    cooling = 0.9995
    max_iter = 500000
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE:
            break
        
        # Choose move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4 and len(points) < N * N:
            # Add a random point not already in set
            candidates = [p for p in all_points if p not in points]
            if not candidates:
                break
            add_p = random.choice(candidates)
            remove_p = None
            new_points = points + [add_p]
        elif move_type < 0.7 and len(points) > 1:
            # Remove a random point
            idx = random.randint(0, len(points) - 1)
            remove_p = points[idx]
            add_p = None
            new_points = points[:idx] + points[idx+1:]
        else:
            # Swap: remove one, add another
            if len(points) < 1:
                continue
            idx = random.randint(0, len(points) - 1)
            remove_p = points[idx]
            candidates = [p for p in all_points if p not in points]
            if not candidates:
                continue
            add_p = random.choice(candidates)
            new_points = points[:idx] + [add_p] + points[idx+1:]
        
        # Compute new conflict count
        new_conflicts = full_conflicts(new_points)
        new_size = len(new_points)
        
        # Objective: maximize size, minimize conflicts
        # Use a combined score: size * 100 - conflicts
        old_score = len(points) * 100 - full_conflicts(points)
        new_score = new_size * 100 - new_conflicts
        delta = new_score - old_score
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            points = new_points
            
            # Track best
            if new_conflicts == 0 and new_size > best_size:
                best_size = new_size
                best_points = list(points)
                best_conflicts = 0
            elif new_size > best_size and new_conflicts < best_conflicts:
                best_size = new_size
                best_points = list(points)
                best_conflicts = new_conflicts
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Restart temperature occasionally
        if iteration % 10000 == 0 and T < 1.0:
            T = 20.0
    
    # Final: return best valid solution (zero conflicts)
    if best_conflicts == 0:
        return best_points
    else:
        # Fallback: greedily fix the best candidate
        # Try to remove points to eliminate conflicts
        result = list(best_points)
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for i in range(len(result)):
                for j in range(i + 1, len(result)):
                    d_ij = dist2(result[i], result[j])
                    for k in range(len(result)):
                        if k == i or k == j:
                            continue
                        d_ik = dist2(result[i], result[k])
                        d_jk = dist2(result[j], result[k])
                        if d_ik == d_ij or d_jk == d_ij or d_ik == d_jk:
                            # Conflict found; remove point k (or i or j)
                            # Remove the one that causes most conflicts
                            # Simple: remove k
                            result.pop(k)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        
        # Verify
        if full_conflicts(result) == 0:
            return result
        else:
            # Last resort: return empty or minimal valid set
            return []

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} points")
    print(f"Conflicts: {full_conflicts(result)}")
    print(result[:10])
