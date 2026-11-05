"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random

DEADLINE = time.time() + 280  # 280s safety margin before 300s kill

def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle with any two existing points."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        d_pa = _dist_sq(p, a)
        for j in range(i + 1, n):
            b = chosen[j]
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False

def _is_maximal(chosen):
    """Check if the set is maximal: no grid point can be added without creating an isosceles triangle."""
    chosen_set = set(chosen)
    for x in range(N):
        for y in range(N):
            p = (x, y)
            if p in chosen_set:
                continue
            if not _creates_isosceles(p, chosen):
                return False
    return True

def _greedy_construct(order):
    """Build a maximal no-isosceles set by greedily adding points in the given order."""
    chosen = []
    for p in order:
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
    return chosen

def _local_search(chosen, max_iters=5000):
    """Try to improve a maximal set by removing points and re-adding others."""
    chosen = list(chosen)
    best_size = len(chosen)
    
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Try removing each point and see if we can add more
        improved = False
        for i in range(len(chosen)):
            if time.time() > DEADLINE:
                break
            
            # Remove point i
            removed = chosen.pop(i)
            removed_set = set(chosen)
            
            # Try to add any point that was previously blocked
            added_count = 0
            for x in range(N):
                if time.time() > DEADLINE:
                    break
                for y in range(N):
                    if time.time() > DEADLINE:
                        break
                    p = (x, y)
                    if p in removed_set or p == removed:
                        continue
                    if not _creates_isosceles(p, chosen):
                        chosen.append(p)
                        removed_set.add(p)
                        added_count += 1
            
            if len(chosen) > best_size:
                best_size = len(chosen)
                improved = True
                # Keep this improved set
                break
            else:
                # Restore the removed point
                chosen.append(removed)
                removed_set.add(removed)
        
        if not improved:
            break
    
    return chosen

def construct():
    """Construct a minimal maximal no-isosceles set."""
    random.seed(42)
    
    # Start with a greedy construction in row-major order
    base_order = [(x, y) for y in range(N) for x in range(N)]
    chosen = _greedy_construct(base_order)
    best = chosen
    best_size = len(chosen)
    
    # Try multiple random orderings to find smaller maximal sets
    attempts = 0
    while time.time() < DEADLINE:
        attempts += 1
        
        # Create a randomized ordering
        order = list(base_order)
        random.shuffle(order)
        
        # Greedy construction with this order
        candidate = _greedy_construct(order)
        
        # Local search to improve
        candidate = _local_search(candidate, max_iters=2000)
        
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
        
        # Early termination if we've found a very small set
        if best_size <= 10:
            break
    
    # Verify maximality of the best result
    if not _is_maximal(best):
        # If not maximal, force maximality by adding points
        chosen_set = set(best)
        for x in range(N):
            for y in range(N):
                p = (x, y)
                if p not in chosen_set:
                    if not _creates_isosceles(p, best):
                        best.append(p)
                        chosen_set.add(p)
    
    return best
# EVOLVE-BLOCK-END
