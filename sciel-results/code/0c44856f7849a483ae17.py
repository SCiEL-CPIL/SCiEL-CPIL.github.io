"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools


def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            a, b = chosen[i], chosen[j]
            d_pa = _dist_sq(p, a)
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False


def _blocked_by(p, chosen):
    """Return set of points that would be blocked if p is added to chosen."""
    # A point q is blocked if (p, a, b) forms isosceles for some a,b in chosen,
    # OR if adding q to (chosen + [p]) creates isosceles.
    # Actually, we want: which candidate points q (not in chosen, not p) 
    # become blocked when we add p to chosen.
    # q is blocked if _creates_isosceles(q, chosen + [p]) is True.
    # But that's expensive. Let's compute it differently.
    # For each pair (a, b) in chosen + [p], q is blocked if dist(q,a)==dist(q,b) or dist(q,a)==dist(a,b) or dist(q,b)==dist(a,b).
    # This is still expensive. Let's just check a sample or use a simpler heuristic.
    pass


def _greedy_randomized(all_points, rng):
    """Greedy construction with randomization and 'most blocking' heuristic."""
    chosen = []
    remaining = list(all_points)
    rng.shuffle(remaining)
    
    while remaining:
        # Find all candidates that can be added
        valid = []
        for p in remaining:
            if not _creates_isosceles(p, chosen):
                valid.append(p)
        
        if not valid:
            break
        
        # Heuristic: pick the point that blocks the most other valid points
        # For each valid point, count how many other valid points it would block
        best_p = None
        best_block_count = -1
        # Sample if too many
        if len(valid) > 20:
            sample = rng.sample(valid, 20)
        else:
            sample = valid
            
        for p in sample:
            # Count how many other valid points would be blocked
            block_count = 0
            for q in valid:
                if q == p:
                    continue
                # Check if q would still be valid after adding p
                if _creates_isosceles(q, chosen + [p]):
                    block_count += 1
            if block_count > best_block_count:
                best_block_count = block_count
                best_p = p
        
        if best_p is None:
            best_p = rng.choice(valid)
        
        chosen.append(best_p)
        remaining.remove(best_p)
    
    return chosen


def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    
    best_set = None
    best_size = float('inf')
    
    # Multiple restarts with different seeds
    for seed in range(50):
        rng = random.Random(seed)
        result = _greedy_randomized(all_points, rng)
        if len(result) < best_size:
            best_size = len(result)
            best_set = result
    
    return best_set
# EVOLVE-BLOCK-END
