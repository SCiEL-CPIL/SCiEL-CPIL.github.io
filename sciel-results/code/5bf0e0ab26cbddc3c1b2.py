"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 5000
DEADLINE = 320  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, all_primes):
    """Check if points form an admissible tuple."""
    n = len(points)
    if n == 0:
        return True
    for p in all_primes:
        if p > n:
            break
        residues = set()
        for x in points:
            residues.add(x % p)
        if len(residues) == p:
            return False
    return True


def _greedy_construct(order, all_primes, deadline):
    """Greedy construction: add elements in given order if they don't violate admissibility."""
    in_set = [False] * (N + 1)
    current_size = 0
    
    # Track residue counts for all primes
    residue_counts = {}
    missing_counts = {}
    
    for p in all_primes:
        residue_counts[p] = [0] * p
        missing_counts[p] = p
    
    points = []
    
    for x in order:
        if time.time() > deadline:
            break
        
        # Check which primes are active (p <= current_size + 1)
        target_size = current_size + 1
        valid = True
        
        for p in all_primes:
            if p > target_size:
                break
            if missing_counts[p] == 1:
                r = x % p
                if residue_counts[p][r] == 0:
                    valid = False
                    break
        
        if valid:
            in_set[x] = True
            points.append(x)
            current_size += 1
            
            # Update counts for all primes
            for p in all_primes:
                r = x % p
                if residue_counts[p][r] == 0:
                    missing_counts[p] -= 1
                residue_counts[p][r] += 1
    
    return points


def _repair(points, all_primes, deadline):
    """Repair phase: remove elements to fix admissibility violations."""
    points_set = set(points)
    
    # Build residue counts
    residue_counts = {}
    missing_counts = {}
    for p in all_primes:
        residue_counts[p] = [0] * p
        missing_counts[p] = p
    
    for x in points_set:
        for p in all_primes:
            r = x % p
            if residue_counts[p][r] == 0:
                missing_counts[p] -= 1
            residue_counts[p][r] += 1
    
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
        
        n = len(points_set)
        for p in all_primes:
            if p > n:
                break
            
            if missing_counts[p] == 0:
                # Find the least populated residue class
                min_count = min(residue_counts[p])
                target_residues = [r for r, c in enumerate(residue_counts[p]) if c == min_count]
                target = target_residues[0]
                
                # Remove all elements with this residue
                to_remove = [x for x in points_set if x % p == target]
                
                if not to_remove:
                    continue
                
                for x in to_remove:
                    points_set.remove(x)
                    for q in all_primes:
                        r = x % q
                        residue_counts[q][r] -= 1
                        if residue_counts[q][r] == 0:
                            missing_counts[q] += 1
                
                changed = True
                break
    
    return list(points_set)


def _local_search(points, all_primes, deadline):
    """Local search: try to add back elements that were not in the initial set."""
    points_set = set(points)
    
    # Build residue counts
    residue_counts = {}
    missing_counts = {}
    for p in all_primes:
        residue_counts[p] = [0] * p
        missing_counts[p] = p
    
    for x in points_set:
        for p in all_primes:
            r = x % p
            if residue_counts[p][r] == 0:
                missing_counts[p] -= 1
            residue_counts[p][r] += 1
    
    # Candidates: elements not in points_set
    candidates = [x for x in range(N + 1) if x not in points_set]
    random.shuffle(candidates)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            current_size = len(points_set)
            target_size = current_size + 1
            valid = True
            
            for p in all_primes:
                if p > target_size:
                    break
                if missing_counts[p] == 1:
                    r = x % p
                    if residue_counts[p][r] == 0:
                        valid = False
                        break
            
            if valid:
                points_set.add(x)
                for p in all_primes:
                    r = x % p
                    if residue_counts[p][r] == 0:
                        missing_counts[p] -= 1
                    residue_counts[p][r] += 1
                candidates.remove(x)
                improved = True
    
    return list(points_set)


def construct():
    start_time = time.time()
    all_primes = _primes(N + 1)
    
    best_points = []
    best_size = 0
    
    # Multiple randomized restarts
    num_restarts = 0
    while time.time() - start_time < DEADLINE - 30:
        num_restarts += 1
        
        # Create a randomized order
        # Mix: start with some structure, then randomize
        base_order = list(range(N + 1))
        
        # Strategy: prioritize numbers that are "less restrictive"
        # Heuristic: numbers with more missing residue classes are better
        # For now, use random shuffle with some structure
        
        # Create weighted random order: favor numbers that appear in many residue gaps
        # Simplified: just shuffle, but keep 0 and 1 early as they're often useful
        order = base_order[:]
        random.shuffle(order)
        
        # Ensure 0 and 1 are considered early (they're often in optimal solutions)
        if 0 in order:
            idx0 = order.index(0)
            order[idx0], order[0] = order[0], order[idx0]
        if 1 in order and len(order) > 1:
            idx1 = order.index(1)
            order[idx1], order[1] = order[1], order[idx1]
        
        # Phase 1: Greedy construction
        greedy_points = _greedy_construct(order, all_primes, time.time() + 20)
        
        # Phase 2: Repair
        repaired_points = _repair(greedy_points, all_primes, time.time() + 30)
        
        # Phase 3: Local search
        final_points = _local_search(repaired_points, all_primes, time.time() + 60)
        
        # Update best
        if len(final_points) > best_size:
            best_size = len(final_points)
            best_points = final_points[:]
        
        # If we've used most of our time, stop
        if time.time() - start_time > DEADLINE - 40:
            break
    
    # Final verification
    if not _is_admissible(best_points, all_primes):
        # Fallback to a simple valid solution
        best_points = [1]
    
    return best_points
# EVOLVE-BLOCK-END
