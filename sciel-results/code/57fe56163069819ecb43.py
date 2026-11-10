"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import product

N = 5000

# EVOLVE-BLOCK-START

def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, primes_to_check):
    """Check if points form an admissible tuple for the given primes."""
    n = len(points)
    if n == 0:
        return True
    for p in primes_to_check:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True

def _count_residues(points, p):
    """Count how many points fall into each residue class mod p."""
    counts = [0] * p
    for v in points:
        counts[v % p] += 1
    return counts

def _remove_least_populated(points, p):
    """Remove the least populated residue class mod p from points."""
    counts = _count_residues(points, p)
    min_count = min(counts)
    # Find the residue class with the minimum count
    # If multiple, pick the first one
    target_residue = counts.index(min_count)
    return [v for v in points if v % p != target_residue]

def _greedy_finish(points, primes):
    """
    Greedily repair admissibility by removing least-populated residue classes
    for primes that are fully covered.
    """
    n = len(points)
    if n == 0:
        return points
    # Only need to check primes <= len(points)
    relevant_primes = [p for p in primes if p <= n]
    
    changed = True
    while changed:
        changed = False
        for p in relevant_primes:
            if p > len(points):
                continue
            residues = set(v % p for v in points)
            if len(residues) == p:
                points = _remove_least_populated(points, p)
                changed = True
                break
    return points

def _construct_with_seed(seed_residues, primes, deadline):
    """
    Construct a tuple given specific residue assignments for the first few primes.
    seed_residues: dict mapping prime -> residue class to remove
    """
    # Start with all integers in [0, N]
    points = list(range(N + 1))
    
    # Apply the seed residue removals
    for p, residue in seed_residues.items():
        if p in primes:
            points = [v for v in points if v % p != residue]
    
    # Now greedily finish
    points = _greedy_finish(points, primes)
    
    return points

def _try_branching(first_primes, primes, deadline):
    """
    Try all combinations of residue removals for the first few primes,
    then greedily finish each branch.
    """
    best_points = None
    best_size = -1
    
    # Get the primes we want to branch on
    branch_primes = [p for p in first_primes if p in primes]
    
    # Generate all combinations of residues for these primes
    residue_ranges = [range(p) for p in branch_primes]
    
    for combo in product(*residue_ranges):
        if time.time() > deadline:
            break
        
        seed_residues = dict(zip(branch_primes, combo))
        points = _construct_with_seed(seed_residues, primes, deadline)
        
        size = len(points)
        if size > best_size:
            best_size = size
            best_points = points
    
    return best_points, best_size

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10 second safety margin
    
    # Get all primes up to N
    all_primes = _primes(N)
    
    # The seed program's approach: start with 1 and primes > sqrt(N)
    # But we'll do better by branching on small primes
    
    # First, let's try the original seed approach as a baseline
    points_baseline = [1] + [p for p in all_primes if p * p > N]
    points_baseline = _greedy_finish(points_baseline, all_primes)
    best_points = points_baseline
    best_size = len(best_points)
    
    # Now try branching on the first 4 primes: 2, 3, 5, 7
    # This explores 2*3*5*7 = 210 configurations
    if time.time() < deadline:
        branch_primes = [2, 3, 5, 7]
        # Filter to only those that are in all_primes and <= N
        branch_primes = [p for p in branch_primes if p in all_primes]
        
        if branch_primes:
            branched_points, branched_size = _try_branching(branch_primes, all_primes, deadline)
            if branched_points is not None and branched_size > best_size:
                best_points = branched_points
                best_size = branched_size
    
    # If we still have time, try branching on more primes
    if time.time() < deadline - 10:
        # Try branching on first 5 primes: 2, 3, 5, 7, 11
        branch_primes_5 = [2, 3, 5, 7, 11]
        branch_primes_5 = [p for p in branch_primes_5 if p in all_primes]
        
        if len(branch_primes_5) > len([p for p in [2, 3, 5, 7] if p in all_primes]):
            branched_points_5, branched_size_5 = _try_branching(branch_primes_5, all_primes, deadline)
            if branched_points_5 is not None and branched_size_5 > best_size:
                best_points = branched_points_5
                best_size = branched_size_5
    
    # Verify admissibility of the best result
    n = len(best_points)
    relevant_primes = [p for p in all_primes if p <= n]
    if not _is_admissible(best_points, relevant_primes):
        # If somehow not admissible, do a final repair
        best_points = _greedy_finish(best_points, all_primes)
    
    return best_points

# EVOLVE-BLOCK-END
