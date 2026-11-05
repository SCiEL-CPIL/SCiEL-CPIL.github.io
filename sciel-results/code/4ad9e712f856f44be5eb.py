"""
Strategy: Global Optimization via Max-Flow (Min-Cut)
Unlike the local repair heuristic, this formulates the problem as finding the minimum set of points to remove such that for every prime p <= len(points), at least one residue class mod p is empty.
This is a hitting set / deletion problem. We approximate it by constructing a bipartite graph between "Points" and "Constraint Violations" (specific prime-residue class fullness) and using Max-Flow to find the minimum cut.
Since the constraint is "some class is missed", we can model it as: for each prime p, we must "break" the fullness of at least one class.
A simpler, more direct global approach for this specific structure (admissible tuples) is often related to finding a large subset with no complete residue system.
However, a robust global method is to use a Max-Flow based deletion.
Let's use a different, more direct global strategy: 
Iterative Deepening with Randomized Restarts and Greedy Construction, but with a "Lookahead" or "Backtracking" mechanism for the critical small primes.
Actually, the most effective "different family" for admissible tuples is often **Greedy Construction with Pruning** or **Simulated Annealing**.
Let's implement **Simulated Annealing** to escape local optima of the deletion heuristic.
We start with a full set or a large random set, and iteratively remove/swap elements to minimize the "violation count" (number of primes p where all classes are covered).
Objective: Maximize size subject to Violation Count == 0.
Penalty: Size - Lambda * Violation_Count.
We anneal Lambda to force feasibility.
"""

import time
import random
import math
from collections import defaultdict

N = 5000
DEADLINE = 325.0  # Safety margin

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

# Pre-compute primes up to N
PRIMES_N = _primes(N)
# Map prime to index for quick access? Not strictly needed, list is fine.

def _get_relevant_primes(k):
    """Return primes p <= k."""
    # Since k <= N, we can just filter PRIMES_N
    # Optimization: Pre-compute prefixes? Or just use bisect.
    # For k up to 5000, list comprehension is fast enough if done sparingly.
    # But we do this in the inner loop. Let's optimize.
    pass

# Pre-compute list of primes up to N once.
# For a given tuple size K, we only care about primes <= K.
# Let's create a sorted list of primes.
ALL_PRIMES = _primes(N)

def _violations(points):
    """
    Returns the number of primes p <= len(points) for which the points cover ALL residue classes mod p.
    We want this to be 0.
    """
    k = len(points)
    if k == 0:
        return 0
    
    # Only check primes <= k
    # Find primes <= k
    # Using binary search on ALL_PRIMES
    import bisect
    idx = bisect.bisect_right(ALL_PRIMES, k)
    relevant_primes = ALL_PRIMES[:idx]
    
    if not relevant_primes:
        return 0
        
    # For efficiency, we can check primes in increasing order.
    # If a prime p is "bad" (all classes covered), we count it.
    violations = 0
    for p in relevant_primes:
        # Check if all residues 0..p-1 are present
        # If p > k, it's impossible to cover all residues (pigeonhole), so p is always "good" (misses some class).
        # So we only need to check p <= k. This is already handled by relevant_primes.
        
        # Efficient check: 
        # If k >= p, we need to see if the set of residues is exactly {0, ..., p-1}.
        # We can use a boolean array or a set.
        # Since p can be up to 5000, a set is O(k). 
        # Total cost per violation check is O(K * pi(K)).
        # This might be slow for K=5000. pi(5000) ~ 669. 5000 * 669 ~ 3.3M ops per eval.
        # In SA, we do many evals. We need speed.
        
        # Optimization: 
        # Only primes p <= k are candidates.
        # For small p, checking is fast.
        # For large p (close to k), it's rare that all classes are covered unless the set is specifically constructed.
        
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                break
        if len(residues) == p:
            violations += 1
            
    return violations

def construct():
    start_time = time.time()
    
    # Initial state: Start with a large set, e.g., all integers [0, N-1] is too big? 
    # The problem asks for a tuple in [0, N]. 
    # The seed starts with a specific set. 
    # Let's start with a random subset of [0, N] of size ~N/2 or similar, or just [0, N-1].
    # Actually, the "admissible" property is local. 
    # A known good starting point is the set of integers that are NOT divisible by small primes? No, that's for primality.
    # Admissible tuple: For every p, some residue is missing.
    # The set {0, 1, ..., K-1} is NOT admissible for p <= K because it covers all residues.
    # So we must remove elements.
    
    # Strategy: Simulated Annealing
    # State: A set of points (use a list for iteration, but check membership via set for speed? 
    #        Actually, we need to remove/swap. A list is easier to mutate. 
    #        We'll maintain a list `points` and a set `point_set` for O(1) lookups if needed, 
    #        but for SA moves, we pick an index from the list.
    
    # Initial Points: Start with a dense set, e.g., range(0, N). 
    # This has many violations. We will remove points to fix violations.
    # Better initial: Start with a set that is "almost" admissible.
    # The seed program's logic: Start with primes > sqrt(N) and 1. 
    # Let's use a similar "sparse" start to be closer to optimal, then add points back if possible?
    # No, SA is better for maximizing size. 
    # Let's start with a random permutation of [0, N] and take the first M points?
    # Or start with all points and remove.
    
    # Let's try: Start with points = list(range(N))
    # This is very large, but has huge violations.
    # SA Move: 
    # 1. Pick a random point to remove. (Decrease size, likely decrease violations)
    # 2. Pick a random point to add (from those not in set). (Increase size, likely increase violations)
    # 3. Swap a point in with a point out.
    
    # Objective: Maximize (Size - Penalty * Violations)
    # We want Violations = 0.
    
    # To make it faster, we can track violations incrementally? 
    # That's hard because removing a point affects many primes.
    # However, if we only check "critical" primes (small ones), it's faster.
    # Most violations come from small primes.
    
    # Let's use a simpler, faster heuristic that is globally better than local repair:
    # **Greedy with Backtracking on Small Primes**
    # 1. Start with a candidate set (e.g., all numbers 0..N-1).
    # 2. Identify the "most violated" prime (smallest p where all classes are covered).
    # 3. For that prime, we MUST remove at least one number from each "critical" class? No, just one class needs to be empty.
    #    So we must remove ALL numbers in one specific residue class mod p.
    #    To minimize loss, we should remove the class with the FEWEST numbers.
    # 4. This is exactly what the seed program does! 
    #    So why is it stuck? Because it's greedy. It might pick a class that hurts future primes.
    
    # Different Strategy: **Branch and Bound** or **Local Search with Multiple Restarts**.
    # Let's implement **Iterated Local Search (ILS)**.
    # 1. Run the seed's repair algorithm to get a local optimum.
    # 2. Perturb the result (randomly remove some points, or swap points).
    # 3. Re-run repair.
    # 4. Keep the best.
    # This is a different *search metaheuristic* family (Global vs Local).
    
    def _repair(points):
        """The seed's repair logic, optimized."""
        points = list(set(points)) # Ensure distinct
        changed = True
        while changed:
            changed = False
            k = len(points)
            if k == 0:
                break
            # Find primes <= k
            # We can pre-calculate primes up to N.
            # For efficiency in this loop, we check primes in order.
            for p in ALL_PRIMES:
                if p > k:
                    break
                classes = defaultdict(list)
                for v in points:
                    classes[v % p].append(v)
                if len(classes) == p:
                    # All classes covered. Remove the smallest one.
                    smallest_class = min(classes.values(), key=len)
                    # Remove all points in this class
                    remove_set = set(smallest_class)
                    points = [v for v in points if v not in remove_set]
                    changed = True
                    break
        return points

    best_points = []
    best_size = 0
    
    # Initial run
    # Start with a set that is likely to yield a good result.
    # The seed starts with [1] + primes > sqrt(N).
    # Let's start with a slightly larger random set or the full range?
    # Starting with full range [0, N-1] leads to massive deletions.
    # Starting with a "random" subset of size N/2 might be better.
    
    # Let's try starting with the "Seed Initial" but then using ILS.
    # Seed Initial: [1] + [p for p in PRIMES_N if p*p > N]
    init_points = [1] + [p for p in PRIMES_N if p * p > N]
    
    current_points = _repair(init_points)
    best_points = current_points
    best_size = len(current_points)
    
    # ILS Loop
    # Time budget: 320s
    while time.time() - start_time < DEADLINE - 5:
        # Perturb
        # Randomly remove a few points from the current best
        perturbed = list(best_points)
        if not perturbed:
            break
            
        # Number of perturbations: proportional to size? 
        # Let's try removing 5-10% of points randomly.
        num_remove = max(1, int(len(perturbed) * 0.05))
        if num_remove >= len(perturbed):
            num_remove = len(perturbed) - 1 if len(perturbed) > 1 else 0
            
        if num_remove > 0:
            random_indices = random.sample(range(len(perturbed)), num_remove)
            remove_set = set(perturbed[i] for i in random_indices)
            perturbed = [v for v in perturbed if v not in remove_set]
        
        # Re-repair
        new_points = _repair(perturbed)
        
        if len(new_points) > best_size:
            best_size = len(new_points)
            best_points = new_points
            
            # If we found a perfect admissible tuple (which is the goal), 
            # can we ADD points back?
            # The repair function only removes. 
            # We can try to "grow" the tuple.
            # Try adding random points from [0, N] and repairing.
            # This is a "Grow" phase.
            
    # Final Growth Phase
    # Try to add points to the best tuple found.
    # For each point in [0, N] not in best_points:
    #   If adding it and repairing results in a larger set, keep it.
    # This is expensive. 
    # Instead, let's do a final pass of "Greedy Add".
    
    # Check time
    if time.time() - start_time < DEADLINE - 10:
        current = list(best_points)
        current_set = set(current)
        # Try adding points in random order
        candidates = [i for i in range(N) if i not in current_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() - start_time > DEADLINE - 10:
                break
            # Tentatively add
            trial = current + [c]
            # Repair
            repaired = _repair(trial)
            if len(repaired) > len(current):
                current = repaired
                current_set = set(current)
                best_points = current
                best_size = len(current)

    return best_points

# Ensure the function signature matches
def construct():
    # The implementation above is inside a dummy scope, let's restructure properly.
    pass

# Re-defining construct properly to avoid the dummy scope issue
def construct():
    start_time = time.time()
    
    def _repair(points):
        """The seed's repair logic, optimized."""
        points = list(set(points)) 
        changed = True
        while changed:
            changed = False
            k = len(points)
            if k == 0:
                break
            for p in ALL_PRIMES:
                if p > k:
                    break
                classes = defaultdict(list)
                for v in points:
                    classes[v % p].append(v)
                if len(classes) == p:
                    smallest_class = min(classes.values(), key=len)
                    remove_set = set(smallest_class)
                    points = [v for v in points if v not in remove_set]
                    changed = True
                    break
        return points

    # Initial state
    init_points = [1] + [p for p in ALL_PRIMES if p * p > N]
    
    best_points = _repair(init_points)
    best_size = len(best_points)
    
    # ILS Loop
    while time.time() - start_time < DEADLINE - 5:
        perturbed = list(best_points)
        if not perturbed:
            break
            
        num_remove = max(1, int(len(perturbed) * 0.05))
        if num_remove >= len(perturbed):
            num_remove = len(perturbed) - 1 if len(perturbed) > 1 else 0
            
        if num_remove > 0:
            random_indices = random.sample(range(len(perturbed)), num_remove)
            remove_set = set(perturbed[i] for i in random_indices)
            perturbed = [v for v in perturbed if v not in remove_set]
        
        new_points = _repair(perturbed)
        
        if len(new_points) > best_size:
            best_size = len(new_points)
            best_points = new_points

    # Final Growth Phase
    if time.time() - start_time < DEADLINE - 10:
        current = list(best_points)
        current_set = set(current)
        candidates = [i for i in range(N) if i not in current_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() - start_time > DEADLINE - 10:
                break
            trial = current + [c]
            repaired = _repair(trial)
            if len(repaired) > len(current):
                current = repaired
                current_set = set(current)
                best_points = current
                best_size = len(current)

    return best_points

