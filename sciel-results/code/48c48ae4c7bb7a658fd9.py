import math
import time
from collections import defaultdict

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    limit_time = start_time + 290.0

    primes = _primes(N)
    
    # Strategy: For each prime p <= current_size, we must have at least one residue class mod p empty.
    # We want to minimize the number of points removed.
    
    # Key insight: For small primes, the choice of which residue class to empty has huge impact.
    # Use beam search for the first few primes, then greedy for the rest.
    
    # Phase 1: Beam search for first K primes
    # A state is: (removed_set, list_of_chosen_residues_for_primes_so_far)
    # We track removed points as a set for efficiency
    
    K_BEAM = 8  # Number of primes to use beam search for
    BEAM_WIDTH = 50
    
    # Precompute residue for each x and each prime in first K_BEAM primes
    beam_primes = primes[:K_BEAM]
    
    # Initial state: no points removed
    # State: (frozenset of removed points, list of (prime, chosen_residue) pairs)
    # For efficiency, represent removed as a boolean array or set
    
    # Actually, let's think differently. For each prime p, we choose a residue r_p.
    # The removed set is the union of {x : x % p == r_p for some p in processed primes}.
    # We want to minimize |removed_set|.
    
    # For beam search, state = (set of removed points, list of chosen residues)
    # But sets are expensive. Let's use a different representation.
    
    # Alternative: For each prime p_i, the set of points removed due to p_i is R_i = {x : x % p_i == r_i}.
    # Total removed = union of R_i.
    # |union| = sum |R_i| - sum |R_i ∩ R_j| + ... (inclusion-exclusion, too complex)
    
    # Better: Just track the actual set of removed points. For N=5000, a set of ints is fine.
    
    # Beam search state: (removed_frozenset, chosen_residues_list)
    # But frozenset hashing is expensive. Let's use a different approach.
    
    # Actually, for the beam search, let's just track:
    # - The set of removed points (as a Python set)
    # - The list of (prime, residue) choices made so far
    
    # For efficiency, let's precompute for each prime p and residue r, the set of points in that class
    class_sets = {}
    for p in beam_primes:
        for r in range(p):
            class_sets[(p, r)] = set(range(r, N + 1, p))
    
    # Beam search
    # State: (removed_set, choices_list) where choices_list = [(p1, r1), (p2, r2), ...]
    # Score: -len(removed_set) (we want to minimize removals)
    
    beam = [(set(), [])]  # Start with empty removed set
    
    for idx, p in enumerate(beam_primes):
        if time.time() > limit_time:
            break
        new_beam = []
        for removed_set, choices in beam:
            # For each possible residue r for this prime p
            for r in range(p):
                # New removed set = removed_set ∪ class_sets[(p, r)]
                new_removed = removed_set | class_sets[(p, r)]
                new_choices = choices + [(p, r)]
                new_beam.append((new_removed, new_choices))
        
        # Sort by size of removed_set (ascending) and keep top BEAM_WIDTH
        new_beam.sort(key=lambda x: len(x[0]))
        beam = new_beam[:BEAM_WIDTH]
    
    # Pick the best state from the beam
    if beam:
        beam.sort(key=lambda x: len(x[0]))
        best_removed = beam[0][0]
        best_choices = beam[0][1]
    else:
        best_removed = set()
        best_choices = []
    
    # Phase 2: Greedy for remaining primes
    # For each remaining prime p, choose the residue class that minimizes additional removals
    # i.e., the class with the fewest points NOT already in best_removed
    
    active = [True] * (N + 1)
    for x in best_removed:
        active[x] = False
    active_count = N + 1 - len(best_removed)
    
    # Continue with remaining primes
    for p in primes[K_BEAM:]:
        if time.time() > limit_time:
            break
        if p > active_count:
            break
        
        # Count active points in each residue class mod p
        residue_counts = [0] * p
        for x in range(N + 1):
            if active[x]:
                residue_counts[x % p] += 1
        
        if min(residue_counts) > 0:
            # Choose the class with fewest active points
            min_count = min(residue_counts)
            min_classes = [r for r in range(p) if residue_counts[r] == min_count]
            chosen_class = min_classes[0]
            
            for x in range(chosen_class, N + 1, p):
                if active[x]:
                    active[x] = False
                    active_count -= 1
    
    # Phase 3: Recovery - try to add back removed points
    current_set = [x for x in range(N + 1) if active[x]]
    
    # Precompute coverage for each prime
    coverage = []
    for p in primes:
        counts = [0] * p
        for x in current_set:
            counts[x % p] += 1
        coverage.append(counts)
    
    missed = [sum(1 for c in cov if c == 0) for cov in coverage]
    
    unique_missed_res = [None] * len(primes)
    for i, cov in enumerate(coverage):
        zeros = [r for r, c in enumerate(cov) if c == 0]
        if len(zeros) == 1:
            unique_missed_res[i] = zeros[0]
    
    removed = [x for x in range(N + 1) if not active[x]]
    
    point_cost = {}
    for x in removed:
        cost = 0
        for i, p in enumerate(primes):
            if unique_missed_res[i] is not None and x % p == unique_missed_res[i]:
                cost += 1
        point_cost[x] = cost
    
    removed_sorted = sorted(removed, key=lambda x: point_cost[x])
    
    # Multiple passes of recovery
    for pass_num in range(3):
        if time.time() > limit_time:
            break
        
        for x in removed_sorted:
            if time.time() > limit_time:
                break
            
            if active[x]:
                continue  # Already added back
            
            new_size = active_count + 1
            valid = True
            for i, p in enumerate(primes):
                if p > new_size:
                    break
                r = x % p
                old_missed = missed[i]
                if coverage[i][r] == 0:
                    new_missed = old_missed - 1
                else:
                    new_missed = old_missed
                if new_missed == 0:
                    valid = False
                    break
            
            if valid:
                active[x] = True
                active_count += 1
                current_set.append(x)
                for i, p in enumerate(primes):
                    r = x % p
                    coverage[i][r] += 1
                    if coverage[i][r] == 1:
                        missed[i] -= 1
                    if missed[i] <= 1:
                        zeros = [r2 for r2, c in enumerate(coverage[i]) if c == 0]
                        if len(zeros) == 1:
                            unique_missed_res[i] = zeros[0]
                        else:
                            unique_missed_res[i] = None
    
    return sorted(current_set)
# EVOLVE-BLOCK-END
