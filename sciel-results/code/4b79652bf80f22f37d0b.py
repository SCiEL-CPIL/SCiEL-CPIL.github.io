"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = 90.0
_SAFETY_MARGIN = 5.0

def _collision_score(candidate_set):
    """Count the number of unordered pair-sum collisions (including diagonals).
    For a set S, count pairs (i,j) with i<=j such that sum_i + sum_j appears
    for multiple distinct (i,j) pairs. Higher = more collisions = harder to
    find large Sidon subsets."""
    sums = {}
    for i in range(len(candidate_set)):
        for j in range(i, len(candidate_set)):
            s = candidate_set[i] + candidate_set[j]
            sums[s] = sums.get(s, 0) + 1
    collisions = sum(c - 1 for c in sums.values() if c > 1)
    return collisions

def _max_sidon_subset_size_bruteforce(ambient_set, max_size_to_check=6):
    """Check if there exists a Sidon subset of a given size.
    Returns the largest size found (up to max_size_to_check)."""
    n = len(ambient_set)
    
    def _is_sidon(subset):
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Check from largest possible down
    for size in range(min(max_size_to_check, n), 2, -1):
        for combo in combinations(ambient_set, size):
            if _is_sidon(list(combo)):
                return size
    return 2

def _greedy_build_collision_density(used_set, all_candidates, current_collisions):
    """Select next element that maximizes collision density.
    Returns (best_element, new_collision_count)."""
    best_elem = None
    best_score = -1
    best_collisions = 0
    
    for cand in all_candidates:
        if cand in used_set:
            continue
        # Compute collision score if we add this candidate
        new_set = list(used_set) + [cand]
        new_set.sort()
        score = _collision_score(new_set)
        if score > best_score:
            best_score = score
            best_elem = cand
            best_collisions = score
    
    return best_elem, best_collisions

def _local_search_improve(ambient_set, time_limit):
    """Try to improve the ambient set via local search (swap one element)."""
    best_set = list(ambient_set)
    best_max_sidon = _max_sidon_subset_size_bruteforce(best_set, 5)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(len(best_set)):
            for new_val in range(129):
                if new_val in best_set and new_val != best_set[i]:
                    continue
                trial = list(best_set)
                trial[i] = new_val
                trial.sort()
                # Quick collision score check
                cs = _collision_score(trial)
                if cs >= _collision_score(best_set):
                    ms = _max_sidon_subset_size_bruteforce(trial, 5)
                    if ms < best_max_sidon:
                        best_set = trial
                        best_max_sidon = ms
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    return best_set, best_max_sidon

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY_MARGIN
    
    # Phase 1: Greedy build maximizing collision density
    used = set()
    ambient = []
    
    # Start with a seed - try multiple starting points
    best_ambient = None
    best_max_sidon = 12  # worst case
    
    random.seed(42)
    
    # Try several random starting seeds
    for seed_attempt in range(20):
        if time.time() > deadline:
            break
        
        # Initialize with 2 random elements
        init_pool = list(range(129))
        random.shuffle(init_pool)
        ambient = []
        used = set()
        
        # Pick first 2 elements
        for _ in range(2):
            e = init_pool.pop()
            ambient.append(e)
            used.add(e)
        
        # Greedily add remaining 10 elements
        for _ in range(10):
            if time.time() > deadline:
                break
            
            # Get all candidates not yet used
            candidates = [x for x in range(129) if x not in used]
            
            best_elem = None
            best_score = -1
            
            for cand in candidates:
                new_set = list(ambient) + [cand]
                score = _collision_score(new_set)
                if score > best_score:
                    best_score = score
                    best_elem = cand
            
            if best_elem is None:
                break
            
            ambient.append(best_elem)
            used.add(best_elem)
        
        if len(ambient) == 12:
            ambient.sort()
            # Evaluate
            ms = _max_sidon_subset_size_bruteforce(ambient, 5)
            if ms < best_max_sidon:
                best_max_sidon = ms
                best_ambient = list(ambient)
            
            if best_max_sidon <= 3:
                break
    
    # Phase 2: Local search to improve
    if best_ambient is not None and time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time > 2.0:
            improved_set, improved_ms = _local_search_improve(
                best_ambient, remaining_time - 1.0
            )
            if improved_ms < best_max_sidon:
                best_ambient = improved_set
                best_max_sidon = improved_ms
    
    # Fallback
    if best_ambient is None:
        best_ambient = list(range(12))
    
    return best_ambient
# EVOLVE-BLOCK-END
