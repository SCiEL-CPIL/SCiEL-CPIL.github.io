"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _score(u):
    """Compute the score for set u (list of nonnegative ints containing 0)."""
    u_set = set(u)
    n = len(u_set)
    max_u = max(u_set)
    if max_u == 0:
        return 0.0
    # Compute sumset U+U
    sumset = set()
    for a in u_set:
        for b in u_set:
            sumset.add(a + b)
    # Compute difference set U-U
    diffset = set()
    for a in u_set:
        for b in u_set:
            diffset.add(a - b)
    len_sum = len(sumset)
    len_diff = len(diffset)
    if len_sum == 0:
        return 0.0
    ratio = len_diff / len_sum
    if ratio <= 1.0:
        return 1.0  # lower bound is at least 1
    denom = math.log(2 * max_u + 1, 2)
    if denom == 0:
        return 1.0
    val = 1.0 + math.log(ratio) / denom
    return val


def _try_add(u, x):
    """Try adding x to u, return new score and new set if improved."""
    if x in u or x < 0:
        return None, None
    new_u = list(u) + [x]
    new_u.sort()
    s = _score(new_u)
    return s, new_u


def _try_remove(u, idx):
    """Remove element at idx, return new score and new set."""
    if len(u) <= 1:
        return None, None
    new_u = [u[i] for i in range(len(u)) if i != idx]
    if 0 not in new_u:
        return None, None
    s = _score(new_u)
    return s, new_u


def _local_search(u, time_limit, rng):
    """Local search: try adding, removing, and swapping elements."""
    u = sorted(u)
    current_score = _score(u)
    best_u = list(u)
    best_score = current_score
    
    deadline = time.time() + time_limit
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
        
        improved = False
        
        # Try adding a new element
        max_u = max(u)
        # Candidate values to try adding
        candidates = set()
        # Small additions
        for d in range(1, 20):
            candidates.add(max_u + d)
            candidates.add(max_u + 2 * d)
            candidates.add(max_u + 3 * d)
        # Some random candidates
        for _ in range(30):
            candidates.add(rng.randint(1, max_u * 3 + 50))
        # Also try filling gaps
        for i in range(len(u) - 1):
            if u[i+1] - u[i] > 1:
                for gap_val in range(u[i] + 1, u[i+1]):
                    candidates.add(gap_val)
        
        for x in sorted(candidates):
            if time.time() >= deadline:
                break
            s, new_u = _try_add(u, x)
            if s is not None and s > current_score + 1e-12:
                u = new_u
                current_score = s
                if s > best_score:
                    best_score = s
                    best_u = list(u)
                improved = True
                break
        
        # Try removing an element (not 0)
        if not improved:
            removable = [i for i in range(len(u)) if u[i] != 0]
            if removable:
                idx = rng.choice(removable)
                s, new_u = _try_remove(u, idx)
                if s is not None and s > current_score + 1e-12:
                    u = new_u
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_u = list(u)
                    improved = True
        
        # Try swapping: remove one, add another
        if not improved:
            removable = [i for i in range(len(u)) if u[i] != 0]
            if removable:
                idx = rng.choice(removable)
                # Try replacing u[idx] with a random value
                max_u = max(u)
                for _ in range(5):
                    if time.time() >= deadline:
                        break
                    new_val = rng.randint(1, max_u * 2 + 30)
                    if new_val in u:
                        continue
                    # Create new set with swap
                    new_u = [u[i] for i in range(len(u)) if i != idx]
                    new_u.append(new_val)
                    new_u.sort()
                    if 0 not in new_u:
                        continue
                    s = _score(new_u)
                    if s > current_score + 1e-12:
                        u = new_u
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_u = list(u)
                        improved = True
                        break
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If no improvement for a while, do a random restart with perturbation
        if no_improve_count % 500 == 499 and no_improve_count > 0:
            # Perturb: remove 1-2 random non-zero elements and re-run local search
            perturbed = list(best_u)
            removable = [i for i in range(len(perturbed)) if perturbed[i] != 0]
            n_remove = min(len(removable), rng.randint(1, 2))
            for _ in range(n_remove):
                if not removable:
                    break
                idx = rng.choice(removable)
                perturbed.pop(idx)
                removable = [i for i in range(len(perturbed)) if perturbed[i] != 0]
            if 0 in perturbed and len(perturbed) > 0:
                perturbed.sort()
                s = _score(perturbed)
                if s > current_score - 0.02:  # allow slight degradation
                    u = perturbed
                    current_score = s
                    no_improve_count = 0
    
    return best_u


def construct():
    """Construct a finite set U of nonnegative integers containing 0."""
    start_time = time.time()
    total_budget = 140.0  # safety margin below 150s
    
    rng = random.Random(42)
    
    # Start with a good initial set based on geometric-like progression
    # [0, 1, 3, 7, 15] gave ~1.098 in analysis
    # Let's try several initial seeds and pick the best
    seeds = [
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 7, 13],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15, 30],
        [0, 1, 2, 4, 8, 15],
        [0, 1, 3, 6, 12, 23],
        [0, 1, 3, 7, 15, 28],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 4, 7, 11, 15],
    ]
    
    # Evaluate all seeds
    best_seed = seeds[0]
    best_seed_score = _score(best_seed)
    for s in seeds[1:]:
        sc = _score(s)
        if sc > best_seed_score:
            best_seed_score = sc
            best_seed = s
    
    # Run local search on the best seed
    remaining = total_budget - (time.time() - start_time)
    if remaining < 5.0:
        return sorted(best_seed)
    
    # Do multiple local search passes with different time allocations
    best_result = list(best_seed)
    best_result_score = best_seed_score
    
    # First pass: main local search
    t1 = time.time()
    result = _local_search(best_seed, remaining * 0.6, rng)
    elapsed = time.time() - t1
    if result:
        s = _score(result)
        if s > best_result_score:
            best_result_score = s
            best_result = result
    
    # Second pass: local search on a perturbed version
    remaining = total_budget - (time.time() - start_time)
    if remaining > 10.0:
        # Perturb the best result
        perturbed = list(best_result)
        removable = [i for i in range(len(perturbed)) if perturbed[i] != 0]
        if removable:
            idx = rng.choice(removable)
            perturbed.pop(idx)
            # Add a random new element
            max_u = max(perturbed) if perturbed else 0
            new_val = rng.randint(1, max_u * 2 + 30)
            if new_val not in perturbed:
                perturbed.append(new_val)
            perturbed.sort()
            if 0 in perturbed and len(perturbed) > 0:
                t2 = time.time()
                result2 = _local_search(perturbed, min(remaining * 0.3, 40.0), rng)
                if result2:
                    s2 = _score(result2)
                    if s2 > best_result_score:
                        best_result_score = s2
                        best_result = result2
    
    # Final: try a few more direct additions to the best result
    remaining = total_budget - (time.time() - start_time)
    if remaining > 3.0:
        max_u = max(best_result)
        for x in range(max_u + 1, max_u + 40):
            if time.time() > start_time + total_budget - 1.0:
                break
            s, new_u = _try_add(best_result, x)
            if s is not None and s > best_result_score + 1e-12:
                best_result_score = s
                best_result = new_u
                break
    
    return sorted(best_result)

# EVOLVE-BLOCK-END
