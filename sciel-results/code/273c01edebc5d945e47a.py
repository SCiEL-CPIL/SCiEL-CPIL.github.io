"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import Counter
from itertools import combinations

def _count_duplicate_sums(subset):
    """Count how many unordered pair sums (including diagonals) are duplicated."""
    n = len(subset)
    sums = [subset[i] + subset[j] for i in range(n) for j in range(i, n)]
    c = Counter(sums)
    return sum(k * (k - 1) // 2 for k in c.values())

def _is_sidon(vals):
    """Check if a list of values forms a Sidon set (unique pair sums including diagonals)."""
    s = len(vals)
    seen = set()
    for i in range(s):
        vi = vals[i]
        for j in range(i, s):
            ps = vi + vals[j]
            if ps in seen:
                return False
            seen.add(ps)
    return True

def _max_sidon_subset_size_bruteforce(subset, deadline=None):
    """
    Find largest subset with all unordered pair sums unique (including diagonals).
    Optimized for n=12.
    """
    n = len(subset)
    # If n is small, just check
    # We iterate sizes from n down to 1
    for size in range(n, 0, -1):
        if deadline and time.time() > deadline:
            break
        # Generate combinations
        # For n=12, C(12, 12)=1, C(12, 11)=12, C(12, 10)=66, C(12, 9)=220, C(12, 8)=495, C(12, 7)=792, C(12, 6)=924, C(12, 5)=792
        # Total checks in worst case ~ 3300. Very fast.
        for combo in combinations(range(n), size):
            if deadline and time.time() > deadline:
                break
            vals = [subset[i] for i in combo]
            if _is_sidon(vals):
                return size
    return 0

def _local_search_collision(subset, deadline, max_passes=20):
    """
    Local search to maximize the number of duplicate sums.
    """
    subset = list(subset)
    n = len(subset)
    current_dups = _count_duplicate_sums(subset)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(n))
        random.shuffle(indices)
        for i in indices:
            old_val = subset[i]
            best_new_val = old_val
            best_new_dups = current_dups
            
            # Try all possible new values
            for new_val in range(129):
                if new_val == old_val:
                    continue
                # Check uniqueness
                # Optimization: if new_val is in subset (excluding i), skip
                # Since n=12, this is fast
                # Robust uniqueness check
                is_unique = True
                for j in range(n):
                    if j != i and subset[j] == new_val:
                        is_unique = False
                        break
                if not is_unique:
                    continue

                subset[i] = new_val
                new_dups = _count_duplicate_sums(subset)
                
                if new_dups > best_new_dups:
                    best_new_dups = new_dups
                    best_new_val = new_val
            
            if best_new_val != old_val:
                subset[i] = best_new_val
                current_dups = best_new_dups
                improved = True
        
        if not improved:
            break
            
    return subset

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_subset = None
    best_score = 13  # Max possible is 12
    
    random.seed(42)
    
    # 1. Generate Initial Candidates
    candidates = []
    
    # Random candidates
    for _ in range(30):
        candidates.append(random.sample(range(129), 12))
        
    # Structured: Arithmetic Progressions
    for start in range(0, 110, 5):
        for d in [1, 2, 3, 4, 5, 6, 7, 8]:
            ap = [start + i * d for i in range(12)]
            if all(0 <= x <= 128 for x in ap):
                candidates.append(ap)
                
    # Structured: Symmetric/Clustered
    # E.g., pairs summing to constant
    for center in range(60, 100, 5):
        # Try to form 6 pairs summing to 2*center
        # c-5, c+5, c-4, c+4, ...
        # But they must be distinct and in range
        pairs = []
        used = set()
        for offset in range(1, 10):
            v1 = center - offset
            v2 = center + offset
            if 0 <= v1 <= 128 and 0 <= v2 <= 128 and v1 not in used and v2 not in used:
                pairs.append(v1)
                pairs.append(v2)
                used.add(v1)
                used.add(v2)
                if len(pairs) == 12:
                    break
        if len(pairs) == 12:
            candidates.append(pairs)
            
    # 2. Process Candidates
    # We will evaluate a subset of them with the exact solver
    # To save time, we first improve them with local search (collision maximization)
    # Then evaluate the top ones.
    
    evaluated_candidates = []
    
    for candidate in candidates:
        if time.time() > deadline - 5:
            break
            
        # Local search to maximize collisions
        # Limit passes to ensure we process more candidates
        improved_candidate = _local_search_collision(candidate, deadline, max_passes=5)
        evaluated_candidates.append(improved_candidate)
        
    # 3. Evaluate Top Candidates
    # Sort by collision count? Or just evaluate all?
    # Collision count is a good proxy. Let's sort by it descending.
    evaluated_candidates.sort(key=_count_duplicate_sums, reverse=True)
    
    # Evaluate the top 50 candidates (or all if less)
    num_to_eval = min(50, len(evaluated_candidates))
    
    for i in range(num_to_eval):
        if time.time() > deadline - 2:
            break
            
        cand = evaluated_candidates[i]
        score = _max_sidon_subset_size_bruteforce(cand, deadline)
        
        if score < best_score:
            best_score = score
            best_subset = cand
            # If we find a very good score (e.g., 4 or 5), we can stop early
            if best_score <= 4:
                break
                
    # 4. Final Polish
    if best_subset is None:
        # Fallback
        best_subset = list(range(12))
        best_score = _max_sidon_subset_size_bruteforce(best_subset, deadline)
    else:
        # Do a deeper local search on the best subset found
        # Try to reduce the score further by accepting moves that keep score same but increase collisions
        # Or just continue collision maximization
        if time.time() < deadline - 10:
            # Run local search with more passes
            polished = _local_search_collision(best_subset, deadline, max_passes=20)
            new_score = _max_sidon_subset_size_bruteforce(polished, deadline)
            if new_score < best_score:
                best_score = new_score
                best_subset = polished
            
            # Try random restarts from the best subset
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                perturbed = list(best_subset)
                # Perturb 2 random elements
                idx1 = random.randint(0, 11)
                idx2 = random.randint(0, 11)
                while idx2 == idx1:
                    idx2 = random.randint(0, 11)
                perturbed[idx1] = random.randint(0, 128)
                perturbed[idx2] = random.randint(0, 128)
                
                # Ensure distinct
                if len(set(perturbed)) < 12:
                    continue
                    
                improved_perturbed = _local_search_collision(perturbed, deadline, max_passes=10)
                score_p = _max_sidon_subset_size_bruteforce(improved_perturbed, deadline)
                if score_p < best_score:
                    best_score = score_p
                    best_subset = improved_perturbed
                    if best_score <= 4:
                        break

    return sorted(best_subset)
# EVOLVE-BLOCK-END
