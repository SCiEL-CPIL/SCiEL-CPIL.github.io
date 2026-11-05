import time
from fractions import Fraction
import itertools

def _sieve(n):
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def _compute_L_from_counts(counts, max_check=100000):
    """Compute L from coverage counts array."""
    L = 0
    limit = min(max_check, len(counts) - 1)
    for i in range(1, limit + 1):
        if counts[i] > 0:
            L = i
        else:
            break
    return L

def _build_counts(classes, max_L=100000):
    """Build coverage counts array from classes."""
    counts = [0] * (max_L + 1)
    for p, r in classes:
        start_k = r if r > 0 else p
        for k in range(start_k, max_L + 1, p):
            counts[k] += 1
    return counts

def _greedy_assign(P_sorted, max_L=100000):
    """Greedy: for n=1..max_L, if not covered, assign smallest unused prime."""
    counts = [0] * (max_L + 1)
    residues = {}
    used_primes = set()
    
    for n in range(1, max_L + 1):
        if counts[n] > 0:
            continue
        assigned = False
        for p in P_sorted:
            if p in used_primes:
                continue
            r = n % p
            residues[p] = r
            used_primes.add(p)
            start_k = r if r > 0 else p
            for k in range(start_k, max_L + 1, p):
                counts[k] += 1
            assigned = True
            break
        if not assigned:
            break
    
    L = _compute_L_from_counts(counts, max_L)
    classes = [[p, residues[p]] for p in P_sorted if p in residues]
    return L, classes, counts

def _local_search_fast(classes, P_sorted, max_L=100000, deadline=None):
    """Fast local search using count-based evaluation."""
    used_primes = set(p for p, r in classes)
    best_classes = [list(c) for c in classes]
    counts = _build_counts(best_classes, max_L)
    best_L = _compute_L_from_counts(counts, max_L)
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        gap = best_L + 1
        if gap > max_L:
            break
        
        # For each prime, try residue that covers the gap
        for idx, (p, r) in enumerate(best_classes):
            if deadline is not None and time.time() > deadline:
                break
            
            new_r = gap % p
            if new_r == r:
                continue
            
            # Tentatively change
            old_r = best_classes[idx][1]
            best_classes[idx][1] = new_r
            
            # Update counts: remove old coverage, add new
            old_start = old_r if old_r > 0 else p
            new_start = new_r if new_r > 0 else p
            
            for k in range(old_start, max_L + 1, p):
                counts[k] -= 1
            for k in range(new_start, max_L + 1, p):
                counts[k] += 1
            
            new_L = _compute_L_from_counts(counts, max_L)
            
            if new_L > best_L:
                best_L = new_L
                improved = True
                # Keep change
                break
            else:
                # Revert
                best_classes[idx][1] = old_r
                for k in range(old_start, max_L + 1, p):
                    counts[k] -= 1
                for k in range(new_start, max_L + 1, p):
                    counts[k] += 1
        
        # Try adding a new prime if budget allows
        if not improved and deadline is None or time.time() < deadline:
            current_sum = sum(Fraction(1, p) for p, r in best_classes)
            for p in P_sorted:
                if p in used_primes:
                    continue
                if current_sum + Fraction(1, p) > 1:
                    continue
                
                new_r = gap % p
                test_classes = [list(c) for c in best_classes] + [[p, new_r]]
                
                new_counts = counts.copy()
                start_k = new_r if new_r > 0 else p
                for k in range(start_k, max_L + 1, p):
                    new_counts[k] += 1
                
                new_L = _compute_L_from_counts(new_counts, max_L)
                if new_L > best_L:
                    best_L = new_L
                    best_classes = test_classes
                    used_primes.add(p)
                    counts = new_counts
                    improved = True
                    break
    
    return best_L, best_classes

def _multi_start_local_search(classes, P_sorted, max_L=100000, deadline=None, n_starts=5):
    """Multiple random restarts of local search."""
    best_L = 0
    best_classes = classes
    
    for start in range(n_starts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Create perturbed version
        perturbed = [list(c) for c in classes]
        if start > 0:
            # Randomly perturb some residues
            import random
            rng = random.Random(start * 12345 + 678)
            n_perturb = max(1, len(perturbed) // 5)
            for _ in range(n_perturb):
                idx = rng.randrange(len(perturbed))
                p = perturbed[idx][0]
                perturbed[idx][1] = rng.randrange(p)
        
        L, cls = _local_search_fast(perturbed, P_sorted, max_L, deadline)
        if L > best_L:
            best_L = L
            best_classes = cls
    
    return best_L, best_classes

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    primes = _sieve(1000)
    small_primes = [2, 3, 5, 7, 11]
    large_primes = [p for p in primes if p > 11]
    
    # Generate candidate prime sets
    candidate_sets = []
    seen_sets = set()
    
    # Seed: primes >= 13
    seed_set = [p for p in primes if p >= 13]
    candidate_sets.append(seed_set)
    seen_sets.add(tuple(sorted(seed_set)))
    
    # Systematic: subsets of small primes + fill with large primes
    for r in range(len(small_primes) + 1):
        for subset in itertools.combinations(small_primes, r):
            current_sum = Fraction(0)
            for p in subset:
                current_sum += Fraction(1, p)
            if current_sum > 1:
                continue
            
            current_set = list(subset)
            for p in large_primes:
                if current_sum + Fraction(1, p) <= 1:
                    current_set.append(p)
                    current_sum += Fraction(1, p)
            
            key = tuple(sorted(current_set))
            if key not in seen_sets:
                seen_sets.add(key)
                candidate_sets.append(current_set)
    
    # Targeted swaps: seed minus some primes plus smaller ones
    swap_candidates = [
        ([53, 59], [11]),
        ([13], [7]),
        ([13, 17], [5]),
        ([13, 17, 19, 23, 29], [3]),
        ([p for p in seed_set if p <= 47], [2]),
        ([13, 53, 59], [7, 11]),
        ([13, 17, 53, 59], [5, 11]),
        ([13, 17, 19], [7, 11]),
        ([13, 17, 19, 23], [5, 7]),
        ([13, 17, 19, 23, 29], [5, 7, 11]),
        ([13, 17, 19, 23, 29, 31, 37], [3, 5]),
        ([13, 17, 19, 23, 29, 31, 37, 41, 43], [3]),
        ([p for p in seed_set if p <= 61], [2, 3]),
        ([p for p in seed_set if p <= 79], [2, 3, 5]),
    ]
    for to_remove, to_add in swap_candidates:
        new_set = [p for p in seed_set if p not in to_remove]
        current_sum = sum(Fraction(1, p) for p in new_set)
        valid_add = []
        for p in to_add:
            if p in new_set:
                continue
            if current_sum + Fraction(1, p) <= 1:
                valid_add.append(p)
                current_sum += Fraction(1, p)
        new_set.extend(valid_add)
        key = tuple(sorted(new_set))
        if key not in seen_sets:
            seen_sets.add(key)
            candidate_sets.append(new_set)
    
    # Evaluate each candidate
    best_L = 0
    best_classes = []
    
    # First pass: quick greedy on all candidates
    candidate_scores = []
    for P in candidate_sets:
        if time.time() > deadline - 40:
            break
        P_sorted = sorted(P)
        L, classes, _ = _greedy_assign(P_sorted, max_L=100000)
        candidate_scores.append((L, P_sorted, classes))
        if L > best_L:
            best_L = L
            best_classes = classes
    
    candidate_scores.sort(key=lambda x: -x[0])
    
    # Second pass: local search on top candidates
    n_local = min(15, len(candidate_scores))
    for L, P_sorted, classes in candidate_scores[:n_local]:
        if time.time() > deadline - 15:
            break
        new_L, new_classes = _multi_start_local_search(classes, P_sorted, max_L=100000, deadline=deadline - 5, n_starts=3)
        if new_L > best_L:
            best_L = new_L
            best_classes = new_classes
    
    # Final polish: more iterations on the best
    if time.time() < deadline - 5:
        P_best = sorted(set(p for p, r in best_classes))
        final_L, final_classes = _multi_start_local_search(best_classes, P_best, max_L=100000, deadline=deadline - 2, n_starts=10)
        if final_L > best_L:
            best_L = final_L
            best_classes = final_classes
    
    # Verify budget
    s = sum(Fraction(1, p) for p, r in best_classes)
    if s > 1:
        # Fallback to seed
        seed_set = [p for p in primes if p >= 13]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]
    
    return best_classes

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s, Classes: {len(result)}")
    # Compute L
    max_L = 100000
    covered = [False] * (max_L + 1)
    for p, r in result:
        start_k = r if r > 0 else p
        for k in range(start_k, max_L + 1, p):
            covered[k] = True
    L = 0
    for i in range(1, max_L + 1):
        if covered[i]:
            L = i
        else:
            break
    print(f"L = {L}")
    s = sum(Fraction(1, p) for p, r in result)
    print(f"Sum 1/p = {float(s):.6f}")
