import time
from fractions import Fraction
import random
import itertools

def _sieve(n):
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def _build_counts(primes_residues, max_n):
    """Build coverage count array: count[i] = number of classes covering i."""
    counts = [0] * (max_n + 1)
    for p, r in primes_residues:
        start = r if r > 0 else p
        for k in range(start, max_n + 1, p):
            counts[k] += 1
    return counts

def _compute_L_from_counts(counts, max_n):
    """Find largest L such that all 1..L have count > 0."""
    L = 0
    for i in range(1, max_n + 1):
        if counts[i] > 0:
            L = i
        else:
            break
    return L

def _greedy_assign(primes, max_n=100000):
    """Greedy: for n=1..max_n, if not covered, assign smallest unused prime."""
    covered = [False] * (max_n + 1)
    residues = {}
    used_primes = set()
    primes_sorted = sorted(primes)
    
    for n in range(1, max_n + 1):
        if covered[n]:
            continue
        assigned = False
        for p in primes_sorted:
            if p in used_primes:
                continue
            r = n % p
            residues[p] = r
            used_primes.add(p)
            start_k = r if r > 0 else p
            for k in range(start_k, max_n + 1, p):
                covered[k] = True
            assigned = True
            break
        if not assigned:
            break
    
    L = _compute_L(covered, max_n)
    classes = [[p, residues[p]] for p in primes_sorted if p in residues]
    return L, classes

def _local_search_fast(classes, primes_pool, max_n=100000, deadline=None):
    """Fast local search using count-based evaluation."""
    best_classes = [list(c) for c in classes]
    current_sum = sum(Fraction(1, p) for p, r in best_classes)
    
    # Initial evaluation
    counts = _build_counts(best_classes, max_n)
    best_L = _compute_L_from_counts(counts, max_n)
    best_counts = counts[:]
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        gap = best_L + 1
        if gap > max_n:
            break
        
        # Try changing residues of existing primes to cover the gap
        for idx in range(len(best_classes)):
            if deadline is not None and time.time() > deadline:
                return best_L, best_classes
            p, old_r = best_classes[idx]
            
            # Only try residue that covers the gap
            new_r = gap % p
            if new_r == old_r:
                continue
            
            # Update counts: remove old residue's coverage, add new
            old_start = old_r if old_r > 0 else p
            for k in range(old_start, max_n + 1, p):
                best_counts[k] -= 1
            
            new_start = new_r if new_r > 0 else p
            for k in range(new_start, max_n + 1, p):
                best_counts[k] += 1
            
            new_L = _compute_L_from_counts(best_counts, max_n)
            
            if new_L > best_L:
                best_L = new_L
                best_classes[idx][1] = new_r
                improved = True
            else:
                # Revert
                for k in range(new_start, max_n + 1, p):
                    best_counts[k] -= 1
                for k in range(old_start, max_n + 1, p):
                    best_counts[k] += 1
        
        # Try adding a new prime if budget allows
        if not improved:
            for p in primes_pool:
                if deadline is not None and time.time() > deadline:
                    return best_L, best_classes
                if any(pp == p for pp, _ in best_classes):
                    continue
                if current_sum + Fraction(1, p) > 1:
                    continue
                
                new_r = gap % p
                # Try adding this prime
                new_start = new_r if new_r > 0 else p
                for k in range(new_start, max_n + 1, p):
                    best_counts[k] += 1
                
                new_L = _compute_L_from_counts(best_counts, max_n)
                
                if new_L > best_L:
                    best_L = new_L
                    best_classes.append([p, new_r])
                    current_sum += Fraction(1, p)
                    improved = True
                else:
                    for k in range(new_start, max_n + 1, p):
                        best_counts[k] -= 1
    
    return best_L, best_classes

def _hill_climb_two_swap(classes, primes_pool, max_n=100000, deadline=None):
    """Try swapping two residues simultaneously."""
    best_classes = [list(c) for c in classes]
    best_L = _compute_L_from_counts(_build_counts(best_classes, max_n), max_n)
    best_counts = _build_counts(best_classes, max_n)
    
    n_classes = len(best_classes)
    
    for i in range(n_classes):
        if deadline is not None and time.time() > deadline:
            return best_L, best_classes
        p1, r1 = best_classes[i]
        for j in range(i + 1, n_classes):
            if deadline is not None and time.time() > deadline:
                return best_L, best_classes
            p2, r2 = best_classes[j]
            
            gap = best_L + 1
            if gap > max_n:
                return best_L, best_classes
            
            # Try setting both to cover the gap
            new_r1 = gap % p1
            new_r2 = gap % p2
            
            if new_r1 == r1 and new_r2 == r2:
                continue
            
            # Update counts
            old_start1 = r1 if r1 > 0 else p1
            old_start2 = r2 if r2 > 0 else p2
            for k in range(old_start1, max_n + 1, p1):
                best_counts[k] -= 1
            for k in range(old_start2, max_n + 1, p2):
                best_counts[k] -= 1
            
            new_start1 = new_r1 if new_r1 > 0 else p1
            new_start2 = new_r2 if new_r2 > 0 else p2
            for k in range(new_start1, max_n + 1, p1):
                best_counts[k] += 1
            for k in range(new_start2, max_n + 1, p2):
                best_counts[k] += 1
            
            new_L = _compute_L_from_counts(best_counts, max_n)
            
            if new_L > best_L:
                best_L = new_L
                best_classes[i][1] = new_r1
                best_classes[j][1] = new_r2
            else:
                # Revert
                for k in range(new_start1, max_n + 1, p1):
                    best_counts[k] -= 1
                for k in range(new_start2, max_n + 1, p2):
                    best_counts[k] -= 1
                for k in range(old_start1, max_n + 1, p1):
                    best_counts[k] += 1
                for k in range(old_start2, max_n + 1, p2):
                    best_counts[k] += 1
    
    return best_L, best_classes

def _multi_start_local_search(primes_pool, max_n=100000, deadline=None, n_starts=5):
    """Multiple random starts with local search."""
    best_L = 0
    best_classes = []
    
    for start_idx in range(n_starts):
        if deadline is not None and time.time() > deadline:
            break
        
        # Create a random initial assignment
        random_primes = sorted(random.sample(primes_pool, min(len(primes_pool), 40)))
        L, classes = _greedy_assign(random_primes, max_n)
        
        # Local search
        L2, classes2 = _local_search_fast(classes, primes_pool, max_n, deadline)
        
        if L2 > best_L:
            best_L = L2
            best_classes = classes2
    
    return best_L, best_classes

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    primes = _sieve(1000)
    max_n = 100000
    
    # Generate candidate prime sets
    candidate_sets = []
    seen_sets = set()
    
    # Seed: primes >= 13
    seed_set = [p for p in primes if p >= 13]
    candidate_sets.append(seed_set)
    seen_sets.add(tuple(sorted(seed_set)))
    
    # Systematic: subsets of small primes + fill with large primes
    small_primes = [2, 3, 5, 7, 11]
    large_primes = [p for p in primes if p > 11]
    
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
    
    # Targeted swaps
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
    
    best_L = 0
    best_classes = []
    
    # First pass: quick greedy on all candidates
    for P in candidate_sets:
        if time.time() > deadline - 30:
            break
        P_sorted = sorted(P)
        L, classes = _greedy_assign(P_sorted, max_n)
        if L > best_L:
            best_L = L
            best_classes = classes
    
    # Second pass: local search on top candidates
    candidate_scores = []
    for P in candidate_sets:
        if time.time() > deadline - 20:
            break
        P_sorted = sorted(P)
        L, classes = _greedy_assign(P_sorted, max_n)
        candidate_scores.append((L, P_sorted, classes))
    
    candidate_scores.sort(key=lambda x: -x[0])
    
    # Local search on top candidates
    for L, P_sorted, classes in candidate_scores[:30]:
        if time.time() > deadline - 15:
            break
        new_L, new_classes = _local_search_fast(classes, primes, max_n, deadline - 5)
        if new_L > best_L:
            best_L = new_L
            best_classes = new_classes
        
        # Also try two-swap hill climb
        if time.time() > deadline - 10:
            break
        new_L2, new_classes2 = _hill_climb_two_swap(new_classes, primes, max_n, deadline - 5)
        if new_L2 > best_L:
            best_L = new_L2
            best_classes = new_classes2
    
    # Multi-start if time remains
    if time.time() < deadline - 10:
        ms_L, ms_classes = _multi_start_local_search(primes, max_n, deadline - 5, n_starts=3)
        if ms_L > best_L:
            best_L = ms_L
            best_classes = ms_classes
    
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
    max_n = 100000
    counts = _build_counts(result, max_n)
    L = _compute_L_from_counts(counts, max_n)
    print(f"L = {L}")
    s = sum(Fraction(1, p) for p, r in result)
    print(f"Sum 1/p = {float(s):.6f}")
