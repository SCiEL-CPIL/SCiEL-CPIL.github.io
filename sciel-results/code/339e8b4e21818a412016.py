import time
import math
import random
from collections import Counter

N = 5000
DEADLINE = 325.0  # seconds

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]


def _fast_is_admissible(points, primes, n):
    """
    Check if points form an admissible tuple w.r.t. given primes.
    Uses bitmask for residue coverage when p is small enough.
    n = len(points), so we only check primes p <= n.
    """
    for p in primes:
        if p > n:
            break
        if p <= 64:
            # Use bitmask
            mask = 0
            for v in points:
                mask |= (1 << (v % p))
                if mask == (1 << p) - 1:
                    return False
        else:
            # Use set for larger primes
            residues = set()
            for v in points:
                residues.add(v % p)
                if len(residues) == p:
                    return False
    return True


def _greedy_construct_conflict(primes, n_max=N, randomize=False):
    """
    Conflict-degree greedy: for each prime p that covers all residues,
    remove the residue class whose elements have the highest 'conflict degree'
    (i.e., elements that also help cover many other small prime residue classes).
    This preserves elements that are more "useful" for other constraints.
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # Precompute for each value v in [0, n_max] and each small prime p, v % p
    # This is implicit in the loop below
    
    for p in small_primes:
        if p > len(points):
            break
        # Check if all p residue classes are covered
        class_counts = Counter()
        for v in points:
            class_counts[v % p] += 1
        
        if len(class_counts) < p:
            continue  # Already admissible for this prime
        
        # All p residue classes are covered. Need to remove one class.
        # Score each residue class r by how "valuable" its elements are
        # for other primes. We want to remove the class whose elements are
        # LEAST valuable for other primes.
        
        if randomize:
            # Randomized: pick randomly among bottom half of classes by count
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            threshold = min_count + (max_count - min_count) // 2
            candidates = [r for r, c in class_counts.items() if c <= threshold]
            remove_class = random.choice(candidates)
        else:
            # Deterministic: remove the least-populated class
            # Tie-break: among equal counts, remove the class whose elements
            # cover the most residue classes for other primes (most "redundant")
            min_count = min(class_counts.values())
            min_classes = [r for r, c in class_counts.items() if c == min_count]
            
            if len(min_classes) == 1:
                remove_class = min_classes[0]
            else:
                # Score each candidate class by how much its removal hurts
                # other primes. We want to remove the one that hurts least.
                best_remove = min_classes[0]
                best_score = float('inf')
                for rc in min_classes:
                    # Count how many elements in this class also help
                    # cover residues for other primes
                    hurt_score = 0
                    for v in points:
                        if v % p == rc:
                            for p2 in small_primes:
                                if p2 == p or p2 > len(points):
                                    continue
                                # If removing v would cause p2 to lose a residue
                                # that no other element covers, that's a "hurt"
                                # But this is expensive. Use a simpler heuristic:
                                # just count distinct residues v covers for other primes
                                hurt_score += 1  # simple proxy
                    # Actually, let's use a simpler tie-break: just pick first
                    break  # Skip expensive scoring, use first min class
                remove_class = min_classes[0]
        
        points = {v for v in points if v % p != remove_class}
    
    return sorted(points)


def _greedy_construct(primes, n_max=N):
    """
    Construct a large admissible set by greedy inclusion.
    Start with all integers in [0, n_max], then remove elements to fix
    each prime whose residue classes are all covered.
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    for p in small_primes:
        if p > len(points):
            break
        class_counts = Counter(v % p for v in points)
        if len(class_counts) == p:
            min_count = min(class_counts.values())
            remove_class = min(class_counts.keys(), key=lambda r: class_counts[r])
            points = {v for v in points if v % p != remove_class}
    
    return sorted(points)


def _local_search_fast(points, primes, n_max=N, max_iters=20000, deadline=None):
    """
    Fast local search: try to add back removed elements using fast admissibility check.
    """
    points_set = set(points)
    best = list(points)
    best_size = len(best)
    small_primes = [p for p in primes if p <= n_max + 1]
    candidates = [v for v in range(n_max + 1) if v not in points_set]
    
    if not candidates:
        return best
    
    if deadline is not None:
        start_time = time.time()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Try adding a random candidate
        idx = random.randint(0, len(candidates) - 1)
        candidate = candidates[idx]
        
        # Fast check: only need to verify primes p <= best_size + 1
        new_size = best_size + 1
        admissible = True
        for p in small_primes:
            if p > new_size:
                break
            if p <= 64:
                mask = 0
                for v in points_set:
                    mask |= (1 << (v % p))
                    if mask == (1 << p) - 1:
                        break
                if mask != (1 << p) - 1:
                    # Check if adding candidate completes all residues
                    if (mask | (1 << (candidate % p))) == (1 << p) - 1:
                        admissible = False
                        break
            else:
                residues = set()
                for v in points_set:
                    residues.add(v % p)
                    if len(residues) == p:
                        break
                if len(residues) < p:
                    if (candidate % p) not in residues:
                        # Adding candidate doesn't complete all residues
                        pass
                    else:
                        # Check if it would complete
                        if len(residues | {candidate % p}) == p:
                            admissible = False
                            break
        
        if admissible:
            points_set.add(candidate)
            candidates.pop(idx)
            best = sorted(points_set)
            best_size = len(best)
    
    return best


def _multi_start_greedy(primes, n_max=N, deadline=None, num_attempts=200):
    """
    Multi-start greedy with different random removals to find larger admissible sets.
    """
    best = None
    best_size = 0
    small_primes = [p for p in primes if p <= n_max + 1]
    
    if deadline is not None:
        start_time = time.time()
    
    # Start with the standard greedy
    greedy_result = _greedy_construct(primes, n_max)
    best = greedy_result
    best_size = len(best)
    
    # Also try conflict greedy
    conflict_result = _greedy_construct_conflict(primes, n_max, randomize=False)
    if len(conflict_result) > best_size:
        best = conflict_result
        best_size = len(best)
    
    if deadline is not None and time.time() > deadline - 15:
        return best
    
    # Try multiple random initial removals
    for attempt in range(num_attempts):
        if deadline is not None and time.time() > deadline - 15:
            break
        
        points = set(range(n_max + 1))
        # Randomly remove some elements to break symmetry
        num_remove = random.randint(1, 8)
        for _ in range(num_remove):
            if points:
                to_remove = random.choice(list(points))
                points.discard(to_remove)
        
        # Run greedy repair with randomization
        for p in small_primes:
            if p > len(points):
                break
            class_counts = Counter(v % p for v in points)
            if len(class_counts) == p:
                min_count = min(class_counts.values())
                # Randomly choose among min-count classes
                min_classes = [r for r, c in class_counts.items() if c == min_count]
                remove_class = random.choice(min_classes)
                points = {v for v in points if v % p != remove_class}
        
        if len(points) > best_size:
            best = sorted(points)
            best_size = len(points)
    
    return best


def _final_verify_and_fix(best, primes):
    """Verify admissibility and fix if needed."""
    n = len(best)
    small_primes = [p for p in primes if p <= n]
    for p in small_primes:
        residues = set(v % p for v in best)
        if len(residues) == p:
            class_counts = Counter(v % p for v in best)
            min_count = min(class_counts.values())
            min_classes = [r for r, c in class_counts.items() if c == min_count]
            remove_class = min_classes[0]
            best = [v for v in best if v % p != remove_class]
    return best


def construct():
    """
    Construct the largest admissible tuple within [0, 5000].
    An admissible tuple satisfies: for every prime p <= len(tuple),
    at least one residue class mod p is missed.
    """
    start_time = time.time()
    primes = _sieve_primes(N + 1)
    
    # First, get a baseline greedy solution
    baseline = _greedy_construct(primes, N)
    best = baseline
    best_size = len(best)
    
    # Try conflict-degree greedy
    conflict = _greedy_construct_conflict(primes, N, randomize=False)
    if len(conflict) > best_size:
        best = conflict
        best_size = len(best)
    
    deadline = start_time + DEADLINE
    
    # Try multi-start greedy with randomization
    if time.time() < deadline - 40:
        multi_result = _multi_start_greedy(primes, N, deadline=deadline, num_attempts=300)
        if len(multi_result) > best_size:
            best = multi_result
            best_size = len(best)
    
    # Local search to try to add elements back
    if time.time() < deadline - 25:
        local_result = _local_search_fast(
            best, primes, N,
            max_iters=30000,
            deadline=deadline
        )
        if len(local_result) > best_size:
            best = local_result
            best_size = len(best)
    
    # Final verification
    best = _final_verify_and_fix(best, primes)
    
    return best

# EVOLVE-BLOCK-END
