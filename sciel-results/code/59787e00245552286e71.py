"""Construct the largest admissible tuple within [0, 5000]."""
import time
import random
import itertools
import numpy as np

N = 5000
DEADLINE = 330.0
SAFETY_MARGIN = 5.0

def _sieve_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i, v in enumerate(sieve) if v]

def _is_admissible(points, primes):
    n = len(points)
    for p in primes:
        if p > n:
            break
        covered = set(v % p for v in points)
        if len(covered) == p:
            return False
    return True

def _compute_residue_state(points, primes):
    """Compute covered residue classes for each prime."""
    n = len(points)
    state = {}
    for p in primes:
        if p > n:
            break
        covered = set(v % p for v in points)
        state[p] = covered
    return state

def _can_add_fast(candidate, residue_state, primes):
    """Check if adding candidate maintains admissibility using precomputed state."""
    new_n = len(residue_state)  # approximation; actual check below
    # We need to check all primes p <= new_n
    # But residue_state only has primes <= current_n
    # For simplicity, do a full check for small primes
    return True  # placeholder, replaced by actual check in caller

def _greedy_construct(missed_residues, primes, start_time, max_time):
    """
    Greedily construct an admissible tuple given fixed missed residue classes
    for the first k primes. missed_residues is a dict {prime: missed_class}.
    """
    point_set = set()
    points = []
    
    # Precompute: for each candidate value, check if it's compatible with
    # the fixed missed residue classes
    # A value v is compatible if for each (p, r) in missed_residues, v % p != r
    # (we want to MISS residue class r mod p, so we should NOT include values
    #  that would cover all classes - but actually we need to be more careful)
    
    # Actually, the constraint is: for each prime p <= len(tuple), we must miss
    # at least one residue class. If we fix that we miss class r_p mod p, then
    # we should exclude all values v where v % p == r_p for those small primes.
    # But that's too restrictive. Instead, we just need to ensure that at the end,
    # for each p <= len(tuple), not all p classes are covered.
    
    # Better approach: build greedily, checking admissibility at each step.
    # For speed, track residue counts incrementally.
    
    # For the fixed small primes, we'll bias toward avoiding the missed classes
    # to make it easier to maintain admissibility.
    
    # Start with a seed: include 1 and large primes
    sqrt_n = int(N ** 0.5) + 1
    for p in primes:
        if p > sqrt_n and p <= N:
            points.append(p)
            point_set.add(p)
    if 1 not in point_set:
        points.append(1)
        point_set.add(1)
    
    # Sort candidates: prefer those that avoid the fixed missed classes for small primes
    # This biases the greedy construction toward maintaining admissibility
    candidates = [v for v in range(N + 1) if v not in point_set]
    
    # Score each candidate: lower score = better (prefer values that avoid
    # the fixed missed residue classes for small primes)
    def _candidate_score(v):
        score = 0
        for p, r in missed_residues.items():
            if v % p == r:
                score += p  # penalize values in the missed class
        return score
    
    candidates.sort(key=_candidate_score)
    
    # Track residue coverage for fast admissibility checks
    # Only track primes up to a reasonable bound
    max_prime_track = 50  # track primes up to 50 for speed
    tracked_primes = [p for p in primes if p <= max_prime_track]
    
    # residue_counts[p] = list of counts for each residue class mod p
    residue_counts = {}
    for p in tracked_primes:
        counts = [0] * p
        for v in points:
            counts[v % p] += 1
        residue_counts[p] = counts
    
    def _check_admissible_incremental():
        """Check if current set is admissible using tracked primes + full check for larger."""
        n = len(points)
        # Check tracked primes
        for p in tracked_primes:
            if p > n:
                break
            counts = residue_counts[p]
            if all(c > 0 for c in counts):
                return False
        # Check untracked primes (p > max_prime_track and p <= n)
        for p in primes:
            if p > max_prime_track and p <= n:
                covered = set(v % p for v in points)
                if len(covered) == p:
                    return False
        return True
    
    # Greedily add candidates
    for v in candidates:
        if time.time() - start_time > max_time:
            break
        if v in point_set:
            continue
        
        # Check if adding v maintains admissibility
        # Tentatively add
        point_set.add(v)
        points.append(v)
        n_new = len(points)
        
        # Update residue counts for tracked primes
        for p in tracked_primes:
            residue_counts[p][v % p] += 1
        
        # Check admissibility
        if _check_admissible_incremental():
            # Keep v
            continue
        else:
            # Remove v
            points.pop()
            point_set.discard(v)
            for p in tracked_primes:
                residue_counts[p][v % p] -= 1
    
    # Repair phase: for any prime where all classes are covered, remove from least-populated
    changed = True
    while changed:
        if time.time() - start_time > max_time:
            break
        changed = False
        n = len(points)
        for p in primes:
            if p > n:
                break
            # Count residue classes
            class_counts = {}
            for v in points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            if len(class_counts) == p:
                min_class = min(class_counts, key=class_counts.get)
                candidates_to_remove = [v for v in points if v % p == min_class]
                candidates_to_remove.sort(reverse=True)
                to_remove = candidates_to_remove[0]
                points.remove(to_remove)
                point_set.discard(to_remove)
                # Update residue counts
                for pp in tracked_primes:
                    if pp == p:
                        residue_counts[pp][to_remove % pp] -= 1
                changed = True
                break
    
    return points

def _hill_climb(points, primes, start_time, max_time, max_steps=10000):
    """
    Hill-climb: try to add random points that maintain admissibility.
    """
    point_set = set(points)
    best = list(points)
    best_len = len(best)
    
    # Precompute residue state
    n = len(points)
    residue_state = {}
    for p in primes:
        if p > n:
            break
        covered = set(v % p for v in points)
        residue_state[p] = covered
    
    step = 0
    while step < max_steps and time.time() - start_time < max_time:
        step += 1
        # Pick a random candidate
        v = random.randint(0, N)
        if v in point_set:
            continue
        
        # Check if adding v maintains admissibility
        new_n = len(points) + 1
        valid = True
        for p in primes:
            if p > new_n:
                break
            if p in residue_state:
                covered = residue_state[p]
            else:
                covered = set(v2 % p for v2 in points)
                residue_state[p] = covered
            if len(covered) == p - 1 and v % p not in covered:
                # Adding v would cover all p classes
                valid = False
                break
            elif len(covered) == p:
                valid = False
                break
        
        if valid:
            points.append(v)
            point_set.add(v)
            # Update residue state
            for p in primes:
                if p <= new_n:
                    if p not in residue_state:
                        residue_state[p] = set(v2 % p for v2 in points)
                    else:
                        residue_state[p].add(v % p)
            
            if len(points) > best_len:
                best = list(points)
                best_len = len(best)
    
    return best

def construct():
    start_time = time.time()
    max_time = DEADLINE - SAFETY_MARGIN
    
    primes = _sieve_primes(N + 1)
    
    # Branch over missed residue classes for first 5 primes: 2, 3, 5, 7, 11
    small_primes = [2, 3, 5, 7, 11]
    
    best_points = []
    best_len = 0
    
    # Try all combinations of missed residues for small primes
    # That's 2*3*5*7*11 = 2310 combinations
    # But we'll sample a subset if time is tight
    combos = list(itertools.product(*[range(p) for p in small_primes]))
    random.seed(42)
    random.shuffle(combos)
    
    # Time budget per combo
    n_combos = len(combos)
    time_per_combo = max(0.5, (max_time - 10) / n_combos)  # leave 10s for polish
    
    for combo in combos:
        if time.time() - start_time > max_time - 5:
            break
        
        missed_residues = dict(zip(small_primes, combo))
        
        # Greedy construction
        points = _greedy_construct(missed_residues, primes, start_time, max_time)
        
        if len(points) > best_len:
            best_points = list(points)
            best_len = len(points)
        
        # Hill-climb polish
        points = _hill_climb(points, primes, start_time, max_time, max_steps=2000)
        
        if len(points) > best_len:
            best_points = list(points)
            best_len = len(points)
    
    # Final verification
    if not _is_admissible(best_points, primes):
        # Fallback
        best_points = [1] + [p for p in primes if p * p > N and p <= N]
    
    return best_points
