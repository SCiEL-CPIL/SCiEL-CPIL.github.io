import time
import random
import math
import numpy as np

N = 5000
MAX_TIME = 290.0  # Safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _initial_greedy():
    """Fast greedy to get a good starting point for SA."""
    S = list(range(N + 1))
    while True:
        len_S = len(S)
        viol_p = None
        for p in PRIMES:
            if p > len_S:
                break
            residues = set(x % p for x in S)
            if len(residues) == p:
                viol_p = p
                break
        if viol_p is None:
            break
        p = viol_p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
        min_count = min(counts)
        r_to_remove = min(r for r in range(p) if counts[r] == min_count)
        S = [x for x in S if x % p != r_to_remove]
    return S

def _count_violations(present, primes, n_elements):
    """Count how many primes p <= n_elements are violated (all residues present)."""
    violations = 0
    for p in primes:
        if p > n_elements:
            break
        seen = [0] * p
        for x in present:
            seen[x % p] = 1
        if all(seen):
            violations += 1
    return violations

def _simulate_annealing(start_set, primes, max_time):
    """
    Simulated annealing over element inclusion.
    State: set of included elements.
    Moves: add a random excluded element, or remove a random included element.
    Energy: number of violated primes (we want 0).
    Objective: maximize |S| subject to 0 violations.
    """
    start_time = time.time()
    
    # Work with a list for efficiency; maintain a set for O(1) membership
    included = list(start_set)
    included_set = set(included)
    excluded = [x for x in range(N + 1) if x not in included_set]
    excluded_set = set(excluded)
    
    # Precompute residue tables for speed
    # For each prime p, precompute x % p for all x
    # But that's memory heavy for large p. Instead, compute on the fly but cache per-prime.
    
    # Initial energy
    n_elem = len(included)
    energy = _count_violations(included, primes, n_elem)
    
    # We track the best valid solution found
    best_included = list(included) if energy == 0 else []
    best_size = len(included) if energy == 0 else 0
    
    # SA parameters
    T0 = 5.0
    Tend = 0.01
    total_steps = 0
    
    # Precompute which primes are relevant
    # For a set of size k, only primes p <= k matter
    # We'll maintain the list of active primes
    
    # For efficiency, maintain residue counts per prime
    # residue_counts[p][r] = number of elements in included with x % p == r
    # This is expensive to maintain for all primes, so let's use a different approach:
    # Only track violations incrementally.
    
    # Actually, let's use a smarter approach:
    # Maintain for each prime p <= N: a bitmask or count array of residues present.
    # When we add/remove x, update only the primes that x affects.
    
    # Let's precompute for each x, the list of (p, x%p) for all primes p <= N
    # That's too much memory. Instead, for each x, compute x % p on the fly for active primes.
    
    # Alternative: since N=5000, primes up to 5000 are about 669 primes.
    # For each move, we need to check if adding/removing x changes any violation.
    # A prime p is violated iff all p residues are present.
    # Adding x can only turn a non-violated prime into violated if x fills the last missing residue.
    # Removing x can only turn a violated prime into non-violated if x was the only element in that residue.
    
    # Let's maintain: for each prime p (up to max relevant), residue_count[p][r]
    # and residue_present[p][r] = residue_count[p][r] > 0
    # and violated[p] = all(residue_present[p])
    
    max_p = N  # primes up to N
    # Only primes <= len(included) can be violated, but let's track all for simplicity
    # Actually, let's track primes up to N but only check up to len(included)
    
    # Memory: 669 primes * up to 5000 residues each = too much for large p
    # For p > 100, we have at most 5000 elements, so at most 5000 distinct residues (but p > 5000 means no violation possible)
    # Actually for p > len(included), violation is impossible.
    # So we only need to track primes p <= len(included).
    # But len(included) changes... 
    
    # Let's use a simpler approach: recompute energy from scratch but efficiently.
    # For N=5000, checking all primes up to 5000 with 5000 elements each is ~669*5000 = 3.3M ops per check.
    # That's too slow for many steps.
    
    # Better: maintain per-prime residue counts incrementally.
    # For primes p where p <= N, maintain an array of counts.
    # Total memory: sum of p for all primes p <= N ≈ sum of first 669 primes ≈ 669^2/2 * ~1.5 ≈ ~330K. Fine.
    
    # Let's precompute primes and their indices
    prime_indices = {p: i for i, p in enumerate(PRIMES)}
    
    # residue_counts: dict from p to list of counts
    residue_counts = {}
    # violated: dict from p to bool
    violated = {}
    
    # Initialize
    for p in PRIMES:
        if p > N:
            break
        counts = [0] * p
        for x in included:
            counts[x % p] += 1
        residue_counts[p] = counts
        violated[p] = all(c > 0 for c in counts)
    
    # Count total violations
    def _total_violations():
        n = len(included)
        cnt = 0
        for p in PRIMES:
            if p > n:
                break
            if violated[p]:
                cnt += 1
        return cnt
    
    energy = _total_violations()
    if energy == 0:
        best_included = list(included)
        best_size = len(included)
    
    rng = random.Random(12345)
    
    while time.time() - start_time < max_time:
        # Pick a move: add or remove
        if rng.random() < 0.5 and excluded:
            # Add a random excluded element
            idx = rng.randrange(len(excluded))
            x = excluded[idx]
            # Update residue counts
            for p in PRIMES:
                if p > N:
                    break
                r = x % p
                old_violated = violated[p]
                old_count = residue_counts[p][r]
                residue_counts[p][r] = old_count + 1
                new_violated = all(c > 0 for c in residue_counts[p])
                violated[p] = new_violated
                if old_violated and not new_violated:
                    # Can't happen when adding
                    pass
                elif not old_violated and new_violated:
                    energy += 1
                elif old_violated and new_violated:
                    pass
                else:
                    # was not violated, still not violated
                    pass
            included.append(x)
            included_set.add(x)
            excluded.pop(idx)
            excluded_set.discard(x)
        elif included:
            # Remove a random included element
            idx = rng.randrange(len(included))
            x = included[idx]
            # Update residue counts
            for p in PRIMES:
                if p > N:
                    break
                r = x % p
                old_violated = violated[p]
                old_count = residue_counts[p][r]
                residue_counts[p][r] = old_count - 1
                new_violated = all(c > 0 for c in residue_counts[p])
                violated[p] = new_violated
                if old_violated and not new_violated:
                    energy -= 1
                elif not old_violated and new_violated:
                    energy += 1
            included.pop(idx)
            included_set.discard(x)
            excluded.append(x)
            excluded_set.add(x)
        
        n_elem = len(included)
        # Recompute energy properly (the incremental might have errors for primes > n_elem)
        # Actually, for primes p > n_elem, they can't be violated. Let's just recompute.
        # For speed, let's trust the incremental for now but verify periodically.
        
        # SA acceptance
        # We want to minimize energy (violations) and maximize size.
        # Use a combined objective: energy - alpha * size
        # Or: accept if energy decreases, or if energy same and size increases, or probabilistically.
        
        # Simpler: use temperature-based acceptance on energy
        if energy == 0:
            # Valid solution, track best
            if n_elem > best_size:
                best_size = n_elem
                best_included = list(included)
            # Still continue to try to grow
            # Accept with high probability
            if rng.random() < 0.9:
                pass  # accept
            else:
                # Reject: undo the move
                if included:
                    # This is getting complex. Let's simplify.
                    pass
        else:
            # Invalid, try to accept based on temperature
            # T decreases over time
            frac = min(1.0, total_steps / max(1, int(max_time * 100)))
            T = T0 * (Tend / T0) ** frac
            # Accept with probability exp(-energy/T)
            if rng.random() < math.exp(-energy / max(T, 1e-10)):
                pass  # accept
            else:
                # Reject: undo
                # This is complex. Let's use a different approach.
                pass
        
        total_steps += 1
    
    return best_included

def _sa_with_proper_undo(start_set, primes, max_time):
    """
    Simulated annealing with proper undo mechanism.
    State: list of included elements + set for membership.
    We store the full state and undo by re-adding/removing.
    """
    start_time = time.time()
    rng = random.Random(54321)
    
    included = list(start_set)
    included_set = set(included)
    excluded = [x for x in range(N + 1) if x not in included_set]
    
    # Precompute: for each x in [0, N], and for each prime p, x % p
    # Too much memory. Instead, compute on the fly.
    
    # Maintain per-prime residue presence (boolean) and count
    # Only for primes p <= N
    active_primes = [p for p in PRIMES if p <= N]
    
    residue_count = {}
    for p in active_primes:
        rc = [0] * p
        for x in included:
            rc[x % p] += 1
        residue_count[p] = rc
    
    def _is_violated(p):
        rc = residue_count[p]
        return all(c > 0 for c in rc)
    
    def _energy():
        n = len(included)
        cnt = 0
        for p in active_primes:
            if p > n:
                break
            if _is_violated(p):
                cnt += 1
        return cnt
    
    energy = _energy()
    best_included = list(included) if energy == 0 else []
    best_size = len(included) if energy == 0 else 0
    
    T0 = 3.0
    Tend = 0.005
    step = 0
    total_planned = int(max_time * 500)  # estimate ~500 steps/sec
    
    while time.time() - start_time < max_time:
        step += 1
        frac = step / max(1, total_planned)
        T = T0 * (Tend / T0) ** frac
        
        # Choose move
        do_add = rng.random() < 0.55  # slightly favor adding to grow
        if do_add and excluded:
            idx = rng.randrange(len(excluded))
            x = excluded[idx]
            # Apply: add x
            for p in active_primes:
                residue_count[p][x % p] += 1
            included.append(x)
            included_set.add(x)
            excluded.pop(idx)
            # Compute delta energy
            new_energy = _energy()
            delta_e = new_energy - energy
            # Accept?
            if delta_e <= 0 or (T > 1e-10 and rng.random() < math.exp(-delta_e / T)):
                energy = new_energy
                if energy == 0 and len(included) > best_size:
                    best_size = len(included)
                    best_included = list(included)
            else:
                # Undo
                for p in active_primes:
                    residue_count[p][x % p] -= 1
                included.pop()
                included_set.discard(x)
                excluded.append(x)
        elif not do_add and included:
            idx = rng.randrange(len(included))
            x = included[idx]
            # Apply: remove x
            for p in active_primes:
                residue_count[p][x % p] -= 1
            included.pop(idx)
            included_set.discard(x)
            excluded.append(x)
            new_energy = _energy()
            delta_e = new_energy - energy
            if delta_e <= 0 or (T > 1e-10 and rng.random() < math.exp(-delta_e / T)):
                energy = new_energy
                if energy == 0 and len(included) > best_size:
                    best_size = len(included)
                    best_included = list(included)
            else:
                # Undo
                for p in active_primes:
                    residue_count[p][x % p] += 1
                included.insert(idx, x)
                included_set.add(x)
                excluded.pop()
    
    return best_included

def _local_improve_fast(S, max_time):
    """
    Fast local improvement: try to add back excluded elements.
    Use incremental violation checking.
    """
    start_time = time.time()
    included = list(S)
    included_set = set(included)
    excluded = [x for x in range(N + 1) if x not in included_set]
    rng = random.Random(999)
    rng.shuffle(excluded)
    
    active_primes = [p for p in PRIMES if p <= N]
    
    # Build residue counts
    residue_count = {}
    for p in active_primes:
        rc = [0] * p
        for x in included:
            rc[x % p] += 1
        residue_count[p] = rc
    
    def _energy():
        n = len(included)
        cnt = 0
        for p in active_primes:
            if p > n:
                break
            if all(c > 0 for c in residue_count[p]):
                cnt += 1
        return cnt
    
    energy = _energy()
    assert energy == 0, f"Initial state should be admissible, got energy {energy}"
    
    for x in excluded:
        if time.time() - start_time > max_time:
            break
        # Check if adding x keeps admissibility
        n_new = len(included) + 1
        violated = False
        for p in active_primes:
            if p > n_new:
                break
            r = x % p
            # After adding x, residue r will have count >= 1 (it already has >= 0, and we add 1)
            # The prime p becomes violated iff all residues have count > 0 after adding.
            # Before adding, if residue_count[p][r] == 0, then adding x makes it 1.
            # Check if all other residues already have count > 0.
            if residue_count[p][r] == 0:
                # This residue was missing. Check if all others are present.
                all_others = all(residue_count[p][r2] > 0 for r2 in range(p) if r2 != r)
                if all_others:
                    violated = True
                    break
            # If residue_count[p][r] > 0, adding x doesn't change whether p is violated
            # (it was already violated or not, and adding to an already-present residue doesn't change that)
            # Wait: if p was already violated (all residues present), it stays violated.
            # If p was not violated, and residue_count[p][r] > 0, then some other residue is missing,
            # and adding x to residue r doesn't fill that other missing residue. So still not violated.
            # So only the case residue_count[p][r] == 0 matters.
        
        if not violated:
            # Add x
            for p in active_primes:
                residue_count[p][x % p] += 1
            included.append(x)
            included_set.add(x)
    
    return included

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy initial solution
    S_greedy = _initial_greedy()
    best = list(S_greedy)
    best_size = len(best)
    
    # Phase 2: Local improvement on greedy
    time_left = MAX_TIME - (time.time() - start_time)
    if time_left > 10:
        S_improved = _local_improve_fast(S_greedy, max_time=min(30, time_left - 5))
        if len(S_improved) > best_size:
            best = S_improved
            best_size = len(S_improved)
    
    # Phase 3: Simulated annealing from the improved solution
    time_left = MAX_TIME - (time.time() - start_time)
    if time_left > 20:
        S_sa = _sa_with_proper_undo(best, PRIMES, max_time=min(time_left - 5, 200))
        if len(S_sa) > best_size:
            best = S_sa
            best_size = len(S_sa)
    
    # Phase 4: More local improvement
    time_left = MAX_TIME - (time.time() - start_time)
    if time_left > 10:
        S_final = _local_improve_fast(best, max_time=min(time_left - 5, 60))
        if len(S_final) > best_size:
            best = S_final
            best_size = len(S_final)
    
    # Phase 5: If time remains, try SA again from best
    time_left = MAX_TIME - (time.time() - start_time)
    if time_left > 30:
        S_sa2 = _sa_with_proper_undo(best, PRIMES, max_time=min(time_left - 5, 150))
        if len(S_sa2) > best_size:
            best = S_sa2
            best_size = len(S_sa2)
    
    return best

# Precompute primes at module level for speed
PRIMES = _primes(N)
active_primes_global = [p for p in PRIMES if p <= N]
