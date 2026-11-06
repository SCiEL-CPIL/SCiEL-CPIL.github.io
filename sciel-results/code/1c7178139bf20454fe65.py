"""Improved program: deterministic construction + local search to maximize disjoint residue classes."""
import math
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pair_disjoint(r1, m1, r2, m2):
    """Two residue classes are disjoint iff gcd(m1,m2) does not divide (r2-r1)."""
    g = _gcd(m1, m2)
    if g == 1:
        return True
    return ((r2 - r1) % g) != 0

def _valid(residue, modulus, existing):
    """Check if (residue, modulus) is disjoint from all existing classes."""
    for (r, m) in existing:
        if not _pair_disjoint(residue, modulus, r, m):
            return False
    return True

def _try_add(existing, modulus):
    """Try to find a valid residue for the given modulus. Returns residue or None."""
    for r in range(modulus):
        if _valid(r, modulus, existing):
            return r
    return None

def _deterministic_base():
    """Build a strong base using prime powers (mutually coprime) + some primes."""
    # All prime powers <= 128 are mutually coprime -> automatically disjoint
    # 2,3,4,5,7,8,9,11,13,16,17,19,23,25,27,29,31,32,37,41,43,47,49,53,59,61,64,67,71,73,79,81,83,89,97,101,103,107,109,113,121,127
    # That's 42 moduli! All pairwise coprime.
    prime_powers = []
    for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127]:
        pk = p
        while pk <= 128:
            prime_powers.append(pk)
            pk *= p
    # Assign residue 0 to all (works since they're pairwise coprime)
    existing = []
    for m in sorted(prime_powers):
        existing.append([0, m])
    return existing

def _local_search(existing, deadline):
    """Try to add more moduli via greedy + local search."""
    used_moduli = set(m for _, m in existing)
    candidates = [m for m in range(2, 129) if m not in used_moduli]
    
    # Sort candidates: prefer those with more "conflicts" first (harder to place)
    # Actually, prefer those that are coprime to more existing moduli (easier)
    # Then try to add them
    added = 0
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try each candidate
        for m in candidates[:]:
            if m in used_moduli:
                continue
            r = _try_add(existing, m)
            if r is not None:
                existing.append([r, m])
                used_moduli.add(m)
                candidates.remove(m)
                added += 1
                changed = True
    return existing, added

def _swap_search(existing, deadline):
    """Try removing one class and adding two others to net gain."""
    for i in range(len(existing)):
        if time.time() > deadline:
            break
        # Try removing existing[i]
        r_i, m_i = existing[i]
        temp = existing[:i] + existing[i+1:]
        used = set(m for _, m in temp)
        # Try to add two moduli not in used
        added_count = 0
        new_classes = []
        for m in range(2, 129):
            if m in used or m == m_i:
                continue
            r = _try_add(temp + new_classes, m)
            if r is not None:
                new_classes.append([r, m])
                used.add(m)
                added_count += 1
                if added_count >= 2:
                    break
        if added_count > 1:
            existing = temp + new_classes
    return existing

def _random_residue_search(existing, deadline, seed=42):
    """Randomly try different residue assignments to free up moduli."""
    rng = random.Random(seed)
    best = existing[:]
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        # Try a random permutation of residues for existing classes
        trial = []
        valid = True
        for i, (r, m) in enumerate(existing):
            # Try a few random residues
            found = False
            for _ in range(5):
                new_r = rng.randint(0, m - 1)
                if _valid(new_r, m, trial):
                    trial.append([new_r, m])
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid and len(trial) == len(existing):
            # Now try to add more
            used = set(m for _, m in trial)
            for m in range(2, 129):
                if m not in used:
                    r = _try_add(trial, m)
                    if r is not None:
                        trial.append([r, m])
                        used.add(m)
            if len(trial) > len(best):
                best = trial[:]
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    # Step 1: Deterministic base (prime powers, all coprime)
    existing = _deterministic_base()
    
    # Step 2: Greedy add remaining moduli
    existing, _ = _local_search(existing, deadline)
    
    # Step 3: Swap search (remove one, add two)
    if time.time() < deadline - 5:
        existing = _swap_search(existing, deadline)
    
    # Step 4: Random residue search
    if time.time() < deadline - 2:
        existing = _random_residue_search(existing, deadline)
    
    # Final greedy pass
    used = set(m for _, m in existing)
    for m in range(2, 129):
        if m not in used:
            r = _try_add(existing, m)
            if r is not None:
                existing.append([r, m])
                used.add(m)
    
    return existing

# EVOLVE-BLOCK-END
