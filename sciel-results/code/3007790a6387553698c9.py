"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import deque

MAX_MOD = 128
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _disjoint(r1, m1, r2, m2):
    """Two residue classes are disjoint iff gcd(m1,m2) does not divide (r2-r1)."""
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _is_valid_solution(pairs):
    n = len(pairs)
    for i in range(n):
        r1, m1 = pairs[i]
        for j in range(i+1, n):
            r2, m2 = pairs[j]
            if not _disjoint(r1, m1, r2, m2):
                return False
    return True

def _find_residue(m, existing_pairs):
    """Find a residue r for modulus m that is disjoint from all existing pairs."""
    for r in range(m):
        ok = True
        for (r2, m2) in existing_pairs:
            if not _disjoint(r, m, r2, m2):
                ok = False
                break
        if ok:
            return r
    return None

def _greedy_build(moduli_order):
    pairs = []
    for m in moduli_order:
        r = _find_residue(m, pairs)
        if r is not None:
            pairs.append([r, m])
    return pairs

def _ac3_prune(moduli, residues_map, all_moduli_set):
    """
    Apply AC-3 arc consistency to prune residue domains.
    residues_map: dict mapping modulus -> set of allowed residues.
    Returns True if no domain became empty.
    """
    # Build constraint graph: for each pair of moduli (m1, m2) with gcd > 1,
    # we have a constraint. If gcd == 1, they are always disjoint (since (r2-r1)%1==0 always,
    # wait. gcd(m1,m2)=1 => (r2-r1)%1 == 0 is always true. So _disjoint returns False?
    # Let's re-read: "disjoint iff gcd(m1,m2) does not divide (r2-r1)"
    # If g=1, 1 divides everything. So (r2-r1)%1 == 0. So _disjoint returns False.
    # This means coprime moduli CANNOT coexist?
    # Let's check the task description: "Every two infinite residue classes must be disjoint".
    # If m1=2, m2=3. r1=0, r2=0. Class 1: 0, 2, 4... Class 2: 0, 3, 6...
    # Intersection: 0, 6, 12... They are NOT disjoint.
    # So yes, coprime moduli cannot coexist if we want disjoint classes.
    # This implies we can only pick moduli from a single "chain" of divisibility or similar.
    # Actually, if m1 | m2, we can pick residues such that they are disjoint.
    # If gcd(m1, m2) = g > 1, we need (r2-r1) % g != 0.
    
    # The constraint is: for m1, m2, if g = gcd(m1, m2) > 1, then r2 != r1 (mod g).
    # If g == 1, then r2 != r1 (mod 1) is impossible (since mod 1 is always 0).
    # So ANY pair of coprime moduli violates the disjointness condition.
    # Therefore, all selected moduli must be pairwise non-coprime?
    # No, that's not right.
    # Let's re-verify the disjointness condition.
    # Sets: S1 = {x | x % m1 == r1}, S2 = {x | x % m2 == r2}.
    # They are disjoint iff the system x % m1 == r1 AND x % m2 == r2 has NO solution.
    # By CRT, a solution exists iff (r2 - r1) % gcd(m1, m2) == 0.
    # So they are disjoint iff (r2 - r1) % gcd(m1, m2) != 0.
    # If gcd(m1, m2) == 1, then (r2 - r1) % 1 == 0 is ALWAYS true.
    # So a solution ALWAYS exists.
    # So coprime moduli ALWAYS intersect.
    # Thus, we CANNOT have any two coprime moduli in our set.
    # This means all moduli must share a common factor? Or at least pairwise gcd > 1.
    # If we pick all even numbers, they are pairwise non-coprime (gcd >= 2).
    # If we pick 2, 3, 5... 2 and 3 are coprime -> intersect.
    # So the set of moduli must be such that no two are coprime.
    # The largest such set is likely all even numbers, or all multiples of 3, etc.
    # Even numbers: 2, 4, ..., 128. Count = 64.
    # Multiples of 3: 3, 6, ..., 126. Count = 42.
    # So we should focus on even moduli primarily.
    
    # AC-3 implementation for the selected moduli set.
    # Queue of arcs (i, j) where i, j are indices in the moduli list.
    # We only care about moduli in the current candidate set.
    
    # For efficiency, we only prune if the set of moduli is fixed.
    # But here we are building the set.
    # Let's use AC-3 as a filter for a given subset of moduli.
    
    # This function is not directly used in the main loop below in its current form,
    # but the logic informs the strategy: prefer moduli with high gcd overlap.
    pass

def _local_search(pairs, rng, time_limit):
    """
    Local search to improve the solution.
    Moves:
    1. Add a new modulus (if possible).
    2. Change residue of an existing modulus (to allow adding others).
    3. Remove a modulus (if it helps add more later - rare).
    """
    current_pairs = [p[:] for p in pairs]
    current_size = len(current_pairs)
    best_pairs = [p[:] for p in pairs]
    best_size = current_size
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcds for speed?
    # gcd is fast enough.
    
    def _try_add(m, ps):
        r = _find_residue(m, ps)
        if r is not None:
            return [p[:] for p in ps] + [[r, m]]
        return None
    
    def _try_change(idx, ps):
        r_old, m = ps[idx]
        others = [p for i, p in enumerate(ps) if i != idx]
        # Try random residues for m
        residues = list(range(m))
        rng.shuffle(residues)
        for r in residues:
            if r == r_old:
                continue
            ok = True
            for (r2, m2) in others:
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                new_ps = [p[:] for p in ps]
                new_ps[idx] = [r, m]
                return new_ps
        return None

    while time.time() < time_limit:
        current_mods = [m for _, m in current_pairs]
        available = [m for m in all_moduli if m not in current_mods]
        
        # Prioritize even moduli for adding
        # Since coprime moduli are forbidden, we should stick to a "family"
        # e.g., all even, or all multiples of 3.
        # The current best likely contains mostly evens.
        
        if not available:
            break
            
        # Weighted random choice from available
        # Bias towards even numbers
        evens_avail = [m for m in available if m % 2 == 0]
        odds_avail = [m for m in available if m % 2 != 0]
        
        if evens_avail and (not odds_avail or rng.random() < 0.8):
            m = rng.choice(evens_avail)
        else:
            m = rng.choice(available)
            
        new_pairs = _try_add(m, current_pairs)
        if new_pairs is not None:
            current_pairs = new_pairs
            current_size = len(current_pairs)
            if current_size > best_size:
                best_size = current_size
                best_pairs = [p[:] for p in current_pairs]
        
        # Try to change a residue to enable future additions
        if current_pairs and rng.random() < 0.3:
            idx = rng.randrange(len(current_pairs))
            new_ps = _try_change(idx, current_pairs)
            if new_ps is not None:
                current_pairs = new_ps
                # Size doesn't change, but state does.
                
    return best_pairs

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    all_moduli = list(range(2, MAX_MOD + 1))
    
    # Strategy:
    # 1. Since coprime moduli cannot coexist, we should focus on a single "family".
    # The largest family is even numbers (64 items).
    # Next is multiples of 3 (42 items).
    # We can try to build a set of evens, then see if we can add any odds
    # that are non-coprime with all evens in the set.
    # An odd number m is non-coprime with an even number e if gcd(m, e) > 1.
    # Since m is odd, gcd(m, e) must be an odd factor of e.
    # So e must be a multiple of some odd factor of m.
    # If we pick all evens, e.g., 2, 4, 6, 8...
    # 6 is in the set. 6 = 2*3.
    # Can we add 9? gcd(9, 2)=1 -> No.
    # So 9 cannot coexist with 2.
    # Thus, if 2 is in the set, NO odd number can be added (since all odds are coprime to 2).
    # So if we include 2, we are restricted to even moduli only.
    # If we EXCLUDE 2, can we add odds?
    # If we exclude 2, we can use 4, 6, 8, ...
    # Can we add 9?
    # 9 is coprime to 4 (gcd 1). No.
    # 9 is coprime to 8 (gcd 1). No.
    # 9 shares factor 3 with 6, 12, 18...
    # But 9 is coprime to 4. So 9 cannot coexist with 4.
    # If we exclude 4 and 2?
    # Use 6, 8, 10, 12...
    # 9 vs 8: gcd 1. No.
    # It seems hard to mix families.
    # The "even" family is the largest.
    # Let's try to maximize the even subset.
    
    # Greedy on evens
    evens = [m for m in all_moduli if m % 2 == 0]
    # Sort by descending value? Or ascending?
    # Larger moduli have more residues, so easier to place.
    order1 = sorted(evens, reverse=True)
    p1 = _greedy_build(order1)
    
    # Also try ascending
    order2 = sorted(evens)
    p2 = _greedy_build(order2)
    
    best_pairs = p1 if len(p1) > len(p2) else p2
    
    # Local search to improve
    time_left = DEADLINE - (time.time() - start_time)
    if time_left > 5:
        best_pairs = _local_search(best_pairs, rng, time.time() + time_left - 2)
        
    # Verify
    if not _is_valid_solution(best_pairs):
        # Fallback to a known valid small set
        best_pairs = [[0, 2], [1, 4], [2, 8], [3, 16], [4, 32], [5, 64], [6, 128]]
        # Check this fallback
        # 2,4: gcd 2. (1-0)%2=1!=0. OK.
        # 4,8: gcd 4. (2-1)%4=1!=0. OK.
        # 8,16: gcd 8. (3-2)%8=1!=0. OK.
        # 16,32: gcd 16. (4-3)%16=1!=0. OK.
        # 32,64: gcd 32. (5-4)%32=1!=0. OK.
        # 64,128: gcd 64. (6-5)%64=1!=0. OK.
        # Cross: 2,8: gcd 2. (2-0)%2=0. FAIL.
        # So that fallback is invalid.
        # Let's just return an empty list or a single element if validation fails.
        # But greedy should always produce a valid list.
        # If validation fails, return the first valid prefix.
        valid_prefix = []
        for p in best_pairs:
            # Check p against valid_prefix
            ok = True
            for q in valid_prefix:
                if not _disjoint(p[0], p[1], q[0], q[1]):
                    ok = False
                    break
            if ok:
                valid_prefix.append(p)
        best_pairs = valid_prefix
        
    return best_pairs
# EVOLVE-BLOCK-END
