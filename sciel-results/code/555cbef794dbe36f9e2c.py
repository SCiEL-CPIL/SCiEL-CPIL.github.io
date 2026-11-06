import math
import random
import time

MAX_MOD = 128
DEADLINE_MARGIN = 8.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _num_distinct_prime_factors(x):
    count = 0
    d = 2
    while d * d <= x:
        if x % d == 0:
            count += 1
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        count += 1
    return count

def _solve_mrv(moduli, time_limit):
    """
    Solve using MRV (Minimum Remaining Values) heuristic with forward checking.
    At each step, pick the unassigned modulus with the fewest valid residues.
    """
    n = len(moduli)
    if n == 0:
        return []
    
    residues = [None] * n
    assigned = [False] * n
    count_assigned = 0
    
    # Precompute gcd matrix
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            gcds[i][j] = math.gcd(moduli[i], moduli[j])
            gcds[j][i] = gcds[i][j]
    
    def get_valid_residues(idx):
        """Get all valid residues for modulus at idx given current assignments."""
        m = moduli[idx]
        valid = []
        for r in range(m):
            ok = True
            for j in range(n):
                if assigned[j] and j != idx:
                    if not _disjoint(r, m, residues[j], moduli[j]):
                        ok = False
                        break
            if ok:
                valid.append(r)
        return valid
    
    while count_assigned < n:
        if time.time() > time_limit:
            return None
        
        # MRV: find unassigned modulus with fewest valid residues
        best_idx = -1
        best_count = float('inf')
        best_residues = []
        
        for i in range(n):
            if assigned[i]:
                continue
            valid = get_valid_residues(i)
            if len(valid) == 0:
                return None  # Dead end
            if len(valid) < best_count:
                best_count = len(valid)
                best_idx = i
                best_residues = valid
                if best_count == 1:
                    break
        
        if best_idx == -1:
            return None
        
        # Pick a residue (random from valid, or first)
        r = random.choice(best_residues)
        residues[best_idx] = r
        assigned[best_idx] = True
        count_assigned += 1
    
    return [(residues[i], moduli[i]) for i in range(n)]

def _solve_greedy_random(moduli, time_limit):
    """
    Randomized greedy: process moduli in random order, pick random valid residue.
    Multiple restarts.
    """
    n = len(moduli)
    if n == 0:
        return []
    
    for attempt in range(200):
        if time.time() > time_limit:
            return None
        
        order = list(range(n))
        random.shuffle(order)
        
        residues = [None] * n
        assigned = [False] * n
        success = True
        
        for idx in order:
            m = moduli[idx]
            # Find valid residues
            valid = []
            for r in range(m):
                ok = True
                for j in range(n):
                    if assigned[j] and j != idx:
                        if not _disjoint(r, m, residues[j], moduli[j]):
                            ok = False
                            break
                if ok:
                    valid.append(r)
            
            if not valid:
                success = False
                break
            
            r = random.choice(valid)
            residues[idx] = r
            assigned[idx] = True
        
        if success:
            return [(residues[i], moduli[i]) for i in range(n)]
    
    return None

def _partial_greedy(moduli, time_limit):
    """
    Greedy that returns the largest valid subset it can find.
    Processes moduli in order, skips those that can't be assigned.
    """
    n = len(moduli)
    if n == 0:
        return []
    
    order = list(range(n))
    random.shuffle(order)
    
    residues = [None] * n
    assigned = [False] * n
    result = []
    
    for idx in order:
        if time.time() > time_limit:
            break
        m = moduli[idx]
        valid = []
        for r in range(m):
            ok = True
            for j in range(n):
                if assigned[j] and j != idx:
                    if not _disjoint(r, m, residues[j], moduli[j]):
                        ok = False
                        break
            if ok:
                valid.append(r)
        
        if valid:
            r = random.choice(valid)
            residues[idx] = r
            assigned[idx] = True
            result.append((r, m))
    
    return result

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Strategy: use all even moduli (64 of them) as the base set.
    # These are the largest pairwise non-coprime subset.
    even_mods = [m for m in all_mods if m % 2 == 0]
    
    # Also consider adding some odd moduli if compatible.
    # An odd modulus m can be added if for every even modulus e in our set,
    # gcd(m, e) = gcd(m, e) and we can find a residue r_m such that
    # (r_m - r_e) % gcd(m, e) != 0 for all e.
    
    best_pairs = []
    best_len = 0
    
    # First, try to solve for all even moduli using MRV
    shuffled = even_mods[:]
    random.shuffle(shuffled)
    pairs = _solve_mrv(shuffled, deadline - 15.0)
    if pairs is not None:
        best_pairs = pairs
        best_len = len(pairs)
    
    # If MRV failed, try randomized greedy
    if best_len < len(even_mods):
        pairs = _solve_greedy_random(even_mods, deadline - 10.0)
        if pairs is not None:
            best_pairs = pairs
            best_len = len(pairs)
    
    # If still not full, try partial greedy
    if best_len < len(even_mods):
        pairs = _partial_greedy(even_mods, deadline - 5.0)
        if len(pairs) > best_len:
            best_pairs = pairs
            best_len = len(pairs)
    
    # Now try to add odd moduli to the existing set
    if best_pairs:
        existing = set((r, m) for r, m in best_pairs)
        odd_mods = [m for m in all_mods if m % 2 == 1]
        random.shuffle(odd_mods)
        
        for m in odd_mods:
            if time.time() > deadline - 2.0:
                break
            if m in [pm for _, pm in best_pairs]:
                continue
            
            # Try to find a residue r for m that is disjoint from all existing
            valid = []
            for r in range(m):
                ok = True
                for (r2, m2) in best_pairs:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    valid.append(r)
            
            if valid:
                r = random.choice(valid)
                best_pairs.append((r, m))
                best_len += 1
    
    # Fallback: if we have nothing, return a trivial valid set
    if not best_pairs:
        best_pairs = [(0, 2)]
    
    return [[r, m] for r, m in best_pairs]
