import time
import random
import math

N = 5000
DEADLINE_MARGIN = 5.0
T_START = time.time()

# Precompute primes up to N+10
_MAX_PRIME_LIMIT = N + 10
_primes_cache = {}

def _get_primes(limit):
    if limit in _primes_cache:
        return _primes_cache[limit]
    if limit > 100 and limit <= _MAX_PRIME_LIMIT:
        if _MAX_PRIME_LIMIT not in _primes_cache:
            sieve = bytearray([1]) * (_MAX_PRIME_LIMIT + 1)
            sieve[0] = sieve[1] = 0
            for p in range(2, int(_MAX_PRIME_LIMIT ** 0.5) + 1):
                if sieve[p]:
                    sieve[p * p : _MAX_PRIME_LIMIT + 1 : p] = bytearray(len(sieve[p * p : _MAX_PRIME_LIMIT + 1 : p]))
            _primes_cache[_MAX_PRIME_LIMIT] = [i for i in range(2, _MAX_PRIME_LIMIT + 1) if sieve[i]]
        res = [p for p in _primes_cache[_MAX_PRIME_LIMIT] if p <= limit]
        _primes_cache[limit] = res
        return res
    else:
        if limit < 2:
            res = []
        else:
            sieve = bytearray([1]) * (limit + 1)
            sieve[0] = sieve[1] = 0
            for p in range(2, int(limit ** 0.5) + 1):
                if sieve[p]:
                    sieve[p * p : limit + 1 : p] = bytearray(len(sieve[p * p : limit + 1 : p]))
            res = [i for i in range(2, limit + 1) if sieve[i]]
        _primes_cache[limit] = res
        return res

_all_primes = _get_primes(_MAX_PRIME_LIMIT)

def _time_left():
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)

def _check_admissible(points):
    if not points:
        return True
    m = len(points)
    primes = _get_primes(m)
    for p in primes:
        seen = [0] * p
        for v in points:
            seen[v % p] = 1
        if all(seen):
            return False
    return True

def _greedy_repair(points):
    """Seed repair: delete least-populated class for each violating prime."""
    changed = True
    while changed:
        changed = False
        m = len(points)
        primes = _get_primes(m)
        for p in primes:
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                to_remove = set(smallest)
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points

class AdmissibleSet:
    """
    Maintains a set of points S subset [0, N] and tracks for each prime p <= N:
    - filled[p][r]: count of elements in S with x % p == r
    - zero_residues[p]: number of r with filled[p][r] == 0
    
    Admissible iff for all p <= |S|, zero_residues[p] > 0.
    """
    def __init__(self, N):
        self.N = N
        self.primes = [p for p in _all_primes if p <= N + 1]
        self.filled = {}
        self.zero_residues = {}
        self.S = []
        self.S_set = set()
        for p in self.primes:
            self.filled[p] = [0] * p
            self.zero_residues[p] = p
            
    def add(self, x):
        """Try to add x. Returns True if successful (set remains admissible)."""
        new_size = len(self.S) + 1
        updates = []
        safe = True
        for p in self.primes:
            if p > self.N: break
            r = x % p
            old_filled = self.filled[p][r]
            self.filled[p][r] = old_filled + 1
            updates.append((p, r, old_filled))
            if old_filled == 0:
                self.zero_residues[p] -= 1
                if p <= new_size and self.zero_residues[p] == 0:
                    safe = False
                    break
        if safe:
            self.S.append(x)
            self.S_set.add(x)
            return True
        else:
            # Revert
            for p, r, old_filled in updates:
                self.filled[p][r] = old_filled
                if old_filled == 0:
                    self.zero_residues[p] += 1
            return False

def _greedy_greedy(N, max_time):
    """
    Greedy construction with random ordering.
    We maintain an AdmissibleSet.
    We have a pool of candidate nodes [0, N].
    We iteratively pick a node from the pool that can be added.
    To speed up, we use a priority queue?
    Or simply:
    1. Shuffle candidates.
    2. Iterate through them.
    3. If candidate can be added, add it.
    4. Remove it from candidates.
    5. Repeat until no candidates left or time limit.
    """
    start_time = time.time()
    best_size = 0
    best_set = []
    
    # We will run multiple trials.
    # Each trial: shuffle [0..N], try to add each.
    # This is O(N * P) per trial.
    # P ~ 600. N = 5000.
    # 3M ops per trial.
    # 330s budget.
    
    # Let's just run one long loop of random restarts.
    while _time_left() > 2.0:
        candidates = list(range(N + 1))
        random.shuffle(candidates)
        aset = AdmissibleSet(N)
        
        for x in candidates:
            if aset.add(x):
                # Optional: local search?
                pass
        
        if len(aset.S) > best_size:
            best_size = len(aset.S)
            best_set = aset.S[:]
            
        # Check time
        if time.time() - start_time > max_time:
            break
            
    return best_size, best_set

def _local_improve(points, N, max_time):
    """
    Try to improve a given admissible set by swapping elements.
    For each element x in S, try to remove it and add some y not in S.
    If size doesn't decrease, we might gain flexibility?
    Actually, we want to increase size.
    
    Strategy:
    1. Try to add elements one by one from a random ordering of [0..N] \ S.
    2. If stuck, try swapping: remove x, try to add y. If we can add more than 1 element for each removed, it's good.
    Simpler:
    1. Start with current S.
    2. Create a list of "blocked" elements: elements that cannot be added.
    3. Pick a random x in S. Remove it.
    4. Try to add any element from [0..N] \ (S \ {x}).
    5. If we can add y, keep the change (size same).
    6. Repeat.
    
    This is a random walk on the space of admissible sets of fixed size.
    Once we find a "bottleneck" configuration, we might be able to grow.
    """
    start_time = time.time()
    S = list(points)
    S_set = set(S)
    aset = AdmissibleSet(N)
    # Initialize aset with S
    for x in S:
        for p in aset.primes:
            r = x % p
            aset.filled[p][r] += 1
            if aset.filled[p][r] == 1:
                aset.zero_residues[p] -= 1
        aset.S.append(x)
        aset.S_set.add(x)
        
    # We need to be careful: AdmissibleSet.add assumes S is empty or consistent.
    # Let's just use the manual update logic or a specialized method.
    
    # Actually, let's just use a simpler approach:
    # Maintain counts manually.
    
    primes = [p for p in _all_primes if p <= N + 1]
    filled = {}
    zero_residues = {}
    for p in primes:
        filled[p] = [0] * p
        zero_residues[p] = p
        
    for x in S:
        for p in primes:
            r = x % p
            old = filled[p][r]
            filled[p][r] = old + 1
            if old == 0:
                zero_residues[p] -= 1
                
    def can_add(x, current_size):
        new_size = current_size + 1
        for p in primes:
            r = x % p
            if filled[p][r] == 0:
                if p <= new_size and zero_residues[p] == 1: # Only one zero left, and this is it
                    # Adding x makes this residue non-zero, so zero_residues becomes 0.
                    # If p <= new_size, violation.
                    return False
            # If filled[p][r] > 0, zero_residues doesn't change.
        return True

    def do_add(x):
        for p in primes:
            r = x % p
            if filled[p][r] == 0:
                zero_residues[p] -= 1
            filled[p][r] += 1
        S.append(x)
        S_set.add(x)

    def do_remove(x):
        for p in primes:
            r = x % p
            filled[p][r] -= 1
            if filled[p][r] == 0:
                zero_residues[p] += 1
        S.remove(x)
        S_set.remove(x)
        
    # Try to grow
    # We'll shuffle the complement
    complement = [x for x in range(N + 1) if x not in S_set]
    random.shuffle(complement)
    
    for x in complement:
        if can_add(x, len(S)):
            do_add(x)
            
    # Now try swaps to enable growth
    # Pick random element to remove, then try to add any element.
    # We do this for a fixed number of steps or time.
    
    steps = 0
    while _time_left() > 2.0 and steps < 10000:
        steps += 1
        if not S:
            break
        # Pick random x to remove
        idx = random.randint(0, len(S) - 1)
        x_remove = S[idx]
        
        # We need to check if removing x allows adding something new.
        # Actually, just remove it, then scan for a new addition.
        do_remove(x_remove)
        
        added_new = False
        # Try to add any element from complement (which now includes x_remove? No, we removed it from S, so it's available)
        # We should check a random subset or scan.
        # Scanning 5000 elements * 600 primes = 3M ops. Too slow for many steps.
        
        # Heuristic: Only check a few random candidates.
        # Or, check if any of the previously "blocked" candidates can now be added.
        # Let's just try a few random ones from the full range [0, N].
        
        for _ in range(50):
            cand = random.randint(0, N)
            if cand in S_set:
                continue
            if can_add(cand, len(S)):
                do_add(cand)
                added_new = True
                break
                
        if not added_new:
            # Revert? Or keep it?
            # If we removed x and didn't add anything, size decreased.
            # We should revert to maintain size, or accept it if we think it opens up future possibilities.
            # Usually, we want to maintain size.
            # Let's revert.
            do_add(x_remove) # Add back
            # Note: do_add assumes it's not in S. Since we removed it, it's safe.
            # But we need to be careful with the logic in do_add.
            
    return S

def construct():
    # 1. Get baseline from greedy repair
    points = list(range(N + 1))
    points = _greedy_repair(points)
    best = points if _check_admissible(points) else []
    best_size = len(best)
    
    # 2. Try Greedy Random Restart
    if _time_left() > 10.0:
        try:
            g_size, g_set = _greedy_greedy(N, _time_left() / 2)
            if g_size > best_size and _check_admissible(g_set):
                best = g_set
                best_size = g_size
        except Exception:
            pass
            
    # 3. Try Local Improvement on the best found so far
    if _time_left() > 5.0:
        try:
            improved = _local_improve(best, N, _time_left() / 2)
            if len(improved) > best_size and _check_admissible(improved):
                best = improved
                best_size = len(best)
        except Exception:
            pass
            
    return list(best)
