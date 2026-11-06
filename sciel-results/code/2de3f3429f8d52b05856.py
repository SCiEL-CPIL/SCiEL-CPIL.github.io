<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _positive_diffs(s):
    n = len(s)
    return [s[j] - s[i] for i in range(n) for j in range(i + 1, n)]

def _is_sidon(s):
    d = _positive_diffs(s)
    return len(d) == len(set(d))

def _covers_1_to_30(s):
    d = set(_positive_diffs(s))
    return all(x in d for x in range(1, 31))

def _valid(s):
    if len(s) < 1 or len(s) > 32:
        return False
    if 0 not in s:
        return False
    if max(s) > 10**6:
        return False
    if not _is_sidon(s):
        return False
    if not _covers_1_to_30(s):
        return False
    return True

def _greedy_construct():
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

def _random_local_search():
    best = None
    best_val = 10**7
    rng = random.Random(12345)
    # Start from a few random seeds
    for attempt in range(200):
        if time.time() > _DEADLINE:
            break
        # random initial set containing 0, size 9-12
        size = rng.randint(9, 12)
        upper = rng.randint(60, 200)
        s = {0}
        while len(s) < size:
            s.add(rng.randint(1, upper))
        s = sorted(s)
        if not _is_sidon(s):
            continue
        cur = s
        cur_val = max(cur)
        for step in range(3000):
            if time.time() > _DEADLINE:
                break
            # mutate: add, remove, or shift an element
            op = rng.random()
            if op < 0.4 and len(cur) < 32:
                # add a random element
                cand = cur + [rng.randint(1, max(cur) + 50)]
                cand = sorted(cand)
                if len(set(cand)) == len(cand) and _is_sidon(cand):
                    cur = cand
            elif op < 0.7 and len(cur) > 9:
                # remove a random non-zero element
                idx = rng.randint(1, len(cur) - 1)
                cand = cur[:idx] + cur[idx + 1:]
                if _is_sidon(cand):
                    cur = cand
            else:
                # shift a random non-zero element
                idx = rng.randint(1, len(cur) - 1)
                delta = rng.randint(-10, 10)
                new_val = cur[idx] + delta
                if new_val <= 0:
                    continue
                cand = cur[:idx] + [new_val] + cur[idx + 1:]
                cand = sorted(cand)
                if len(set(cand)) == len(cand) and _is_sidon(cand):
                    cur = cand
            # check if covers 1..30
            if _covers_1_to_30(cur):
                val = max(cur)
                if val < best_val:
                    best_val = val
                    best = cur[:]
    return best

def _greedy_with_pruning():
    """Greedy but try to keep the set compact by choosing the smallest valid x."""
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        # Try adding a single element x such that x - v = d or v - x = d for some v in a
        # and the result is Sidon. Prefer smallest max.
        candidates = []
        for v in a:
            for x in (v + d, v - d):
                if x >= 0 and x not in a:
                    candidates.append(x)
        candidates = sorted(set(candidates))
        for x in candidates:
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            # Add two elements to realize d
            y = max(a) + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

def construct():
    t0 = time.time()
    # Try greedy first (fast, gives a valid baseline)
    g = _greedy_construct()
    best = g
    best_val = max(g)
    # Try improved greedy
    g2 = _greedy_with_pruning()
    if _valid(g2) and max(g2) < best_val:
        best = g2
        best_val = max(g2)
    # Try local search to improve
    ls = _random_local_search()
    if ls is not None and max(ls) < best_val:
        best = ls
        best_val = max(ls)
    return best
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _diffs_set(s):
    """Return set of positive differences for sorted list s."""
    n = len(s)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(s[j] - s[i])
    return diffs

def _is_sidon(s):
    """Check if s is a Sidon set (all positive differences distinct)."""
    n = len(s)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def _missing_diffs(s):
    """Return set of d in 1..30 not present as a difference."""
    ds = _diffs_set(s)
    return set(range(1, 31)) - ds

def _energy(s):
    """
    Energy function: penalize missing differences heavily, then max element.
    Lower is better.
    """
    if not s or 0 not in s or len(s) > 32 or max(s) > 10**6:
        return 10**9
    if not _is_sidon(s):
        # Count duplicate differences as penalty
        n = len(s)
        seen = set()
        dup_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = s[j] - s[i]
                if d in seen:
                    dup_count += 1
                seen.add(d)
        return 10000 * dup_count + max(s)
    missing = _missing_diffs(s)
    if missing:
        return 10000 * len(missing) + max(s)
    return max(s)

def _greedy_init():
    """Greedy construction to get a valid starting point."""
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        for v in a:
            for x in (v + d, v - d):
                if x >= 0 and x not in a:
                    trial = sorted(a + [x])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        found = True
                        break
            if found:
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

def _simulated_annealing(initial_set, rng, max_steps=50000, t_start=100.0, t_end=1.0):
    """
    Simulated annealing to minimize max element while maintaining Sidon property
    and covering differences 1..30.
    """
    cur = sorted(initial_set)
    cur_energy = _energy(cur)
    best = cur[:]
    best_energy = cur_energy
    
    for step in range(max_steps):
        if time.time() > _DEADLINE:
            break
        
        # Temperature schedule: exponential cooling
        t = t_start * (t_end / t_start) ** (step / max_steps)
        
        # Generate a move
        op = rng.random()
        if op < 0.35 and len(cur) < 32:
            # Add a random element near existing ones
            if len(cur) > 1:
                base = rng.choice(cur)
                offset = rng.randint(-20, 30)
                cand_val = base + offset
                if cand_val <= 0 or cand_val in cur:
                    continue
                cand = sorted(cur + [cand_val])
            else:
                cand_val = rng.randint(1, 100)
                cand = sorted(cur + [cand_val])
        elif op < 0.60 and len(cur) > 1:
            # Remove a random non-zero element
            idx = rng.randint(1, len(cur) - 1)
            cand = cur[:idx] + cur[idx + 1:]
        else:
            # Shift a random non-zero element
            if len(cur) <= 1:
                continue
            idx = rng.randint(1, len(cur) - 1)
            delta = rng.randint(-15, 15)
            new_val = cur[idx] + delta
            if new_val <= 0 or new_val in cur:
                continue
            cand = cur[:idx] + [new_val] + cur[idx + 1:]
            cand = sorted(cand)
        
        if len(cand) < 1 or 0 not in cand or len(cand) > 32 or max(cand) > 10**6:
            continue
        
        cand_energy = _energy(cand)
        delta_e = cand_energy - cur_energy
        
        if delta_e < 0 or (t > 0 and rng.random() < math.exp(-delta_e / t)):
            cur = cand
            cur_energy = cand_energy
            if cur_energy < best_energy:
                best_energy = cur_energy
                best = cur[:]
    
    return best

def _local_refine(s, rng, max_steps=2000):
    """
    Local search refinement: try small shifts to reduce max element
    while maintaining validity.
    """
    cur = sorted(s)
    best = cur[:]
    best_val = max(cur)
    
    for step in range(max_steps):
        if time.time() > _DEADLINE:
            break
        
        improved = False
        for i in range(1, len(cur)):
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = cur[i] + delta
                if new_val <= 0 or new_val in cur:
                    continue
                cand = cur[:i] + [new_val] + cur[i + 1:]
                cand = sorted(cand)
                if _is_sidon(cand) and not _missing_diffs(cand):
                    if max(cand) < best_val:
                        best_val = max(cand)
                        best = cand[:]
                        improved = True
        if not improved:
            break
    
    return best

def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    # Get a valid initial solution via greedy
    g = _greedy_init()
    best = g
    best_val = max(g)
    
    # Run multiple simulated annealing restarts
    num_restarts = 0
    while time.time() < _DEADLINE - 10:
        num_restarts += 1
        
        # Perturb the best solution for diversity
        if num_restarts == 1:
            start = best[:]
        else:
            start = best[:]
            # Randomly perturb
            for _ in range(rng.randint(1, 3)):
                if len(start) > 1:
                    idx = rng.randint(1, len(start) - 1)
                    delta = rng.randint(-10, 10)
                    new_val = start[idx] + delta
                    if new_val > 0 and new_val not in start:
                        start[idx] = new_val
                        start = sorted(start)
        
        # Run simulated annealing
        sa_result = _simulated_annealing(start, rng, max_steps=30000, t_start=50.0, t_end=0.5)
        if sa_result and _is_sidon(sa_result) and not _missing_diffs(sa_result):
            if max(sa_result) < best_val:
                best_val = max(sa_result)
                best = sa_result[:]
        
        # Local refinement
        refined = _local_refine(best, rng, max_steps=1000)
        if max(refined) < best_val:
            best_val = max(refined)
            best = refined[:]
        
        # Check time
        if time.time() > _DEADLINE - 5:
            break
    
    # Final local refinement
    final = _local_refine(best, rng, max_steps=500)
    if max(final) < best_val:
        best = final
    
    return best
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
