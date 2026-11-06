import time
import random
import numpy as np

N = 32
M = 512

def _init_score(s):
    """Compute initial diff_count, satisfied set, and pending set."""
    diff_count = 0
    satisfied = set()
    pending = {}
    s_list = list(s)
    s_set = s
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            a, b = s_list[i], s_list[j]
            if a > b:
                a, b = b, a
            d = b - a
            c = b + d
            if c in s_set:
                if d not in satisfied:
                    satisfied.add(d)
                    diff_count += 1
            else:
                if d not in pending:
                    pending[d] = set()
                pending[d].add((a, c))
    return diff_count, satisfied, pending

def _score_delta(s, satisfied, pending, a_out, a_in):
    """Compute score change when removing a_out and adding a_in.
    Returns (delta, new_diff_count, new_satisfied, new_pending)."""
    new_diff_count = len(satisfied)
    new_satisfied = satisfied.copy()
    new_pending = {d: set(ps) for d, ps in pending.items()}
    
    # Remove pairs involving a_out
    for d in list(new_pending.keys()):
        new_pending[d] = {p for p in new_pending[d] if a_out not in p}
        if not new_pending[d]:
            del new_pending[d]
    
    for d in list(new_satisfied.keys()):
        # Check if d still has a valid progression after removing a_out
        still_valid = False
        # We need to check if there's still some a such that a, a+d, a+2d are all in s\{a_out}∪{a_in}
        # This is complex; instead, we'll handle it differently
        # Actually, let's just recompute which satisfied d's are affected
        pass
    
    # This approach is getting complex. Let me use a cleaner method:
    # Just recompute the full score but with the modified set, using the fact
    # that most differences are unaffected.
    
    # Actually, for correctness and simplicity with n=32, let's just do full recompute
    # but optimize it. The incremental approach is error-prone.
    
    # Let me try a different incremental approach:
    # For each d, track the count of valid progressions.
    # When removing a_out: for each d, if a_out was part of a progression, decrement count.
    # If count becomes 0, d is no longer satisfied.
    # When adding a_in: for each d, if a_in forms a new progression, increment count.
    # If count was 0 and becomes 1, d is now satisfied.
    
    # This requires tracking counts, not just sets. Let me redesign.
    
    # For now, let's just do full recompute but make it fast.
    new_s = list(s - {a_out} | {a_in})
    new_s_set = set(new_s)
    diffs = set()
    for i in range(len(new_s)):
        a = new_s[i]
        for j in range(len(new_s)):
            if i == j:
                continue
            b = new_s[j]
            d = b - a
            if d > 0:
                c = a + 2 * d
                if c in new_s_set:
                    diffs.add(d)
    return len(diffs), None, None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple candidate initial sets
    candidates = []
    
    # Candidate 1: Two blocks - small and medium
    s1 = set(range(1, 17)) | set(range(50, 66))
    candidates.append(s1)
    
    # Candidate 2: Three blocks
    s2 = set(range(1, 12)) | set(range(40, 51)) | set(range(80, 91))
    candidates.append(s2)
    
    # Candidate 3: Spread out more
    s3 = set(range(1, 11)) | set(range(30, 40)) | set(range(60, 70)) | set(range(90, 100))
    candidates.append(s3)
    
    # Candidate 4: Random with bias toward small values
    for _ in range(3):
        s4 = set(random.sample(range(1, 200), 32))
        candidates.append(s4)
    
    # Candidate 5: Random
    for _ in range(3):
        s5 = set(random.sample(range(1, 513), 32))
        candidates.append(s5)
    
    # Candidate 6: Geometric-like spread
    s6 = set()
    vals = [1, 2, 3, 5, 7, 11, 13, 17, 21, 25, 31, 37, 43, 50, 58, 67,
            77, 88, 100, 113, 127, 142, 158, 175, 193, 212, 232, 253,
            275, 298, 322, 347]
    s6 = set(v for v in vals if 1 <= v <= 512)
    if len(s6) == 32:
        candidates.append(s6)
    
    # Score all candidates
    best_set = None
    best_score = -1
    for s in candidates:
        if len(s) != 32:
            continue
        sc = _score_delta(s, None, None, 0, 0)  # Full recompute
        if sc > best_score:
            best_score = sc
            best_set = set(s)
    
    current_set = best_set
    current_score = best_score
    
    # Local search with simulated annealing
    T = 10.0
    cooling = 0.9995
    min_T = 0.5
    
    while time.time() < deadline:
        # Generate neighbor
        elem_out = random.choice(list(current_set))
        elem_in = random.randint(1, 512)
        while elem_in in current_set:
            elem_in = random.randint(1, 512)
        
        new_set = set(current_set - {elem_out} | {elem_in})
        new_score = _score_delta(new_set, None, None, 0, 0)
        
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < np.exp(delta / T):
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        
        T = max(T * cooling, min_T)
        
        # Check deadline with safety margin
        if time.time() > deadline - 0.5:
            break
    
    return sorted(best_set)
