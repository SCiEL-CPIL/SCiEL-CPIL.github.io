import random
import time

N = 256
MAX_SUM = 2 * N  # 512

# Bitset representation: sums as a Python int where bit s is set if sum s is in use
# This gives O(1) conflict checks via bit operations

def _sums_to_bits(sums):
    """Convert a set of sums to a bitset integer."""
    bits = 0
    for s in sums:
        bits |= (1 << s)
    return bits

def _bits_to_sums(bits):
    """Convert a bitset integer back to a set of sums."""
    sums = set()
    s = 0
    while bits:
        if bits & 1:
            sums.add(s)
        bits >>= 1
        s += 1
    return sums

def _compute_sums(selected):
    """Compute all unordered pair sums (including a+a) for a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            sums.add(a + selected[j])
    return sums

def _compute_sum_bits(selected):
    """Compute sum bitset for a Sidon set. Returns None if invalid."""
    bits = 0
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if bits & (1 << s):
                return None  # Conflict
            bits |= (1 << s)
    return bits

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered pair sums distinct)."""
    if len(selected) != len(set(selected)):
        return False
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sum_bits):
    """Check if selected is inclusion-maximal using bitset."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if 2x is in sums
        if sum_bits & (1 << (2 * x)):
            continue
        # Check if x+a is in sums for any a in selected
        blocked = False
        for a in selected:
            if sum_bits & (1 << (x + a)):
                blocked = True
                break
        if not blocked:
            return False
    return True

def _get_unblocked(selected, sum_bits):
    """Return list of missing elements that can be added (not blocked)."""
    sel_set = set(selected)
    unblocked = []
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if sum_bits & (1 << (2 * x)):
            continue
        blocked = False
        for a in selected:
            if sum_bits & (1 << (x + a)):
                blocked = True
                break
        if not blocked:
            unblocked.append(x)
    return unblocked

def _try_add(selected, sum_bits, x):
    """Try to add x to selected. Return (success, new_selected, new_sum_bits)."""
    if sum_bits & (1 << (2 * x)):
        return False, selected, sum_bits
    new_bits = sum_bits | (1 << (2 * x))
    for a in selected:
        s = x + a
        if new_bits & (1 << s):
            return False, selected, sum_bits
        new_bits |= (1 << s)
    new_selected = selected + [x]
    return True, new_selected, new_bits

def _blocking_count(selected, sum_bits, x, unblocked):
    """Count how many unblocked elements x would block if added."""
    new_bits = sum_bits | (1 << (2 * x))
    for a in selected:
        new_bits |= (1 << (x + a))
    
    count = 0
    for y in unblocked:
        if y == x:
            continue
        # Check if y would be blocked with new_bits
        if new_bits & (1 << (2 * y)):
            count += 1
            continue
        y_blocked = False
        for a in selected:
            if new_bits & (1 << (y + a)):
                y_blocked = True
                break
        if y_blocked:
            count += 1
    return count

def _greedy_maximize(selected, sum_bits, rng, deadline):
    """Greedily add elements until maximal, preferring those that block the most."""
    selected = list(selected)
    sum_bits = sum_bits
    
    while time.time() < deadline:
        unblocked = _get_unblocked(selected, sum_bits)
        if not unblocked:
            break
        
        # Choose element to add: prefer one that blocks the most other unblocked elements
        best_x = None
        best_score = -1
        for x in unblocked:
            score = _blocking_count(selected, sum_bits, x, unblocked)
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            best_x = rng.choice(unblocked)
        
        ok, selected, sum_bits = _try_add(selected, sum_bits, best_x)
        if not ok:
            break
    
    return selected, sum_bits

def _reduce_by_removal(selected, sum_bits, deadline):
    """Aggressively try to remove elements while maintaining maximality."""
    selected = list(selected)
    sum_bits = sum_bits
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            # Recompute sum bits for candidate
            new_bits = _compute_sum_bits(candidate)
            if new_bits is None:
                continue
            if _is_maximal(candidate, new_bits):
                selected = candidate
                sum_bits = new_bits
                improved = True
                break
    
    return selected, sum_bits

def _swap_search(selected, sum_bits, deadline):
    """Try 2-opt style swaps: remove two, add one, to reduce size."""
    selected = list(selected)
    sum_bits = sum_bits
    sel_set = set(selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(selected)
        for i in range(n):
            for j in range(i + 1, n):
                # Try removing both selected[i] and selected[j]
                candidate = [x for k, x in enumerate(selected) if k != i and k != j]
                if not candidate:
                    continue
                new_bits = _compute_sum_bits(candidate)
                if new_bits is None:
                    continue
                # Now try adding any missing element to see if we can get a smaller valid maximal set
                # Actually, just check if removing both keeps it maximal
                if _is_maximal(candidate, new_bits):
                    selected = candidate
                    sum_bits = new_bits
                    improved = True
                    break
            if improved:
                break
    
    return selected, sum_bits

def _construct_with_target_size(rng, target_size, deadline):
    """Construct a Sidon set by randomly trying elements up to target_size, then maximize."""
    selected = []
    sum_bits = 0
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(selected) >= target_size:
            break
        ok, selected, sum_bits = _try_add(selected, sum_bits, x)
    
    if not selected:
        return None, None
    
    # Greedily maximize
    selected, sum_bits = _greedy_maximize(selected, sum_bits, rng, deadline)
    
    return selected, sum_bits

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Multiple strategies with different random seeds and target sizes
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Vary target size for diversity
        target_size = rng.randint(6, 18)
        
        selected, sum_bits = _construct_with_target_size(rng, target_size, deadline)
        
        if not selected:
            continue
        
        if not _is_valid_sidon(selected):
            continue
        if not _is_maximal(selected, sum_bits):
            continue
        
        if len(selected) < best_size:
            best_size = len(selected)
            best = list(selected)
        
        # Try to reduce further
        if time.time() < deadline:
            reduced, reduced_bits = _reduce_by_removal(selected, sum_bits, deadline)
            if _is_valid_sidon(reduced) and _is_maximal(reduced, reduced_bits):
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best = list(reduced)
        
        # Try swap search for further reduction
        if time.time() < deadline:
            swapped, swapped_bits = _swap_search(reduced, reduced_bits, deadline)
            if _is_valid_sidon(swapped) and _is_maximal(swapped, swapped_bits):
                if len(swapped) < best_size:
                    best_size = len(swapped)
                    best = list(swapped)
        
        # Early termination if we find a very small set
        if best_size <= 10:
            break
    
    # Fallback: deterministic greedy
    if best is None:
        selected = []
        sum_bits = 0
        for x in range(1, N + 1):
            ok, selected, sum_bits = _try_add(selected, sum_bits, x)
        return selected
    
    return best
