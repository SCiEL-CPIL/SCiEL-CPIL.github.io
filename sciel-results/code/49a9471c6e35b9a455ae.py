"""Combined program: Cantor seed + randomized greedy + fast swap local search with bitset verification."""
import time
import random

N = 512
MAX = N

# Precompute for each x the list of (d) values for each of the 4 positions,
# so the inner loop is just iterating a precomputed tuple.
_pos_last   = [tuple(range(1, (x - 1) // 3 + 1)) for x in range(N + 1)]
_pos_first  = [tuple(range(1, (N - x) // 3 + 1)) for x in range(N + 1)]
_pos_second = [tuple(range(1, min(x - 1, (N - x) // 2) + 1)) for x in range(N + 1)]
_pos_third  = [tuple(range(1, min((x - 1) // 2, N - x) + 1)) for x in range(N + 1)]

# Precompute all 4-APs as (a, b, c, d) tuples for fast conflict checking
_all_4aps = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = a + 2 * d
        f = a + 3 * d
        if f <= N:
            _all_4aps.append((a, b, c, f))

# For each element x, precompute which other elements it conflicts with (forms a 4-AP with)
# conflict_map[x] = frozenset of elements y such that x and y are in some 4-AP together
_conflict_map = [set() for _ in range(N + 1)]
for (a, b, c, f) in _all_4aps:
    for x in (a, b, c, f):
        for y in (a, b, c, f):
            if x != y:
                _conflict_map[x].add(y)
_conflict_map = [frozenset(s) for s in _conflict_map]

def _can_add_bitset(x, bits):
    """Check if x can be added to the set represented by bits (int)."""
    for d in _pos_last[x]:
        if (bits >> (x - d)) & 1 and (bits >> (x - 2 * d)) & 1 and (bits >> (x - 3 * d)) & 1:
            return False
    for d in _pos_first[x]:
        if (bits >> (x + d)) & 1 and (bits >> (x + 2 * d)) & 1 and (bits >> (x + 3 * d)) & 1:
            return False
    for d in _pos_second[x]:
        if (bits >> (x - d)) & 1 and (bits >> (x + d)) & 1 and (bits >> (x + 2 * d)) & 1:
            return False
    for d in _pos_third[x]:
        if (bits >> (x - 2 * d)) & 1 and (bits >> (x - d)) & 1 and (bits >> (x + d)) & 1:
            return False
    return True


def _find_blockers(x, bits):
    """Find all elements that block x from being added (form a 4-AP with x and 2 others in bits)."""
    blockers = set()
    for d in _pos_last[x]:
        a, b, c = x - d, x - 2 * d, x - 3 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    for d in _pos_first[x]:
        a, b, c = x + d, x + 2 * d, x + 3 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    for d in _pos_second[x]:
        a, b, c = x - d, x + d, x + 2 * d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    for d in _pos_third[x]:
        a, b, c = x - 2 * d, x - d, x + d
        if ((bits >> a) & 1) and ((bits >> b) & 1) and ((bits >> c) & 1):
            blockers.add(a); blockers.add(b); blockers.add(c)
    return blockers


def _is_4ap_free_bits(bits):
    """Fast bitset-based check for 4-AP-free property."""
    for x in range(1, N + 1):
        if not ((bits >> x) & 1):
            continue
        for d in _pos_last[x]:
            if ((bits >> (x - d)) & 1) and ((bits >> (x - 2 * d)) & 1) and ((bits >> (x - 3 * d)) & 1):
                return False
        for d in _pos_first[x]:
            if ((bits >> (x + d)) & 1) and ((bits >> (x + 2 * d)) & 1) and ((bits >> (x + 3 * d)) & 1):
                return False
        for d in _pos_second[x]:
            if ((bits >> (x - d)) & 1) and ((bits >> (x + d)) & 1) and ((bits >> (x + 2 * d)) & 1):
                return False
        for d in _pos_third[x]:
            if ((bits >> (x - 2 * d)) & 1) and ((bits >> (x - d)) & 1) and ((bits >> (x + d)) & 1):
                return False
    return True


def _greedy_forward_bitset():
    """Fast forward greedy using bitset."""
    bits = 0
    for x in range(1, N + 1):
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _cantor_set_bits():
    """Cantor-like construction: keep numbers whose ternary expansion has no '1' digits."""
    bits = 0
    for x in range(1, N + 1):
        # Check if x in ternary has only 0s and 2s
        t = x
        ok = True
        while t > 0:
            if t % 3 == 1:
                ok = False
                break
            t //= 3
        if ok:
            bits |= (1 << x)
    return bits


def _greedy_density_weighted_bitset():
    """Greedy that prioritizes elements with fewer conflicts (lower density neighbors)."""
    # Score each element by how many potential 4-AP partners it has
    scores = [0] * (N + 1)
    for x in range(1, N + 1):
        cnt = 0
        for d in _pos_last[x]:
            cnt += 1
        for d in _pos_first[x]:
            cnt += 1
        for d in _pos_second[x]:
            cnt += 1
        for d in _pos_third[x]:
            cnt += 1
        scores[x] = cnt
    
    # Sort by score ascending (fewer conflicts first), then by value
    order = sorted(range(1, N + 1), key=lambda x: (scores[x], x))
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_random_order_bitset(rng):
    """Random-order greedy using bitset."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _bits_to_set(bits):
    """Convert bitset to set."""
    return {i for i in range(1, N + 1) if (bits >> i) & 1}


def _local_improve_bitset(bits, deadline):
    """Fast local search using bitset with improved swap strategy."""
    current_bits = bits
    current_size = bin(current_bits).count('1')
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Pass 1: Add any available element
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if not ((current_bits >> x) & 1) and _can_add_bitset(x, current_bits):
                current_bits |= (1 << x)
                current_size += 1
                improved = True
        
        # Pass 2: Try swaps (remove b, add x)
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if (current_bits >> x) & 1:
                continue
            
            # Find blockers using precomputed helper
            blockers = _find_blockers(x, current_bits)
            if not blockers:
                continue
                
            # Try each blocker as a swap candidate
            for b in blockers:
                if time.time() > deadline - 2:
                    break
                # Try swap: remove b, add x
                candidate_bits = current_bits & ~(1 << b)
                if _can_add_bitset(x, candidate_bits):
                    candidate_bits |= (1 << x)
                    
                    # Greedy re-fill on candidate
                    changed = True
                    while changed and time.time() < deadline - 2:
                        changed = False
                        for y in range(1, N + 1):
                            if time.time() > deadline - 2:
                                break
                            if not ((candidate_bits >> y) & 1) and _can_add_bitset(y, candidate_bits):
                                candidate_bits |= (1 << y)
                                changed = True
                                
                    cand_size = bin(candidate_bits).count('1')
                    if cand_size > current_size:
                        current_bits = candidate_bits
                        current_size = cand_size
                        improved = True
                        break
    
    return current_bits


def _greedy_reverse_order_bitset():
    """Reverse-order greedy using bitset."""
    bits = 0
    for x in range(N, 0, -1):
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def _greedy_random_reverse_bitset(rng):
    """Random reverse-order greedy using bitset."""
    order = list(range(N, 0, -1))
    rng.shuffle(order)
    bits = 0
    for x in order:
        if _can_add_bitset(x, bits):
            bits |= (1 << x)
    return bits


def construct():
    deadline = time.time() + 55  # Safety margin within 60s limit

    # 1) Deterministic seeds
    best_bits = _greedy_forward_bitset()
    best_size = bin(best_bits).count('1')
    
    rev_bits = _greedy_reverse_order_bitset()
    rev_size = bin(rev_bits).count('1')
    if rev_size > best_size:
        best_bits = rev_bits
        best_size = rev_size

    cant_bits = _cantor_set_bits()
    cant_size = bin(cant_bits).count('1')
    if cant_size > best_size:
        best_bits = cant_bits
        best_size = cant_size

    dens_bits = _greedy_density_weighted_bitset()
    dens_size = bin(dens_bits).count('1')
    if dens_size > best_size:
        best_bits = dens_bits
        best_size = dens_size

    # Local improvement on each deterministic seed
    for seed_bits in [_greedy_forward_bitset(), _greedy_reverse_order_bitset(), _cantor_set_bits(), _greedy_density_weighted_bitset()]:
        if time.time() > deadline - 10:
            break
        imp = _local_improve_bitset(seed_bits, deadline)
        imp_size = bin(imp).count('1')
        if imp_size > best_size:
            best_bits = imp
            best_size = imp_size

    # 2) Many randomized restarts (both forward and reverse orders)
    rng = random.Random(42)
    while time.time() < deadline - 12:
        r = rng.random()
        if r < 0.4:
            bits = _greedy_random_order_bitset(rng)
        elif r < 0.8:
            bits = _greedy_random_reverse_bitset(rng)
        else:
            bits = _greedy_density_weighted_bitset()
            # Perturb density-weighted with random swaps
            for _ in range(5):
                x = rng.randint(1, N)
                if (bits >> x) & 1:
                    bits &= ~(1 << x)
                elif _can_add_bitset(x, bits):
                    bits |= (1 << x)
        size = bin(bits).count('1')
        if size > best_size:
            best_bits = bits
            best_size = size

    # 3) Final local improvement on the best set found
    if time.time() < deadline - 3:
        improved_bits = _local_improve_bitset(best_bits, deadline)
        improved_size = bin(improved_bits).count('1')
        if improved_size > best_size:
            best_bits = improved_bits
            best_size = improved_size

    # 4) Convert to set for final verification and return
    best_set = _bits_to_set(best_bits)

    # 5) Final verification; fallback if invalid
    if not _is_4ap_free_bits(best_bits):
        best_set = _bits_to_set(_greedy_forward_bitset())

    return sorted(best_set)
