"""Structural change: precomputed-pair-gain hill climber over 2-bit-per-element bitmasks."""
import time
import numpy as np

N = 256
TIME_LIMIT = 90.0
SAFETY = 0.88

# Precompute 3-AP and 4-AP lists
_threes = []
_fours = []

for a in range(1, N + 1):
    max_d = (N - a) // 2
    for d in range(1, max_d + 1):
        b = a + d
        c = a + 2 * d
        _threes.append((a, b, c))
        # 4-AP check: a, b, c, a+3d
        if a + 3 * d <= N:
            _fours.append((a, b, c, a + 3 * d))

# Also need 4-APs where the added element is in positions 2,3,4
# But we handle all 4-APs via the full list; the constraint is: no 4-AP fully contained in A.
# So we precompute all 4-APs in [1,N]:
_fours = []
for a in range(1, N + 1):
    max_d = (N - a) // 3
    for d in range(1, max_d + 1):
        _fours.append((a, a + d, a + 2 * d, a + 3 * d))

# Build bitmasks for each 3-AP and 4-AP
# Using 2 bits per element: bit 2*(x-1) = in_set, bit 2*(x-1)+1 = available (not in 4-AP conflict)
# Actually, let's use a simpler representation:
# in_set_mask: 1 bit per element (256 bits) -> use Python int
# For 4-AP checking: precompute for each element x, the set of 4-APs it participates in
# For 3-AP counting: precompute for each element x, the set of 3-APs it participates in

# Let's use a different approach for speed:
# Represent the set as a Python int bitmask (256 bits)
# Precompute: for each element x, list of 3-AP indices where x appears
# Precompute: for each element x, list of 4-AP indices where x appears

# Actually, for the hill climber, let's use:
# - A boolean array for membership (fast indexing)
# - Precomputed 3-AP gain: for each pair (add_x, remove_y) or (add_x), compute gain quickly
# - Precomputed 4-AP constraint: for each x, which other elements would complete a 4-AP

# Let me think differently. The key insight from inspiration 3 is:
# Precompute all 3-APs and 4-APs, then use bitmask operations.

# Representation: Python int with 256 bits, bit i (0-indexed) represents element i+1
# 3-AP count: for each 3-AP (a,b,c), check if bits a-1, b-1, c-1 are all set
# 4-AP check: for each 4-AP (a,b,c,d), check if bits a-1, b-1, c-1, d-1 are all set

# For speed, precompute:
# three_masks: list of ints, each with 3 bits set (for the 3-AP)
# four_masks: list of ints, each with 4 bits set (for the 4-AP)

_three_masks = []
for (a, b, c) in _threes:
    m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
    _three_masks.append(m)

_four_masks = []
for (a, b, c, d) in _fours:
    m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (d - 1))
    _four_masks.append(m)

# For each element x, precompute which 3-APs it's in and which 4-APs it's in
# three_of[x] = list of indices into _three_masks where x appears
# four_of[x] = list of indices into _four_masks where x appears

_three_of = [[] for _ in range(N + 1)]
for idx, (a, b, c) in enumerate(_threes):
    _three_of[a].append(idx)
    _three_of[b].append(idx)
    _three_of[c].append(idx)

_four_of = [[] for _ in range(N + 1)]
for idx, (a, b, c, d) in enumerate(_fours):
    _four_of[a].append(idx)
    _four_of[b].append(idx)
    _four_of[c].append(idx)
    _four_of[d].append(idx)


def _count_3aps(mask):
    """Count 3-APs using bitmask."""
    count = 0
    for tm in _three_masks:
        if (mask & tm) == tm:
            count += 1
    return count


def _has_4ap(mask):
    """Check if mask contains any 4-AP."""
    for fm in _four_masks:
        if (mask & fm) == fm:
            return True
    return False


def _can_add(mask, x):
    """Check if adding element x to mask creates a 4-AP."""
    bit = 1 << (x - 1)
    new_mask = mask | bit
    for idx in _four_of[x]:
        fm = _four_masks[idx]
        # Check if the other 3 elements of this 4-AP are all in mask
        others = fm & ~bit
        if (new_mask & fm) == fm:
            return False
    return True


def _can_remove(mask, x):
    """Removing x never creates a 4-AP, so always True."""
    return True


def _gain_add(mask, x):
    """Compute gain in 3-AP count when adding x."""
    bit = 1 << (x - 1)
    gain = 0
    for idx in _three_of[x]:
        tm = _three_masks[idx]
        # Check if the other 2 elements are in mask
        others = tm & ~bit
        if (mask & others) == others:
            gain += 1
    return gain


def _loss_remove(mask, x):
    """Compute loss in 3-AP count when removing x."""
    bit = 1 << (x - 1)
    loss = 0
    for idx in _three_of[x]:
        tm = _three_masks[idx]
        # Check if the other 2 elements are in mask
        others = tm & ~bit
        if (mask & others) == others:
            loss += 1
    return loss


def _greedy_init(rng):
    """Greedy construction with random ordering."""
    mask = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if _can_add(mask, x):
            mask |= (1 << (x - 1))
    return mask


def _hill_climb(mask, deadline, rng):
    """Pair-gain hill climbing."""
    current_score = _count_3aps(mask)
    
    while time.time() < deadline:
        best_gain = 0
        best_move = None  # (type, x, y) where type is 'add', 'remove', 'swap'
        
        # Try single adds
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            if mask & bit:
                continue
            if not _can_add(mask, x):
                continue
            gain = _gain_add(mask, x)
            if gain > best_gain:
                best_gain = gain
                best_move = ('add', x, 0)
        
        # Try single removes (only if it enables more adds later, but for now just check net)
        # Actually, removing alone decreases score, so skip unless part of swap
        
        # Try swaps: remove y, add x
        for y in range(1, N + 1):
            bit_y = 1 << (y - 1)
            if not (mask & bit_y):
                continue
            for x in range(1, N + 1):
                bit_x = 1 << (x - 1)
                if mask & bit_x:
                    continue
                # Check if we can add x after removing y
                temp_mask = mask & ~bit_y
                if not _can_add(temp_mask, x):
                    continue
                loss = _loss_remove(mask, y)
                gain = _gain_add(temp_mask, x)
                net = gain - loss
                if net > best_gain:
                    best_gain = net
                    best_move = ('swap', x, y)
        
        if best_move is None or best_gain <= 0:
            # No improving move found; try random perturbations
            if time.time() > deadline:
                break
            # Random restart with slight perturbation
            perturbed = mask
            # Remove 1-3 random elements
            in_set = [i for i in range(1, N + 1) if (mask >> (i - 1)) & 1]
            if in_set:
                for _ in range(min(3, len(in_set))):
                    y = rng.choice(in_set)
                    perturbed &= ~(1 << (y - 1))
                    in_set.remove(y)
            # Add 1-3 random elements
            not_in_set = [i for i in range(1, N + 1) if not (perturbed >> (i - 1)) & 1]
            if not_in_set:
                for _ in range(min(3, len(not_in_set))):
                    x = rng.choice(not_in_set)
                    if _can_add(perturbed, x):
                        perturbed |= (1 << (x - 1))
            mask = perturbed
            current_score = _count_3aps(mask)
            continue
        
        # Apply best move
        mtype, x, y = best_move
        if mtype == 'add':
            mask |= (1 << (x - 1))
        elif mtype == 'swap':
            mask &= ~(1 << (y - 1))
            mask |= (1 << (x - 1))
        
        current_score += best_gain
    
    return mask, current_score


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT * SAFETY
    
    rng = np.random.default_rng(12345)
    
    best_mask = 0
    best_score = -1
    
    # Multiple greedy inits with different seeds
    for trial in range(8):
        if time.time() > deadline:
            break
        mask = _greedy_init(rng)
        score = _count_3aps(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
    
    # Hill climb on best
    if time.time() < deadline:
        best_mask, best_score = _hill_climb(best_mask, deadline, rng)
    
    # Try a few more inits + hill climbs if time remains
    for trial in range(6):
        if time.time() > deadline:
            break
        mask = _greedy_init(rng)
        score = _count_3aps(mask)
        mask, score = _hill_climb(mask, deadline, rng)
        if score > best_score:
            best_score = score
            best_mask = mask
    
    # Convert mask to list
    result = [i for i in range(1, N + 1) if (best_mask >> (i - 1)) & 1]
    
    if not result:
        result = [1]
    
    return result
