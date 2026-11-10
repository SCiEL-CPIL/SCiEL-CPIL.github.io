# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_N = 256
_KILL = 90.0
_MARGIN = 8.0

# Precompute reciprocals
_RECIP = [0.0] + [1.0 / i for i in range(1, _N + 1)]
# Precompute suffix sums for quick upper bound (loose, but fast)
# Actually, sum of available is tighter.
# Let's precompute weight of each bit
_WEIGHTS = _RECIP

def _update_forbidden_add(forbidden, S_bits, x):
    """
    Calculate new forbidden mask when adding x to S.
    S_bits is the bitset of current set (before adding x).
    x is the new element.
    Returns new forbidden mask.
    """
    new_forbidden = forbidden
    # Iterate through all y in S
    s = S_bits
    while s:
        # Get lowest set bit
        lsb = s & -s
        y = lsb.bit_length() - 1
        s ^= lsb
        
        # Compute 3rd terms for (x, y)
        # 1. x, y, z => 2y = x + z => z = 2y - x
        z = 2 * y - x
        if 1 <= z <= _N:
            new_forbidden |= (1 << z)
            
        # 2. y, x, z => 2x = y + z => z = 2x - y
        z = 2 * x - y
        if 1 <= z <= _N:
            new_forbidden |= (1 << z)
            
        # 3. x, z, y => 2z = x + y => z = (x+y)/2
        s_sum = x + y
        if s_sum % 2 == 0:
            z = s_sum // 2
            if 1 <= z <= _N:
                new_forbidden |= (1 << z)
                
    return new_forbidden

def _branch_and_bound(start_time, deadline):
    """
    Branch and bound search using bitsets.
    """
    best_w = -1.0
    best_S = 0
    
    # Initial state
    S_bits = 0
    forbidden = 0
    current_w = 0.0
    
    # Precompute full mask for 1..N
    FULL_MASK = (1 << (_N + 1)) - 2  # Bits 1 to N
    
    def _sum_available(available_mask):
        s = 0.0
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += _RECIP[idx]
            m ^= lsb
        return s

    def _search(S_bits, forbidden, current_w, last_idx):
        nonlocal best_w, best_S
        
        if time.time() > deadline:
            return
            
        # Update best
        if current_w > best_w:
            best_w = current_w
            best_S = S_bits
            
        # Calculate available mask
        # Available: not in S, not in forbidden, index > last_idx (to avoid permutations)
        # Wait, if we enforce increasing index, we might miss sets that don't follow sorted order?
        # No, any set has a sorted order. If we build the set by adding elements in increasing order of index,
        # we cover all subsets exactly once.
        
        # However, the "forbidden" constraint depends on the whole set, not just the last index.
        # But if we always add the next smallest available index, we are fine.
        # Actually, we should iterate through ALL available indices, not just > last_idx,
        # because we might have skipped some smaller ones?
        # No, if we skip a smaller one, we can't add it later if we enforce order?
        # Standard B&B for subsets:
        # Option 1: Include i. Option 2: Exclude i.
        # This is cleaner.
        
        # Let's stick to: Try adding each x > last_idx that is available.
        # This generates each subset exactly once in sorted order.
        
        available_mask = FULL_MASK & ~S_bits & ~forbidden
        # Only consider x > last_idx
        # Mask out bits <= last_idx
        if last_idx >= 0:
            available_mask &= ~((1 << (last_idx + 1)) - 1)
            
        # Pruning: Upper bound
        # Sum of weights of all available nodes
        # If current_w + sum(available) <= best_w, prune
        # Optimization: If available_mask is 0, return
        if available_mask:
            # Quick bound: sum of top K weights?
            # Just sum all available
            ub = current_w + _sum_available(available_mask)
            if ub <= best_w:
                return
        
        # Iterate through available candidates
        m = available_mask
        while m:
            if time.time() > deadline:
                return
            lsb = m & -m
            x = lsb.bit_length() - 1
            m ^= lsb
            
            # Try adding x
            new_S = S_bits | (1 << x)
            new_forbidden = _update_forbidden_add(forbidden, S_bits, x)
            # Note: x itself is now in S, so it shouldn't be in forbidden?
            # The update function adds z based on pairs in OLD S.
            # It does not add x to forbidden.
            # But x is in new_S.
            # The check for validity of x was: x not in forbidden (old).
            # This is correct.
            
            # Recurse
            _search(new_S, new_forbidden, current_w + _RECIP[x], x)
            
            # Backtrack: no need to undo, we pass copies (integers are immutable)
            
    _search(0, 0, 0.0, 0)
    
    return best_S, best_w

def _hill_climb_bitset(S_bits, deadline):
    """
    Hill climbing on bitset representation.
    Tries to add single elements or swap.
    """
    best_S = S_bits
    best_w = sum(_RECIP[i] for i in range(1, _N+1) if (S_bits >> i) & 1)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Compute forbidden
        forbidden = 0
        s = S_bits
        while s:
            lsb = s & -s
            y = lsb.bit_length() - 1
            s ^= lsb
            # We need forbidden for the CURRENT set.
            # This is O(N^2) to compute from scratch.
            # Better to maintain it?
            # For simplicity in hill climb, recompute.
            pass
        
        # Efficient forbidden computation:
        forbidden = 0
        s_list = [i for i in range(1, _N+1) if (S_bits >> i) & 1]
        for i in range(len(s_list)):
            a = s_list[i]
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                z = 2*b - a
                if 1 <= z <= _N: forbidden |= (1 << z)
                z = 2*a - b
                if 1 <= z <= _N: forbidden |= (1 << z)
                s_sum = a + b
                if s_sum % 2 == 0:
                    z = s_sum // 2
                    if 1 <= z <= _N: forbidden |= (1 << z)
        
        best_gain = -1e-9
        best_action = None
        best_x = -1
        best_y = -1
        
        # Try adds
        available = ((1 << (_N + 1)) - 2) & ~S_bits & ~forbidden
        m = available
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            m ^= lsb
            g = _RECIP[x]
            if g > best_gain:
                best_gain = g
                best_action = 'add'
                best_x = x
        
        # Try swaps: remove x, add y
        # Iterate x in S
        s_list = [i for i in range(1, _N+1) if (S_bits >> i) & 1]
        for x in s_list:
            if time.time() > deadline:
                break
            # Remove x
            S_temp = S_bits & ~(1 << x)
            # Recompute forbidden for S_temp? Expensive.
            # Approximation: Just check if y is valid in S_temp?
            # Let's just try adds for now to keep it fast, or simple swaps.
            # Given time, let's stick to adds and single removals if they help?
            # Removals don't help weight directly, but might allow better adds later.
            # Hill climb usually takes improving moves.
            # So only add if gain > 0.
            # Swap: remove x, add y. Gain = 1/y - 1/x.
            # We need to check if y is valid in S \ {x}.
            
            # To make this fast, we can precompute conflicts?
            # Let's skip complex swaps in hill climb and rely on B&B for global structure.
            # Or implement simple swap check.
            
            # For each x in S, try to find best y not in S, not in forbidden(S\{x})
            # This is hard without recomputing forbidden.
            # Let's just do adds in hill climb.
            
        if best_gain > 1e-9 and best_action == 'add':
            S_bits |= (1 << best_x)
            best_w += _RECIP[best_x]
            best_S = S_bits
            improved = True
            
    return best_S, best_w

def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    rng = random.Random(20240517)
    
    best_S_bits = 0
    best_w = -1.0
    
    # Phase 1: Branch and Bound
    bt_deadline = min(start + 40.0, deadline - 10.0)
    bt_S, bt_w = _branch_and_bound(start, bt_deadline)
    if bt_w > best_w:
        best_w = bt_w
        best_S_bits = bt_S
        
    # Phase 2: Hill Climbing from B&B result
    if best_S_bits:
        hc_deadline = deadline - 5.0
        hc_S, hc_w = _hill_climb_bitset(best_S_bits, hc_deadline)
        if hc_w > best_w:
            best_w = hc_w
            best_S_bits = hc_S
            
    # Phase 3: Random Greedy + Hill Climb
    for _ in range(10):
        if time.time() > deadline - 5.0:
            break
        # Random Greedy
        S_bits = 0
        forbidden = 0
        order = list(range(1, _N + 1))
        rng.shuffle(order)
        for x in order:
            if (S_bits >> x) & 1: continue
            if (forbidden >> x) & 1: continue
            S_bits |= (1 << x)
            forbidden = _update_forbidden_add(forbidden, S_bits & ~(1<<x), x) # Careful: update uses OLD S
            # Wait, _update_forbidden_add takes S_bits (old) and x.
            # In the loop, S_bits is old before adding x.
            # So:
            # old_S = S_bits
            # S_bits |= 1<<x
            # forbidden = _update_forbidden_add(forbidden, old_S, x)
            # Let's fix the loop logic.
            
        # Redo random greedy properly
        S_bits = 0
        forbidden = 0
        order = list(range(1, _N + 1))
        rng.shuffle(order)
        for x in order:
            if (S_bits >> x) & 1: continue
            if (forbidden >> x) & 1: continue
            old_S = S_bits
            S_bits |= (1 << x)
            forbidden = _update_forbidden_add(forbidden, old_S, x)
            
        hc_S, hc_w = _hill_climb_bitset(S_bits, deadline - 2.0)
        if hc_w > best_w:
            best_w = hc_w
            best_S_bits = hc_S
            
    # Convert to list
    res = [i for i in range(1, _N + 1) if (best_S_bits >> i) & 1]
    
    if not res:
        res = [1]
        
    # Verification
    Sset = set(res)
    ok = True
    for i, a in enumerate(res):
        for b in res[i+1:]:
            c = 2 * b - a
            if c in Sset and c > b:
                ok = False
            d = 2 * a - b
            if d in Sset and d < a:
                ok = False
            s = a + b
            if s % 2 == 0:
                m = s // 2
                if m in Sset and m != a and m != b:
                    ok = False
    if not ok:
        # Fallback
        res = [1]
        
    return res
# EVOLVE-BLOCK-END
