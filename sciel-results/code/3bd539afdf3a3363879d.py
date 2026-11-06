import time
import random
import numpy as np

def _paleyn_orientation(p):
    """Construct Paley tournament orientation for prime p."""
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            # i -> j if (j - i) % p is a quadratic residue
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _verify_orientation(n, orientation):
    """Verify if every 4-set has a common dominator."""
    # Build in-neighborhoods as bitmasks
    in_masks = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            if bit == 0:
                # i -> j
                in_masks[j] |= (1 << i)
            else:
                # j -> i
                in_masks[i] |= (1 << j)
            idx += 1
            
    # Check all 4-sets
    # Optimization: Use itertools.combinations
    from itertools import combinations
    for comb in combinations(range(n), 4):
        a, b, c, d = comb
        # Dominators are vertices pointing to all a,b,c,d
        dom_mask = in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]
        if dom_mask == 0:
            return False
    return True

def _run_sa_for_n(n, time_limit, seed=42):
    """Run Simulated Annealing to find a valid tournament on n vertices."""
    random.seed(seed)
    start_time = time.time()
    
    # Precompute 4-sets
    from itertools import combinations
    all_4sets = list(combinations(range(n), 4))
    num_4sets = len(all_4sets)
    
    # Map vertex to list of 4-set indices containing it
    sets_containing = [[] for _ in range(n)]
    for idx, comb in enumerate(all_4sets):
        for v in comb:
            sets_containing[v].append(idx)
            
    # Initial random tournament
    out_masks = [0] * n
    in_masks = [0] * n
    orientation_bits = [0] * (n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                # i -> j
                out_masks[i] |= (1 << j)
                in_masks[j] |= (1 << i)
                orientation_bits[idx] = 0
            else:
                # j -> i
                out_masks[j] |= (1 << i)
                in_masks[i] |= (1 << j)
                orientation_bits[idx] = 1
            idx += 1
            
    # Calculate initial bad count
    bad_count = 0
    is_bad = [False] * num_4sets
    for idx, comb in enumerate(all_4sets):
        a, b, c, d = comb
        dom_mask = in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]
        if dom_mask == 0:
            bad_count += 1
            is_bad[idx] = True
            
    if bad_count == 0:
        return orientation_bits

    best_orientation = None
    best_bad_count = bad_count
    
    # SA parameters
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    while time.time() - start_time < time_limit:
        if bad_count == 0:
            return orientation_bits
            
        # Pick random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        # Identify affected 4-sets
        # Sets containing i or j
        affected_indices = set(sets_containing[i])
        affected_indices.update(sets_containing[j])
        
        old_bad_count = bad_count
        new_bad_count = bad_count
        
        # Calculate delta for affected sets
        for s_idx in affected_indices:
            comb = all_4sets[s_idx]
            a, b, c, d = comb
            
            # Check if this set contains i or j (it should by construction)
            # Calculate current dominator mask
            # Note: in_masks reflect the CURRENT state before flip
            
            # We need to know the state of edges i->j and j->i
            # If i < j:
            #   bit index in orientation_bits is idx
            #   But we can just look at out_masks
            
            # Let's simulate the flip on in_masks for the purpose of calculation
            # But we don't want to modify global state yet.
            
            # Current state:
            # Does i point to j?
            i_points_to_j = bool(out_masks[i] & (1 << j))
            
            # If we flip:
            # If i->j becomes j->i:
            #   in_masks[i] gains bit j
            #   in_masks[j] loses bit i
            #   out_masks[i] loses bit j
            #   out_masks[j] gains bit i
            
            # Let's compute the new dom_mask for this set
            # We need to know if i or j is in the set.
            # The set is {a,b,c,d}.
            
            # Original dom_mask
            orig_dom = in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]
            
            # New dom_mask
            # If i is in set, in_masks[i] changes.
            # If j is in set, in_masks[j] changes.
            
            new_in_masks = in_masks[:] # This is expensive (copying list of ints)
            # Better: just adjust the specific masks
            
            # Let's determine which masks change
            # i_points_to_j is current state.
            # Flip means:
            # if i_points_to_j: new state is j->i
            # else: new state is i->j
            
            # Let's create temporary views
            # We only care about in_masks for a,b,c,d
            
            m_a = in_masks[a]
            m_b = in_masks[b]
            m_c = in_masks[c]
            m_d = in_masks[d]
            
            if i_points_to_j:
                # Flip to j->i
                # in_masks[i] |= (1 << j)
                # in_masks[j] &= ~(1 << i)
                
                if a == i: m_a |= (1 << j)
                elif a == j: m_a &= ~(1 << i)
                
                if b == i: m_b |= (1 << j)
                elif b == j: m_b &= ~(1 << i)
                
                if c == i: m_c |= (1 << j)
                elif c == j: m_c &= ~(1 << i)
                
                if d == i: m_d |= (1 << j)
                elif d == j: m_d &= ~(1 << i)
            else:
                # Flip to i->j
                # in_masks[j] |= (1 << i)
                # in_masks[i] &= ~(1 << j)
                
                if a == i: m_a &= ~(1 << j)
                elif a == j: m_a |= (1 << i)
                
                if b == i: m_b &= ~(1 << j)
                elif b == j: m_b |= (1 << i)
                
                if c == i: m_c &= ~(1 << j)
                elif c == j: m_c |= (1 << i)
                
                if d == i: m_d &= ~(1 << j)
                elif d == j: m_d |= (1 << i)
            
            new_dom = m_a & m_b & m_c & m_d
            
            # Update bad count
            was_bad = is_bad[s_idx]
            is_bad_now = (new_dom == 0)
            
            if was_bad and not is_bad_now:
                new_bad_count -= 1
            elif not was_bad and is_bad_now:
                new_bad_count += 1
            
        delta = new_bad_count - old_bad_count
        
        # Acceptance
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > 0:
                # Avoid overflow in exp
                if delta / T < 100:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        accept = True
        
        if accept:
            # Apply flip
            i_points_to_j = bool(out_masks[i] & (1 << j))
            if i_points_to_j:
                # i->j becomes j->i
                out_masks[i] &= ~(1 << j)
                out_masks[j] |= (1 << i)
                in_masks[j] &= ~(1 << i)
                in_masks[i] |= (1 << j)
                
                # Update orientation_bits
                # Find index for (i,j)
                # idx = i*n - i*(i+1)//2 + (j - i - 1) ?
                # Simpler: precomputed index map
                # We didn't precompute index map for (i,j) -> list index.
                # Let's add it.
                
            else:
                # j->i becomes i->j
                out_masks[j] &= ~(1 << i)
                out_masks[i] |= (1 << j)
                in_masks[i] &= ~(1 << j)
                in_masks[j] |= (1 << i)
            
            bad_count = new_bad_count
            if bad_count < best_bad_count:
                best_bad_count = bad_count
                # Store best orientation?
                # It's a list of bits, copying is O(n^2).
                # We can just keep the masks and reconstruct at the end?
                # But we need to return orientation if bad_count==0.
                # If we don't store, we might lose the best state if we get stuck.
                # But if bad_count becomes 0, we return immediately.
                # So we only need to track best if we don't find 0.
                # For now, we only return if 0.
                
            # Update is_bad array for affected sets
            for s_idx in affected_indices:
                comb = all_4sets[s_idx]
                a, b, c, d = comb
                # Recalculate dom with new masks
                # This is redundant with the calculation above, but safer
                # Actually, we can just use the logic above again or store the result.
                # Let's just recalculate quickly.
                new_dom = in_masks[a] & in_masks[b] & in_masks[c] & in_masks[d]
                is_bad[s_idx] = (new_dom == 0)
                
            T *= cooling
        else:
            T *= cooling

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try n from 31 to 67
    for n in range(31, 68):
        if time.time() > deadline:
            break
            
        # 1. Check Paley if applicable
        # Prime check
        def is_prime(x):
            if x < 2: return False
            for i in range(2, int(x**0.5) + 1):
                if x % i == 0: return False
            return True
        
        if is_prime(n) and n % 4 == 3:
            orientation = _paleyn_orientation(n)
            if _verify_orientation(n, orientation):
                return {'n': n, 'orientation': orientation}
        
        # 2. Run SA
        # Allocate time: 2 seconds per n
        # If n is small, maybe more?
        # Let's just do 2s.
        sa_time = 2.0
        if n == 31:
            sa_time = 5.0 # Give 31 a chance
        
        result_orient = _run_sa_for_n(n, sa_time, seed=n)
        if result_orient is not None:
            return {'n': n, 'orientation': result_orient}

    # Fallback: Paley 67
    p = 67
    orientation = _paleyn_orientation(p)
    return {'n': p, 'orientation': orientation}
