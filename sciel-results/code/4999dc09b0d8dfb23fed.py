"""Rewritten: graph/local-search approach for 4-term-progression 3-color constraint."""
import time
import random
import numpy as np

# Precompute all non-constant 4-term progressions among positions 0..127
# A 4-term progression: (a, a+d, a+2d, a+3d) with d>=1 and a+3d <= 127
PROGS = []
for a in range(128):
    for d in range(1, (128 - a) // 3 + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        if e < 128:
            PROGS.append((a, b, c, e))

NP = len(PROGS)

def count_violations(colors):
    """Count 4-term progressions that use fewer than 3 distinct colors."""
    v = 0
    cols = colors
    for (a, b, c, e) in PROGS:
        ca, cb, cc, ce = cols[a], cols[b], cols[c], cols[e]
        # fewer than 3 distinct colors
        if ca == cb == cc == ce:
            v += 1
        elif ca == cb and ca == cc and ca == ce:
            v += 1
        elif ca == cb and ca == ce and cb == cc:
            v += 1
        elif ca == cb and ca == ce and cb == ce:
            v += 1
        elif ca == cb and ca == ce:
            v += 1
        elif ca == cc and ca == ce and cb == cc:
            v += 1
        elif ca == cc and ca == ce and cb == ce:
            v += 1
        elif ca == cc and ca == ce:
            v += 1
        elif ca == ce and cb == cc and cb == ce:
            v += 1
        elif ca == ce and cb == ce:
            v += 1
        elif ca == cb and cb == ce:
            v += 1
        elif ca == cc and cc == ce:
            v += 1
        elif cb == cc and cb == ce:
            v += 1
        # Simpler: count distinct
        # Actually just use set size check via small logic
    # The above is error prone; use a compact distinct-count:
    v = 0
    for (a, b, c, e) in PROGS:
        ca, cb, cc, ce = cols[a], cols[b], cols[c], cols[e]
        # distinct count < 3  <=>  (ca==cb and ca==cc and ca==ce) or
        #                         (ca==cb and ca==cc and cb==ce) etc.
        # Easier: 
        if ca == cb:
            if ca == cc:
                # ca==cb==cc; need ce != ca for 3 distinct? No: if ce==ca then 1 distinct; if ce!=ca then 2 distinct
                if ce == ca:
                    v += 1
                else:
                    v += 1  # 2 distinct
            else:
                # ca==cb != cc
                if ce == ca:
                    v += 1  # 2 distinct
                elif ce == cc:
                    v += 1  # 2 distinct
                else:
                    pass    # 3 distinct
        else:
            # ca != cb
            if ca == cc:
                # ca==cc != cb
                if ce == ca:
                    v += 1  # 2 distinct
                elif ce == cb:
                    v += 1  # 2 distinct
                else:
                    pass
            else:
                # ca != cb, ca != cc
                if cb == cc:
                    # cb==cc != ca
                    if ce == ca:
                        v += 1  # 2 distinct
                    elif ce == cb:
                        v += 1  # 2 distinct
                    else:
                        pass
                else:
                    # all ca,cb,cc distinct
                    pass  # at least 3 distinct already
    return v

def count_violations_fast(colors):
    """Faster: use numpy where possible, but PROGS is list of tuples; do vectorized."""
    cols = np.asarray(colors)
    # Build arrays of indices
    idx = np.array(PROGS)  # shape (NP,4)
    ca = cols[idx[:,0]]
    cb = cols[idx[:,1]]
    cc = cols[idx[:,2]]
    ce = cols[idx[:,3]]
    # distinct < 3 means: the set of {ca,cb,cc,ce} has size 1 or 2
    # size 1: all equal
    # size 2: exactly two values, each appearing at least twice, OR one value thrice + one once
    # Simpler: count distinct via (ca!=cb)+(ca!=cc)+(ca!=ce)+(cb!=cc)+(cb!=ce)+(cc!=ce)
    # If all equal: 0 differences
    # If 2 distinct: at least one pair equal and at least one pair different
    # We want: NOT (3 or 4 distinct). 
    # 4 distinct: all 6 pairs differ
    # 3 distinct: exactly one pair equal (the two of the repeated color) and rest differ
    # 2 distinct: at least 3 pairs equal (since with 2 colors on 4 items, by pigeonhole one color has >=2, so at least 1 equal pair; actually min equal pairs for 2 distinct on 4 items is 1 (2+2) -> 2 equal pairs? 
    # Let's just compute distinct count per progression via a small loop in numpy isn't trivial.
    # Use: distinct = 4 - (number of "collisions" counted cleverly) is messy.
    # Alternative: for each progression, compute max - min == 0 -> 1 distinct; else if (max-min) small... no.
    # Just do: eq12 = (ca==cb); eq13=(ca==cc); eq14=(ca==ce); eq23=(cb==cc); eq24=(cb==ce); eq34=(cc==ce)
    eq12 = (ca==cb).astype(np.int8)
    eq13 = (ca==cc).astype(np.int8)
    eq14 = (ca==ce).astype(np.int8)
    eq23 = (cb==cc).astype(np.int8)
    eq24 = (cb==ce).astype(np.int8)
    eq34 = (cc==ce).astype(np.int8)
    # A progression is BAD (fewer than 3 distinct) iff:
    # - 1 distinct: all 6 eq = 1
    # - 2 distinct: not all 6 eq, but the equality graph is connected with 2 components of sizes (2,2) or (3,1) or (1,3) or (2,2)
    #   In terms of eq counts: for 2 distinct, the number of equal pairs is at least 1 and the structure must be that all equal pairs involve only 2 values.
    # Simplest robust: compute distinct count as:
    #   d = 1 + (eq12==0) + (eq13==0 and eq12==0) + (eq14==0 and eq13==0 and eq12==0)
    #   but that's not right either.
    # Let's just do: for each row, distinct = number of unique values.
    # We'll do it with a small python loop over NP (which is ~ (128*128/6) ~ 2730, actually let's compute: sum over a of (128-a)//3 ~ 128*128/6 ~ 2730). 2730 iterations in python is fine.
    v = 0
    for i in range(NP):
        a = idx[i,0]; b = idx[i,1]; c = idx[i,2]; e = idx[i,3]
        ca_, cb_, cc_, ce_ = cols[a], cols[b], cols[c], cols[e]
        # count distinct
        if ca_ == cb_:
            if ca_ == cc_:
                if ca_ == ce_:
                    v += 1  # 1 distinct
                else:
                    v += 1  # 2 distinct
            else:
                if ce_ == ca_:
                    v += 1  # 2
                elif ce_ == cc_:
                    v += 1  # 2
                else:
                    pass
        else:
            if ca_ == cc_:
                if ce_ == ca_:
                    v += 1
                elif ce_ == cb_:
                    v += 1
                else:
                    pass
            else:
                if cb_ == cc_:
                    if ce_ == ca_:
                        v += 1
                    elif ce_ == cb_:
                        v += 1
                    else:
                        pass
                else:
                    pass
    return v

def construct():
    deadline = time.time() + 80.0
    rng = random.Random(42)
    
    best_colors = None
    best_k = 128
    best_viol = None
    
    # Start with a feasible solution: use the greedy from seed (reimplemented) to get a baseline
    # Actually, let's start with random assignments and run local search.
    
    def greedy_feasible():
        colors = [0]*128
        for x in range(128):
            # try colors 0..127, pick first that keeps all progressions ending at x valid
            # but we also need progressions not yet complete... this is the original greedy.
            # For a feasible start, use 4 colors via mod pattern? 
            # A known construction: color by (i mod 4) might not work. 
            # Let's just use the original greedy logic but optimized.
            pass
        return None
    
    # Use the original seed's greedy to get a feasible starting point (it's slow but once is fine)
    # Actually the original is O(n^3) ~ 128^3 = 2M, acceptable.
    def seed_greedy():
        colors = []
        for x in range(1, 129):
            found = False
            for color in range(128):
                ok = True
                # Check all 4-term progressions where x is the last element
                # x = a+3d => a = x-1-3d in 0-index? Let's use 0-index: position x-1
                pos = x - 1
                for d in range(1, (pos)//3 + 1):
                    a = pos - 3*d
                    if a < 0:
                        break
                    b = a + d
                    c = a + 2*d
                    # colors at a,b,c are already set
                    cs = {colors[a], colors[b], colors[c], color}
                    if len(cs) < 3:
                        ok = False
                        break
                if ok:
                    colors.append(color)
                    found = True
                    break
            if not found:
                # Should not happen with 128 colors
                colors.append(0)
        return colors
    
    start_colors = seed_greedy()
    start_k = len(set(start_colors))
    best_colors = list(start_colors)
    best_k = start_k
    
    # Local search: min-conflicts style
    # State: colors array. Objective: minimize distinct colors, subject to 0 violations.
    # Approach: 
    #   1. Try to reduce number of colors by merging color classes.
    #   2. Use local search to fix violations.
    
    # Better: iterative improvement
    #   - Start with a random coloring using k colors
    #   - Run local search to eliminate violations
    #   - If successful, try to reduce k by 1 and repeat
    
    def local_search(colors, max_iters=200000, deadline_t=None):
        """Min-conflicts: pick random position, change to color that minimizes violations."""
        cols = list(colors)
        # Precompute for each position, which progressions it's in
        pos_to_progs = [[] for _ in range(128)]
        for i, (a,b,c,e) in enumerate(PROGS):
            pos_to_progs[a].append(i)
            pos_to_progs[b].append(i)
            pos_to_progs[c].append(i)
            pos_to_progs[e].append(i)
        
        v = count_violations_fast(cols)
        for it in range(max_iters):
            if deadline_t is not None and time.time() > deadline_t:
                break
            if v == 0:
                return cols, 0
            # Pick a random position involved in a violation
            # For simplicity, pick random position
            pos = rng.randint(0, 127)
            old = cols[pos]
            # Try all colors 0..127, pick one that minimizes violations
            best_color = old
            best_v = v
            # Only try colors that are currently in use or a few extras
            used_colors = set(cols)
            candidates = list(used_colors)
            if len(candidates) < 128:
                candidates += [rng.randint(0,127) for _ in range(3)]
            for newc in candidates:
                if newc == old:
                    continue
                cols[pos] = newc
                nv = count_violations_fast(cols)
                cols[pos] = old
                if nv < best_v:
                    best_v = nv
                    best_color = newc
            cols[pos] = best_color
            v = best_v
        return cols, v
    
    # Try to reduce number of colors
    for target_k in range(best_k, 1, -1):
        if time.time() > deadline:
            break
        # Try to find a valid coloring with <= target_k colors
        # Start from best_colors, try to merge colors
        success = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            # Create a random assignment using target_k colors
            colors = [rng.randint(0, target_k-1) for _ in range(128)]
            # Run local search
            colors, v = local_search(colors, max_iters=50000, deadline_t=deadline)
            if v == 0:
                k_used = len(set(colors))
                if k_used <= target_k:
                    if k_used < best_k:
                        best_k = k_used
                        best_colors = colors
                    success = True
                    break
        
        if success:
            continue
        else:
            # Try one more time with more iterations
            if time.time() < deadline - 5:
                colors = [rng.randint(0, target_k-1) for _ in range(128)]
                colors, v = local_search(colors, max_iters=200000, deadline_t=deadline)
                if v == 0:
                    k_used = len(set(colors))
                    if k_used < best_k:
                        best_k = k_used
                        best_colors = colors
                break
    
    # Final: ensure we return a valid coloring
    # If best_colors has violations, run one more local search
    v = count_violations_fast(best_colors)
    if v > 0:
        best_colors, v = local_search(best_colors, max_iters=100000, deadline_t=deadline)
    
    # Map colors to 0..127 (they already are)
    return best_colors

# Precompute PROGS at module level
PROGS = []
for a in range(128):
    for d in range(1, (128 - a) // 3 + 1):
        e = a + 3 * d
        if e < 128:
            PROGS.append((a, a+d, a+2*d, e))
NP = len(PROGS)
