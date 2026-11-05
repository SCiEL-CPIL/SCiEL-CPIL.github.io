import time
import random
from itertools import combinations

def _build_bitmaps(n):
    """Precompute index mapping for triples (a,b,c) -> bit position."""
    # For each pair (a,b), list of c > b such that (a,b,c) is a triple
    pair_to_cs = {}
    triple_index = {}
    idx = 0
    for a in range(n):
        for b in range(a+1, n):
            cs = list(range(b+1, n))
            pair_to_cs[(a,b)] = cs
            for c in cs:
                triple_index[(a,b,c)] = idx
                idx += 1
    return pair_to_cs, triple_index

def _triple_bit_index(n, triple_index, a, b, c):
    """Get the bit position for triple (a,b,c)."""
    return triple_index[(a,b,c)]

def _find_violations(n, colors, triple_index, pair_to_cs):
    """Find all red K4 and blue K5 violations.
    Returns (red_violations, blue_violations) as lists of tuples."""
    n_t = len(colors)
    # Build masks
    # For each pair (a,b), a bitmask of c's where (a,b,c) is red
    red_mask = {}
    blue_mask = {}
    for (a,b), cs in pair_to_cs.items():
        r = 0
        bl = 0
        for c in cs:
            bi = triple_index[(a,b,c)]
            if colors[bi] == 0:
                r |= (1 << c)
            else:
                bl |= (1 << c)
        red_mask[(a,b)] = r
        blue_mask[(a,b)] = bl
    
    red_violations = []
    blue_violations = []
    
    # Find red K4: for each 4-set, check if all 4 triples are red
    # For triple (a,b,c) red, check if there's d > c with all (a,d),(b,d),(c,d) pairs having red at d
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                bi = triple_index[(a,b,c)]
                if colors[bi] == 0:  # red
                    # Check if (a,b,c) is red
                    # Need d such that (a,b,d),(a,c,d),(b,c,d) all red
                    # i.e., d in red_mask[(a,b)] & red_mask[(a,c)] & red_mask[(b,c)] with d > c
                    candidates = red_mask[(a,b)] & red_mask[(a,c)] & red_mask[(b,c)]
                    # Only d > c
                    mask = candidates & ~((1 << (c+1)) - 1)
                    while mask:
                        lb = mask & (-mask)
                        d = lb.bit_length() - 1
                        mask ^= lb
                        red_violations.append((a,b,c,d))
    
    # Find blue K5: for each 5-set, check if all 10 triples are blue
    # For 5 vertices a<b<c<d<e, all 10 triples must be blue
    # Efficient: for triple (a,b,c) blue, count completions
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                bi = triple_index[(a,b,c)]
                if colors[bi] == 1:  # blue
                    # Need d,e > c such that all triples among {a,b,c,d,e} are blue
                    # (a,b,c) is blue already
                    # Need (a,b,d),(a,c,d),(b,c,d) blue: d in blue_mask[(a,b)]&blue_mask[(a,c)]&blue_mask[(b,c)] with d>c
                    cand_d = blue_mask[(a,b)] & blue_mask[(a,c)] & blue_mask[(b,c)]
                    cand_d &= ~((1 << (c+1)) - 1)
                    if cand_d:
                        # For each d, need e > d such that (a,b,e),(a,c,e),(b,c,e),(a,d,e),(b,d,e),(c,d,e) all blue
                        # i.e., e in blue_mask[(a,b)]&blue_mask[(a,c)]&blue_mask[(b,c)]&blue_mask[(a,d)]&blue_mask[(b,d)]&blue_mask[(c,d)] with e>d
                        while cand_d:
                            lb = cand_d & (-cand_d)
                            d = lb.bit_length() - 1
                            cand_d ^= lb
                            cand_e = blue_mask[(a,b)] & blue_mask[(a,c)] & blue_mask[(b,c)]
                            cand_e &= blue_mask[(a,d)] & blue_mask[(b,d)] & blue_mask[(c,d)]
                            cand_e &= ~((1 << (d+1)) - 1)
                            while cand_e:
                                lb2 = cand_e & (-cand_e)
                                e = lb2.bit_length() - 1
                                cand_e ^= lb2
                                blue_violations.append((a,b,c,d,e))
    
    return red_violations, blue_violations

def _count_violations_fast(n, colors, triple_index, pair_to_cs):
    """Count red K4 and blue K5 violations using bitmask approach."""
    # Build masks
    red_mask = {}
    blue_mask = {}
    for (a,b), cs in pair_to_cs.items():
        r = 0
        bl = 0
        for c in cs:
            bi = triple_index[(a,b,c)]
            if colors[bi] == 0:
                r |= (1 << c)
            else:
                bl |= (1 << c)
        red_mask[(a,b)] = r
        blue_mask[(a,b)] = bl
    
    red_count = 0
    blue_count = 0
    
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                bi = triple_index[(a,b,c)]
                if colors[bi] == 0:
                    candidates = red_mask[(a,b)] & red_mask[(a,c)] & red_mask[(b,c)]
                    mask = candidates & ~((1 << (c+1)) - 1)
                    red_count += bin(mask).count('1')
                else:
                    cand_d = blue_mask[(a,b)] & blue_mask[(a,c)] & blue_mask[(b,c)]
                    cand_d &= ~((1 << (c+1)) - 1)
                    while cand_d:
                        lb = cand_d & (-cand_d)
                        d = lb.bit_length() - 1
                        cand_d ^= lb
                        cand_e = blue_mask[(a,b)] & blue_mask[(a,c)] & blue_mask[(b,c)]
                        cand_e &= blue_mask[(a,d)] & blue_mask[(b,d)] & blue_mask[(c,d)]
                        cand_e &= ~((1 << (d+1)) - 1)
                        blue_count += bin(cand_e).count('1')
    
    return red_count, blue_count

def _local_search(n, colors, triple_index, pair_to_cs, deadline, rng):
    """Run local search to eliminate violations."""
    n_t = len(colors)
    colors = list(colors)
    
    # Build initial masks
    def build_masks(colors):
        red_mask = {}
        blue_mask = {}
        for (a,b), cs in pair_to_cs.items():
            r = 0
            bl = 0
            for c in cs:
                bi = triple_index[(a,b,c)]
                if colors[bi] == 0:
                    r |= (1 << c)
                else:
                    bl |= (1 << c)
            red_mask[(a,b)] = r
            blue_mask[(a,b)] = bl
        return red_mask, blue_mask
    
    def update_masks_for_flip(red_mask, blue_mask, a, b, c, new_color):
        """Update masks after flipping triple (a,b,c) to new_color."""
        pairs = [(a,b), (a,c), (b,c)]
        for p in pairs:
            old = red_mask[p]
            bl_old = blue_mask[p]
            bit = 1 << c
            if new_color == 0:  # red
                red_mask[p] = old | bit
                blue_mask[p] = bl_old & ~bit
            else:  # blue
                red_mask[p] = old & ~bit
                blue_mask[p] = bl_old | bit
    
    def count_violations_with_masks(red_mask, blue_mask):
        red_count = 0
        blue_count = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    bi = triple_index[(a,b,c)]
                    if colors[bi] == 0:
                        candidates = red_mask[(a,b)] & red_mask[(a,c)] & red_mask[(b,c)]
                        mask = candidates & ~((1 << (c+1)) - 1)
                        red_count += bin(mask).count('1')
                    else:
                        cand_d = blue_mask[(a,b)] & blue_mask[(a,c)] & blue_mask[(b,c)]
                        cand_d &= ~((1 << (c+1)) - 1)
                        while cand_d:
                            lb = cand_d & (-cand_d)
                            d = lb.bit_length() - 1
                            cand_d ^= lb
                            cand_e = blue_mask[(a,b)] & blue_mask[(a,c)] & blue_mask[(b,c)]
                            cand_e &= blue_mask[(a,d)] & blue_mask[(b,d)] & blue_mask[(c,d)]
                            cand_e &= ~((1 << (d+1)) - 1)
                            blue_count += bin(cand_e).count('1')
        return red_count, blue_count
    
    red_mask, blue_mask = build_masks(colors)
    red_count, blue_count = count_violations_with_masks(red_mask, blue_mask)
    total = red_count + blue_count
    
    if total == 0:
        return colors, 0
    
    max_iters = 5000
    no_improve = 0
    best_total = total
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if total == 0:
            break
        
        # Find a violating triple to flip
        # Strategy: find a triple that appears in a violation and flip it
        # Pick a random triple weighted by how many violations it's in
        # For simplicity: pick from known violation patterns
        
        # Find all red K4 violations
        red_viols = []
        blue_viols = []
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    bi = triple_index[(a,b,c)]
                    if colors[bi] == 0:
                        candidates = red_mask[(a,b)] & red_mask[(a,c)] & red_mask[(b,c)]
                        mask = candidates & ~((1 << (c+1)) - 1)
                        while mask:
                            lb = mask & (-mask)
                            d = lb.bit_length() - 1
                            mask ^= lb
                            red_viols.append((a,b,c,d))
                    else:
                        cand_d = blue_mask[(a,b)] & blue_mask[(a,c)] & blue_mask[(b,c)]
                        cand_d &= ~((1 << (c+1)) - 1)
                        while cand_d:
                            lb = cand_d & (-cand_d)
                            d = lb.bit_length() - 1
                            cand_d ^= lb
                            cand_e = blue_mask[(a,b)] & blue_mask[(a,c)] & blue_mask[(b,c)]
                            cand_e &= blue_mask[(a,d)] & blue_mask[(b,d)] & blue_mask[(c,d)]
                            cand_e &= ~((1 << (d+1)) - 1)
                            while cand_e:
                                lb2 = cand_e & (-cand_e)
                                e = lb2.bit_length() - 1
                                cand_e ^= lb2
                                blue_viols.append((a,b,c,d,e))
        
        if not red_viols and not blue_viols:
            break
        
        # Choose a triple to flip
        flip_candidates = set()
        for (a,b,c,d) in red_viols[:50]:  # limit for speed
            # The 4 triples in this K4: (a,b,c),(a,b,d),(a,c,d),(b,c,d)
            flip_candidates.add((a,b,c))
            flip_candidates.add((a,b,d))
            flip_candidates.add((a,c,d))
            flip_candidates.add((b,c,d))
        for (a,b,c,d,e) in blue_viols[:30]:
            # 10 triples in K5
            verts = [a,b,c,d,e]
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        flip_candidates.add(tuple(sorted([verts[i], verts[j], verts[k]])))
        
        if not flip_candidates:
            # Random flip
            idx = rng.randrange(n_t)
            a, b, c = next((a,b,c) for (a,b,c), bi in triple_index.items() if bi == idx)
        else:
            # Pick one that reduces violations most (try a few)
            flip_list = list(flip_candidates)
            rng.shuffle(flip_list)
            best_flip = None
            best_delta = float('inf')
            for trial in flip_list[:10]:
                a, b, c = trial
                bi = triple_index.get((a,b,c))
                if bi is None:
                    continue
                old_color = colors[bi]
                new_color = 1 - old_color
                # Estimate: just flip and see
                # For speed, just pick first
                best_flip = (a, b, c, new_color)
                break
            if best_flip is None:
                idx = rng.randrange(n_t)
                a, b, c = next((a,b,c) for (a,b,c), bi in triple_index.items() if bi == idx)
                best_flip = (a, b, c, 1 - colors[idx])
            
            a, b, c, new_color = best_flip
        
        bi = triple_index[(a,b,c)]
        old_color = colors[bi]
        new_color = 1 - old_color
        colors[bi] = new_color
        update_masks_for_flip(red_mask, blue_mask, a, b, c, new_color)
        
        new_red, new_blue = count_violations_with_masks(red_mask, blue_mask)
        new_total = new_red + new_blue
        
        if new_total < total:
            total = new_total
            no_improve = 0
            if total < best_total:
                best_total = total
        else:
            # Revert
            colors[bi] = old_color
            update_masks_for_flip(red_mask, blue_mask, a, b, c, old_color)
            no_improve += 1
            if no_improve > 200:
                # Random restart: flip random set of triples
                for _ in range(5):
                    idx = rng.randrange(n_t)
                    a, b, c = next((a,b,c) for (a,b,c), bi in triple_index.items() if bi == idx)
                    new_c = 1 - colors[idx]
                    colors[idx] = new_c
                    update_masks_for_flip(red_mask, blue_mask, a, b, c, new_c)
                no_improve = 0
                total = best_total
    
    return colors, total

def _initial_coloring(n, strategy, rng):
    """Generate an initial coloring using various strategies."""
    n_t = sum(1 for _ in combinations(range(n), 3))
    
    if strategy == 'sum_mod3':
        # Red if sum % 3 == 0
        colors = []
        for a, b, c in combinations(range(n), 3):
            colors.append(0 if (a + b + c) % 3 == 0 else 1)
    elif strategy == 'sum_mod2':
        colors = []
        for a, b, c in combinations(range(n), 3):
            colors.append(0 if (a + b + c) % 2 == 0 else 1)
    elif strategy == 'product_mod3':
        colors = []
        for a, b, c in combinations(range(n), 3):
            colors.append(0 if (a * b + b * c + c * a) % 3 == 0 else 1)
    elif strategy == 'random':
        colors = [rng.randint(0, 1) for _ in range(n_t)]
    elif strategy == 'biased_random':
        colors = [rng.randint(0, 1) for _ in range(n_t)]
        # Bias toward red for small indices
        for i, (a, b, c) in enumerate(combinations(range(n), 3)):
            if a + b + c < n * 1.5:
                colors[i] = 0
    elif strategy == 'cyclic':
        colors = []
        for a, b, c in combinations(range(n), 3):
            # Red if (b-a) and (c-b) have same parity
            colors.append(0 if ((b-a) % 2 == (c-b) % 2) else 1)
    elif strategy == 'difference_mod3':
        colors = []
        for a, b, c in combinations(range(n), 3):
            colors.append(0 if ((b-a) + (c-b)) % 3 == 0 else 1)
    else:
        colors = [rng.randint(0, 1) for _ in range(n_t)]
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    rng = random.Random(42)
    
    # Try from n=34 down to n=9
    best_result = None
    best_n = 0
    
    for n in range(34, 8, -1):
        if time.time() > deadline - 5:
            break
        
        pair_to_cs, triple_index = _build_bitmaps(n)
        n_t = len(triple_index)
        
        # Try multiple strategies and restarts
        strategies = ['sum_mod3', 'cyclic', 'difference_mod3', 'sum_mod2', 'product_mod3', 'random', 'biased_random']
        
        found_valid = False
        for strategy in strategies:
            if time.time() > deadline - 3:
                break
            
            colors = _initial_coloring(n, strategy, rng)
            
            # Quick check
            red_count, blue_count = _count_violations_fast(n, colors, triple_index, pair_to_cs)
            if red_count == 0 and blue_count == 0:
                found_valid = True
                break
            
            # Run local search
            colors, total_violations = _local_search(n, colors, triple_index, pair_to_cs, deadline - 3, rng)
            
            if total_violations == 0:
                found_valid = True
                break
        
        if found_valid:
            # Verify
            pair_to_cs_v, triple_index_v = _build_bitmaps(n)
            red_count, blue_count = _count_violations_fast(n, colors, triple_index_v, pair_to_cs_v)
            if red_count == 0 and blue_count == 0:
                best_result = {'n': n, 'colors': list(colors)}
                best_n = n
                break
            else:
                # Try more restarts
                for restart in range(3):
                    if time.time() > deadline - 3:
                        break
                    colors = _initial_coloring(n, 'random', rng)
                    colors, total_violations = _local_search(n, colors, triple_index, pair_to_cs, deadline - 3, rng)
                    if total_violations == 0:
                        red_count, blue_count = _count_violations_fast(n, colors, triple_index_v, pair_to_cs_v)
                        if red_count == 0 and blue_count == 0:
                            best_result = {'n': n, 'colors': list(colors)}
                            best_n = n
                            found_valid = True
                            break
        
        if time.time() > deadline - 5:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return a small valid coloring
    # n=9 with sum_mod3 is valid
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
