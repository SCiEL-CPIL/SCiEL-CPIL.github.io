# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys
from collections import defaultdict
from itertools import product

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a
    nonzero square mod 205.

    Uses CRT decomposition (205 = 5 * 41) to:
      - Precompute component-level conflict structures
      - Generate high-quality initial solutions via structured construction
      - Guide local search with component-aware move generation
      - Bitset-based conflict graph for fast compatibility checks
      - Simulated annealing with efficient precomputed moves
      - 1-1, 2-2, and 3-3 swap moves
      - Multi-start with CRT-informed initializations
      - Thorough local search polish after each annealing run
    """
    m = 205
    deadline = time.time() + 580  # safety margin before 630s
    p1, p2 = 5, 41  # 205 = 5 * 41

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_set = forbidden

    # CRT structure: map each residue r in Z/205 to (r%5, r%41)
    # Precompute component-level square sets
    sq_mod5 = set()
    for x in range(1, 5):
        sq_mod5.add(pow(x, 2, 5))
    sq_mod5.discard(0)
    
    sq_mod41 = set()
    for x in range(1, 41):
        sq_mod41.add(pow(x, 2, 41))
    sq_mod41.discard(0)
    
    # A difference d is forbidden in Z/205 iff d%5 is a nonzero square in Z/5
    # OR d%41 is a nonzero square in Z/41 (since d is a square mod 205 iff
    # d is a square mod 5 AND d is a square mod 41 by CRT)
    # Wait - let me reconsider. d is a square mod 205 iff d is a square mod 5
    # AND d is a square mod 41. So forbidden means d IS a square mod 205.
    # So (a-b) mod 205 must NOT be a square mod 205.
    # (a-b) mod 205 is a square mod 205 iff (a-b)%5 is a square mod 5 
    # AND (a-b)%41 is a square mod 41.
    # So we need: NOT[(a-b)%5 in sq_mod5 AND (a-b)%41 in sq_mod41]
    # i.e., (a-b)%5 not in sq_mod5 OR (a-b)%41 not in sq_mod41

    # Build conflict graph as bitsets: conflict_mask[v] is a 205-bit int
    # where bit u is set iff (v - u) % m is in forbidden_set
    conflict_mask = [0] * m
    for v in range(m):
        mask = 0
        for u in range(m):
            if u == v:
                continue
            d = (v - u) % m
            if d in forbidden_set:
                mask |= (1 << u)
        conflict_mask[v] = mask

    # Component-level compatibility: two residues (a1,a2) and (b1,b2) in Z/5 x Z/41
    # are compatible iff NOT[(a1-b1)%5 in sq_mod5 AND (a2-b2)%41 in sq_mod41]
    # Precompute for each pair of components whether they conflict
    def components_conflict(a1, a2, b1, b2):
        d1 = (a1 - b1) % 5
        d2 = (a2 - b2) % 41
        if d1 == 0 and d2 == 0:
            return False  # same element, not a conflict
        return (d1 in sq_mod5) and (d2 in sq_mod41)

    # Precompute component conflict tables
    # comp_conflict_5[a1][b1] = set of b2 values that conflict with (a1, a2) for any a2
    # Actually, let's precompute: for each (a1, a2), which (b1, b2) conflict
    # This is a 5x41 x 5x41 table = 205x205 = 42025 entries, manageable
    
    # For efficiency, precompute for each residue v in Z/205, its components
    v_comp = [(v % 5, v % 41) for v in range(m)]
    
    # Precompute: for each v, which w conflict with v (as a bitmask)
    # Already have conflict_mask, but let's also have component-aware views
    # comp_conflict[v][w] = True iff v and w conflict
    # We already have this via conflict_mask
    
    # For CRT-informed initialization: build sets by component
    # A set S in Z/205 projects to S1 in Z/5 and S2 in Z/41
    # For S to be valid: for any a,b in S, NOT[(a-b)%5 in sq_mod5 AND (a-b)%41 in sq_mod41]
    # This means: for each pair (a1,b1) in S1 with (a1-b1)%5 in sq_mod5,
    # we need (a2-b2)%41 NOT in sq_mod41 for the corresponding (a2,b2) in S2
    # This couples the components, so we can't fully separate them.
    # But we can use component structure to guide search.
    
    # Precompute: for each residue r in Z/205, its "component signature"
    # which residues have the same (mod 5) class
    mod5_classes = defaultdict(list)
    mod41_classes = defaultdict(list)
    for v in range(m):
        mod5_classes[v % 5].append(v)
        mod41_classes[v % 41].append(v)
    
    # For each pair of mod-5 classes (c1, c2) where (c1-c2)%5 is a nonzero square,
    # any two elements from these classes conflict ONLY if their mod-41 difference
    # is also a nonzero square. So within such class pairs, we can still pick
    # multiple elements if their mod-41 differences avoid squares.
    
    # Identify "dangerous" mod-5 class pairs
    dangerous_mod5_pairs = set()
    for c1 in range(5):
        for c2 in range(5):
            if c1 != c2 and (c1 - c2) % 5 in sq_mod5:
                dangerous_mod5_pairs.add((c1, c2))
    
    # Similarly for mod-41
    dangerous_mod41_pairs = set()
    for c1 in range(41):
        for c2 in range(41):
            if c1 != c2 and (c1 - c2) % 41 in sq_mod41:
                dangerous_mod41_pairs.add((c1, c2))

    def is_compatible(candidate, selected_bits):
        """Check if candidate is compatible with all elements in selected_bits (a bitmask)."""
        return (conflict_mask[candidate] & selected_bits) == 0

    def popcount(x):
        return bin(x).count('1')

    def bits_to_list(bits):
        """Convert bitmask to list of vertex indices."""
        result = []
        t = bits
        while t:
            low = t & (-t)
            idx = low.bit_length() - 1
            result.append(idx)
            t ^= low
        return result

    def crt_greedy_init():
        """
        CRT-informed greedy initialization.
        Strategy: process residues in an order that respects component structure.
        For each mod-5 class, try to add elements whose mod-41 component
        avoids conflicts with already-selected elements.
        """
        # Shuffle the mod-5 classes
        class_order = list(range(5))
        random.shuffle(class_order)
        
        selected_bits = 0
        selected_set = set()
        
        for c5 in class_order:
            # Get all residues in this mod-5 class
            candidates = mod5_classes[c5][:]
            random.shuffle(candidates)
            
            for v in candidates:
                if is_compatible(v, selected_bits):
                    selected_bits |= (1 << v)
                    selected_set.add(v)
        
        return selected_bits

    def crt_greedy_init_v2():
        """
        Alternative CRT-informed greedy: process by mod-41 classes, 
        within each class try to maximize mod-5 diversity.
        """
        class_order = list(range(41))
        random.shuffle(class_order)
        
        selected_bits = 0
        selected_set = set()
        
        for c41 in class_order:
            candidates = mod41_classes[c41][:]
            random.shuffle(candidates)
            
            for v in candidates:
                if is_compatible(v, selected_bits):
                    selected_bits |= (1 << v)
                    selected_set.add(v)
        
        return selected_bits

    def crt_greedy_init_v3():
        """
        CRT-informed greedy with component-aware scoring.
        Score each candidate by how many new mod-5 and mod-41 components it covers.
        """
        selected_bits = 0
        selected_set = set()
        used_mod5 = set()
        used_mod41 = set()
        
        remaining = list(range(m))
        random.shuffle(remaining)
        
        for v in remaining:
            if not is_compatible(v, selected_bits):
                continue
            
            # Score: prefer vertices that add new component coverage
            score = 0
            if v % 5 not in used_mod5:
                score += 2
            if v % 41 not in used_mod41:
                score += 1
            
            # With some probability, accept based on score
            if score > 0 or random.random() < 0.3:
                selected_bits |= (1 << v)
                selected_set.add(v)
                used_mod5.add(v % 5)
                used_mod41.add(v % 41)
        
        return selected_bits

    def random_greedy():
        """Randomized greedy construction."""
        order = list(range(m))
        random.shuffle(order)
        selected_bits = 0
        for v in order:
            if is_compatible(v, selected_bits):
                selected_bits |= (1 << v)
        return selected_bits

    def local_search_add_only(selected_bits, max_iters=1000):
        """
        Local search: try to add any compatible vertex repeatedly.
        Returns improved bitmask.
        """
        current = selected_bits
        current_size = popcount(current)
        best = current
        best_size = current_size

        for it in range(max_iters):
            if time.time() > deadline:
                break

            improved = False
            temp = ~current & ((1 << m) - 1)
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                if is_compatible(v, current):
                    current = current | (1 << v)
                    current_size += 1
                    improved = True
                    break
                temp ^= low
            if improved:
                if current_size > best_size:
                    best_size = current_size
                    best = current
                continue
            else:
                break

        return best, best_size

    def local_search_remove_add(selected_bits, max_iters=2000):
        """
        Local search that can remove and re-add vertices to escape local optima.
        Uses CRT component information to guide which vertices to remove.
        """
        current = selected_bits
        current_size = popcount(current)
        best = current
        best_size = current_size
        vertices = bits_to_list(current)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Try to add
            temp = ~current & ((1 << m) - 1)
            added = False
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                if is_compatible(v, current):
                    current = current | (1 << v)
                    current_size += 1
                    added = True
                    break
                temp ^= low

            if added:
                if current_size > best_size:
                    best_size = current_size
                    best = current
                continue

            # No add possible, try remove+add with CRT-informed removal
            if current_size > 0:
                vertices = bits_to_list(current)
                
                # CRT-informed removal: prefer removing vertices that are
                # in "dense" component classes (more conflicts)
                # Count conflicts per vertex
                conflict_counts = []
                for v in vertices:
                    # Count how many other selected vertices conflict with v
                    # (they all do, by definition, but we want to know which
                    # removals free up the most space)
                    # Actually, all pairs conflict or not. Let's count how many
                    # non-selected vertices would become compatible if we remove v.
                    trial = current & ~(1 << v)
                    freed = 0
                    t = ~trial & ((1 << m) - 1)
                    while t:
                        low2 = t & (-t)
                        w = low2.bit_length() - 1
                        if is_compatible(w, trial) and not is_compatible(w, current):
                            freed += 1
                        t ^= low2
                    conflict_counts.append((freed, v))
                
                # Remove the vertex that frees the most space
                conflict_counts.sort(reverse=True)
                rem = conflict_counts[0][1]
                trial = current & ~(1 << rem)
                trial_size = current_size - 1

                # Try to add two vertices to the trial
                temp2 = ~trial & ((1 << m) - 1)
                add_count = 0
                while temp2 and add_count < 2:
                    low = temp2 & (-temp2)
                    v = low.bit_length() - 1
                    if is_compatible(v, trial):
                        trial = trial | (1 << v)
                        trial_size += 1
                        add_count += 1
                    temp2 ^= low

                if trial_size > current_size:
                    current = trial
                    current_size = trial_size
                    if current_size > best_size:
                        best_size = current_size
                        best = current

        return best, best_size

    # (local_search_remove_add already defined above with CRT-informed logic)

    def simulated_annealing(initial_bits, start_temp=1.0, end_temp=0.001, max_iters=10000):
        """
        Simulated annealing with 1-1, 2-2, and 3-3 swap moves.
        Uses CRT component information to guide move generation.
        Returns (best_bits, best_size).
        """
        current = initial_bits
        current_size = popcount(current)
        best = current
        best_size = current_size

        n_iters = max_iters
        for it in range(n_iters):
            if time.time() > deadline:
                break

            # Exponential cooling
            temp = start_temp * (end_temp / start_temp) ** (it / n_iters)

            # Choose move type
            r = random.random()
            if r < 0.35:
                # 1-1 swap with CRT-informed removal
                if current_size > 0:
                    vertices = bits_to_list(current)
                    
                    # CRT-informed: prefer removing vertices whose mod-5 class
                    # has many elements (more "redundant")
                    mod5_count = defaultdict(int)
                    for v in vertices:
                        mod5_count[v % 5] += 1
                    
                    # Weight removal by inverse of class size (remove from larger classes)
                    weights = []
                    for v in vertices:
                        weights.append(mod5_count[v % 5])
                    
                    total_w = sum(weights)
                    if total_w > 0:
                        r = random.random() * total_w
                        cum = 0
                        rem = vertices[0]
                        for i, w in enumerate(weights):
                            cum += w
                            if r <= cum:
                                rem = vertices[i]
                                break
                    else:
                        rem = random.choice(vertices)
                    
                    trial = current & ~(1 << rem)

                    # Find compatible candidates to add
                    cands = []
                    t = ~trial & ((1 << m) - 1)
                    while t:
                        low = t & (-t)
                        idx = low.bit_length() - 1
                        if is_compatible(idx, trial):
                            cands.append(idx)
                        t ^= low

                    if cands:
                        # CRT-informed: prefer adding from mod-5 classes with fewer elements
                        mod5_count_new = defaultdict(int)
                        for v in bits_to_list(trial):
                            mod5_count_new[v % 5] += 1
                        
                        # Score candidates by mod-5 class scarcity
                        scored_cands = []
                        for c in cands:
                            score = 1.0 / (1.0 + mod5_count_new[c % 5])
                            scored_cands.append((score, c))
                        scored_cands.sort(key=lambda x: x[0], reverse=True)
                        
                        # Pick from top candidates with some randomness
                        top_k = min(len(scored_cands), 5)
                        add_v = scored_cands[random.randint(0, top_k - 1)][1]
                        new_state = trial | (1 << add_v)
                        new_size = current_size
                    else:
                        new_state = current
                        new_size = current_size
                else:
                    new_state = current
                    new_size = current_size

            elif r < 0.65:
                # 2-2 swap
                if current_size >= 2:
                    vertices = bits_to_list(current)
                    rem1, rem2 = random.sample(vertices, 2)
                    trial = current & ~(1 << rem1) & ~(1 << rem2)

                    # Find compatible candidates
                    cands = []
                    t = ~trial & ((1 << m) - 1)
                    while t:
                        low = t & (-t)
                        idx = low.bit_length() - 1
                        if is_compatible(idx, trial):
                            cands.append(idx)
                        t ^= low

                    if len(cands) >= 2:
                        add1, add2 = random.sample(cands, 2)
                        # Check if add2 is compatible with trial + add1
                        if is_compatible(add2, trial | (1 << add1)):
                            new_state = trial | (1 << add1) | (1 << add2)
                            new_size = current_size
                        else:
                            # Try just one add
                            new_state = trial | (1 << add1)
                            new_size = current_size - 1
                    elif len(cands) == 1:
                        new_state = trial | (1 << cands[0])
                        new_size = current_size - 1
                    else:
                        new_state = current
                        new_size = current_size
                else:
                    new_state = current
                    new_size = current_size

            elif r < 0.85:
                # 3-3 swap
                if current_size >= 3:
                    vertices = bits_to_list(current)
                    rem1, rem2, rem3 = random.sample(vertices, 3)
                    trial = current & ~(1 << rem1) & ~(1 << rem2) & ~(1 << rem3)

                    # Find compatible candidates
                    cands = []
                    t = ~trial & ((1 << m) - 1)
                    while t:
                        low = t & (-t)
                        idx = low.bit_length() - 1
                        if is_compatible(idx, trial):
                            cands.append(idx)
                        t ^= low

                    if len(cands) >= 3:
                        add1, add2, add3 = random.sample(cands, 3)
                        if (is_compatible(add2, trial | (1 << add1)) and
                            is_compatible(add3, trial | (1 << add1) | (1 << add2))):
                            new_state = trial | (1 << add1) | (1 << add2) | (1 << add3)
                            new_size = current_size
                        else:
                            new_state = current
                            new_size = current_size
                    elif len(cands) >= 1:
                        # Add as many as possible
                        add_count = 0
                        for c in cands[:3]:
                            if is_compatible(c, trial):
                                trial = trial | (1 << c)
                                add_count += 1
                        new_state = trial
                        new_size = current_size - 3 + add_count
                    else:
                        new_state = current
                        new_size = current_size
                else:
                    new_state = current
                    new_size = current_size

            else:
                # Try to add a vertex with CRT-informed selection
                temp_add = ~current & ((1 << m) - 1)
                cands = []
                t = temp_add
                while t:
                    low = t & (-t)
                    idx = low.bit_length() - 1
                    if is_compatible(idx, current):
                        cands.append(idx)
                    t ^= low

                if cands:
                    # CRT-informed: prefer adding from underrepresented mod-5 classes
                    mod5_count = defaultdict(int)
                    for v in bits_to_list(current):
                        mod5_count[v % 5] += 1
                    
                    scored_cands = []
                    for c in cands:
                        score = 1.0 / (1.0 + mod5_count[c % 5])
                        scored_cands.append((score, c))
                    scored_cands.sort(key=lambda x: x[0], reverse=True)
                    
                    top_k = min(len(scored_cands), 5)
                    v = scored_cands[random.randint(0, top_k - 1)][1]
                    new_state = current | (1 << v)
                    new_size = current_size + 1
                else:
                    new_state = current
                    new_size = current_size

            delta = new_size - current_size
            if delta > 0 or random.random() < pow(2.718281828, delta / max(temp, 1e-10)):
                current = new_state
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best = current

        # Final local search polish
        best, best_size = local_search_add_only(best, max_iters=500)
        return best, best_size

    # Main search
    best_bits = 0
    best_size = 0

    # Try multiple initialization strategies
    init_strategies = [
        lambda: random_greedy(),
        lambda: crt_greedy_init(),
        lambda: crt_greedy_init_v2(),
        lambda: crt_greedy_init_v3(),
    ]
    
    # Also try deterministic greedy
    def deterministic_greedy():
        bits = 0
        for v in range(m):
            if is_compatible(v, bits):
                bits |= (1 << v)
        return bits
    
    init_strategies.append(deterministic_greedy)

    for init_func in init_strategies:
        g_bits = init_func()
        g_size = popcount(g_bits)
        
        # Local search polish
        ls_bits, ls_size = local_search_add_only(g_bits, max_iters=500)
        if ls_size > best_size:
            best_size = ls_size
            best_bits = ls_bits
        
        # Remove-add local search
        ra_bits, ra_size = local_search_remove_add(best_bits, max_iters=500)
        if ra_size > best_size:
            best_size = ra_size
            best_bits = ra_bits

    # Multiple restarts with simulated annealing
    restart_count = 0
    while time.time() < deadline - 60:
        restart_count += 1

        # Choose initialization strategy
        if restart_count % 4 == 1:
            g_bits = random_greedy()
        elif restart_count % 4 == 2:
            g_bits = crt_greedy_init()
        elif restart_count % 4 == 3:
            g_bits = crt_greedy_init_v2()
        else:
            g_bits = crt_greedy_init_v3()
        
        g_size = popcount(g_bits)

        # Simulated annealing
        sa_bits, sa_size = simulated_annealing(g_bits, start_temp=1.0, end_temp=0.001, max_iters=6000)

        if sa_size > best_size:
            best_size = sa_size
            best_bits = sa_bits

        # Also try local search on the best found so far
        if restart_count % 3 == 0:
            ls2_bits, ls2_size = local_search_add_only(best_bits, max_iters=500)
            if ls2_size > best_size:
                best_size = ls2_size
                best_bits = ls2_bits

            ra2_bits, ra2_size = local_search_remove_add(best_bits, max_iters=500)
            if ra2_size > best_size:
                best_size = ra2_size
                best_bits = ra2_bits

        # Occasionally try a longer annealing run from the current best
        if restart_count % 7 == 0 and time.time() < deadline - 120:
            sa2_bits, sa2_size = simulated_annealing(best_bits, start_temp=0.5, end_temp=0.001, max_iters=10000)
            if sa2_size > best_size:
                best_size = sa2_size
                best_bits = sa2_bits

    # Final polish with multiple passes
    for _ in range(3):
        final_bits, final_size = local_search_add_only(best_bits, max_iters=1000)
        if final_size > best_size:
            best_size = final_size
            best_bits = final_bits
        
        final_ra_bits, final_ra_size = local_search_remove_add(best_bits, max_iters=500)
        if final_ra_size > best_size:
            best_size = final_ra_size
            best_bits = final_ra_bits

    # Convert bitmask to array
    result = []
    for v in range(m):
        if best_bits & (1 << v):
            result.append(v)

    return np.array(result, dtype=np.int64)


# EVOLVE-BLOCK-END
