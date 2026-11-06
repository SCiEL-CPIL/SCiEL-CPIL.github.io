import time
import random
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    N = 180

    # Precompute all forbidden triples: distinct a,b,c with 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => c = ab/(b-a) for b > a, need c integer, c != a, c != b, c in [1,N]
    # a is always the smallest (since 1/a > 1/b and 1/a > 1/c)
    triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    triples.append((a, b, c))
    # Deduplicate
    triples = list(set(triples))
    num_triples = len(triples)

    # Bitset representation: for each element, which triples contain it
    # triples_by_elem[x] = bitmask of triple indices containing element x
    triples_by_elem = [0] * (N + 1)
    for i, (a, b, c) in enumerate(triples):
        bit = 1 << i
        triples_by_elem[a] |= bit
        triples_by_elem[b] |= bit
        triples_by_elem[c] |= bit

    # For fast "can add z" check: for each z, precompute the list of triples containing z
    # and for each such triple, the two other elements
    elem_triples = [[] for _ in range(N + 1)]  # elem_triples[z] = list of (other1, other2) for triples containing z
    for i, (a, b, c) in enumerate(triples):
        elem_triples[a].append((b, c))
        elem_triples[b].append((a, c))
        elem_triples[c].append((a, b))

    # Precompute: for each pair (x,y) with x<y, the set of z such that {x,y,z} is a triple
    # pair_thirds[(x,y)] = bitmask of elements z that complete a triple with x,y
    pair_thirds = {}
    for (a, b, c) in triples:
        for (x, y, z) in [(a, b, c), (a, c, b), (b, c, a)]:
            key = (min(x, y), max(x, y))
            if key not in pair_thirds:
                pair_thirds[key] = 0
            pair_thirds[key] |= (1 << z)

    # Also: for each element z, the bitmask of pairs (x,y) such that {x,y,z} is a triple
    # This is useful for branch-and-bound
    # Actually, let's use a different approach for B&B:
    # State: current set S (as bitmask), candidate set C (elements not in S that could still be added)
    # A candidate z can be added iff for all triples (z, u, v), NOT (u in S and v in S)
    # Equivalently: for all triples containing z, the other two elements are not BOTH in S

    def can_add(z, S_mask):
        """Check if element z (1-indexed) can be added to set S_mask without creating a triple."""
        for (u, v) in elem_triples[z]:
            if (S_mask >> u) & 1 and (S_mask >> v) & 1:
                return False
        return True

    def count_violations(S_mask):
        """Count number of forbidden triples fully contained in S_mask."""
        count = 0
        for (a, b, c) in triples:
            if (S_mask >> a) & 1 and (S_mask >> b) & 1 and (S_mask >> c) & 1:
                count += 1
        return count

    def is_valid(S_mask):
        return count_violations(S_mask) == 0

    # --- Branch and Bound ---
    # We want to find a maximum-size valid subset.
    # B&B: recursively decide to include or exclude each element.
    # Upper bound: current_size + number of remaining candidates that can potentially be added
    # (optimistic: all remaining candidates could be added, but we can tighten this)
    
    # Better bound: for the remaining candidates, compute a greedy upper bound
    # on how many more we can add (greedy independent set on the induced subproblem)
    
    def greedy_upper_bound(candidates_mask, S_mask):
        """Upper bound on how many elements from candidates_mask can be added to S_mask.
        Uses a greedy approach as an upper bound (may overestimate)."""
        # Simple upper bound: just count candidates that can individually be added
        count = 0
        for z in range(1, N + 1):
            if (candidates_mask >> z) & 1:
                if can_add(z, S_mask):
                    count += 1
        return count

    # Tighter bound: run a quick greedy on the candidates
    def greedy_greedy_bound(candidates_mask, S_mask):
        """Run greedy on candidates to get a lower bound on max addable, 
        but for upper bound we use: max independent set size in the conflict graph of candidates.
        Simpler: just use len of candidates that can be added individually as upper bound."""
        count = 0
        for z in range(1, N + 1):
            if (candidates_mask >> z) & 1:
                if can_add(z, S_mask):
                    count += 1
        return count

    # Actually, for a good B&B, let's use a different approach:
    # Order elements by "constraint degree" (number of triples they're in)
    # and use a recursive search with pruning.
    
    # Precompute element degrees
    elem_degree = [0] * (N + 1)
    for (a, b, c) in triples:
        elem_degree[a] += 1
        elem_degree[b] += 1
        elem_degree[c] += 1

    # Order elements: most constrained first (highest degree)
    elem_order = list(range(1, N + 1))
    elem_order.sort(key=lambda x: -elem_degree[x])
    
    # B&B search
    best_size = 0
    best_mask = 0
    search_calls = [0]
    
    def branch_and_bound(idx, S_mask, candidates_mask):
        """Try to build a valid set from elements in elem_order[idx:].
        S_mask: current selected set
        candidates_mask: elements that are still candidates (not yet decided)
        """
        nonlocal best_size, best_mask
        search_calls[0] += 1
        
        if time.time() > deadline - 5:
            return
        
        current_size = bin(S_mask).count('1')
        
        # Upper bound: current + number of candidates that can be added
        # Quick bound: count candidates that can individually be added
        ub = current_size
        for z in range(1, N + 1):
            if (candidates_mask >> z) & 1:
                if can_add(z, S_mask):
                    ub += 1
        
        if ub <= best_size:
            return
        
        if idx >= len(elem_order):
            if current_size > best_size:
                best_size = current_size
                best_mask = S_mask
            return
        
        z = elem_order[idx]
        new_candidates = candidates_mask & ~(1 << z)
        
        # Option 1: exclude z
        branch_and_bound(idx + 1, S_mask, new_candidates)
        
        # Option 2: include z (if possible)
        if can_add(z, S_mask):
            branch_and_bound(idx + 1, S_mask | (1 << z), new_candidates)

    # Run B&B but it might be too slow for 180 elements. Let's limit it.
    # Instead, use a hybrid: greedy initialization + local search + limited B&B on subsets
    
    # --- Strategy 1: Greedy construction with smart ordering ---
    def greedy_construct(order):
        S_mask = 0
        for z in order:
            if can_add(z, S_mask):
                S_mask |= (1 << z)
        return S_mask

    # --- Strategy 2: Local search (add/remove/swap) ---
    def local_search(S_mask, max_rounds=500):
        S_mask = S_mask
        best_local = S_mask
        best_local_size = bin(S_mask).count('1')
        
        for _ in range(max_rounds):
            if time.time() > deadline - 8:
                break
            
            improved = False
            
            # Try adding elements
            for z in range(1, N + 1):
                if not (S_mask >> z) & 1:
                    if can_add(z, S_mask):
                        S_mask |= (1 << z)
                        improved = True
                        if bin(S_mask).count('1') > best_local_size:
                            best_local = S_mask
                            best_local_size = bin(S_mask).count('1')
            
            # Try swaps: remove one element, add another
            if not improved:
                in_S = [z for z in range(1, N + 1) if (S_mask >> z) & 1]
                not_in_S = [z for z in range(1, N + 1) if not (S_mask >> z) & 1]
                
                if in_S and not_in_S:
                    # Try more random swaps with higher probability of finding improvements
                    for _ in range(50):
                        if time.time() > deadline - 8:
                            break
                        x = random.choice(in_S)
                        y = random.choice(not_in_S)
                        new_mask = (S_mask & ~(1 << x)) | (1 << y)
                        if is_valid(new_mask):
                            S_mask = new_mask
                            in_S = [z for z in range(1, N + 1) if (S_mask >> z) & 1]
                            not_in_S = [z for z in range(1, N + 1) if not (S_mask >> z) & 1]
                            if bin(S_mask).count('1') > best_local_size:
                                best_local = S_mask
                                best_local_size = bin(S_mask).count('1')
                            improved = True
                            break
                    
                    # Try 2-for-1 swaps: remove two elements, add one
                    if not improved and len(in_S) >= 2:
                        for _ in range(30):
                            if time.time() > deadline - 8:
                                break
                            idx1, idx2 = random.sample(range(len(in_S)), 2)
                            x, y = in_S[idx1], in_S[idx2]
                            z = random.choice(not_in_S)
                            new_mask = (S_mask & ~(1 << x) & ~(1 << y)) | (1 << z)
                            if is_valid(new_mask):
                                S_mask = new_mask
                                in_S = [z for z in range(1, N + 1) if (S_mask >> z) & 1]
                                not_in_S = [z for z in range(1, N + 1) if not (S_mask >> z) & 1]
                                if bin(S_mask).count('1') > best_local_size:
                                    best_local = S_mask
                                    best_local_size = bin(S_mask).count('1')
                                improved = True
                                break
        
        return best_local

    # --- Strategy 3: Greedy removal from full set ---
    def greedy_remove():
        S_mask = (1 << (N + 1)) - 2  # all elements 1..N set
        
        # Count violations per element
        while True:
            if time.time() > deadline - 25:
                break
            
            # Find all violated triples
            violated_elems = [0] * (N + 1)
            has_violation = False
            for (a, b, c) in triples:
                if (S_mask >> a) & 1 and (S_mask >> b) & 1 and (S_mask >> c) & 1:
                    has_violation = True
                    violated_elems[a] += 1
                    violated_elems[b] += 1
                    violated_elems[c] += 1
            
            if not has_violation:
                break
            
            # Remove element with max violations
            max_v = 0
            max_elem = -1
            for e in range(1, N + 1):
                if (S_mask >> e) & 1 and violated_elems[e] > max_v:
                    max_v = violated_elems[e]
                    max_elem = e
            
            if max_elem == -1:
                break
            
            S_mask &= ~(1 << max_elem)
        
        return S_mask

    # Run strategies
    best_mask = 0
    best_size = 0

    # 1. Greedy removal
    if time.time() < deadline - 35:
        rm_mask = greedy_remove()
        rm_size = bin(rm_mask).count('1')
        if rm_size > best_size:
            best_mask = rm_mask
            best_size = rm_size

    # 2. Local search from greedy removal result
    if time.time() < deadline - 28 and best_size > 0:
        ls_mask = local_search(best_mask, max_rounds=300)
        ls_size = bin(ls_mask).count('1')
        if ls_size > best_size:
            best_mask = ls_mask
            best_size = ls_size

    # 3. Multiple greedy constructions with different orderings
    if time.time() < deadline - 22:
        # Order by decreasing degree
        order_desc = sorted(range(1, N + 1), key=lambda x: -elem_degree[x])
        gc_mask = greedy_construct(order_desc)
        gc_size = bin(gc_mask).count('1')
        if gc_size > best_size:
            best_mask = gc_mask
            best_size = gc_size

        # Order by increasing degree
        order_asc = sorted(range(1, N + 1), key=lambda x: elem_degree[x])
        gc_mask2 = greedy_construct(order_asc)
        gc_size2 = bin(gc_mask2).count('1')
        if gc_size2 > best_size:
            best_mask = gc_mask2
            best_size = gc_size2

        # Random orders
        for _ in range(40):
            if time.time() > deadline - 18:
                break
            rand_order = list(range(1, N + 1))
            random.shuffle(rand_order)
            gc_mask3 = greedy_construct(rand_order)
            gc_size3 = bin(gc_mask3).count('1')
            if gc_size3 > best_size:
                best_mask = gc_mask3
                best_size = gc_size3
            # Also try local search from this
            if time.time() < deadline - 12:
                ls_from_gc = local_search(gc_mask3, max_rounds=80)
                ls_size = bin(ls_from_gc).count('1')
                if ls_size > best_size:
                    best_mask = ls_from_gc
                    best_size = ls_size

    # 4. Start from high-number seed and add lower elements
    if time.time() < deadline - 18:
        # Seed: all elements from 91 to 180
        seed_mask = 0
        for z in range(91, N + 1):
            seed_mask |= (1 << z)
        
        # Add lower elements greedily
        for z in range(1, 91):
            if can_add(z, seed_mask):
                seed_mask |= (1 << z)
        
        seed_size = bin(seed_mask).count('1')
        if seed_size > best_size:
            best_mask = seed_mask
            best_size = seed_size
        
        # Local search from seed
        if time.time() < deadline - 12:
            ls_seed = local_search(seed_mask, max_rounds=150)
            ls_seed_size = bin(ls_seed).count('1')
            if ls_seed_size > best_size:
                best_mask = ls_seed
                best_size = ls_seed_size

    # 5. Additional diverse seeds
    if time.time() < deadline - 15:
        # Try different seed ranges
        for low, high in [(1, 60), (61, 120), (121, 180), (1, 90), (91, 180)]:
            if time.time() > deadline - 10:
                break
            seed_mask = 0
            for z in range(low, high + 1):
                seed_mask |= (1 << z)
            
            # Add remaining elements greedily
            for z in range(1, N + 1):
                if not (seed_mask >> z) & 1:
                    if can_add(z, seed_mask):
                        seed_mask |= (1 << z)
            
            seed_size = bin(seed_mask).count('1')
            if seed_size > best_size:
                best_mask = seed_mask
                best_size = seed_size
            
            # Local search from seed
            if time.time() < deadline - 8:
                ls_seed = local_search(seed_mask, max_rounds=100)
                ls_seed_size = bin(ls_seed).count('1')
                if ls_seed_size > best_size:
                    best_mask = ls_seed
                    best_size = ls_seed_size

    # 6. Final local search on best result
    if time.time() < deadline - 5 and best_size > 0:
        final_mask = local_search(best_mask, max_rounds=400)
        final_size = bin(final_mask).count('1')
        if final_size > best_size:
            best_mask = final_mask
            best_size = final_size

    # Convert bitmask to list
    result = []
    for z in range(1, N + 1):
        if (best_mask >> z) & 1:
            result.append(z)

    if not result:
        result = [1]

    return result
