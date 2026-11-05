import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s eval limit

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency as bitsets
    adj = [0] * N
    forbidden = [False] * N

    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden[i] = True
        else:
            for sq in squares:
                u = sq - v
                if 1 <= u <= N:
                    j = u - 1
                    if not forbidden[j]:
                        adj[i] |= (1 << j)

    candidates = [i for i in range(N) if not forbidden[i]]
    num_candidates = len(candidates)

    cand_mask = 0
    for i in candidates:
        cand_mask |= (1 << i)

    def greedy_init(seed=None):
        """Greedy min-degree MIS with optional randomization."""
        if seed is not None:
            random.seed(seed)
        available_mask = cand_mask
        selected_mask = 0
        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def greedy_random():
        """Greedy with random tie-breaking and noise."""
        available_mask = cand_mask
        selected_mask = 0
        while available_mask:
            best_v = -1
            min_deg = N + 1
            m = available_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & available_mask).bit_count()
                deg += random.uniform(0, 1.0)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            if best_v == -1:
                break
            selected_mask |= (1 << best_v)
            available_mask &= ~((1 << best_v) | (adj[best_v] & available_mask))
        return selected_mask

    def local_improve(mask):
        """Add any vertex that doesn't conflict with current set."""
        improved = True
        while improved:
            improved = False
            not_in_mask = cand_mask & ~mask
            m = not_in_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                if (adj[v] & mask) == 0:
                    mask |= lsb
                    improved = True
                    break
        return mask

    def try_exchange_11(mask):
        """Try 1-1 exchange: remove u, add v where v was blocked only by u."""
        m = mask
        while m:
            lsb_u = m & -m
            u = lsb_u.bit_length() - 1
            m ^= lsb_u
            potential = adj[u] & ~mask & cand_mask
            pm = potential
            while pm:
                lsb_v = pm & -pm
                v = lsb_v.bit_length() - 1
                pm ^= lsb_v
                if (adj[v] & mask) == lsb_u:
                    return mask ^ lsb_u | lsb_v
        return mask

    def local_search(mask, max_rounds=50):
        """Iterate local_improve and exchanges until no improvement."""
        for _ in range(max_rounds):
            if time.time() - start_time > DEADLINE:
                break
            old_mask = mask
            mask = local_improve(mask)
            for _ in range(10):
                new_mask = try_exchange_11(mask)
                if new_mask != mask:
                    mask = new_mask
                    break
            if mask == old_mask:
                break
        return mask

    def perturb(mask, strength=3):
        """Remove random vertices and optionally add random non-conflicting ones."""
        current = mask
        count = current.bit_count()
        remove_k = min(strength, count)
        for _ in range(remove_k):
            if not current:
                break
            bits = []
            m = current
            while m:
                lsb = m & -m
                bits.append(lsb)
                m ^= lsb
            if bits:
                current ^= random.choice(bits)
        for _ in range(random.randint(0, 2)):
            not_in = cand_mask & ~current
            if not_in:
                m = not_in
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    if (adj[v] & current) == 0:
                        current |= lsb
                        break
        return current

    def max_matching_mask(mask):
        """Compute maximum cardinality matching on subgraph induced by mask using greedy + augmenting paths."""
        # Build adjacency list for vertices in mask
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        n = len(verts)
        if n == 0:
            return 0
        
        # Map vertex index to position
        vpos = {}
        for i, v in enumerate(verts):
            vpos[v] = i
        
        # Build adjacency list
        adj_list = [[] for _ in range(n)]
        for i, v in enumerate(verts):
            m2 = adj[v] & mask
            while m2:
                lsb2 = m2 & -m2
                u = lsb2.bit_length() - 1
                m2 ^= lsb2
                j = vpos.get(u)
                if j is not None and j > i:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        
        # Greedy matching
        match = [-1] * n
        for i in range(n):
            if match[i] == -1:
                for j in adj_list[i]:
                    if match[j] == -1:
                        match[i] = j
                        match[j] = i
                        break
        
        matching_size = sum(1 for i in range(n) if match[i] != -1 and i < match[i])
        
        # Try to augment with BFS (Hopcroft-Karp simplified for small graphs)
        # For n up to ~200, simple augmenting path search works
        def try_augment():
            nonlocal match, matching_size
            # Find augmenting path from each unmatched vertex
            for start in range(n):
                if match[start] != -1:
                    continue
                # BFS to find augmenting path
                parent = [-1] * n
                visited = [False] * n
                queue = [start]
                visited[start] = True
                found = False
                while queue and not found:
                    u = queue.pop(0)
                    for w in adj_list[u]:
                        if not visited[w]:
                            visited[w] = True
                            parent[w] = u
                            if match[w] == -1:
                                # Found augmenting path
                                found = True
                                break
                            else:
                                queue.append(match[w])
                if found:
                    # Trace back and flip
                    v = w
                    while v != -1:
                        pv = parent[v]
                        match[v] = pv
                        if pv != -1:
                            match[pv] = v
                        v = pv
                        if v == start:
                            break
                    matching_size += 1
                    return True
            return False
        
        # Try a few augmentations
        for _ in range(20):
            if not try_augment():
                break
        
        return matching_size

    def lovasz_bound(mask):
        """Upper bound on independence number: |V| - |max matching|."""
        n = mask.bit_count()
        if n == 0:
            return 0
        mm = max_matching_mask(mask)
        return n - mm

    def branch_and_bound(remaining_mask, current_size, best_size_ref):
        """Branch and bound MIS with Lovasz bound. Returns best mask found."""
        nonlocal start_time, DEADLINE
        
        if time.time() - start_time > DEADLINE:
            return 0, best_size_ref
        
        if remaining_mask == 0:
            return 0, best_size_ref
        
        # Upper bound: current_size + lovasz_bound(remaining_mask)
        ub = current_size + lovasz_bound(remaining_mask)
        if ub <= best_size_ref:
            return 0, best_size_ref
        
        # Pick a vertex to branch on (highest degree in remaining)
        m = remaining_mask
        best_v = -1
        max_deg = -1
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            deg = (adj[v] & remaining_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        if best_v == -1:
            return 0, best_size_ref
        
        v_bit = 1 << best_v
        
        # Branch 1: include best_v
        new_remaining = remaining_mask & ~v_bit & ~adj[best_v]
        res1, best_size_ref = branch_and_bound(new_remaining, current_size + 1, best_size_ref)
        
        # Branch 2: exclude best_v
        new_remaining2 = remaining_mask & ~v_bit
        res2, best_size_ref = branch_and_bound(new_remaining2, current_size, best_size_ref)
        
        # Return the better result
        if res1 + current_size + 1 > res2 + current_size:
            # We need to reconstruct... this is getting complex
            # For now, just track size
            return max(res1, res2), best_size_ref
        return max(res1, res2), best_size_ref

    def bb_solve(remaining_mask, initial_mask, best_size):
        """Branch and bound that returns the actual mask."""
        nonlocal start_time, DEADLINE, best_mask, best_size
        
        # Use a different approach: branch on vertices, track the actual solution
        # For simplicity, use a recursive approach that builds the solution
        
        def _bb(rem, sel, size):
            nonlocal start_time, DEADLINE, best_mask, best_size
            
            if time.time() - start_time > DEADLINE:
                return
            
            if rem == 0:
                if size > best_size:
                    best_size = size
                    best_mask = sel
                return
            
            ub = size + lovasz_bound(rem)
            if ub <= best_size:
                return
            
            # Pick vertex with max degree in rem
            m = rem
            bv = -1
            md = -1
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = (adj[v] & rem).bit_count()
                if deg > md:
                    md = deg
                    bv = v
            
            if bv == -1:
                return
            
            vb = 1 << bv
            
            # Include bv
            _bb(rem & ~vb & ~adj[bv], sel | vb, size + 1)
            
            # Exclude bv
            _bb(rem & ~vb, sel, size)
        
        _bb(remaining_mask, initial_mask, initial_mask.bit_count())

    # Main: Iterated Local Search + B&B
    best_mask = 0
    best_size = 0

    # Multiple initializations
    for init_seed in [None, 42, 123, 777, 2024]:
        if time.time() - start_time > DEADLINE:
            break
        if init_seed is None:
            mask = greedy_init()
        else:
            mask = greedy_init(seed=init_seed)
        mask = local_search(mask)
        if mask.bit_count() > best_size:
            best_size = mask.bit_count()
            best_mask = mask

    # Try branch and bound from the best solution found
    # The remaining vertices are those not in best_mask
    remaining = cand_mask & ~best_mask
    if remaining and time.time() - start_time < DEADLINE - 10:
        bb_solve(remaining, best_mask, best_size)

    # ILS loop
    no_improve_count = 0
    while time.time() - start_time < DEADLINE - 5:
        # Perturb
        current = perturb(best_mask, strength=random.randint(2, 6))
        
        # Rebuild with greedy fill
        current = local_improve(current)
        
        # Local search
        current = local_search(current)
        
        if current.bit_count() > best_size:
            best_size = current.bit_count()
            best_mask = current
            no_improve_count = 0
            # Try B&B from this new best
            remaining = cand_mask & ~best_mask
            if remaining and time.time() - start_time < DEADLINE - 10:
                bb_solve(remaining, best_mask, best_size)
        else:
            no_improve_count += 1
        
        if no_improve_count % 10 == 0 or random.random() < 0.3:
            mask_r = greedy_random()
            mask_r = local_search(mask_r)
            if mask_r.bit_count() > best_size:
                best_size = mask_r.bit_count()
                best_mask = mask_r
                no_improve_count = 0
                remaining = cand_mask & ~best_mask
                if remaining and time.time() - start_time < DEADLINE - 10:
                    bb_solve(remaining, best_mask, best_size)

    # Final polish
    best_mask = local_improve(best_mask)
    best_size = best_mask.bit_count()

    # Final B&B attempt with tight time budget
    remaining = cand_mask & ~best_mask
    if remaining and time.time() - start_time < DEADLINE - 2:
        bb_solve(remaining, best_mask, best_size)

    # Convert to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v + 1)
        m ^= lsb

    return sorted(result)
