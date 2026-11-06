"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math

    def bron_kerbosch_bitmask(n, adj_mask, cliques):
        """Bron-Kerbosch with pivoting, using bitmasks. Collects cliques of size >= 2."""
        def bk(R, P, X):
            if P == 0 and X == 0:
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            # Pivot: choose v in P|X maximizing |P & adj[v]|
            PX = P | X
            if PX == 0:
                return
            best_v = -1
            best_count = -1
            # Iterate over set bits in PX
            temp = PX
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                cnt = (P & adj_mask[v]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
                temp ^= lsb
            # Candidates: P minus neighbors of pivot
            if best_v >= 0:
                candidates = P & ~adj_mask[best_v]
            else:
                candidates = P
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                bk(R | lsb, P & adj_mask[v], X & adj_mask[v])
                P ^= lsb
                X |= lsb
                temp ^= lsb

        bk(0, (1 << n) - 1, 0)

    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def bitmask_to_list(mask):
        return [i for i in range(mask.bit_length()) if mask & (1 << i)]

    def greedy_transversal(cliques_masks, n, weight_fn=None):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        if not cliques_masks:
            return 0

        # Precompute incidence: for each vertex, list of clique indices it belongs to
        m = len(cliques_masks)
        vertex_cliques = [[] for _ in range(n)]
        for ci, cm in enumerate(cliques_masks):
            temp = cm
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                vertex_cliques[v].append(ci)
                temp ^= lsb

        if weight_fn is not None:
            # Custom weighting: use it to break ties or scale
            vertex_weight = weight_fn(cliques_masks, n)

        uncovered_count = [1] * m  # 1 = uncovered
        selected = 0
        selected_list = []

        while True:
            # Check if all covered
            all_covered = True
            for ci in range(m):
                if uncovered_count[ci] > 0:
                    all_covered = False
                    break
            if all_covered:
                break

            # Find vertex covering most uncovered cliques
            best_v = -1
            best_count = -1
            for v in range(n):
                if selected & (1 << v):
                    continue
                count = 0
                for ci in vertex_cliques[v]:
                    if uncovered_count[ci] > 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v < 0 or best_count <= 0:
                break
            selected |= (1 << best_v)
            selected_list.append(best_v)
            # Mark cliques containing best_v as covered
            for ci in vertex_cliques[best_v]:
                uncovered_count[ci] = 0

        return selected, selected_list

    def iterated_local_search(selected_mask, cliques_masks, n, deadline, max_restarts=20):
        """Iterated local search with perturbations to escape local optima."""
        best_mask = selected_mask
        best_size = selected_mask.bit_count()
        current_mask = selected_mask
        current_size = current_mask.bit_count()
        
        def is_valid(mask):
            for cm in cliques_masks:
                if not (cm & mask):
                    return False
            return True
        
        def local_search_reduce(mask, cliques_masks, n, max_iters=100):
            improved = True
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                for v in range(n):
                    if not (mask & (1 << v)):
                        continue
                    trial = mask ^ (1 << v)
                    ok = True
                    for cm in cliques_masks:
                        if not (cm & trial):
                            ok = False
                            break
                    if ok:
                        mask = trial
                        improved = True
                        break
            return mask
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Local search reduction
            current_mask = local_search_reduce(current_mask, cliques_masks, n)
            current_size = current_mask.bit_count()
            
            if current_size < best_size:
                best_size = current_size
                best_mask = current_mask
            
            # Perturbation: remove 1-2 random vertices and re-add via greedy
            if current_size > 0:
                # Remove 1-2 random vertices
                num_remove = random.randint(1, min(2, current_size))
                bits = []
                temp_mask = current_mask
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    bits.append(lsb)
                    temp_mask ^= lsb
                
                if bits:
                    removed = random.sample(bits, num_remove)
                    perturbed = current_mask
                    for lsb in removed:
                        perturbed ^= lsb
                    
                    # Re-add vertices via greedy to restore validity
                    # Find which cliques are now uncovered
                    uncovered = [cm for cm in cliques_masks if not (cm & perturbed)]
                    
                    if uncovered:
                        # Greedily add vertices to cover uncovered cliques
                        while uncovered and time.time() < deadline - 5:
                            best_v = -1
                            best_count = -1
                            for v in range(n):
                                if perturbed & (1 << v):
                                    continue
                                count = sum(1 for cm in uncovered if cm & (1 << v))
                                if count > best_count:
                                    best_count = count
                                    best_v = v
                            if best_v < 0 or best_count <= 0:
                                break
                            perturbed |= (1 << best_v)
                            uncovered = [cm for cm in uncovered if not (cm & (1 << best_v))]
                        
                        # Only accept if valid and not worse
                        if is_valid(perturbed):
                            perturbed_size = perturbed.bit_count()
                            if perturbed_size <= current_size + 1:  # Allow slight increase
                                current_mask = perturbed
                                current_size = perturbed_size
                            else:
                                # Revert to previous
                                current_mask = local_search_reduce(best_mask, cliques_masks, n)
                                current_size = current_mask.bit_count()
        
        return best_mask

    def local_search_reduce(selected_mask, cliques_masks, n, max_iters=200):
        """Try to remove vertices; if it breaks coverage, stop."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(n):
                if not (selected_mask & (1 << v)):
                    continue
                trial = selected_mask ^ (1 << v)
                # Check all cliques are still covered
                ok = True
                for cm in cliques_masks:
                    if not (cm & trial):
                        ok = False
                        break
                if ok:
                    selected_mask = trial
                    improved = True
                    break
        return selected_mask

    def local_search_swap(selected_mask, cliques_masks, n, max_iters=100):
        """Try removing one and adding another to reduce size.
        A swap can reduce size when a vertex in the set covers cliques that are
        also covered by a vertex NOT in the set, allowing removal of the in-set vertex
        without adding a new one (net -1 if the in-set vertex becomes redundant after
        the swap logic). Actually, a true 1-exchange: remove v_in, add v_out.
        Size stays same. To reduce size, we need: remove v_in, and v_out is already
        in the set... that's just removal.
        
        Better: try all pairs (v_in in set, v_out not in set). If removing v_in
        and adding v_out keeps validity AND the new set has a vertex that can be
        removed, we've effectively reduced. But simpler: just try removals first
        (handled by local_search_reduce), then try 1-exchanges to escape local
        minima where no single removal works but a swap + removal does.
        
        Implementation: for each v_in in set, for each v_out not in set:
          trial = (set - {v_in}) | {v_out}
          if valid(trial), accept and restart reduction.
        """
        if not cliques_masks:
            return selected_mask

        def is_valid(mask):
            for cm in cliques_masks:
                if not (cm & mask):
                    return False
            return True

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Collect vertices in set
            in_set = []
            temp = selected_mask
            while temp:
                lsb = temp & (-temp)
                in_set.append(lsb.bit_length() - 1)
                temp ^= lsb

            # Collect vertices not in set
            not_in_set = [v for v in range(n) if not (selected_mask & (1 << v))]

            for v_in in in_set:
                for v_out in not_in_set:
                    trial = (selected_mask ^ (1 << v_in)) | (1 << v_out)
                    if is_valid(trial):
                        # Check if we can then reduce
                        reduced = trial
                        # Try to remove any vertex from trial
                        for v in range(n):
                            if not (reduced & (1 << v)):
                                continue
                            t2 = reduced ^ (1 << v)
                            if is_valid(t2):
                                reduced = t2
                                break
                        if reduced.bit_count() < selected_mask.bit_count():
                            selected_mask = reduced
                            improved = True
                            break
                        elif reduced.bit_count() == selected_mask.bit_count():
                            # Same size, accept to escape local minimum
                            selected_mask = trial
                            improved = True
                            break
                if improved:
                    break
        return selected_mask

    def simulated_annealing(selected_mask, cliques_masks, n, start_temp=10.0, end_temp=0.1, max_iters=500, deadline=None):
        """SA to try to reduce the transversal size."""
        if not cliques_masks:
            return selected_mask

        def is_valid(mask):
            for cm in cliques_masks:
                if not (cm & mask):
                    return False
            return True

        def size(mask):
            return mask.bit_count()

        current = selected_mask
        current_size = size(current)
        best = current
        best_size = current_size

        temp = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters))

        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break

            # Generate neighbor: remove a random vertex, add a random vertex, or swap
            r = random.random()
            if r < 0.3 and current.bit_count() > 0:
                # Remove a random vertex
                bits = []
                temp_mask = current
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    bits.append(lsb)
                    temp_mask ^= lsb
                if not bits:
                    continue
                lsb = random.choice(bits)
                neighbor = current ^ lsb
            elif r < 0.6:
                # Add a random vertex not in current
                available = ((1 << n) - 1) & ~current
                if available == 0:
                    continue
                temp_mask = available
                bits = []
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    bits.append(lsb)
                    temp_mask ^= lsb
                lsb = random.choice(bits)
                neighbor = current | lsb
            else:
                # Swap: remove one in-set, add one not-in-set
                in_bits = []
                temp_mask = current
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    in_bits.append(lsb)
                    temp_mask ^= lsb
                avail = ((1 << n) - 1) & ~current
                if not in_bits or avail == 0:
                    continue
                out_bits = []
                temp_mask = avail
                while temp_mask:
                    lsb = temp_mask & (-temp_mask)
                    out_bits.append(lsb)
                    temp_mask ^= lsb
                lsb_in = random.choice(in_bits)
                lsb_out = random.choice(out_bits)
                neighbor = (current ^ lsb_in) | lsb_out

            if not is_valid(neighbor):
                continue

            neighbor_size = size(neighbor)
            delta = neighbor_size - current_size

            if delta <= 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current = neighbor
                current_size = neighbor_size
                if current_size < best_size:
                    best = current
                    best_size = current_size

            temp *= cooling

        return best

    def solve_instance(n, p, seed, deadline):
        adj_mask = build_graph(n, p, seed)
        cliques = []
        bron_kerbosch_bitmask(n, adj_mask, cliques)

        if not cliques:
            return []

        # Try multiple greedy strategies
        best_mask = None
        best_size = n + 1

        # Strategy 1: standard greedy (most cliques covered)
        mask1, _ = greedy_transversal(cliques, n)
        mask1 = local_search_reduce(mask1, cliques, n)
        s1 = mask1.bit_count()
        if s1 < best_size:
            best_size = s1
            best_mask = mask1

        # Strategy 2: weighted by clique size (larger cliques first)
        def weight_fn_size(cliques_masks, n):
            w = [0] * n
            for cm in cliques_masks:
                sz = cm.bit_count()
                temp = cm
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    w[v] += sz
                    temp ^= lsb
            return w

        mask2, _ = greedy_transversal(cliques, n, weight_fn_size)
        mask2 = local_search_reduce(mask2, cliques, n)
        s2 = mask2.bit_count()
        if s2 < best_size:
            best_size = s2
            best_mask = mask2

        # Strategy 3: weighted by 1/clique_size (smaller cliques first)
        def weight_fn_inv_size(cliques_masks, n):
            w = [0.0] * n
            for cm in cliques_masks:
                sz = max(1, cm.bit_count())
                temp = cm
                while temp:
                    lsb = temp & (-temp)
                    v = lsb.bit_length() - 1
                    w[v] += 1.0 / sz
                    temp ^= lsb
            return w

        mask3, _ = greedy_transversal(cliques, n, weight_fn_inv_size)
        mask3 = local_search_reduce(mask3, cliques, n)
        s3 = mask3.bit_count()
        if s3 < best_size:
            best_size = s3
            best_mask = mask3

        # Apply local search swap to escape 1-removal local optima
        if time.time() < deadline - 10:
            best_mask = local_search_swap(best_mask, cliques, n, max_iters=50)
            best_size = best_mask.bit_count()

        # Apply simulated annealing to the best found
        if time.time() < deadline - 10:
            best_mask = simulated_annealing(best_mask, cliques, n, start_temp=5.0, end_temp=0.01, max_iters=400, deadline=deadline - 10)
            best_size = best_mask.bit_count()
            # After SA, try reduction again
            best_mask = local_search_reduce(best_mask, cliques, n)
            best_size = best_mask.bit_count()

        # Apply iterated local search with perturbations
        if time.time() < deadline - 3:
            best_mask = iterated_local_search(best_mask, cliques, n, deadline, max_restarts=15)
            best_size = best_mask.bit_count()

        return bitmask_to_list(best_mask)

    start_time = time.time()
    deadline = start_time + 130

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            # Fallback: all vertices
            out.append(list(range(n)))
            continue
        result = solve_instance(n, p, seed, deadline)
        out.append(result)

    return out
# EVOLVE-BLOCK-END
