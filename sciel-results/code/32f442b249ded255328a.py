"""Combined: bitmask coprime checks + heuristic local search + branch-and-bound MWIS polish."""
import math
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 138.0  # safety margin before 150s kill
    N = 512

    # Precompute prime factor bitmasks for all numbers 1..511
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N, i):
                is_prime[j] = False

    primes = [i for i in range(2, N) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}

    prime_factors_mask = [0] * N
    for a in range(2, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_index[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_index[x])
        prime_factors_mask[a] = mask
    prime_factors_mask[1] = 0

    def score(arr):
        return sum(1.0 / (N - a) for a in arr)

    # Conflict graph: two numbers conflict if not coprime
    # For MWIS: we want max-weight subset with no conflicts
    # Vertex a (index 0..511, value 1..511) has weight w[a] = 1/(512-a)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    # Conflict adjacency: conflict_mask[a] is a bitmask of which other numbers conflict with a
    # Since we have 512 vertices, we use Python integers as bitsets
    conflict_mask = [0] * N
    for a in range(1, N):
        for b in range(a + 1, N):
            if (prime_factors_mask[a] & prime_factors_mask[b]) != 0:
                # They share a prime factor, so not coprime -> conflict
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    # Greedy from largest to smallest (baseline)
    def greedy_desc():
        chosen = []
        used_mask = 0
        for a in range(511, 0, -1):
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    best = greedy_desc()
    best_score = score(best)

    # Randomized greedy with multiple orderings
    rng = random.Random(42)
    all_nums = list(range(1, 512))

    def greedy_with_order(order):
        chosen = []
        used_mask = 0
        for a in order:
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    for trial in range(300):
        if time.time() > deadline:
            break
        # Biased random order
        weights_list = [1.0 / (N - a) for a in all_nums]
        remaining = list(all_nums)
        rem_weights = list(weights_list)
        order = []
        for _ in range(len(remaining)):
            total_w = sum(rem_weights)
            if total_w <= 0:
                order.extend(remaining)
                break
            r = rng.random() * total_w
            cumulative = 0
            for i, w in enumerate(rem_weights):
                cumulative += w
                if cumulative >= r:
                    order.append(remaining.pop(i))
                    rem_weights.pop(i)
                    break
            else:
                order.extend(remaining)
                break
        cand = greedy_with_order(order)
        s = score(cand)
        if s > best_score:
            best_score = s
            best = cand

    # Local search
    def local_search(chosen, max_iters=200):
        chosen_set = set(chosen)
        current_score = sum(1.0 / (N - a) for a in chosen)
        used_mask = 0
        for a in chosen:
            used_mask |= prime_factors_mask[a]

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1

            # Try adding a new element
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if (prime_factors_mask[a] & used_mask) == 0:
                    chosen.append(a)
                    chosen_set.add(a)
                    used_mask |= prime_factors_mask[a]
                    current_score += 1.0 / (N - a)
                    improved = True
                    break

            if improved:
                continue

            # Try single removal + greedy refill
            sorted_chosen = sorted(chosen, key=lambda x: 1.0 / (N - x))
            for a in sorted_chosen:
                if time.time() > deadline:
                    break
                remaining = [x for x in chosen if x != a]
                rem_set = set(remaining)
                rem_mask = 0
                for x in remaining:
                    rem_mask |= prime_factors_mask[x]

                new_chosen = list(remaining)
                new_set = set(remaining)
                new_mask = rem_mask
                new_score = sum(1.0 / (N - x) for x in new_chosen)

                for c in range(511, 0, -1):
                    if c in new_set:
                        continue
                    if (prime_factors_mask[c] & new_mask) == 0:
                        new_chosen.append(c)
                        new_set.add(c)
                        new_mask |= prime_factors_mask[c]
                        new_score += 1.0 / (N - c)

                if new_score > current_score + 1e-12:
                    chosen = new_chosen
                    chosen_set = new_set
                    used_mask = new_mask
                    current_score = new_score
                    improved = True
                    break

            if improved:
                continue

            # Try double removal
            if len(chosen) >= 2:
                for i in range(len(sorted_chosen)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(sorted_chosen)):
                        a, b = sorted_chosen[i], sorted_chosen[j]
                        remaining = [x for x in chosen if x != a and x != b]
                        rem_set = set(remaining)
                        rem_mask = 0
                        for x in remaining:
                            rem_mask |= prime_factors_mask[x]

                        new_chosen = list(remaining)
                        new_set = set(remaining)
                        new_mask = rem_mask
                        new_score = sum(1.0 / (N - x) for x in new_chosen)
                        added_count = 0

                        for c in range(511, 0, -1):
                            if c in new_set:
                                continue
                            if (prime_factors_mask[c] & new_mask) == 0:
                                new_chosen.append(c)
                                new_set.add(c)
                                new_mask |= prime_factors_mask[c]
                                new_score += 1.0 / (N - c)
                                added_count += 1
                                if added_count >= 4:
                                    break

                        if new_score > current_score + 1e-12:
                            chosen = new_chosen
                            chosen_set = new_set
                            used_mask = new_mask
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break

        return chosen, current_score

    best, best_score = local_search(best)

    # Random restarts with local search
    for restart in range(150):
        if time.time() > deadline - 20:  # reserve time for B&B
            break

        rng_seed = 1000 + restart * 7919
        rng2 = random.Random(rng_seed)
        strategy = restart % 5

        if strategy == 0:
            seed_candidates = list(range(511, 300, -1))
        elif strategy == 1:
            seed_candidates = list(range(511, 200, -1))
        elif strategy == 2:
            seed_candidates = list(range(511, 100, -1))
        elif strategy == 3:
            seed_candidates = [a for a in range(511, 400, -1) if rng2.random() < 0.5]
        else:
            seed_candidates = list(range(511, 450, -1)) + [rng2.randint(1, 511) for _ in range(20)]

        rng2.shuffle(seed_candidates)
        num_seeds = 2 + (restart % 5)
        chosen = []
        used_mask = 0
        for a in seed_candidates:
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
            if len(chosen) >= num_seeds:
                break

        if not chosen:
            continue

        for a in range(511, 0, -1):
            if a in chosen:
                continue
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]

        chosen, s = local_search(chosen, max_iters=100)
        if s > best_score:
            best_score = s
            best = chosen

    # Branch-and-bound MWIS polish
    # Use the current best as incumbent; try to find a better independent set
    # The conflict graph has 512 vertices. We'll use a recursive B&B with:
    # - Candidate set represented as a bitset (Python int)
    # - Upper bound: sum of weights of candidates (optimistic) + greedy coloring bound
    # - Branch on highest-weight vertex in candidate set

    # For efficiency, precompute: for each vertex, its conflict set within the full graph
    # We'll work with vertex indices 1..511 mapped to 0..510 for bit positions
    # Actually, let's use 0-based indexing internally: vertex a (1..511) -> index a-1

    # Convert to 0-based: vertex index i (0..511) corresponds to number i+1
    # weight[i] = 1/(512 - (i+1)) = 1/(511-i)
    w0 = [0.0] * 512
    for i in range(512):
        w0[i] = 1.0 / (511 - i)

    # Conflict in 0-based: conflict0[i] is bitmask of conflicts for vertex i
    conflict0 = [0] * 512
    for i in range(512):
        a = i + 1
        for j in range(512):
            if i == j:
                continue
            b = j + 1
            if (prime_factors_mask[a] & prime_factors_mask[b]) != 0:
                conflict0[i] |= (1 << j)

    # Greedy coloring upper bound for a candidate set (bitset)
    # This is an approximation; exact coloring is NP-hard but greedy gives a valid upper bound
    # for the independent set problem via the "color bound" technique
    def color_bound(candidates_bitset, current_weight):
        """
        Upper bound on the weight of any independent set within candidates_bitset.
        Uses a greedy sequential coloring: assign colors such that adjacent vertices
        get different colors. The max color class weight gives a bound.
        Actually, the standard technique: the size of a max independent set <= number of colors.
        For weighted: we use a simpler bound - sum of top k weights where k is an upper bound
        on independence number.
        
        Simpler approach: just return sum of all weights in candidates (optimistic bound).
        For tighter: use the fact that vertices sharing a prime form cliques.
        """
        # Simple optimistic bound: sum of all weights in candidate set
        total = 0.0
        c = candidates_bitset
        while c:
            # Get lowest set bit
            lsb = c & (-c)
            idx = lsb.bit_length() - 1
            total += w0[idx]
            c &= c - 1
        return current_weight + total

    # Better bound: use clique cover. Vertices that share a prime factor p form a clique.
    # For each prime p, the vertices in candidates that are divisible by p form a clique,
    # so we can pick at most one from each such group.
    # Upper bound: for each prime, take max weight in that prime's candidate group,
    # plus vertices not divisible by any "small" prime (primes <= 22).
    # This is still an upper bound.

    def clique_bound(candidates_bitset, current_weight):
        """
        Upper bound using prime-based clique structure.
        For each prime p, at most one vertex divisible by p can be in the independent set.
        So: sum over primes p of max weight among candidates divisible by p,
        plus weight of candidates not divisible by any prime (i.e., only 1, but 1 is coprime to all).
        Actually vertex 1 (index 0) has no prime factors, so it's in no clique.
        """
        # For each prime, find the max weight candidate divisible by that prime
        # and not already "used up" by a higher prime's contribution
        # This is complex; simpler: for each prime, max weight in its group
        # The sum of these maxima is an upper bound (may overcount if a vertex is in multiple groups,
        # but since we take max per group and a vertex belongs to multiple groups, it's still valid
        # as an upper bound because the true max independent set weight <= sum of group maxima)
        
        # Precompute: for each prime index, which vertex indices are divisible by that prime
        # We'll compute on the fly for the candidate set
        
        bound = 0.0
        # For each prime, find max weight in candidates divisible by p
        for pi, p in enumerate(primes):
            # Find candidates divisible by p
            max_w = 0.0
            found = False
            c = candidates_bitset
            while c:
                lsb = c & (-c)
                idx = lsb.bit_length() - 1
                a = idx + 1
                if a % p == 0:
                    if w0[idx] > max_w:
                        max_w = w0[idx]
                    found = True
                c &= c - 1
            if found:
                bound += max_w
        
        # Add vertex 1 (index 0) if in candidates and not counted (it's not divisible by any prime)
        if candidates_bitset & 1:  # vertex 0 (number 1) is in candidates
            bound += w0[0]
        
        return current_weight + bound

    # Branch and bound
    best_incumbent = list(best)
    best_incumbent_score = best_score
    bb_start = time.time()
    bb_deadline = min(deadline - 5, bb_start + 60)  # max 60s for B&B
    
    nodes_explored = 0

    def branch_and_bound(candidates_bitset, current_set_bitset, current_weight):
        nonlocal best_incumbent, best_incumbent_score, nodes_explored
        
        if time.time() > bb_deadline:
            return
        
        nodes_explored += 1
        
        if candidates_bitset == 0:
            if current_weight > best_incumbent_score:
                best_incumbent_score = current_weight
                # Reconstruct set
                best_incumbent = []
                c = current_set_bitset
                while c:
                    lsb = c & (-c)
                    idx = lsb.bit_length() - 1
                    best_incumbent.append(idx + 1)
                    c &= c - 1
            return
        
        # Upper bound check
        ub = clique_bound(candidates_bitset, current_weight)
        if ub <= best_incumbent_score + 1e-12:
            return
        
        # Pick a vertex to branch on: highest weight in candidates
        # Find the vertex with highest weight in candidates_bitset
        best_idx = -1
        best_w = -1.0
        c = candidates_bitset
        while c:
            lsb = c & (-c)
            idx = lsb.bit_length() - 1
            if w0[idx] > best_w:
                best_w = w0[idx]
                best_idx = idx
            c &= c - 1
        
        if best_idx < 0:
            return
        
        # Branch 1: include best_idx
        # Remove best_idx and all its conflicts from candidates
        new_candidates = candidates_bitset & ~(1 << best_idx)
        new_candidates &= ~conflict0[best_idx]
        new_set = current_set_bitset | (1 << best_idx)
        branch_and_bound(new_candidates, new_set, current_weight + w0[best_idx])
        
        # Branch 2: exclude best_idx
        new_candidates2 = candidates_bitset & ~(1 << best_idx)
        branch_and_bound(new_candidates2, current_set_bitset, current_weight)

    # Start B&B with all vertices as candidates
    all_candidates = (1 << 512) - 1  # all 512 vertices
    # Actually, we only care about vertices 1..511, so indices 0..511
    # But vertex 0 (number 1) is coprime to all, so it's always beneficial to include
    # Let's start with all vertices
    branch_and_bound(all_candidates, 0, 0.0)

    if best_incumbent_score > best_score:
        best = best_incumbent
        best_score = best_incumbent_score

    # Final polish: one more local search pass
    if time.time() < deadline - 2:
        best, final_score = local_search(best, max_iters=50)
        if final_score > best_score:
            best_score = final_score

    # Ensure we have a valid answer
    if not best:
        best = [1]  # 1 is coprime to everything

    return best

# EVOLVE-BLOCK-END
