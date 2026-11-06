import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (150s kill, 35s buffer)

    # Precompute all triples
    all_triples = list(combinations(range(16), 3))
    n_triples = len(all_triples)  # 560

    # Precompute the conflict graph
    # Two triples conflict if there exist three disjoint pairs such that all 8 transversals are present.
    # For triple T = {a,b,c}, it conflicts with T' = {a',b',c'} if T and T' are part of a K(2,2,2)
    # i.e., the 6 points form a 3-partite structure with all 8 transversals.
    #
    # A K(2,2,2) is defined by three disjoint pairs (x1,x2), (y1,y2), (z1,z2).
    # The 8 transversals are all (xi, yj, zk) for i,j in {1,2}.
    # Two triples T and T' conflict if they are both among the 8 transversals of some K(2,2,2).
    #
    # Strategy: For each pair of triples that share at most 2 points and together use at most 6 points,
    # check if they belong to the same K(2,2,2).
    #
    # More efficiently: enumerate all K(2,2,2) configurations (choosing 3 disjoint pairs from 16 points),
    # and for each, mark all pairs of its 8 transversals as conflicting.

    # Build conflict adjacency as a list of bitmasks (560 bits each -> 560/64 = 9 64-bit words, or use Python ints)
    # Python arbitrary-precision ints work well as bitsets.
    conflict = [0] * n_triples

    # Map triple -> index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}

    # Enumerate all ways to pick 3 disjoint pairs from 16 points
    # A K(2,2,2) is determined by choosing 6 points and partitioning them into 3 pairs.
    # For 6 points, there are 15 ways to partition into 3 pairs.
    # Number of 6-subsets of 16: C(16,6) = 8008. Total K(2,2,2) configs: 8008 * 15 = 120120.

    # For each 6-subset, enumerate all 15 pairings
    def pairings_of_6(points):
        """Yield all 15 ways to partition 6 points into 3 unordered pairs."""
        p0, p1, p2, p3, p4, p5 = points
        # Fix p0's partner, then pair up the remaining 4
        for i in range(1, 6):
            partner = p1 + i - 1
            remaining = [p for p in (p1, p2, p3, p4, p5) if p != partner]
            # remaining has 4 elements; 3 ways to pair
            r0, r1, r2, r3 = remaining
            yield (p0, partner), (r0, r1), (r2, r3)
            yield (p0, partner), (r0, r2), (r1, r3)
            yield (p0, partner), (r0, r3), (r1, r2)

    # Enumerate all K(2,2,2) and mark conflicts
    for six_pts in combinations(range(16), 6):
        for pairs in pairings_of_6(six_pts):
            (x1, x2), (y1, y2), (z1, z2) = pairs
            # 8 transversals
            transversals = [
                (x1, y1, z1), (x1, y1, z2), (x1, y2, z1), (x1, y2, z2),
                (x2, y1, z1), (x2, y1, z2), (x2, y2, z1), (x2, y2, z2)
            ]
            # Get indices
            indices = []
            valid = True
            for t in transversals:
                st = tuple(sorted(t))
                if st in triple_to_idx:
                    indices.append(triple_to_idx[st])
                else:
                    valid = False
                    break
            if not valid:
                continue
            # Mark all pairs as conflicting
            for i in range(8):
                for j in range(i + 1, 8):
                    conflict[indices[i]] |= (1 << indices[j])
                    conflict[indices[j]] |= (1 << indices[i])

    # Now we have the conflict graph. We want to find the maximum independent set.
    # Use Simulated Annealing.

    # Precompute for each node: list of non-conflicting neighbors (for fast moves)
    # Actually, for SA we need: given current set S, can we add triple t?
    # t can be added iff no member of S conflicts with t.
    # Precompute for each triple: bitmask of triples it conflicts with.
    # conflict[i] is already a bitmask of conflicting triples.

    # SA State: a set of indices (as a Python set for membership, plus a bitmask for fast conflict checks)
    # For speed, represent the current solution as a bitmask (Python int with 560 bits).

    def can_add(sol_mask, idx):
        """Check if triple idx can be added to the solution represented by sol_mask."""
        # Check if idx is already in sol
        if (sol_mask >> idx) & 1:
            return True
        # Check if any member of sol conflicts with idx
        # conflict[idx] has bits set for all triples that conflict with idx
        # We need: (sol_mask & conflict[idx]) == 0
        return (sol_mask & conflict[idx]) == 0

    def popcount(x):
        return bin(x).count('1')

    # Phase 1: Greedy initialization (fast, using precomputed conflicts)
    best_mask = 0
    best_size = 0

    for attempt in range(20):
        if time.time() > deadline - 60:
            break

        sol = 0
        order = list(range(n_triples))
        random.shuffle(order)

        for idx in order:
            if can_add(sol, idx):
                sol |= (1 << idx)

        sz = popcount(sol)
        if sz > best_size:
            best_size = sz
            best_mask = sol

    # Phase 2: Simulated Annealing
    # Move operators:
    #   1. Add a random non-conflicting triple not in sol
    #   2. Remove a random triple from sol and add a random conflicting replacement
    #   3. Swap: remove one, add one (net zero or +1)
    #   4. Kick: remove k triples, add k+1 triples

    sol = best_mask
    sol_size = best_size
    best_mask_sa = best_mask
    best_size_sa = best_size

    T = 5.0
    T_min = 0.01
    cooling = 0.9997

    # Precompute list of all indices for random selection
    all_indices = list(range(n_triples))

    sa_start = time.time()
    max_sa_time = deadline - 10.0

    while time.time() < max_sa_time:
        # Choose a random move type
        move_type = random.random()

        if move_type < 0.4:
            # Try to add a random triple
            idx = random.choice(all_indices)
            if not ((sol >> idx) & 1) and can_add(sol, idx):
                new_sol = sol | (1 << idx)
                delta = 1
                sol = new_sol
                sol_size += 1
                if sol_size > best_size_sa:
                    best_size_sa = sol_size
                    best_mask_sa = sol
            else:
                # Can't add directly; try a kick: remove a conflicting triple
                blockers = sol & conflict[idx]
                if blockers:
                    # Pick a random blocker
                    b_idx = random.choice([i for i in range(n_triples) if (blockers >> i) & 1])
                    # Remove blocker, try to add idx
                    new_sol = (sol & ~(1 << b_idx))
                    if can_add(new_sol, idx):
                        new_sol |= (1 << idx)
                        delta = 0  # net zero (removed 1, added 1)
                        # Accept with probability based on delta
                        if delta >= 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                            sol = new_sol
                            # sol_size unchanged
                            if sol_size > best_size_sa:
                                best_size_sa = sol_size
                                best_mask_sa = sol
        elif move_type < 0.7:
            # Remove a random triple from sol (regression move)
            in_sol = [i for i in range(n_triples) if (sol >> i) & 1]
            if in_sol:
                rm_idx = random.choice(in_sol)
                new_sol = sol & ~(1 << rm_idx)
                delta = -1
                if random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    sol = new_sol
                    sol_size -= 1
        elif move_type < 0.85:
            # Swap: remove one from sol, add one not in sol
            in_sol = [i for i in range(n_triples) if (sol >> i) & 1]
            if in_sol:
                rm_idx = random.choice(in_sol)
                # Try to add a random triple after removing rm_idx
                for _ in range(5):
                    add_idx = random.choice(all_indices)
                    if ((sol >> add_idx) & 1) == 0:
                        new_sol = sol & ~(1 << rm_idx)
                        if can_add(new_sol, add_idx):
                            new_sol |= (1 << add_idx)
                            delta = 0
                            if delta >= 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                                sol = new_sol
                            break
        else:
            # Larger kick: remove 2, try to add 2
            in_sol = [i for i in range(n_triples) if (sol >> i) & 1]
            if len(in_sol) >= 2:
                rm1, rm2 = random.sample(in_sol, 2)
                new_sol = sol & ~(1 << rm1) & ~(1 << rm2)
                added = 0
                for _ in range(10):
                    add_idx = random.choice(all_indices)
                    if ((new_sol >> add_idx) & 1) == 0 and can_add(new_sol, add_idx):
                        new_sol |= (1 << add_idx)
                        added += 1
                        if added >= 2:
                            break
                delta = added - 2
                if delta >= 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-10)):
                    sol = new_sol
                    sol_size += delta
                    if sol_size > best_size_sa:
                        best_size_sa = sol_size
                        best_mask_sa = sol

        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min

    # Phase 3: Final greedy polish on the best SA solution
    # Try to add any remaining triple that fits
    polish_mask = best_mask_sa
    for idx in all_indices:
        if not ((polish_mask >> idx) & 1) and can_add(polish_mask, idx):
            polish_mask |= (1 << idx)

    polish_size = popcount(polish_mask)
    if polish_size > best_size_sa:
        best_mask_sa = polish_mask
        best_size_sa = polish_size

    # Also try greedy polish from the initial best
    polish2 = best_mask
    for idx in all_indices:
        if not ((polish2 >> idx) & 1) and can_add(polish2, idx):
            polish2 |= (1 << idx)
    if popcount(polish2) > best_size_sa:
        best_mask_sa = polish2
        best_size_sa = popcount(polish2)

    # Convert best_mask_sa to list of triples
    result = []
    for i in range(n_triples):
        if (best_mask_sa >> i) & 1:
            result.append(list(all_triples[i]))

    # Fallback if empty
    if not result:
        result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]

    return result
