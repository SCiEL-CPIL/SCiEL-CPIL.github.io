import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    # ---------- Precomputation ----------
    # Map every t-subset to a bit index
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)

    # Precompute, for each 2-subset, the list of 4-subsets that contain it.
    # Adding a point x to a block B creates new 4-subsets that use x together
    # with exactly 3 points already in B; each such 4-subset contains a 2-subset
    # of B plus x.  We aggregate over all 2-subsets of B.
    # For speed we store the list as a Python int bitset.
    pair_to_bits = {}
    for i in range(v):
        for j in range(i + 1, v):
            bits = 0
            for a in range(v):
                if a == i or a == j:
                    continue
                for b in range(a + 1, v):
                    if b == i or b == j:
                        continue
                    quad = tuple(sorted((i, j, a, b)))
                    bits |= (1 << t_sub_to_idx[quad])
            pair_to_bits[(i, j)] = bits

    # ---------- Helper: gain of adding point x to current block B ----------
    # Returns a bitset of the 4-subsets newly covered.
    def gain_bits(B, x):
        # B is a set of ints (current block, size < k)
        # New 4-subsets created by adding x: those containing x and 3 points of B
        # = union over all 2-subsets (a,b) of B of pair_to_bits[(a,b)] restricted
        # to 4-subsets containing x.  But pair_to_bits[(a,b)] already only covers
        # 4-subsets containing a and b; adding x means the 4-subset is {x,a,b,c}
        # for c in B\{a,b}.  Simpler: iterate 3-subsets of B and form the 4-subset
        # with x.
        bits = 0
        for triple in itertools.combinations(B, 3):
            quad = tuple(sorted((x,) + triple))
            bits |= (1 << t_sub_to_idx[quad])
        return bits

    def block_bits(B):
        bits = 0
        for quad in itertools.combinations(B, t):
            bits |= (1 << t_sub_to_idx[tuple(quad)])
        return bits

    full_mask = (1 << num_t) - 1

    best_blocks = None
    best_count = float('inf')

    def try_reduce(blocks):
        """Local search: delete redundant blocks, shrink blocks 7->6->5."""
        nonlocal best_blocks, best_count
        blocks = [set(b) for b in blocks]
        # Build coverage bitset per block
        cov = [block_bits(b) for b in blocks]

        def coverage_all():
            c = 0
            for m in cov:
                c |= m
            return c

        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            # Delete redundant blocks
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                rest = 0
                for j in range(len(blocks)):
                    if j != i:
                        rest |= cov[j]
                if (cov[i] | rest) == full_mask:
                    blocks.pop(i)
                    cov.pop(i)
                    changed = True

        # Try to shrink each block: for each block B, try removing one point
        # and check coverage still holds.
        for i in range(len(blocks)):
            if time.time() > deadline - 2.0:
                break
            B = blocks[i]
            for x in list(B):
                B2 = B - {x}
                if len(B2) < t:
                    continue
                cov2 = block_bits(B2)
                rest = 0
                for j in range(len(blocks)):
                    if j != i:
                        rest |= cov[j]
                if (cov2 | rest) == full_mask:
                    blocks[i] = B2
                    cov[i] = cov2
                    changed = True
                    # recompute B
                    B = blocks[i]

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [sorted(b) for b in blocks]

    # ---------- Randomized greedy construction, many seeds ----------
    rng = random.Random(12345)
    run = 0
    while time.time() < deadline - 8.0:
        run += 1
        rng.seed(run * 7919 + 13)

        uncovered = full_mask
        blocks = []

        while uncovered != 0:
            # Pick a random uncovered 4-subset to seed a block
            # Find an uncovered bit
            # Use a random position
            # To pick a random uncovered index efficiently, sample and test
            seed_idx = rng.randrange(num_t)
            while ((uncovered >> seed_idx) & 1) == 0:
                seed_idx = rng.randrange(num_t)
            seed_subset = all_t_subsets[seed_idx]
            B = list(seed_subset)
            Bset = set(B)

            while len(B) < k:
                best_gain = -1
                best_x = -1
                tied = []
                for x in range(v):
                    if x in Bset:
                        continue
                    g = gain_bits(Bset, x) & uncovered
                    cnt = g.bit_count()
                    if cnt > best_gain:
                        best_gain = cnt
                        best_x = x
                        tied = [x]
                    elif cnt == best_gain:
                        tied.append(x)
                if best_x == -1:
                    break
                if len(tied) > 1:
                    best_x = tied[rng.randrange(len(tied))]
                B.append(best_x)
                Bset.add(best_x)

            B.sort()
            blocks.append(B)
            uncovered &= ~block_bits(set(B))

        # Local reduction
        try_reduce(blocks)

        # Occasionally, also try a "merge" local search: replace two blocks
        # with one if possible (rare, expensive) -- skip for time.

    if best_blocks is None:
        # Fallback: simple deterministic greedy
        uncovered = full_mask
        blocks = []
        while uncovered != 0:
            seed_idx = 0
            while ((uncovered >> seed_idx) & 1) == 0:
                seed_idx += 1
            B = list(all_t_subsets[seed_idx])
            Bset = set(B)
            while len(B) < k:
                best_gain = -1
                best_x = -1
                for x in range(v):
                    if x in Bset:
                        continue
                    g = gain_bits(Bset, x) & uncovered
                    cnt = g.bit_count()
                    if cnt > best_gain:
                        best_gain = cnt
                        best_x = x
                if best_x == -1:
                    break
                B.append(best_x)
                Bset.add(best_x)
            B.sort()
            blocks.append(B)
            uncovered &= ~block_bits(set(B))
        best_blocks = blocks

    return best_blocks
