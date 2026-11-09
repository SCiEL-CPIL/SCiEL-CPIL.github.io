"""Covering design C(v,k,t) — Random initialization + greedy repair with local search."""
import itertools
import time
import random
import math

def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 115.0  # safety margin before 120s kill

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # Precompute: for each block, the list of t-subset indices it covers
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(subs)

    # Precompute: for each t-subset, the list of block indices that cover it
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    # Precompute block gain (number of t-subsets each block covers)
    block_size = [len(subs) for subs in block_tsubs]

    def greedy_repair(uncovered_set, existing_blocks, seed=0):
        """Add blocks to cover remaining uncovered t-subsets."""
        rng = random.Random(seed)
        is_covered = bytearray(N_t)
        for si in existing_blocks:
            for s in block_tsubs[si]:
                is_covered[s] = 1
        
        # Compute current uncovered
        uncovered = N_t
        for si in range(N_t):
            if not is_covered[si]:
                uncovered += 1
            else:
                uncovered -= 1
        uncovered = N_t - uncovered  # recalculate properly
        
        # Recompute uncovered count
        uncovered = 0
        for si in range(N_t):
            if not is_covered[si]:
                uncovered += 1

        gains = block_size[:]
        selected = list(existing_blocks)

        while uncovered > 0 and time.time() < DEADLINE:
            best_gain = -1
            candidates = []
            for i in range(N_blocks):
                g = gains[i]
                if g > best_gain:
                    best_gain = g
                    candidates = [i]
                elif g == best_gain:
                    candidates.append(i)

            if best_gain <= 0:
                break

            if len(candidates) > 1:
                best_i = candidates[rng.randrange(len(candidates))]
            else:
                best_i = candidates[0]

            selected.append(best_i)

            for si in block_tsubs[best_i]:
                if not is_covered[si]:
                    is_covered[si] = 1
                    uncovered -= 1
                    for b_idx in tsub_blocks[si]:
                        gains[b_idx] -= 1

        return selected

    def try_reduce(block_indices):
        """Remove redundant blocks iteratively."""
        if not block_indices:
            return block_indices

        n = len(block_indices)
        cov = [0] * N_t
        for bi in block_indices:
            for si in block_tsubs[bi]:
                cov[si] += 1

        removed = [False] * n
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                removable = True
                for si in block_tsubs[block_indices[i]]:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    removed[i] = True
                    changed = True
                    for si in block_tsubs[block_indices[i]]:
                        cov[si] -= 1

        return [block_indices[i] for i in range(n) if not removed[i]]

    def local_search(block_indices):
        """Try to remove or swap blocks to reduce count."""
        if not block_indices:
            return block_indices

        n = len(block_indices)
        cov = [0] * N_t
        for bi in block_indices:
            for si in block_tsubs[bi]:
                cov[si] += 1

        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(n):
                if time.time() > DEADLINE:
                    break
                removable = True
                for si in block_tsubs[block_indices[i]]:
                    if cov[si] <= 1:
                        removable = False
                        break
                if removable:
                    for si in block_tsubs[block_indices[i]]:
                        cov[si] -= 1
                    block_indices.pop(i)
                    n -= 1
                    improved = True
                    break

        return block_indices

    def random_init(m, seed=0):
        """Sample m blocks uniformly at random."""
        rng = random.Random(seed)
        return [rng.randrange(N_blocks) for _ in range(m)]

    # Poisson estimate for number of blocks needed
    # Expected uncovered = N_t * (1 - C(k,t)/C(v,t))^m
    # Set to 1: m = ln(1) / ln(1 - C(k,t)/C(v,t)) but ln(1)=0
    # Better: m = -ln(epsilon) / ln(1 - C(k,t)/C(v,t)) where epsilon = 1/N_t
    p = math.comb(k, t) / math.comb(v, t)
    # m such that N_t * (1-p)^m = 1
    m_est = -math.log(1.0 / N_t) / math.log(1.0 - p)
    m_est = int(m_est) + 5  # add small buffer

    best_solution = None
    best_len = float('inf')

    # Strategy 1: Random init + greedy repair
    for r in range(30):
        if time.time() > DEADLINE - 10:
            break
        m = max(1, m_est - 20 + r * 2)  # vary around estimate
        init_blocks = random_init(m, seed=r)
        # Remove duplicates
        init_blocks = list(set(init_blocks))
        sol = greedy_repair(set(init_blocks), [], seed=r)
        sol = try_reduce(sol)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Strategy 2: Pure greedy restarts (original approach)
    for r in range(20):
        if time.time() > DEADLINE - 5:
            break
        is_covered = bytearray(N_t)
        uncovered = N_t
        gains = block_size[:]
        selected = []
        rng = random.Random(1000 + r)

        while uncovered > 0 and time.time() < DEADLINE - 5:
            best_gain = -1
            candidates = []
            for i in range(N_blocks):
                g = gains[i]
                if g > best_gain:
                    best_gain = g
                    candidates = [i]
                elif g == best_gain:
                    candidates.append(i)

            if best_gain <= 0:
                break

            if len(candidates) > 1:
                best_i = candidates[rng.randrange(len(candidates))]
            else:
                best_i = candidates[0]

            selected.append(best_i)

            for si in block_tsubs[best_i]:
                if not is_covered[si]:
                    is_covered[si] = 1
                    uncovered -= 1
                    for b_idx in tsub_blocks[si]:
                        gains[b_idx] -= 1

        sol = try_reduce(selected)
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    # Final local search / reduction pass
    if best_solution and time.time() < DEADLINE - 3:
        best_solution = try_reduce(best_solution)
        best_solution = local_search(best_solution)

    if best_solution is None:
        # Fallback: single greedy
        is_covered = bytearray(N_t)
        uncovered = N_t
        gains = block_size[:]
        selected = []
        while uncovered > 0:
            best_gain = -1
            best_i = 0
            for i in range(N_blocks):
                if gains[i] > best_gain:
                    best_gain = gains[i]
                    best_i = i
            if best_gain <= 0:
                break
            selected.append(best_i)
            for si in block_tsubs[best_i]:
                if not is_covered[si]:
                    is_covered[si] = 1
                    uncovered -= 1
                    for b_idx in tsub_blocks[si]:
                        gains[b_idx] -= 1
        best_solution = try_reduce(selected)

    # Convert block indices to actual blocks
    return [all_blocks[bi] for bi in best_solution]
