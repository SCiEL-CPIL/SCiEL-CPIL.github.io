"""Evolved program: genetic algorithm + local-search repair for covering design C(v,k,t)."""
# EVOLVE-BLOCK-START
import random
import time


def _gain(block, x, uncovered):
    """Number of uncovered t-subsets that would be covered by adding x to block."""
    cnt = 0
    for rest in _combos(block, t - 1):
        if tuple(sorted(rest + (x,))) in uncovered:
            cnt += 1
    return cnt


def _combos(seq, r):
    """itertools.combinations but local for speed."""
    return list(_itertools_combinations(seq, r))


def _itertools_combinations(iterable, r):
    import itertools
    return itertools.combinations(iterable, r)


def _tsubsets(block):
    return _itertools_combinations(block, t)


def _make_block(uncovered, rng, k):
    """Build one k-block greedily to cover many uncovered t-subsets."""
    if not uncovered:
        return sorted(rng.sample(range(v), k))
    # start from an uncovered t-subset
    seed = next(iter(uncovered))
    block = list(seed)
    while len(block) < k:
        best_x, best_g = None, -1
        for x in range(v):
            if x in block:
                continue
            g = _gain(block, x, uncovered)
            if g > best_g:
                best_g, best_x = g, x
        if best_x is None:
            break
        block.append(best_x)
    # pad with random if still short
    while len(block) < k:
        cand = rng.randrange(v)
        if cand not in block:
            block.append(cand)
    return sorted(block)


def _uncovered_set(blocks):
    """Set of t-subsets not contained in any block."""
    covered = set()
    for b in blocks:
        for ts in _tsubsets(b):
            covered.add(ts)
    all_ts = set(_itertools_combinations(range(v), t))
    return all_ts - covered


def _repair(blocks, uncovered, rng, budget=250000):
    """Greedy fill of uncovered t-subsets until covered or budget exhausted."""
    blocks = [list(b) for b in blocks]
    start = time.time()
    while uncovered and (time.time() - start) < 10.0 and budget > 0:
        budget -= 1
        block = _make_block(uncovered, rng, k)
        blocks.append(block)
        for ts in _tsubsets(block):
            uncovered.discard(ts)
    return blocks


def _fitness(blocks, uncovered):
    """Lower is better: blocks count + big penalty per uncovered t-subset."""
    return len(blocks) + 100000 * len(uncovered)


def _crossover(a, b, rng):
    """Uniform crossover of two block lists."""
    if len(a) < len(b):
        a, b = b, a
    child = []
    for blk in a:
        if rng.random() < 0.5 and rng.randrange(len(b)) < len(b):
            child.append(list(b[rng.randrange(len(b))]))
        else:
            child.append(list(blk))
    return child


def _mutate(block, rng, rate=0.3):
    """Swap some points in a block for random others."""
    block = list(block)
    n = len(block)
    for _ in range(rng.randint(1, max(1, int(n * rate) + 1))):
        i = rng.randrange(n)
        x = rng.randrange(v)
        if x in block:
            continue
        block[i] = x
        block = sorted(block)
    return block


def construct(v, k, t):
    global v, k, t
    v, k, t = v, k, t
    rng = random.Random(12345)
    start = time.time()
    deadline = start + 105.0  # leave headroom under 120s

    # --- Initial population: greedy coverings with different seeds ---
    all_ts = set(_itertools_combinations(range(v), t))
    pop = []
    for trial in range(4):
        unc = set(all_ts)
        blocks = []
        while unc:
            blocks.append(_make_block(unc, rng, k))
            for ts in _tsubsets(blocks[-1]):
                unc.discard(ts)
        pop.append(blocks)

    # Evaluate
    def eval_pop(blocks):
        unc = _uncovered_set(blocks)
        return _fitness(blocks, unc), unc

    best_blocks = None
    best_fit = float('inf')
    for blocks in pop:
        fit, unc = eval_pop(blocks)
        if fit < best_fit:
            best_fit = fit
            best_blocks = [list(b) for b in blocks]

    # --- Genetic evolution loop ---
    pop_size = 6
    pop = [list(b) for b in best_blocks]
    while len(pop) < pop_size:
        pop.append([list(b) for b in rng.choice(pop)])

    gen = 0
    while time.time() < deadline - 12.0:
        gen += 1
        # Evaluate current population
        scored = []
        for blocks in pop:
            fit, unc = eval_pop(blocks)
            scored.append((fit, blocks, unc))
        scored.sort(key=lambda x: x[0])
        if scored[0][0] < best_fit:
            best_fit = scored[0][0]
            best_blocks = [list(b) for b in scored[0][1]]

        # Elitism: keep top half
        new_pop = [list(b) for _, b, _ in scored[:pop_size // 2]]

        # Generate offspring via crossover + mutation
        while len(new_pop) < pop_size:
            p1 = rng.choice(scored[:pop_size // 2 + 1])[1]
            p2 = rng.choice(scored[:pop_size // 2 + 1])[1]
            child = _crossover(p1, p2, rng)
            # Mutate a few blocks
            for i in range(rng.randint(0, 3)):
                if child:
                    child[i] = _mutate(child[i], rng)
            # Repair to validity
            unc = _uncovered_set(child)
            if unc:
                child = _repair(child, unc, rng, budget=50000)
            new_pop.append(child)

        pop = new_pop

    # --- Final: take best, repair to fully valid, dedup ---
    blocks = [list(b) for b in best_blocks]
    unc = _uncovered_set(blocks)
    if unc:
        blocks = _repair(blocks, unc, rng, budget=1000000)
    # Deduplicate
    seen = set()
    dedup = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            dedup.append(b)
    blocks = dedup

    # Final safety: ensure valid within remaining time
    unc = _uncovered_set(blocks)
    while unc and time.time() < deadline - 1.0:
        blocks.append(_make_block(unc, rng, k))
        for ts in _tsubsets(blocks[-1]):
            unc.discard(ts)

    return blocks
# EVOLVE-BLOCK-END
