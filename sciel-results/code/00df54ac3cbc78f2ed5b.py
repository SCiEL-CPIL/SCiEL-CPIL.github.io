<<<<<<< SEARCH
import itertools
import random
import time
from collections import defaultdict
import math


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(frozenset(c) for c in itertools.combinations(range(v), t))
    blocks = []
    block_set = set()

    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(tuple(uncovered))
        block = list(seed)
        rng.shuffle(block)

        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)

            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            blocks.append(b_tuple)
            block_set.add(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))

    return blocks
=======
import itertools
import random
import time
from collections import defaultdict
import math


def _precompute_triple_indices(v, t):
    """Precompute mapping from frozenset of t-subset to integer index."""
    triple_to_idx = {}
    idx_to_triple = []
    idx = 0
    for c in itertools.combinations(range(v), t):
        f = frozenset(c)
        triple_to_idx[f] = idx
        idx_to_triple.append(f)
        idx += 1
    return triple_to_idx, idx_to_triple


def _greedy_cover(v, k, t, rng, triple_to_idx=None):
    """Fast greedy construction to get a valid starting covering."""
    if triple_to_idx is None:
        triple_to_idx, _ = _precompute_triple_indices(v, t)
    
    total_triples = len(triple_to_idx)
    covered = bytearray(total_triples)  # 0 or 1
    blocks = []
    block_set = set()
    num_covered = 0

    # Precompute for each point, the list of triples containing it (as indices)
    # triple_containing[x] = list of triple indices that include point x
    triple_containing = defaultdict(list)
    for idx, f in enumerate(triple_to_idx.values()):
        for x in f:
            triple_containing[x].append(idx)

    while num_covered < total_triples:
        # Find an uncovered triple
        seed_idx = None
        # Try random sampling first
        for _ in range(100):
            ci = rng.randrange(total_triples)
            if not covered[ci]:
                seed_idx = ci
                break
        if seed_idx is None:
            # Linear scan
            for ci in range(total_triples):
                if not covered[ci]:
                    seed_idx = ci
                    break
        if seed_idx is None:
            break
            
        seed_triple = triple_to_idx.get(seed_idx)
        if seed_triple is None:
            # Find the actual triple
            for idx, f in enumerate(triple_to_idx.items()):
                if idx == seed_idx:
                    seed_triple = f[1]
                    break
        
        block = list(seed_triple)
        rng.shuffle(block)
        block_set_current = set(block)

        # Greedily add points to maximize coverage, with load-balancing bias
        while len(block) < k:
            best_x = None
            best_gain = -1
            best_load = float('inf')
            cands = [x for x in range(v) if x not in block_set_current]
            rng.shuffle(cands)

            for x in cands:
                gain = 0
                load_penalty = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    tri_idx = triple_to_idx.get(fs, -1)
                    if tri_idx >= 0:
                        if not covered[tri_idx]:
                            gain += 1
                        else:
                            load_penalty += 1
                # Prefer high gain, low load penalty
                score = gain * 10 - load_penalty
                if score > best_gain or (score == best_gain and load_penalty < best_load):
                    best_gain = score
                    best_load = load_penalty
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set_current.add(best_x)

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            blocks.append(b_tuple)
            block_set.add(b_tuple)
            for c in itertools.combinations(block, t):
                f = frozenset(c)
                tri_idx = triple_to_idx.get(f, -1)
                if tri_idx >= 0 and not covered[tri_idx]:
                    covered[tri_idx] = 1
                    num_covered += 1

    return blocks
>>>>>>> REPLACE

