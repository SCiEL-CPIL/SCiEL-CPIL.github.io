"""Covering design C(v,k,t) via fast greedy with precomputed triple tables."""
import itertools
import time


def _triple_index(v, t):
    """Map each t-subset to a unique integer index, and build list of triples."""
    # For small v,t we can just enumerate and use a dict
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {tr: i for i, tr in enumerate(triples)}
    return triples, triple_to_idx


def _block_triples(block, triple_to_idx):
    """Return list of triple indices covered by this block."""
    return [triple_to_idx[t] for t in itertools.combinations(block, t) if t in triple_to_idx]


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    triples, triple_to_idx = _triple_index(v, t)
    num_triples = len(triples)

    # uncovered[i] = True if triple i is not yet covered
    uncovered = bytearray(b'\x01') * num_triples
    uncovered_count = num_triples

    blocks = []

    while uncovered_count > 0:
        if time.time() > deadline:
            break

        # Find first uncovered triple
        first_idx = -1
        for i in range(num_triples):
            if uncovered[i]:
                first_idx = i
                break
        if first_idx < 0:
            break

        seed_triple = triples[first_idx]
        block = list(seed_triple)

        # Greedily add points to maximize coverage of uncovered triples
        while len(block) < k:
            if time.time() > deadline:
                break

            best_x = -1
            best_gain = -1
            block_set = set(block)

            # Precompute the (t-1)-subsets of current block for fast gain calc
            block_subsets = list(itertools.combinations(block, t - 1))

            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for sub in block_subsets:
                    # Check if tuple(sorted(sub + (x,))) is an uncovered triple
                    combined = tuple(sorted(sub + (x,)))
                    idx = triple_to_idx.get(combined, -1)
                    if idx >= 0 and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    if best_gain == len(block_subsets):
                        break

            if best_x < 0:
                break
            block.append(best_x)

        block.sort()

        # Check if this block is already in our list (avoid duplicates)
        block_tuple = tuple(block)
        if block_tuple not in [tuple(b) for b in blocks]:
            # Cover all triples in this block
            for idx in _block_triples(block, triple_to_idx):
                if uncovered[idx]:
                    uncovered[idx] = 0
                    uncovered_count -= 1
            blocks.append(block)

    return blocks
