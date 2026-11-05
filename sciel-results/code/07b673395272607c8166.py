import numpy as np
import itertools
import time


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Precompute all t-subsets and their indices
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}

    best_blocks = None
    best_count = float('inf')

    def do_greedy(rng=None):
        if time.time() > deadline:
            return None

        cover_count = np.zeros(n_triples, dtype=np.int32)
        blocks = []

        while True:
            if time.time() > deadline:
                return None

            uncovered_mask = cover_count == 0
            if not uncovered_mask.any():
                break

            uncovered_indices = np.where(uncovered_mask)[0]
            if rng is not None:
                start_idx = uncovered_indices[rng.integers(len(uncovered_indices))]
            else:
                start_idx = uncovered_indices[0]

            start_triple = all_triples[start_idx]
            block = list(start_triple)
            block_set = set(block)

            # Greedily add points until block has k elements
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        triple = tuple(sorted(combo + (p,)))
                        idx = triple_to_idx[triple]
                        if cover_count[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                if best_point < 0:
                    break
                block.append(best_point)
                block_set.add(best_point)

            block.sort()
            blocks.append(block)

            for triple in itertools.combinations(block, t):
                cover_count[triple_to_idx[triple]] += 1

        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                cover_count[triple_to_idx[triple]] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for block in blocks:
                redundant = True
                for triple in itertools.combinations(block, t):
                    if cover_count[triple_to_idx[triple]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for triple in itertools.combinations(block, t):
                        cover_count[triple_to_idx[triple]] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            blocks = new_blocks
        return blocks

    # Deterministic pass
    blocks = do_greedy(None)
    if blocks:
        blocks = remove_redundant(blocks)
        best_blocks = blocks
        best_count = len(blocks)

    # Randomized restarts
    rng = np.random.default_rng(12345)
    while time.time() < deadline - 10:
        blocks = do_greedy(rng)
        if blocks is None:
            break
        blocks = remove_redundant(blocks)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            if best_count <= 69:
                break

    return best_blocks
