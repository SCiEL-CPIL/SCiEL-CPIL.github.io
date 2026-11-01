"""Covering design C(24,7,3): fast greedy with randomized restarts and local search."""
import itertools
import time
import random
from collections import defaultdict


def _build_greedy(v, k, t, uncovered, rng):
    """Build a greedy covering from the given uncovered set."""
    blocks = []
    uncovered = set(uncovered)
    while uncovered:
        # Pick a random uncovered triple to start (or the first for determinism in some runs)
        start_triple = rng.choice(list(uncovered))
        block = list(start_triple)
        
        while len(block) < k:
            # Compute gain for each candidate point
            best_x = -1
            best_gain = -1
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered triples would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    triple = frozenset(rest + (x,))
                    if triple in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove all covered triples
        for triple in itertools.combinations(block, t):
            uncovered.discard(frozenset(triple))
    
    return blocks


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce the number of blocks via local search."""
    blocks = [list(b) for b in blocks]
    n = len(blocks)
    
    # Build coverage count: for each triple, how many blocks cover it
    triple_count = defaultdict(int)
    block_triples = []
    for b in blocks:
        triples = [frozenset(c) for c in itertools.combinations(b, t)]
        block_triples.append(triples)
        for tr in triples:
            triple_count[tr] += 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing each block
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                break
            # Check if block i is redundant
            redundant = True
            for tr in block_triples[i]:
                if triple_count[tr] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for tr in block_triples[i]:
                    triple_count[tr] -= 1
                    if triple_count[tr] == 0:
                        del triple_count[tr]
                blocks.pop(i)
                block_triples.pop(i)
                n -= 1
                improved = True
        
        # Try replacing two blocks with one
        if n >= 2 and time.time() < deadline:
            for i in range(n - 1):
                if time.time() > deadline:
                    break
                for j in range(n - 1, i, -1):
                    if time.time() > deadline:
                        break
                    # Check if union of triples covered by blocks i and j
                    # can be covered by a single k-subset
                    union_triples = set(block_triples[i]) | set(block_triples[j])
                    # Find all points that appear in any triple in union_triples
                    points = set()
                    for tr in union_triples:
                        points.update(tr)
                    if len(points) <= k:
                        # We can form a block from these points (pad if needed)
                        new_block = sorted(points)
                        if len(new_block) < k:
                            # Pad with arbitrary points
                            for p in range(v):
                                if p not in new_block:
                                    new_block.append(p)
                                if len(new_block) == k:
                                    break
                        new_block = sorted(new_block[:k])
                        # Verify this new block covers all union_triples
                        new_block_set = set(new_block)
                        covers_all = all(
                            set(tr).issubset(new_block_set) 
                            for tr in union_triples
                        )
                        if covers_all:
                            # Replace blocks i and j with new_block
                            for tr in block_triples[i]:
                                triple_count[tr] -= 1
                                if triple_count[tr] == 0:
                                    del triple_count[tr]
                            for tr in block_triples[j]:
                                triple_count[tr] -= 1
                                if triple_count[tr] == 0:
                                    del triple_count[tr]
                            for tr in itertools.combinations(new_block, t):
                                triple_count[frozenset(tr)] += 1
                            blocks[i] = new_block
                            block_triples[i] = [frozenset(c) for c in itertools.combinations(new_block, t)]
                            blocks.pop(j)
                            block_triples.pop(j)
                            n -= 1
                            improved = True
                            break
                if improved:
                    break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    best_blocks = None
    best_count = float('inf')
    
    # First run: deterministic greedy (use first uncovered triple)
    rng = random.Random(42)
    uncovered = set(all_tsets)
    blocks = _build_greedy(v, k, t, uncovered, rng)
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = [list(b) for b in blocks]
    
    # Local search on best solution
    if time.time() < deadline:
        ls_blocks = _local_search(best_blocks, v, k, t, deadline)
        if len(ls_blocks) < best_count:
            best_count = len(ls_blocks)
            best_blocks = ls_blocks
    
    # Randomized restarts
    attempt = 0
    while time.time() < deadline - 5:  # leave 5s buffer
        attempt += 1
        rng = random.Random(attempt * 1000 + 7)
        uncovered = set(all_tsets)
        blocks = _build_greedy(v, k, t, uncovered, rng)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            # Try local search on this new best
            if time.time() < deadline - 5:
                ls_blocks = _local_search(best_blocks, v, k, t, deadline - 5)
                if len(ls_blocks) < best_count:
                    best_count = len(ls_blocks)
                    best_blocks = ls_blocks
    
    return best_blocks
