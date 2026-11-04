"""Covering design C(19,7,4) via fast greedy + local search."""
import time
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Precompute all C(19,4)=3876 t-subsets as frozensets
    all_tsubs = [frozenset(c) for c in itertools.combinations(range(v), t)]
    n_tsubs = len(all_tsubs)

    # Precompute for each point, the list of t-subset indices it participates in
    # point_to_tsubs[p] = list of indices i where p in all_tsubs[i]
    point_to_tsubs = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_to_tsubs[p].append(i)

    # Greedy construction with numpy coverage vector
    covered = np.zeros(n_tsubs, dtype=np.uint8)
    blocks = []

    while True:
        if time.time() > deadline:
            break
        uncovered_count = int(np.count_nonzero(1 - covered))
        if uncovered_count == 0:
            break

        # Find the best starting t-subset (one with fewest already-covered supersets? 
        # Actually just pick any uncovered one - first uncovered)
        # For speed, pick the first uncovered t-subset
        uncovered_indices = np.where(covered == 0)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[0]
        start_tsub = all_tsubs[start_idx]
        block = list(start_tsub)

        # Greedily add points to maximize uncovered coverage
        while len(block) < k:
            if time.time() > deadline:
                break
            best_gain = -1
            best_point = -1
            block_set = set(block)
            for p in range(v):
                if p in block_set:
                    continue
                # Count how many currently-uncovered t-subsets would be covered
                # by adding p to the block
                # A t-subset is covered by block if it's a subset of block
                # With p added, new t-subsets covered are those containing p 
                # and (t-1) elements from block
                gain = 0
                # Enumerate (t-1)-subsets of current block
                for rest in itertools.combinations(block, t - 1):
                    ts = frozenset(rest + (p,))
                    # Find index of this t-subset
                    # We need a fast lookup
                    # Precompute a dict from t-subset to index
                    pass
                # This is too slow - let me use a different approach
                # Count uncovered t-subsets that contain p and are subset of block+[p]
                # i.e., t-subsets S where p in S, S \ {p} subset of block, and not covered
                for idx in point_to_tsubs[p]:
                    if covered[idx]:
                        continue
                    ts = all_tsubs[idx]
                    # Check if ts \ {p} is subset of block
                    if all(x in block_set for x in ts if x != p):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point == -1 or best_gain <= 0:
                # Can't improve, pick arbitrary
                for p in range(v):
                    if p not in block_set:
                        best_point = p
                        break
            block.append(best_point)

        block.sort()
        blocks.append(block)

        # Update coverage
        block_set = set(block)
        for idx in range(n_tsubs):
            if covered[idx]:
                continue
            ts = all_tsubs[idx]
            if all(x in block_set for x in ts):
                covered[idx] = 1

    # Verify coverage
    if np.any(covered == 0):
        # Emergency: add blocks for remaining uncovered
        while np.any(covered == 0):
            uncovered_indices = np.where(covered == 0)[0]
            start_idx = uncovered_indices[0]
            start_tsub = all_tsubs[start_idx]
            block = list(start_tsub)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for p in range(v):
                    if p in block_set:
                        continue
                    gain = 0
                    for idx in point_to_tsubs[p]:
                        if covered[idx]:
                            continue
                        ts = all_tsubs[idx]
                        if all(x in block_set for x in ts if x != p):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                if best_point == -1:
                    for p in range(v):
                        if p not in block_set:
                            best_point = p
                            break
                block.append(best_point)
                block_set.add(best_point)
            block.sort()
            blocks.append(block)
            for idx in range(n_tsubs):
                if covered[idx]:
                    continue
                ts = all_tsubs[idx]
                if all(x in set(block) for x in ts):
                    covered[idx] = 1

    # Local search: try to replace a block with a better one
    # Or try to remove a block if its t-subsets are covered by others
    elapsed = time.time() - start_time
    if elapsed < 100:  # have time for local search
        # Try removing redundant blocks
        for attempt in range(200):
            if time.time() > deadline:
                break
            removed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break
                # Check if block i is redundant
                block_i = set(blocks[i])
                redundant = True
                for idx in range(n_tsubs):
                    ts = all_tsubs[idx]
                    if not all(x in block_i for x in ts):
                        continue
                    # This t-subset is covered by block_i; check if covered by another block
                    found = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if all(x in set(blocks[j]) for x in ts):
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    removed = True
                    break
            if not removed:
                break

        # Try block replacement: replace one block with a different one that covers more
        for attempt in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                # Try to find a better replacement for block i
                # Compute which t-subsets are ONLY covered by block i
                block_i_set = set(blocks[i])
                only_by_i = []
                for idx in range(n_tsubs):
                    ts = all_tsubs[idx]
                    if not all(x in block_i_set for x in ts):
                        continue
                    # Check if covered by any other block
                    only = True
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if all(x in set(blocks[j]) for x in ts):
                            only = False
                            break
                    if only:
                        only_by_i.append(idx)
                
                if not only_by_i:
                    continue  # redundant, already handled
                
                # Try to find a new block that covers all only_by_i t-subsets
                # and maximizes additional coverage
                # The new block must contain all points in the intersection of only_by_i t-subsets
                # Actually it must cover each t-subset, so for each t-subset S in only_by_i,
                # the new block must contain S.
                # So the new block must contain union of all S in only_by_i? No, it must contain each S individually.
                # So new_block must be a superset of each S in only_by_i.
                # That means new_block must contain the union of all points in all S in only_by_i.
                required = set()
                for idx in only_by_i:
                    required.update(all_tsubs[idx])
                
                if len(required) > k:
                    continue  # can't fit in k elements
                
                # Build new block: start with required, fill rest greedily
                new_block = list(required)
                new_block_set = set(required)
                while len(new_block) < k:
                    best_gain = -1
                    best_point = -1
                    for p in range(v):
                        if p in new_block_set:
                            continue
                        gain = 0
                        for idx in point_to_tsubs[p]:
                            if covered[idx]:
                                continue
                            ts = all_tsubs[idx]
                            if all(x in new_block_set for x in ts if x != p):
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = p
                    if best_point == -1:
                        break
                    new_block.append(best_point)
                    new_block_set.add(best_point)
                
                if len(new_block) < k:
                    continue
                
                # Check if new block is different from old
                if sorted(new_block) == sorted(blocks[i]):
                    continue
                
                # Check if new block covers all only_by_i
                new_block_set2 = set(new_block)
                valid = True
                for idx in only_by_i:
                    if not all(x in new_block_set2 for x in all_tsubs[idx]):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Accept the replacement
                blocks[i] = sorted(new_block)
                improved = True
                break
            if not improved:
                break

    return blocks
