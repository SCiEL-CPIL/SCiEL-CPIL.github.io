"""Improved covering construction with greedy + local search and deadline guard."""
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 140  # safety margin below 150s

    # Precompute all t-subsets
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    n_t = len(all_t)

    # Precompute, for each point, the list of t-subset indices that contain it
    # Actually, for each pair (point, t-subset) we need efficient lookup.
    # Better: for each t-subset index, store which points are in it (already known)
    # For gain computation: when we have a partial block B, and we want to add point x,
    # the new t-subsets covered are those that contain x and (t-1) points from B.

    # Build: for each (t-1)-subset, list of t-subsets that extend it
    t_minus_1 = t - 1
    # Map from (t-1)-subset tuple to list of t-subset indices containing it
    ext_map = defaultdict(list)
    for ts in all_t:
        for i in range(t):
            sub = ts[:i] + ts[i+1:]
            ext_map[sub].append(t_index[ts])

    # For each point x, and each (t-1)-subset not containing x, 
    # the t-subsets gained = ext_map[(t-1)-subset with x added]
    # Actually, we need: for partial block B and candidate x not in B,
    # gain = number of uncovered t-subsets S such that x in S and |S ∩ B| >= t-1
    # i.e., S contains x and (t-1) elements from B.
    # So for each (t-1)-subset A of B, the t-subsets containing A∪{x} are ext_map[A∪{x}]
    # But A∪{x} must be a (t-1)+1 = t-subset... wait, A has t-1 elements, x not in A,
    # so A∪{x} has t elements, which IS a t-subset. So ext_map isn't the right structure.
    # Let me rethink: I want, given (t-1)-tuple A (sorted), and point x not in A,
    # the single t-subset A∪{x} (if it exists in our universe). Since all t-subsets of range(v) are in all_t,
    # A∪{x} is always in all_t if x < v and x not in A.
    # So gain for adding x to block B = count of uncovered t-subsets that contain x and (t-1) points from B.
    # For each (t-1)-subset A of B, the t-subset A∪{x} is unique. So gain = sum over A in C(B, t-1) of [A∪{x} is uncovered]
    # This is efficient if t is small.

    def greedy_cover(rng, time_limit):
        uncovered = [True] * n_t
        blocks = []
        remaining = n_t
        
        while remaining > 0 and time.time() < time_limit:
            # Pick the first uncovered t-subset as seed (or randomize slightly)
            # Find first uncovered
            idx = -1
            for i in range(n_t):
                if uncovered[i]:
                    idx = i
                    break
            if idx == -1:
                break
            
            # Start block with the t-subset's points
            block = set(all_t[idx])
            
            # Greedily add points to maximize coverage
            while len(block) < k and time.time() < time_limit:
                best_gain = -1
                best_x = None
                for x in range(v):
                    if x in block:
                        continue
                    # Compute gain
                    gain = 0
                    for A in itertools.combinations(sorted(block), t_minus_1):
                        # A is (t-1)-tuple from block
                        ts = tuple(sorted(A + (x,)))
                        if ts in t_index:
                            if uncovered[t_index[ts]]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None:
                        # Tie-breaking: random
                        if rng.random() < 0.5:
                            best_x = x
                if best_x is None:
                    break
                block.add(best_x)
            
            block = sorted(block)
            blocks.append(block)
            
            # Update uncovered
            for ts in itertools.combinations(block, t):
                i = t_index[ts]
                if uncovered[i]:
                    uncovered[i] = False
                    remaining -= 1
        
        return blocks

    def is_valid(blocks):
        covered = [False] * n_t
        for block in blocks:
            for ts in itertools.combinations(block, t):
                covered[t_index[ts]] = True
        return all(covered)

    def local_search(blocks, time_limit):
        """Try to remove or replace blocks to reduce count."""
        if not is_valid(blocks):
            return blocks
        
        best_blocks = blocks[:]
        n = len(best_blocks)
        
        # First: try removing redundant blocks
        i = 0
        while i < len(best_blocks) and time.time() < time_limit:
            # Check if removing block i still covers everything
            candidate = best_blocks[:i] + best_blocks[i+1:]
            if is_valid(candidate):
                best_blocks = candidate
                # Don't increment i, check next
            else:
                i += 1
        
        # Second: try replacing pairs of blocks with one better block
        # This is expensive; only do if time permits
        if time.time() < time_limit - 30:
            improved = True
            while improved and time.time() < time_limit - 5:
                improved = False
                for i in range(len(best_blocks)):
                    if time.time() > time_limit - 5:
                        break
                    for j in range(i+1, len(best_blocks)):
                        if time.time() > time_limit - 5:
                            break
                        # Try to find a single block that covers all t-subsets 
                        # currently only covered by blocks i or j
                        # This is complex; skip for now
                        pass
        
        return best_blocks

    # Main: run multiple randomized greedy, keep best
    best_result = None
    best_len = float('inf')
    
    # Run a few randomized greedy attempts
    for trial in range(5):
        if time.time() > deadline - 20:
            break
        rng = random.Random(trial * 1000 + 42)
        result = greedy_cover(rng, deadline - 15)
        if is_valid(result):
            if len(result) < best_len:
                best_len = len(result)
                best_result = result
    
    # Local search on best
    if best_result is not None and time.time() < deadline - 10:
        improved = local_search(best_result, deadline - 5)
        if is_valid(improved):
            best_result = improved

    if best_result is None:
        # Fallback: pure greedy
        rng = random.Random(0)
        best_result = greedy_cover(rng, deadline)

    return best_result
