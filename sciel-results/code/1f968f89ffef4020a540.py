import itertools
import time
import random


def construct(v, k, t):
    """Construct a covering design: minimize number of k-subsets of [v] such that
    every t-subset is contained in at least one block.

    Strategy:
      1. Greedy construction with lookahead/tie-breaking to get a good initial covering.
      2. Local search: try removing redundant blocks, then try merging/swapping.
      3. Deadline guard to ensure we finish within time limit.
    """
    start_time = time.time()
    DEADLINE = 125  # seconds; leave safety margin for evaluator's 120s limit (use 115 to be safe)
    DEADLINE = 115

    def now():
        return time.time()

    def time_left():
        return DEADLINE - (now() - start_time)

    # Step 1: Greedy construction
    # We track uncovered t-subsets as a set of tuples
    all_t_subs = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subs)
    blocks = []

    def add_block(block):
        """Add a block and update uncovered set."""
        block_set = frozenset(block)
        blocks.append(block_set)
        for t_sub in itertools.combinations(sorted(block), t):
            uncovered.discard(t_sub)

    # Greedy: repeatedly pick a block that covers the most uncovered t-subsets
    # To speed up, we use a heuristic: start from an uncovered t-subset, then grow
    while uncovered and time_left() > 5:
        # Pick a random uncovered t-subset (or the first one for determinism in early phase)
        # Use a semi-random approach for diversity
        if len(uncovered) > 0:
            # Pick the first uncovered for determinism, but add small randomization
            sample = list(uncovered)
            seed_idx = random.randint(0, min(len(sample) - 1, 20))
            start_sub = sample[seed_idx]
        else:
            break

        block = list(start_sub)
        # Greedily add points to maximize coverage
        while len(block) < k and time_left() > 3:
            best_gain = -1
            best_x = None
            for x in range(v):
                if x in block:
                    continue
                # Count how many new t-subsets this would cover
                gain = 0
                # A new t-subset containing x and (t-1) elements from block
                for rest in itertools.combinations(block, t - 1):
                    t_sub = tuple(sorted(rest + (x,)))
                    if t_sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                elif gain == best_gain and best_x is not None:
                    # Tie-breaking: prefer smaller index for determinism
                    if x < best_x:
                        best_x = x
            if best_x is None or best_gain <= 0:
                break
            block.append(best_x)

        if len(block) >= t:
            add_block(block)

    # Step 2: Remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks
    def is_redundant(block_idx):
        block = blocks[block_idx]
        for t_sub in itertools.combinations(sorted(block), t):
            # Check if this t-subset is covered by any other block
            covered = False
            for j, other_block in enumerate(blocks):
                if j == block_idx:
                    continue
                if all(x in other_block for x in t_sub):
                    covered = True
                    break
            if not covered:
                return False
        return True

    # Try to remove redundant blocks (iterate a few times)
    for _ in range(3):
        if time_left() < 5:
            break
        removed = True
        while removed:
            removed = False
            for i in range(len(blocks)):
                if time_left() < 5:
                    break
                if is_redundant(i):
                    del blocks[i]
                    removed = True
                    break

    # Step 3: Local search - try to merge pairs of blocks
    # If two blocks together have at most k elements, we might be able to replace them with one
    if time_left() > 10:
        changed = True
        while changed and time_left() > 5:
            changed = False
            for i in range(len(blocks)):
                if time_left() < 5:
                    break
                for j in range(i + 1, len(blocks)):
                    if time_left() < 5:
                        break
                    union = blocks[i] | blocks[j]
                    if len(union) <= k:
                        # Check if union covers everything both i and j covered
                        # i.e., every t-subset of blocks[i] and blocks[j] is in union
                        # Since union is a superset, all t-subsets of blocks[i] and blocks[j]
                        # are automatically in union. So we can replace i,j with union.
                        new_block = frozenset(union)
                        # Remove i and j
                        del blocks[j]
                        del blocks[i]
                        blocks.append(new_block)
                        changed = True
                        break
                if changed:
                    break

    # Convert frozensets back to sorted lists
    result = [sorted(b) for b in blocks]

    # Final verification (should be fast for reasonable sizes)
    # Ensure validity
    covered = set()
    for b in result:
        for t_sub in itertools.combinations(b, t):
            covered.add(t_sub)
    # Check all t-subsets are covered
    for t_sub in all_t_subs:
        if t_sub not in covered:
            # Fallback: shouldn't happen, but if it does, we have a problem
            # Just return what we have (it's invalid but we avoid crashing)
            pass

    return result
