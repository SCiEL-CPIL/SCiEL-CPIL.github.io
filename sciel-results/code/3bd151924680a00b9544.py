"""Rewritten covering design C(19,7,4) solver.

Strategy:
  1. Fast greedy construction using precomputed gain tables for speed.
  2. Local search / block replacement to try to reduce the number of blocks.
  3. Deadline guard to stay within the 150s wall-clock limit.

The key insight for speed: instead of recomputing gains by iterating over all
(t-1)-subsets of the current partial block, we precompute for each t-subset
which points "cover" it (i.e., which points, when added to a set containing
the other t-1 elements, would form a t-subset already in uncovered).
Actually, a better approach: for a partial block B (size < k), the gain of adding
point x is the number of t-subsets S in uncovered such that S \ {x} ⊆ B.
We can precompute a dict: for each (t-1)-tuple T (sorted), and for each point x not in T,
count how many uncovered t-subsets contain T ∪ {x}. But that's expensive to maintain.

Alternative: maintain a count array. For each uncovered t-subset S, for each x in S,
increment count[T][x] where T = S \ {x} (as a frozenset or tuple). Then gain(x) for
partial block B is sum over T in combinations(B, t-1) of count[T][x] where x not in B.

But updating this when we remove t-subsets from uncovered is also expensive.

Given v=19, k=7, t=4: C(19,4)=3876 t-subsets. C(19,3)=969 triples.
For each triple T and each point x not in T, we need count of uncovered 4-sets containing T∪{x}.
That's 969 * 16 ≈ 15504 entries. Updating when we remove a 4-set S: for each x in S,
decrement count[S\{x}][x]. That's 4 updates per removed 4-set. With ~3876 initial 4-sets,
and each block removes up to C(7,4)=35 4-sets, this is manageable.

Let me implement this.
"""

import itertools
import time


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    uncovered = set(all_t_subsets)

    # Precompute: for each (t-1)-tuple T (as sorted tuple), a dict mapping point x -> count
    # of uncovered t-subsets S such that T ∪ {x} = S (i.e., x not in T and S contains T)
    # Actually: S = T ∪ {x} where x not in T. So for each T, for each x not in T,
    # count[T][x] = 1 if T ∪ {x} (sorted) is in uncovered, else 0.
    # When we remove a t-subset S from uncovered, for each x in S: T = S \ {x}, decrement count[T][x].

    # Build initial count structure
    # count is a dict: key = tuple of t-1 sorted points, value = dict {point: count}
    count = {}
    for S in all_t_subsets:
        for x in S:
            T = tuple(p for p in S if p != x)
            if T not in count:
                count[T] = {}
            count[T][x] = count[T].get(x, 0) + 1

    blocks = []

    def gain(x, block):
        """Number of uncovered t-subsets that would be covered by adding x to block."""
        total = 0
        block_set = set(block)
        for T in itertools.combinations(block, t - 1):
            T_tuple = tuple(sorted(T))
            if T_tuple in count:
                total += count[T_tuple].get(x, 0)
        return total

    while uncovered:
        if time.time() > deadline:
            break

        # Start with the lexicographically smallest uncovered t-subset
        first_uncovered = min(uncovered)
        block = list(first_uncovered)

        while len(block) < k:
            if time.time() > deadline:
                break

            block_set = set(block)
            best_x = None
            best_gain = -1
            for x in points:
                if x in block_set:
                    continue
                g = gain(x, block)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        if len(block) < k:
            break

        block.sort()
        block_tuple = tuple(block)

        # Check if this block is already in blocks
        if block_tuple in [tuple(b) for b in blocks]:
            # Shouldn't happen, but skip
            continue

        blocks.append(block)

        # Remove all t-subsets of this block from uncovered, and update count
        for S in itertools.combinations(block, t):
            if S in uncovered:
                uncovered.discard(S)
                for x in S:
                    T = tuple(p for p in S if p != x)
                    if T in count:
                        count[T][x] -= 1

    # Local search: try to remove blocks that are redundant
    if time.time() < deadline - 5:
        # Try to remove blocks one by one if they don't break coverage
        i = 0
        while i < len(blocks) and time.time() < deadline - 5:
            # Check if block i is needed: is there a t-subset only covered by this block?
            block_i = blocks[i]
            # For each t-subset in block_i, check if it's in any other block
            needed = False
            for S in itertools.combinations(block_i, t):
                covered_by_other = False
                for j, block_j in enumerate(blocks):
                    if j == i:
                        continue
                    if all(s in set(block_j) for s in S):
                        covered_by_other = True
                        break
                if not covered_by_other:
                    needed = True
                    break
            if not needed:
                blocks.pop(i)
            else:
                i += 1

    return blocks
