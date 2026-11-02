"""Covering design C(19,8,5): fast greedy with deadline guard."""
import time
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin before 120s kill

    # Enumerate all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    # Map each t-subset (tuple) to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # uncovered: numpy bool array, True if still uncovered
    uncovered = np.ones(n_tsubs, dtype=bool)

    blocks = []
    block_cover = []  # list of boolean arrays per block (which tsubs it covers)

    def time_ok():
        return time.time() < DEADLINE

    while uncovered.any() and time_ok():
        # Start a block with the first uncovered t-subset's first element
        first_uncovered_idx = int(np.argmax(uncovered))
        seed_block = list(all_tsubs[first_uncovered_idx])
        block = [seed_block[0]]

        # Precompute: for the current block, which (t-1)-subsets are "active"
        # i.e., the (t-1)-subsets of the block that still have at least one
        # uncovered completion (adding some x not in block).
        # For each candidate x, gain = number of uncovered tsubs that are
        # (a (t-1)-subset of block) + x.

        # We'll maintain: for each point x, a count of how many uncovered tsubs
        # would be covered if we add x to the current block.
        # A tsub is covered by adding x iff:
        #   tsub contains x, and tsub \ {x} is a (t-1)-subset of block.

        # Efficient approach: for each (t-1)-subset S of block, the tsubs that
        # contain S and one more point x are: for each x not in S, the tsub S ∪ {x}.
        # We only count those that are currently uncovered.

        # Precompute for each (t-1)-subset of block the list of x values
        # such that S ∪ {x} is a valid t-subset (i.e., x not in S).
        # Then gain(x) = sum over S in C(block, t-1) of [uncovered[idx(S∪{x})] and x not in S]

        # For speed, precompute a list of (S_tuple, [x options]) but that's large.
        # Alternative: iterate over uncovered tsubs and for each, check if
        # tsub \ {x} ⊆ block for some x not in block. That's O(n_uncovered * t) per gain call.

        # Better: vectorized. For each candidate x, compute:
        #   count of uncovered tsubs that contain x and whose other t-1 elements are all in block.
        # We can precompute a bitmask or use numpy.

        # Let's use a different strategy: maintain an array `gain_counts` of length v.
        # For the current block B, for each (t-1)-subset S of B:
        #   for each x not in S (and x not in B... actually x can be anything not in B for the block):
        #     ts = S ∪ {x}; if ts is a valid t-subset and uncovered[ts]: gain_counts[x] += 1
        # But we only want to add x not in B.

        # Since block grows from 1 to k, and t=5, the number of (t-1)-subsets of block is C(|B|,4).
        # When |B|=1: C(1,4)=0, so gain is 0 for all? No wait: t-1=4, C(1,4)=0.
        # That means when block has 1 element, no (t-1)-subset of block exists, so gain=0 for all.
        # That's a problem. The seed code handles this by starting block with the full first uncovered t-subset.

        # Let me re-read the seed: block = list(min(uncovered)), so block starts with t elements!
        # Then it adds k-t more elements.

        # Let me restart the block construction:
        # Start block with the first uncovered t-subset (t elements).
        # Then add k-t more elements, each chosen to maximize gain.

        # When block has b elements (b >= t), the (t-1)-subsets of block: C(b, t-1).
        # For each such S, and each x not in S (x can be in or not in block, but we only consider x not in block):
        #   ts = S ∪ {x}; if valid and uncovered: gain[x] += 1

        # Let me implement this.
        block = list(all_tsubs[first_uncovered_idx])  # t elements
        block_set = set(block)

        while len(block) < k and time_ok():
            # Compute gain for each candidate x not in block
            gain = np.zeros(v, dtype=np.int32)
            # For each (t-1)-subset S of block:
            for S in itertools.combinations(block, t - 1):
                S_set = set(S)
                # For each x not in S (and we'll filter x not in block later):
                # Actually, we want x such that S ∪ {x} is a t-subset. x can be anything not in S.
                # But we only care about x not in block (since x in block would mean ts ⊆ block, already covered).
                # Let's just compute for all x not in S, then zero out x in block.
                for x in range(v):
                    if x in S_set:
                        continue
                    ts = tuple(sorted(S + (x,)))
                    idx = tsub_to_idx.get(ts, -1)
                    if idx >= 0 and uncovered[idx]:
                        gain[x] += 1
            # Zero out candidates already in block
            for x in block:
                gain[x] = 0
            # Pick the x with max gain; if all zero, pick any not in block
            candidates = [x for x in range(v) if x not in block_set]
            if not candidates:
                break
            best_x = max(candidates, key=lambda x: gain[x])
            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered: mark all t-subsets of this block as covered
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            uncovered[idx] = False

        # If we're running out of time, do a final check and return
        if not time_ok():
            break

    # Final: if we ran out of time, we might not have a valid covering.
    # Check if uncovered is empty. If not, we need to return something.
    # The task says: "Valid when every t-subset lies in at least one block."
    # If invalid, score is 0. So we must ensure validity.
    # If we didn't finish, we should try to complete quickly or return best effort.
    # But the problem is hard; let's just return what we have and hope it's valid.
    # Actually, let's check:
    if uncovered.any():
        # We didn't finish. Try to quickly cover remaining with a few more blocks.
        # Use a simple loop without gain optimization to be fast.
        while uncovered.any() and time.time() < DEADLINE - 5:
            idx = int(np.argmax(uncovered))
            ts = all_tsubs[idx]
            block = list(ts)
            block_set = set(block)
            while len(block) < k:
                # Pick a random-ish x not in block to extend
                for x in range(v):
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                        break
                else:
                    break
            block.sort()
            blocks.append(block)
            for ts2 in itertools.combinations(block, t):
                uncovered[tsub_to_idx[ts2]] = False

    return blocks
