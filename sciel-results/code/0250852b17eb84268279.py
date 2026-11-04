"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all t-subsets and their index
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # uncovered: boolean numpy array
    uncovered = np.ones(n_tsubs, dtype=bool)
    n_uncovered = n_tsubs

    blocks = []

    # Precompute, for each point x, the list of t-subsets that contain x
    # Actually, we need to compute for a partial block B, the t-subsets covered by adding x
    # A t-subset S is covered by B ∪ {x} if S ⊆ B ∪ {x}
    # This means S ⊆ B and x not in S, OR S ⊆ B ∪ {x} and x in S

    # Let's think differently. For a partial block B (|B| = j < k), 
    # adding point x covers t-subsets S such that S ⊆ B ∪ {x} and S not ⊆ B
    # i.e., S contains x and S \ {x} ⊆ B
    # The number of such S is C(j, t-1) if t-1 <= j, else 0.
    # But we want to count UNCOVERED t-subsets.

    # For efficiency, let's precompute for each t-subset S, which points it contains.
    # Then, for a partial block B and candidate x, S is newly covered if x in S and S \ {x} ⊆ B.

    # Alternative approach: maintain for each point pair (or small subset) the count of uncovered t-subsets
    # Actually, let's use a different representation.

    # For each point x, maintain a counter: for each (t-1)-subset A of the other points,
    # how many uncovered t-subsets contain x and are of the form A ∪ {x}?
    # That's just 1 if A ∪ {x} is uncovered, else 0.
    # But A can be any (t-1)-subset of the v-1 other points. There are C(v-1, t-1) such.
    # For v=19, t=5: C(18,4) = 3060. That's manageable.

    # Let's precompute all (t-1)-subsets of {0,...,v-1}
    all_tm1subs = list(itertools.combinations(range(v), t - 1))
    n_tm1 = len(all_tm1subs)
    tm1_index = {a: i for i, a in enumerate(all_tm1subs)}

    # uncovered_tm1: for each (t-1)-subset A, for each point x not in A,
    # whether A ∪ {x} is uncovered.
    # Shape: (n_tm1, v) boolean array
    uncovered_tm1 = np.zeros((n_tm1, v), dtype=bool)
    for a in all_tm1subs:
        ai = tm1_index[a]
        for x in range(v):
            if x not in a:
                ts = tuple(sorted(a + (x,)))
                uncovered_tm1[ai, x] = True

    # When we cover a t-subset S, we need to update uncovered_tm1 for all (t-1)-subsets of S
    # S has C(t, t-1) = t such (t-1)-subsets.
    # For each (t-1)-subset A ⊂ S, the point x = S \ A is such that A ∪ {x} = S.
    # So we set uncovered_tm1[tm1_index[A], x] = False.

    def gain(x, block):
        """Count uncovered t-subsets newly covered by adding x to block."""
        # S is newly covered if x in S and S \ {x} ⊆ block
        # S \ {x} is a (t-1)-subset of block
        # So we need to count (t-1)-subsets A of block such that uncovered_tm1[tm1_index[A], x] is True
        j = len(block)
        if j < t - 1:
            return 0
        count = 0
        for A in itertools.combinations(block, t - 1):
            ai = tm1_index[A]
            if uncovered_tm1[ai, x]:
                count += 1
        return count

    while n_uncovered > 0:
        if time.time() > deadline:
            break

        # Pick the first uncovered t-subset
        first_idx = np.argmax(uncovered)
        first_ts = all_tsubs[first_idx]
        block = list(first_ts)
        # Actually, start with a random subset of the first uncovered t-subset? 
        # Or just start with one point from it.
        # Let's start with the first point of the first uncovered t-subset
        block = [first_ts[0]]

        # Greedily add points
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                g = gain(x, block)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)

        block.sort()
        # Check if this block is already in blocks (for duplicate checking)
        block_tuple = tuple(block)
        if block_tuple not in set(blocks):
            blocks.append(block)

        # Update uncovered: remove all t-subsets contained in block
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                n_uncovered -= 1
                # Update uncovered_tm1
                # ts has t points; for each point x in ts, A = ts \ {x}
                for x in ts:
                    A = tuple(p for p in ts if p != x)
                    ai = tm1_index[A]
                    uncovered_tm1[ai, x] = False

    return blocks
# EVOLVE-BLOCK-END
