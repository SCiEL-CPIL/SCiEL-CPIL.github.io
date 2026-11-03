"""Improved covering design search for C(19,8,5)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math
from collections import defaultdict

# Precompute all 5-subsets of 19 points
V, K, T = 19, 8, 5
ALL_TSUBS = list(itertools.combinations(range(V), T))
N_TSUBS = len(ALL_TSUBS)
TSUB_INDEX = {ts: i for i, ts in enumerate(ALL_TSUBS)}

# Precompute for each point, the set of 4-subsets it forms (for gain computation)
# For a point x and a (t-1)-subset s, the t-subset is s + {x}
# We precompute a mapping: for each 4-subset s, the list of points x such that s+{x} is a 5-subset
# Actually, for each 4-subset s (of 14 points), the 5 points NOT in s can complete it to a 5-subset.
# But we need: for a given partial block B (size < k), and candidate point x not in B,
# gain(x) = number of uncovered 5-subsets of the form (4-subset of B) + {x}

# Precompute: for each 4-subset s, the tuple of 5 points that can complete it (i.e., points not in s)
# Then for a partial block B and candidate x, gain = sum over 4-subsets s of B of [1 if x completes s and s+{x} is uncovered]

# More efficient: precompute for each 4-subset s, a bitmask of which points complete it.
# But with 19 points, we can use a different approach.

# Precompute: for each 4-subset s (as a tuple), the list of points x not in s.
# There are C(19,4) = 3876 four-subsets.
FOUR_SUBS = list(itertools.combinations(range(V), 4))
FOUR_SUB_INDEX = {s: i for i, s in enumerate(FOUR_SUBS)}

# For each 4-subset s, precompute the 5-subsets that contain s (i.e., s + {x} for x not in s)
# There are 15 such 5-subsets per 4-subset.
FOUR_TO_FIVE = []
for s in FOUR_SUBS:
    s_set = set(s)
    five_list = []
    for x in range(V):
        if x not in s_set:
            five_list.append(TSUB_INDEX[tuple(sorted(s + (x,)))])
    FOUR_TO_FIVE.append(five_list)

# For fast gain computation:
# Given a partial block B (list of points, len < K) and candidate x not in B:
# gain(x) = number of uncovered 5-subsets that are of the form (4-subset of B) + {x}
# We can precompute for each 4-subset of B, the 5-subsets it forms with x.

# Better approach: for each 4-subset s, we have FOUR_TO_FIVE[i] which is a list of 5-subset indices.
# For candidate x, we check which of those 5-subsets contain x.
# Precompute: for each 4-subset index i, and each point x, whether x is in the completion.
# Actually, let's precompute: FOUR_TO_FIVE_POINT[i][x] = list of 5-subset indices that contain both s and x.
# This is equivalent to: for 4-subset s, the 5-subsets containing s and x is exactly one if x not in s, else 0.
# Wait, no: a 5-subset containing s and x is determined by s and x (since |s|=4, |s∪{x}|=5).
# So for each 4-subset s and point x not in s, there is exactly one 5-subset: s ∪ {x}.
# So FOUR_TO_FIVE[i] already lists all 15 such 5-subsets.
# For candidate x, gain contribution from 4-subset s is 1 if (s∪{x}) is uncovered.

# Let's precompute a 2D structure: for each 4-subset index i, a list of (x, five_sub_index) pairs.
# Then for a given partial block B, we iterate over all 4-subsets of B, and for each, check if the corresponding 5-subset with x is uncovered.

# For efficiency, let's precompute for each 4-subset s, a dictionary mapping point x -> 5-subset index.
FOUR_TO_FIVE_MAP = []
for i, s in enumerate(FOUR_SUBS):
    s_set = set(s)
    d = {}
    for x in range(V):
        if x not in s_set:
            five = tuple(sorted(s + (x,)))
            d[x] = TSUB_INDEX[five]
    FOUR_TO_FIVE_MAP.append(d)

# Now, for a partial block B (list of points) and candidate x:
# gain(x) = sum over all 4-subsets s of B: 1 if (s ∪ {x}) is uncovered
# We can compute this by iterating over 4-subsets of B.


def greedy_cover(deadline=None):
    """Run greedy set cover with efficient gain computation."""
    uncovered = [True] * N_TSUBS
    n_uncovered = N_TSUBS
    blocks = []
    
    while n_uncovered > 0:
        if deadline and time.time() > deadline:
            break
        
        # Find the uncovered 5-subset with the smallest index (for determinism)
        # Actually, pick any uncovered one; use the first for simplicity
        min_uncovered = -1
        for i in range(N_TSUBS):
            if uncovered[i]:
                min_uncovered = i
                break
        if min_uncovered < 0:
            break
        
        block = list(ALL_TSUBS[min_uncovered])
        
        while len(block) < K:
            if deadline and time.time() > deadline:
                break
            
            block_set = set(block)
            # Compute gain for each candidate point
            best_x = -1
            best_gain = -1
            
            # Precompute 4-subsets of current block
            block_four_subs = list(itertools.combinations(block, 4))
            
            for x in range(V):
                if x in block_set:
                    continue
                gain = 0
                for fs in block_four_subs:
                    fs_idx = FOUR_SUB_INDEX[fs]
                    fs_map = FOUR_TO_FIVE_MAP[fs_idx]
                    if x in fs_map:
                        five_idx = fs_map[x]
                        if uncovered[five_idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                elif gain == best_gain and x < best_x:
                    best_x = x
            
            if best_x < 0:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered
        block_set = set(block)
        for ts in itertools.combinations(block, T):
            idx = TSUB_INDEX[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                n_uncovered -= 1
    
    return blocks


def local_search(blocks, uncovered, n_uncovered, deadline=None, max_iter=50):
    """Try to remove or replace blocks to reduce count."""
    blocks = list(blocks)
    
    # Build coverage count for each 5-subset
    cover_count = [0] * N_TSUBS
    for block in blocks:
        block_set = set(block)
        for ts in itertools.combinations(block, T):
            idx = TSUB_INDEX[ts]
            cover_count[idx] += 1
    
    improved = True
    while improved and n_uncovered == 0:
        improved = False
        if deadline and time.time() > deadline:
            break
        
        # Try to remove a block that has no uniquely covered 5-subset
        for i in range(len(blocks)):
            if deadline and time.time() > deadline:
                break
            
            block = blocks[i]
            # Check if any 5-subset is only covered by this block
            has_unique = False
            for ts in itertools.combinations(block, T):
                idx = TSUB_INDEX[ts]
                if cover_count[idx] == 1:
                    has_unique = True
                    break
            
            if not has_unique:
                # Remove this block
                for ts in itertools.combinations(block, T):
                    idx = TSUB_INDEX[ts]
                    cover_count[idx] -= 1
                
                blocks.pop(i)
                improved = True
                break
    
    return blocks, cover_count


def construct(v, k, t):
    """Construct a covering design."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Run greedy
    blocks = greedy_cover(deadline)
    
    if deadline and time.time() > deadline:
        # Return what we have, but need to verify it's a valid covering
        # If we broke early, we might not have a complete covering
        # Let's check
        covered = [False] * N_TSUBS
        for block in blocks:
            for ts in itertools.combinations(block, T):
                covered[TSUB_INDEX[ts]] = True
        if all(covered):
            return [list(b) for b in blocks]
        else:
            # Complete with greedy
            blocks = greedy_cover(deadline)
            return [list(b) for b in blocks]
    
    # Local search to remove redundant blocks
    # First, build coverage
    cover_count = [0] * N_TSUBS
    for block in blocks:
        for ts in itertools.combinations(block, T):
            cover_count[TSUB_INDEX[ts]] += 1
    
    n_uncovered = sum(1 for c in cover_count if c == 0)
    
    if n_uncovered == 0:
        blocks, cover_count = local_search(blocks, cover_count, n_uncovered, deadline)
    
    # Verify
    final_covered = [False] * N_TSUBS
    for block in blocks:
        for ts in itertools.combinations(block, T):
            final_covered[TSUB_INDEX[ts]] = True
    
    if not all(final_covered):
        # Fallback: run greedy again
        blocks = greedy_cover(deadline)
        return [list(b) for b in blocks]
    
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
