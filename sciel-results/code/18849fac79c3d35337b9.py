"""Optimized covering design for C(v,k,t)."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
from collections import defaultdict

_DEADLINE = 140.0


def _build_masks(v, k, t):
    """Precompute for each point x, the mask of t-subsets that contain x.
    Returns (all_t, masks_by_point) where all_t is array of all t-subsets,
    masks_by_point[x] is a boolean array of length C(v,t) marking t-subsets containing x.
    """
    all_t = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    n = all_t.shape[0]
    # For each t-subset, find which points it contains
    # masks_by_point[x]: boolean array, True if x is in t-subset
    masks_by_point = [np.zeros(n, dtype=bool) for _ in range(v)]
    for i in range(n):
        for x in all_t[i]:
            masks_by_point[x][i] = True
    return all_t, masks_by_point


def _greedy_fast(v, k, t, deadline, rng):
    """Fast greedy: build blocks on uncovered t-subsets using precomputed masks."""
    all_t, masks_by_point = _build_masks(v, k, t)
    n = all_t.shape[0]
    uncovered = np.ones(n, dtype=bool)
    blocks = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        idx = int(np.argmax(uncovered))
        seed = list(all_t[idx])
        block = list(seed)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            # For each candidate x not in block, compute gain
            # gain(x) = number of uncovered t-subsets that contain x AND are "reachable"
            # A t-subset is reachable if it contains some (t-1)-subset of block
            # More precisely: gain(x) = |uncovered ∩ {S : S contains x and S contains some (t-1)-subset of block}|
            
            # Compute the union of masks for all (t-1)-subsets of block
            # For each (t-1)-subset R of block, a t-subset S is "covered by adding x" 
            # if S contains R and S contains x, i.e., S contains R ∪ {x}
            # So gain(x) = |uncovered ∩ masks_for_R∪{x}| where we OR over all R in C(block, t-1)
            
            # Precompute: for each possible (t-1)-subset, we need the mask of t-subsets containing it
            # But that's too many. Instead:
            # gain(x) = |uncovered ∩ (OR over R in C(block, t-1) of {S : x in S and R ⊆ S})|
            # {S : x in S and R ⊆ S} = {S : (R ∪ {x}) ⊆ S}
            # So we need, for each (t-1)-subset R of block, the mask of t-subsets containing R∪{x}
            
            # Alternative: for each x, gain = |uncovered ∩ masks_by_point[x] ∩ (OR of masks for (t-1)-subsets)|
            # But that's not quite right either.
            
            # Let me think differently. A t-subset S is covered by block∪{x} iff S ⊆ block∪{x}.
            # The NEW t-subsets covered by adding x to block are those S such that:
            #   S ⊆ block∪{x}, S ⊄ block, and S is uncovered.
            # S ⊆ block∪{x} and S ⊄ block means x ∈ S and S\{x} ⊆ block.
            # Since |S| = t and x ∈ S, |S\{x}| = t-1, so S\{x} is a (t-1)-subset of block.
            # So: S = R ∪ {x} where R is a (t-1)-subset of block.
            
            # So gain(x) = |{R ∪ {x} : R ∈ C(block, t-1), R ∪ {x} is uncovered}|
            # = |uncovered ∩ {R ∪ {x} for R in C(block, t-1)}|
            
            # We can compute this as: for each R in C(block, t-1), check if R∪{x} is uncovered.
            # But we need to do this efficiently.
            
            # Precompute: for each (t-1)-subset R, find the index of R∪{x} in all_t for each x not in R.
            # Actually, let's just iterate.
            
            best_x, best_g = None, -1
            for x in range(v):
                if x in block:
                    continue
                g = 0
                for R in itertools.combinations(block, t - 1):
                    # R ∪ {x} is a t-subset
                    S = tuple(sorted(R + (x,)))
                    # Find if S is in all_t and uncovered
                    # We need to find the index of S in all_t
                    # This is slow. Let's use a dict.
                if g > best_g:
                    best_g, best_x = g, x
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        # mark covered
        for c in itertools.combinations(block, t):
            # find index of c in all_t
            mask = np.all(all_t == np.array(list(c)), axis=1)
            uncovered &= ~mask
    
    return blocks


def _greedy_v2(v, k, t, deadline, rng):
    """Greedy with precomputed lookup for t-subset indices."""
    all_t = np.array(list(itertools.combinations(range(v), t)), dtype=np.int64)
    n = all_t.shape[0]
    
    # Build a dict from t-subset (as tuple) to index
    tset_to_idx = {}
    for i in range(n):
        tset_to_idx[tuple(all_t[i])] = i
    
    uncovered = np.ones(n, dtype=bool)
    blocks = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        idx = int(np.argmax(uncovered))
        seed = list(all_t[idx])
        block = list(seed)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            best_x, best_g = None, -1
            for x in range(v):
                if x in block:
                    continue
                g = 0
                for R in itertools.combinations(block, t - 1):
                    S = tuple(sorted(R + (x,)))
                    if S in tset_to_idx:
                        si = tset_to_idx[S]
                        if uncovered[si]:
                            g += 1
                if g > best_g:
                    best_g, best_x = g, x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        # mark covered
        for c in itertools.combinations(block, t):
            ct = tuple(c)
            if ct in tset_to_idx:
                uncovered[tset_to_idx[ct]] = False
    
    return blocks


def _is_valid(blocks, v, t):
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, t):
            covered.add(c)
    total = len(list(itertools.combinations(range(v), t)))
    return len(covered) == total


def _remove_redundant(blocks, v, t, deadline):
    """Remove blocks that don't uniquely cover any t-subset."""
    if not blocks:
        return blocks
    cover_by = defaultdict(set)
    for i, b in enumerate(blocks):
        for c in itertools.combinations(b, t):
            cover_by[c].add(i)
    
    removed = [False] * len(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if removed[i]:
                continue
            if time.time() > deadline:
                break
            unique = False
            for c in itertools.combinations(blocks[i], t):
                if len(cover_by[c]) == 1:
                    unique = True
                    break
            if not unique:
                removed[i] = True
                for c in itertools.combinations(blocks[i], t):
                    cover_by[c].discard(i)
                changed = True
    
    return [blocks[j] for j in range(len(blocks)) if not removed[j]]


def construct(v, k, t):
    start = time.time()
    deadline = start + _DEADLINE
    
    # Use the v2 greedy which is faster
    blocks = _greedy_v2(v, k, t, deadline, None)
    
    # Remove redundant blocks
    blocks = _remove_redundant(blocks, v, t, deadline)
    
    # Ensure validity
    if not _is_valid(blocks, v, t):
        # Fallback: rebuild without removing redundant
        blocks = _greedy_v2(v, k, t, deadline, None)
    
    return blocks
# EVOLVE-BLOCK-END
