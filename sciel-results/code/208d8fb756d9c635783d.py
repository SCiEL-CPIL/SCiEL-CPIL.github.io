"""
Greedy covering with precomputed incidence for C(v,k,t).
Key idea: maintain a count array over all t-subsets (indexed by combinatorial
numbering) so that adding a point to a partial block lets us count newly
covered t-subsets in O(C(k-1, t-1)) time instead of scanning the whole set.
"""
import time
import itertools
from math import comb

# Deadline: 150s wall-clock, leave a safety margin
DEADLINE = 145.0


def _combinatorial_index(n, r, combo):
    """Compute the lexicographic index of a combination among C(n, r)."""
    # Use the standard combinadic formula
    idx = 0
    for i, x in enumerate(combo):
        idx += comb(x, r - i)
    return idx


def construct(v, k, t):
    start_time = time.time()
    
    total_t_subsets = comb(v, t)
    
    # Count how many times each t-subset is covered
    # Use a dict mapping frozenset/tuple -> count, or an array if feasible
    # For v=19, t=4: C(19,4)=3876, small enough for an array
    # For general v, k, t: use a dict if total is too large
    
    if total_t_subsets <= 200000:
        # Use array for speed
        uncovered_count = [1] * total_t_subsets  # each t-subset needs 1 cover
        remaining = total_t_subsets
        
        # Precompute: for each point x and each (t-1)-subset of remaining points,
        # the index of the t-subset. Actually, we compute on the fly.
        
        # To speed up: precompute for each point x, and for each subset S of size t-1
        # not containing x, the index of S ∪ {x}.
        # But that's C(v, t-1) * v entries which could be large.
        # Alternative: precompute a list of (index, tuple) for all t-subsets.
        
        # Build mapping from tuple to index
        tuple_to_idx = {}
        all_t_subs = []
        for i, c in enumerate(itertools.combinations(range(v), t)):
            tuple_to_idx[c] = i
            all_t_subs.append(c)
        
        blocks = []
        
        while remaining > 0:
            if time.time() - start_time > DEADLINE:
                break
            
            # Find the least-covered t-subset (all have count 1 initially, so any uncovered)
            # Pick the first uncovered one
            target = None
            for i in range(total_t_subsets):
                if uncovered_count[i] > 0:
                    target = i
                    break
            if target is None:
                break
            
            target_combo = all_t_subs[target]
            block = list(target_combo)
            block_set = set(block)
            
            # Greedily add points to maximize new coverage
            while len(block) < k:
                if time.time() - start_time > DEADLINE:
                    break
                
                best_x = -1
                best_gain = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    # A t-subset S is covered by block ∪ {x} if S ⊆ (block ∪ {x}) and x ∈ S
                    # i.e., S = {x} ∪ T where T is a (t-1)-subset of block
                    gain = 0
                    if len(block) >= t - 1:
                        for T in itertools.combinations(block, t - 1):
                            # S = sorted(T + (x,))
                            S = tuple(sorted(T + (x,)))
                            if S in tuple_to_idx:
                                idx = tuple_to_idx[S]
                                gain += uncovered_count[idx]
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x < 0:
                    break
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered counts for all t-subsets within this block
            for S in itertools.combinations(block, t):
                idx = tuple_to_idx[S]
                if uncovered_count[idx] > 0:
                    uncovered_count[idx] -= 1
                    remaining -= 1
        
        return blocks
    else:
        # Fallback: use dict-based approach for larger instances
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            if time.time() - start_time > DEADLINE:
                break
            
            block = list(min(uncovered))
            block_set = set(block)
            
            while len(block) < k:
                if time.time() - start_time > DEADLINE:
                    break
                
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    if len(block) >= t - 1:
                        for T in itertools.combinations(block, t - 1):
                            S = tuple(sorted(T + (x,)))
                            if S in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x < 0:
                    break
                block.append(best_x)
                block_set.add(best_x)
            
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        
        return blocks
