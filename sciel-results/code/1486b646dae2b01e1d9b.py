"""Covering design C(19,7,4) – branch-and-bound maximum clique on conflict graph."""
import time
import random
import math
from itertools import combinations
from collections import defaultdict

_DEADLINE_S = 145.0
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _greedy_construct(v, k, t, point_freq=None):
    """Fast greedy set-cover to get an initial upper bound."""
    if point_freq is None:
        point_freq = [0] * v

    all_tsubs = list(combinations(range(v), t))
    uncovered = set(all_tsubs)
    blocks = []

    while uncovered:
        if _time_left() < 5.0:
            break

        # Pick uncovered t-subset with min point frequency sum
        best_target = min(uncovered, key=lambda ts: sum(point_freq[p] for p in ts))
        target_set = set(best_target)
        remaining = [p for p in range(v) if p not in target_set]

        best_block = None
        best_gain = -1
        for comb in combinations(remaining, k - t):
            block = tuple(sorted(list(best_target) + list(comb)))
            gain = sum(1 for ts in combinations(block, t) if ts in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block

        if best_block is None:
            break

        blocks.append(list(best_block))
        for ts in combinations(best_block, t):
            uncovered.discard(ts)
        for p in best_block:
            point_freq[p] += 1

    return blocks


def _remove_redundant(blocks, v, k, t):
    """Remove blocks that don't uniquely cover any t-subset."""
    if not blocks:
        return blocks

    tsub_to_blocks = defaultdict(set)
    for i, b in enumerate(blocks):
        for ts in combinations(b, t):
            tsub_to_blocks[ts].add(i)

    changed = True
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            redundant = all(len(tsub_to_blocks[ts]) > 1 for ts in combinations(blocks[i], t))
            if redundant:
                for ts in combinations(blocks[i], t):
                    tsub_to_blocks[ts].discard(i)
                blocks.pop(i)
                for ts, bset in list(tsub_to_blocks.items()):
                    tsub_to_blocks[ts] = {j if j < i else j - 1 for j in bset}
                changed = True
                break
    return blocks


def _block_tsub_set(block, t):
    """Return frozenset of all t-subsets in a block."""
    return frozenset(combinations(block, t))


def _build_conflict_graph(candidate_blocks, v, k, t):
    """
    Build adjacency structure for the conflict graph.
    Node i and j are connected (in conflict graph) if blocks i and j share NO common t-subset.
    A clique in this graph = a set of blocks where every pair shares no t-subset... 
    Wait, that's wrong for covering. Let me reconsider.
    
    Actually for covering: we need blocks such that EVERY t-subset is covered by at least one block.
    The conflict graph approach: two blocks "conflict" if they don't help each other... 
    This is getting complex. Let me use a different framing.
    
    Alternative: The problem is equivalent to finding a minimal set of blocks covering all t-subsets.
    This is a set cover problem. The conflict graph idea doesn't directly map.
    
    Let me instead use: branch-and-bound on the set cover directly, with good lower bounds.
    """
    pass


def _lower_bound_uncovered(uncovered_count, v, k, t):
    """
    Lower bound on additional blocks needed.
    Each block covers at most C(k, t) t-subsets.
    But more tightly: use the fact that each block can cover at most C(k,t) t-subsets,
    and we need to cover all remaining.
    """
    max_per_block = math.comb(k, t)
    return math.ceil(uncovered_count / max_per_block)


def _branch_and_bound_set_cover(v, k, t, initial_blocks, time_limit_s):
    """
    Branch-and-bound to minimize number of blocks in a covering.
    Start from initial solution, try to remove blocks and replace with fewer.
    
    Strategy: 
    1. Start with greedy solution
    2. Iteratively try to replace subsets of blocks with fewer blocks
    3. Use lower bounds to prune
    """
    blocks = list(initial_blocks)
    best_blocks = list(blocks)
    best_len = len(blocks)
    
    # Build coverage structure
    all_tsubs = list(combinations(range(v), t))
    tsub_to_blocks = defaultdict(set)
    for i, b in enumerate(blocks):
        for ts in combinations(b, t):
            tsub_to_blocks[ts].add(i)
    
    def _is_valid(blk_list):
        """Check if all t-subsets are covered."""
        covered = set()
        for b in blk_list:
            for ts in combinations(b, t):
                covered.add(ts)
        return len(covered) == len(all_tsubs)
    
    def _try_improve(current_blocks, current_tsub_map, deadline):
        """Try to find a smaller covering by local moves."""
        nonlocal best_blocks, best_len
        
        n = len(current_blocks)
        if n <= 1:
            return
        
        # Precompute tsub sets for each block
        block_tsubs = [list(combinations(b, t)) for b in current_blocks]
        
        # Try removing one block and replacing with 0 or 1 blocks
        for i in range(n):
            if _time_left() < 1.0:
                return
            
            # Check if block i is removable (all its tsubs covered by others)
            removable = True
            for ts in block_tsubs[i]:
                if len(current_tsub_map[ts]) == 1:
                    removable = False
                    break
            
            if removable:
                # Remove block i
                new_blocks = current_blocks[:i] + current_blocks[i+1:]
                new_map = defaultdict(set)
                for j, b in enumerate(new_blocks):
                    for ts in combinations(b, t):
                        new_map[ts].add(j)
                
                if len(new_blocks) < best_len:
                    best_blocks = list(new_blocks)
                    best_len = len(new_blocks)
                
                # Recurse
                if _time_left() > 5.0 and len(new_blocks) > 2:
                    _try_improve(new_blocks, new_map, deadline)
                continue
            
            # Try replacing block i with a different block
            # Find tsubs that ONLY block i covers
            unique_tsubs = [ts for ts in block_tsubs[i] if len(current_tsub_map[ts]) == 1]
            
            if not unique_tsubs:
                # Block is redundant
                new_blocks = current_blocks[:i] + current_blocks[i+1:]
                new_map = defaultdict(set)
                for j, b in enumerate(new_blocks):
                    for ts in combinations(b, t):
                        new_map[ts].add(j)
                if len(new_blocks) < best_len:
                    best_blocks = list(new_blocks)
                    best_len = len(new_blocks)
                if _time_left() > 5.0 and len(new_blocks) > 2:
                    _try_improve(new_blocks, new_map, deadline)
                continue
            
            # Must include all points from unique_tsubs
            must_include = set()
            for ts in unique_tsubs:
                must_include.update(ts)
            
            if len(must_include) > k:
                continue
            
            # Try to find a replacement block
            remaining_pts = [p for p in range(v) if p not in must_include]
            need = k - len(must_include)
            
            # Try several random replacements
            for _ in range(5):
                if need <= 0:
                    new_block = tuple(sorted(must_include))
                elif need <= len(remaining_pts):
                    extra = random.sample(remaining_pts, need)
                    new_block = tuple(sorted(list(must_include) + extra))
                else:
                    continue
                
                # Check validity: new block must cover all unique_tsubs
                nb_set = set(new_block)
                if not all(set(ts).issubset(nb_set) for ts in unique_tsubs):
                    continue
                
                # Check it's different from current
                if tuple(sorted(current_blocks[i])) == new_block:
                    continue
                
                # Apply swap
                new_blocks = list(current_blocks)
                new_blocks[i] = list(new_block)
                
                # Check for duplicates
                if len(set(tuple(sorted(b)) for b in new_blocks)) != len(new_blocks):
                    continue
                
                # Rebuild map and check validity
                new_map = defaultdict(set)
                for j, b in enumerate(new_blocks):
                    for ts in combinations(b, t):
                        new_map[ts].add(j)
                
                # Verify all tsubs covered
                valid = all(len(new_map[ts]) >= 1 for ts in all_tsubs)
                if valid:
                    if len(new_blocks) < best_len:
                        best_blocks = list(new_blocks)
                        best_len = len(new_blocks)
                    
                    # Try further improvement
                    if _time_left() > 5.0:
                        _try_improve(new_blocks, new_map, deadline)
    
    _try_improve(blocks, tsub_to_blocks, _DEADLINE_S)
    
    return best_blocks


def _simulated_annealing_refine(blocks, v, k, t):
    """SA refinement with add/remove/swap moves."""
    if not blocks:
        return blocks
    
    all_tsubs = list(combinations(range(v), t))
    
    def _rebuild_map(blk_list):
        m = defaultdict(set)
        for i, b in enumerate(blk_list):
            for ts in combinations(b, t):
                m[ts].add(i)
        return m
    
    tsub_to_blocks = _rebuild_map(blocks)
    best_blocks = list(blocks)
    best_len = len(blocks)
    temp = 1.0
    cooling = 0.9995
    
    while _time_left() > 10.0 and temp > 0.01:
        move_type = random.random()
        
        if move_type < 0.3 and len(blocks) < 250:
            # Add random block, then remove redundants
            new_block = tuple(sorted(random.sample(range(v), k)))
            if new_block in set(tuple(sorted(b)) for b in blocks):
                temp *= cooling
                continue
            
            blocks.append(list(new_block))
            for ts in combinations(new_block, t):
                tsub_to_blocks[ts].add(len(blocks) - 1)
            
            # Remove redundants
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    if all(len(tsub_to_blocks[ts]) > 1 for ts in combinations(blocks[i], t)):
                        for ts in combinations(blocks[i], t):
                            tsub_to_blocks[ts].discard(i)
                        blocks.pop(i)
                        for ts, bset in list(tsub_to_blocks.items()):
                            tsub_to_blocks[ts] = {j if j < i else j - 1 for j in bset}
                        changed = True
                        break
        
        elif move_type < 0.6:
            # Try removing a redundant block
            if blocks:
                i = random.randrange(len(blocks))
                if all(len(tsub_to_blocks[ts]) > 1 for ts in combinations(blocks[i], t)):
                    for ts in combinations(blocks[i], t):
                        tsub_to_blocks[ts].discard(i)
                    blocks.pop(i)
                    for ts, bset in list(tsub_to_blocks.items()):
                        tsub_to_blocks[ts] = {j if j < i else j - 1 for j in bset}
        
        else:
            # Swap a block
            if not blocks:
                temp *= cooling
                continue
            i = random.randrange(len(blocks))
            unique_tsubs = [ts for ts in combinations(blocks[i], t) if len(tsub_to_blocks[ts]) == 1]
            
            if not unique_tsubs:
                # Redundant, remove it
                for ts in combinations(blocks[i], t):
                    tsub_to_blocks[ts].discard(i)
                blocks.pop(i)
                for ts, bset in list(tsub_to_blocks.items()):
                    tsub_to_blocks[ts] = {j if j < i else j - 1 for j in bset}
            else:
                must_include = set()
                for ts in unique_tsubs:
                    must_include.update(ts)
                
                if len(must_include) <= k:
                    remaining = [p for p in range(v) if p not in must_include]
                    need = k - len(must_include)
                    
                    for _ in range(10):
                        if need <= 0:
                            new_block = tuple(sorted(must_include))
                        elif need <= len(remaining):
                            extra = random.sample(remaining, need)
                            new_block = tuple(sorted(list(must_include) + extra))
                        else:
                            continue
                        
                        nb_set = set(new_block)
                        if not all(set(ts).issubset(nb_set) for ts in unique_tsubs):
                            continue
                        if tuple(sorted(blocks[i])) == new_block:
                            continue
                        
                        # Apply
                        for ts in combinations(blocks[i], t):
                            tsub_to_blocks[ts].discard(i)
                        blocks[i] = list(new_block)
                        for ts in combinations(new_block, t):
                            tsub_to_blocks[ts].add(i)
                        break
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
        
        temp *= cooling
    
    return best_blocks


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Phase 1: Greedy construction
    blocks = _greedy_construct(v, k, t)
    
    if _time_left() < 5.0:
        return blocks
    
    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant(blocks, v, k, t)
    
    if _time_left() < 10.0:
        return blocks
    
    # Phase 3: Branch-and-bound style improvement
    blocks = _branch_and_bound_set_cover(v, k, t, blocks, _DEADLINE_S - 30)
    
    if _time_left() < 15.0:
        return blocks
    
    # Phase 4: SA refinement
    blocks = _simulated_annealing_refine(blocks, v, k, t)
    
    # Phase 5: Final cleanup
    blocks = _remove_redundant(blocks, v, k, t)
    
    return blocks
