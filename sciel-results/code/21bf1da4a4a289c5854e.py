import itertools
import random
import time
import numpy as np
from collections import defaultdict


def _precompute_triple_index(v, t):
    """Create a mapping from triple to integer index for fast lookup."""
    # Total number of t-subsets
    n_triples = 1
    for i in range(t):
        n_triples *= (v - i)
        n_triples //= (i + 1)
    return n_triples


def _triple_to_idx(triple, v, t):
    """Convert a t-tuple (sorted) to a unique index using combinatorial number system."""
    # Combinatorial number system for C(v, t)
    idx = 0
    for i in range(t):
        # C(triple[i], i+1) in the combinatorial number system
        # Actually use the standard ranking formula
        pass
    # Simpler: use a dict-based approach or a precomputed table
    # For v=24, t=3, C(24,3)=2024, so we can use a direct mapping
    # Use the formula: rank = C(a1,1) + C(a2,2) + ... for sorted elements
    # Actually for C(v,t) ranking:
    # rank of (x1, x2, ..., xt) with 0<=x1<x2<...<xt<v
    # = C(x1,1) + C(x2,2) + C(x3,3) + ... + C(xt,t)
    # Wait, that's not quite right. Let me use the standard formula.
    # The rank of combination (c1, c2, ..., ck) from {0, ..., n-1} is:
    # sum_{i=1}^{k} C(ci, i)
    # where ci is the i-th element (0-indexed) and we use C(n,r) = 0 if n < r
    rank = 0
    for i in range(t):
        rank += _comb(triple[i], i + 1)
    return rank


def _comb(n, r):
    """Compute C(n, r) with C(n,r)=0 if n<r."""
    if n < r or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    r = min(r, n - r)
    result = 1
    for i in range(r):
        result = result * (n - i) // (i + 1)
    return result


def _build_triple_rank_table(v, t):
    """Precompute rank for all t-subsets using combinatorial number system."""
    # For v=24, t=3: C(24,3) = 2024
    # Precompute all C(n, r) values needed
    max_n = v
    table = {}
    for i in range(t):
        for n in range(max_n + 1):
            table[(n, i + 1)] = _comb(n, i + 1)
    return table


def _greedy_cover_fast(v, k, t, rng, deadline, triple_table):
    """Fast greedy covering using precomputed tables and vectorized operations."""
    n_triples = _comb(v, t)
    # uncovered: boolean array of size n_triples
    uncovered = np.ones(n_triples, dtype=np.bool_)
    blocks = []
    
    # Precompute for each point x and each (t-1)-subset of remaining points,
    # the list of triples that would be covered
    # For efficiency, precompute all (t-1)-subsets and their triple contributions
    
    while uncovered.any():
        if time.time() > deadline:
            return None
        
        # Pick a random uncovered triple to start
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = rng.choice(uncovered_indices)
        
        # Decode start_idx back to triple
        start_triple = _decode_triple(start_idx, v, t)
        block = list(start_triple)
        block_set = set(block)
        
        # Mark this triple as covered
        uncovered[start_idx] = False
        
        # Add remaining points to complete the block
        while len(block) < k:
            if time.time() > deadline:
                return None
            
            # Compute gain for each candidate point
            # For point x, gain = number of uncovered triples of form {x} U S where S is a (t-1)-subset of block
            block_combos = list(itertools.combinations(block, t - 1))
            
            best_gain = -1
            best_point = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in block_combos:
                    ts = tuple(sorted(rest + (x,)))
                    idx = _triple_rank(ts, triple_table)
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_point = rng.choice(candidates)
            
            block.append(best_point)
            block_set.add(best_point)
            
            # Update uncovered: mark all triples in the new block as covered
            for ts in itertools.combinations(block, t):
                idx = _triple_rank(ts, triple_table)
                uncovered[idx] = False
        
        block.sort()
        blocks.append(block)
    
    return blocks


def _decode_triple(idx, v, t):
    """Decode a triple index back to the actual triple."""
    # Using combinatorial number system
    # idx = C(x1, 1) + C(x2, 2) + C(x3, 3)
    triple = []
    remaining = idx
    for i in range(t):
        r = i + 1
        # Find x such that C(x, r) <= remaining
        # Binary search
        lo, hi = r, v - 1
        best = r
        while lo <= hi:
            mid = (lo + hi) // 2
            c = _comb(mid, r)
            if c <= remaining:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        triple.append(best)
        remaining -= _comb(best, r)
    return tuple(triple)


def _triple_rank(triple, triple_table):
    """Get rank of a triple using precomputed table."""
    rank = 0
    for i in range(len(triple)):
        rank += triple_table.get((triple[i], i + 1), 0)
    return rank


def _remove_redundant_blocks_fast(blocks, v, t, triple_table):
    """Remove blocks that don't cover any unique t-subset, using fast indexing."""
    if not blocks:
        return blocks
    
    # Build coverage count array
    n_triples = _comb(v, t)
    cover_count = np.zeros(n_triples, dtype=np.int32)
    block_triple_indices = []
    
    for block in blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            idx = _triple_rank(ts, triple_table)
            indices.append(idx)
            cover_count[idx] += 1
        block_triple_indices.append(indices)
    
    # Remove redundant blocks (those where all their triples have cover_count > 1)
    kept_indices = []
    for i, indices in enumerate(block_triple_indices):
        all_redundant = True
        for idx in indices:
            if cover_count[idx] == 1:
                all_redundant = False
                break
        if not all_redundant:
            kept_indices.append(i)
    
    if len(kept_indices) < len(blocks):
        # Rebuild cover_count for kept blocks
        cover_count2 = np.zeros(n_triples, dtype=np.int32)
        kept_blocks = []
        kept_indices2 = []
        for i in kept_indices:
            for idx in block_triple_indices[i]:
                cover_count2[idx] += 1
            kept_blocks.append(blocks[i])
            kept_indices2.append(block_triple_indices[i])
        
        # Second pass
        final_blocks = []
        for i, indices in enumerate(kept_indices2):
            all_redundant = True
            for idx in indices:
                if cover_count2[idx] == 1:
                    all_redundant = False
                    break
            if not all_redundant:
                final_blocks.append(kept_blocks[i])
        
        return final_blocks
    
    return blocks


def _verify_covering(blocks, v, t, triple_table):
    """Verify every t-subset is covered by at least one block."""
    n_triples = _comb(v, t)
    covered = np.zeros(n_triples, dtype=np.bool_)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            idx = _triple_rank(ts, triple_table)
            covered[idx] = True
    return covered.all()


def _local_search_fast(blocks, v, k, t, rng, deadline, triple_table):
    """Improved local search with better scoring and multi-point moves."""
    if not blocks:
        return blocks
    
    n_triples = _comb(v, t)
    current_blocks = [list(b) for b in blocks]
    current_count = len(current_blocks)
    best_blocks = [list(b) for b in current_blocks]
    best_count = current_count
    
    # Build coverage structure
    cover_count = np.zeros(n_triples, dtype=np.int32)
    block_triple_indices = []
    
    for block in current_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            idx = _triple_rank(ts, triple_table)
            indices.append(idx)
            cover_count[idx] += 1
        block_triple_indices.append(indices)
    
    # Remove redundant blocks first
    for i in range(len(current_blocks) - 1, -1, -1):
        if time.time() > deadline:
            break
        is_redundant = True
        for idx in block_triple_indices[i]:
            if cover_count[idx] <= 1:
                is_redundant = False
                break
        if is_redundant:
            for idx in block_triple_indices[i]:
                cover_count[idx] -= 1
            current_blocks.pop(i)
            block_triple_indices.pop(i)
            current_count -= 1
            best_blocks = [list(b) for b in current_blocks]
            best_count = current_count
    
    # Local search: try swaps and check for block removal opportunities
    max_iter = 15000
    no_improve_count = 0
    max_no_improve = 2000
    
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
        if not current_blocks:
            break
        if no_improve_count > max_no_improve:
            break
        
        block_idx = rng.randint(0, len(current_blocks) - 1)
        block = current_blocks[block_idx]
        
        # Try multiple swap strategies
        best_move = None
        best_score = float('inf')
        
        # Strategy 1: Single point swap
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            pos = rng.randint(0, len(block) - 1)
            orig_point = block[pos]
            new_point = rng.randint(0, v - 1)
            
            if new_point in block:
                continue
            
            new_block = block.copy()
            new_block[pos] = new_point
            new_block.sort()
            
            old_indices = set(block_triple_indices[block_idx])
            new_indices = set()
            for ts in itertools.combinations(new_block, t):
                idx = _triple_rank(ts, triple_table)
                new_indices.add(idx)
            
            lost = old_indices - new_indices
            gained = new_indices - old_indices
            
            # Check validity: no lost triple should be uniquely covered
            valid = True
            for idx in lost:
                if cover_count[idx] <= 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Score: prefer moves that create redundancy
            lost_unique = sum(1 for idx in lost if cover_count[idx] == 1)
            gained_redundant = sum(1 for idx in gained if cover_count[idx] >= 1)
            
            score = lost_unique * 100 - gained_redundant * 10
            if score < best_score:
                best_score = score
                best_move = ('swap', block_idx, pos, new_point, new_block, new_indices)
        
        # Strategy 2: Try removing a block if possible after a swap
        # (handled below)
        
        if best_move is not None:
            move_type, bidx, pos, new_point, new_block, new_indices = best_move
            
            old_indices = set(block_triple_indices[bidx])
            lost = old_indices - new_indices
            gained = new_indices - old_indices
            
            for idx in lost:
                cover_count[idx] -= 1
            for idx in gained:
                cover_count[idx] += 1
            
            current_blocks[bidx] = new_block
            block_triple_indices[bidx] = list(new_indices)
            no_improve_count = 0
        
        else:
            no_improve_count += 1
        
        # Check for removable blocks
        removed_any = False
        for i in range(len(current_blocks) - 1, -1, -1):
            is_redundant = True
            for idx in block_triple_indices[i]:
                if cover_count[idx] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for idx in block_triple_indices[i]:
                    cover_count[idx] -= 1
                current_blocks.pop(i)
                block_triple_indices.pop(i)
                current_count -= 1
                best_blocks = [list(b) for b in current_blocks]
                best_count = current_count
                removed_any = True
                no_improve_count = 0
                break
        
        if not removed_any and best_move is None:
            no_improve_count += 1
    
    return best_blocks


def construct(v, k, t):
    deadline = time.time() + 145  # Safety margin before 150s limit
    rng = random.Random(42)
    
    # Precompute triple ranking table
    triple_table = _build_triple_rank_table(v, t)
    n_triples = _comb(v, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multiple randomized restarts
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline - 20:
            break
        
        restart_rng = random.Random(42 + restart * 13)
        
        blocks = _greedy_cover_fast(v, k, t, restart_rng, deadline, triple_table)
        if blocks is None:
            continue
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks_fast(blocks, v, t, triple_table)
        
        # Local search refinement
        blocks = _local_search_fast(blocks, v, k, t, restart_rng, deadline, triple_table)
        
        # Verify validity
        if _verify_covering(blocks, v, t, triple_table):
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
        
        # Early exit if excellent solution found
        if best_count <= 73:
            break
    
    # Phase 2: Final polish on best solution
    if best_blocks is not None and time.time() < deadline - 10:
        polish_rng = random.Random(12345)
        polished = _local_search_fast(best_blocks, v, k, t, polish_rng, deadline, triple_table)
        if _verify_covering(polished, v, t, triple_table) and len(polished) < best_count:
            best_blocks = polished
            best_count = len(polished)
    
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    best_x = next(x for x in range(v) if x not in block)
                block.append(best_x)
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        best_blocks = blocks
    
    return best_blocks
