"""Randomized local search for covering design C(24,7,3) with block-intersection pruning."""
import itertools
import random
import time

def _block_triples_list(block, subset_to_idx):
    """Return list of triple-indices covered by a 7-element block."""
    return [subset_to_idx[frozenset(c)] for c in itertools.combinations(block, 3)]

def _greedy_cover(v, k, t, rng, tsubset_points, subset_to_idx):
    """Fast greedy construction: repeatedly pick a random uncovered triple, grow a block."""
    total_t = len(tsubset_points)
    uncovered = set(range(total_t))
    blocks = []
    # Precompute all possible t-1 subsets for each point for speed? No, k is small.
    # Precompute mapping from frozenset of size t-1 to list of points that complete it?
    # That's expensive. Let's just compute gain on the fly.
    
    while uncovered:
        # Pick a random uncovered triple
        seed_idx = rng.choice(tuple(uncovered))
        seed = tsubset_points[seed_idx]
        block = list(seed)
        
        # Grow block to size k
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                # For each (t-1)-subset of current block, check if adding x covers a new triple
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_idx.get(fs)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                # Early break if we found a perfect gain? Max gain is C(len(block), t-1)
                max_possible = 1
                for _ in range(t-1):
                    max_possible *= (len(block) - _)
                    max_possible //= _
                if gain == max_possible:
                    break
            
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        blocks.append(b_tuple)
        
        # Update uncovered
        for c in itertools.combinations(block, t):
            idx = subset_to_idx[frozenset(c)]
            uncovered.discard(idx)
            
    return blocks

def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0  # Safety margin from 120s
    
    # Precompute all t-subsets
    tsubset_points = []
    subset_to_idx = {}
    for c in itertools.combinations(range(v), t):
        fs = frozenset(c)
        subset_to_idx[fs] = len(tsubset_points)
        tsubset_points.append(c)
    
    total_t = len(tsubset_points)
    
    # Initial greedy cover
    blocks = _greedy_cover(v, k, t, rng, tsubset_points, subset_to_idx)
    
    # Initialize state
    counts = [0] * total_t
    block_triples_cache = {}
    for b in blocks:
        tris = _block_triples_list(b, subset_to_idx)
        block_triples_cache[tuple(b)] = tris
        for idx in tris:
            counts[idx] += 1
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    def remove_redundant():
        """Remove blocks that are redundant (all their triples are covered by other blocks)."""
        removed = 0
        i = 0
        while i < len(blocks):
            b = blocks[i]
            is_red = True
            for idx in block_triples_cache[tuple(b)]:
                if counts[idx] <= 1:
                    is_red = False
                    break
            if is_red:
                for idx in block_triples_cache[tuple(b)]:
                    counts[idx] -= 1
                blocks.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    def try_swap(i):
        """Try to replace block i with a new block that covers all critical triples and more."""
        old_b = blocks[i]
        old_tris = block_triples_cache[tuple(old_b)]
        
        # Find critical points (from triples with count 1)
        critical_points = set()
        for idx in old_tris:
            if counts[idx] == 1:
                critical_points.update(tsubset_points[idx])
        
        if len(critical_points) > k:
            return False
        
        new_set = set(critical_points)
        cands = [x for x in range(v) if x not in new_set]
        rng.shuffle(cands)
        
        # Greedy fill to size k, prioritizing points that cover more critical triples
        while len(new_set) < k and cands:
            best_x = None
            best_gain = -1
            for x in cands:
                if x in new_set:
                    continue
                gain = 0
                for rest in itertools.combinations(new_set, t-1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_idx.get(fs)
                    if idx is not None and counts[idx] == 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                if best_gain == 0 and gain == 0:
                    # If no gain, just pick the first one to fill
                    break
            if best_x is None:
                # Fill with any remaining candidate
                for x in cands:
                    if x not in new_set:
                        new_set.add(x)
                        break
                else:
                    return False
            else:
                new_set.add(best_x)
                cands.remove(best_x)
        
        if len(new_set) != k:
            return False
        
        new_b = tuple(sorted(new_set))
        if new_b == tuple(old_b):
            return False
        
        new_tris = _block_triples_list(new_b, subset_to_idx)
        new_tris_set = set(new_tris)
        
        # Check validity: all count==1 triples of old_b must be in new_tris
        valid = True
        for idx in old_tris:
            if counts[idx] == 1 and idx not in new_tris_set:
                valid = False
                break
        
        if not valid:
            return False
        
        # Apply
        for idx in old_tris:
            counts[idx] -= 1
        for idx in new_tris:
            counts[idx] += 1
        
        blocks[i] = list(new_b)
        block_triples_cache[new_b] = new_tris
        # Clean up cache for old_b if it's no longer used (it's not, since blocks are distinct)
        if tuple(old_b) in block_triples_cache and tuple(old_b) != new_b:
            # Check if any other block has this triple set? No, blocks are distinct.
            # But we might have multiple blocks with same triples? Unlikely but possible.
            # For safety, we don't delete from cache unless we are sure.
            pass
        return True

    def try_merge_two_blocks(i, j):
        """Try to replace blocks i and j with a single block that covers all critical triples of both."""
        b1 = blocks[i]
        b2 = blocks[j]
        tris1 = block_triples_cache[tuple(b1)]
        tris2 = block_triples_cache[tuple(b2)]
        
        # Find critical points from both blocks
        critical_points = set()
        for idx in tris1:
            if counts[idx] == 1:
                critical_points.update(tsubset_points[idx])
        for idx in tris2:
            if counts[idx] == 1:
                critical_points.update(tsubset_points[idx])
        
        if len(critical_points) > k:
            return False
        
        new_set = set(critical_points)
        cands = [x for x in range(v) if x not in new_set]
        rng.shuffle(cands)
        
        # Greedy fill
        while len(new_set) < k and cands:
            best_x = None
            best_gain = -1
            for x in cands:
                if x in new_set:
                    continue
                gain = 0
                for rest in itertools.combinations(new_set, t-1):
                    fs = frozenset(list(rest) + [x])
                    idx = subset_to_idx.get(fs)
                    if idx is not None and counts[idx] == 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                if best_gain == 0 and gain == 0:
                    break
            if best_x is None:
                for x in cands:
                    if x not in new_set:
                        new_set.add(x)
                        break
                else:
                    return False
            else:
                new_set.add(best_x)
                cands.remove(best_x)
        
        if len(new_set) != k:
            return False
        
        new_b = tuple(sorted(new_set))
        
        # Check if new_b is already in blocks (excluding i and j)
        new_tris = _block_triples_list(new_b, subset_to_idx)
        new_tris_set = set(new_tris)
        
        # Check validity: all count==1 triples of b1 and b2 must be in new_tris
        valid = True
        for idx in tris1:
            if counts[idx] == 1 and idx not in new_tris_set:
                valid = False
                break
        if valid:
            for idx in tris2:
                if counts[idx] == 1 and idx not in new_tris_set:
                    valid = False
                    break
        
        if not valid:
            return False
        
        # Apply: remove b1 and b2, add new_b
        # First, decrement counts for b1 and b2
        for idx in tris1:
            counts[idx] -= 1
        for idx in tris2:
            counts[idx] -= 1
        
        # Increment counts for new_b
        for idx in new_tris:
            counts[idx] += 1
        
        # Remove b2 first (higher index), then b1
        blocks.pop(j)
        blocks.pop(i)
        blocks.append(list(new_b))
        
        # Update cache
        block_triples_cache[new_b] = new_tris
        # Clean up old caches
        if tuple(b1) in block_triples_cache:
            del block_triples_cache[tuple(b1)]
        if tuple(b2) in block_triples_cache:
            del block_triples_cache[tuple(b2)]
        
        return True

    no_improve = 0
    while time.time() - start_time < time_limit:
        # 1. Remove redundants
        if remove_redundant() > 0:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
            no_improve = 0
            continue
        
        # 2. Try swaps and merges
        improved = False
        for _ in range(50):  # Try a few moves
            if not blocks:
                break
            
            move_type = rng.choice(['swap', 'merge'])
            if move_type == 'swap':
                i = rng.randrange(len(blocks))
                if try_swap(i):
                    improved = True
            else:
                if len(blocks) >= 2:
                    i, j = rng.sample(range(len(blocks)), 2)
                    if try_merge_two_blocks(i, j):
                        improved = True
            
            if improved:
                # After a move, new redundancies might appear
                if remove_redundant() > 0:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                break
        
        if not improved:
            no_improve += 1
            if no_improve > 200:
                # Restart with a fresh greedy cover
                new_blocks = _greedy_cover(v, k, t, rng, tsubset_points, subset_to_idx)
                if len(new_blocks) < len(blocks):
                    blocks = [list(b) for b in new_blocks]
                    counts = [0] * total_t
                    block_triples_cache = {}
                    for b in blocks:
                        tris = _block_triples_list(b, subset_to_idx)
                        block_triples_cache[tuple(b)] = tris
                        for idx in tris:
                            counts[idx] += 1
                    remove_redundant()
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                else:
                    no_improve += 100

    remove_redundant()
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = [list(b) for b in blocks]
        
    return best_blocks
