"""Direct structural construction for covering design C(24,7,3)."""
import itertools
import random
import time
import math


def _precompute_triple_indices(v, t):
    """Map each t-subset to an index, and precompute all triples as frozensets."""
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {frozenset(tr): i for i, tr in enumerate(triples)}
    return triples, triple_to_idx


def _block_triple_indices(block, triple_to_idx):
    """Get the set of triple indices covered by a block."""
    return set(triple_to_idx[frozenset(c)] for c in itertools.combinations(block, t))


def _greedy_cover_bitmask(v, k, t, rng, triples, triple_to_idx):
    """
    Fast greedy using bitmask for uncovered triples.
    For each step, pick the point that covers the most uncovered triples.
    """
    n_triples = len(triples)
    # Use a list of booleans for uncovered (faster than set for membership)
    uncovered = [True] * n_triples
    uncovered_count = n_triples
    blocks = []
    seen_blocks = set()
    
    while uncovered_count > 0:
        # Pick the first uncovered triple as seed
        seed_idx = 0
        for i in range(n_triples):
            if uncovered[i]:
                seed_idx = i
                break
        seed = triples[seed_idx]
        block = list(seed)
        block_set = set(block)
        
        # Greedily extend
        while len(block) < k:
            best_x = None
            best_gain = -1
            best_tiebreak = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered triples would be covered
                gain = 0
                # For t=3, k=7: we need C(len(block), t-1) = C(len(block), 2) pairs
                # For each pair (a,b) in block, check if (a,b,x) is uncovered
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        tri = frozenset((block[i], block[j], x))
                        idx = triple_to_idx.get(tri)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                # Tiebreak: prefer points that appear in more uncovered triples overall
                # (computed lazily - skip for speed, use random tiebreak)
                if gain > best_gain or (gain == best_gain and gain > 0 and best_x is not None and rng.random() < 0.5):
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            if best_gain <= 0:
                # No positive gain; pick random point not in block
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_x = rng.choice(candidates)
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in seen_blocks:
            seen_blocks.add(b_tuple)
            blocks.append(b_tuple)
            # Update uncovered
            for c in itertools.combinations(block, t):
                idx = triple_to_idx[frozenset(c)]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
    
    return blocks


def _greedy_cover_demand(v, k, t, rng, triples, triple_to_idx):
    """
    Greedy with demand-based ordering: prioritize uncovered triples that have
    fewer potential covering blocks (higher demand).
    """
    n_triples = len(triples)
    uncovered = [True] * n_triples
    uncovered_count = n_triples
    blocks = []
    seen_blocks = set()
    
    # Precompute for each triple, how many k-subsets contain it (constant for all)
    # C(v-3, k-3) for all triples, so use a different heuristic:
    # Prioritize triples whose points have high "scarcity"
    
    # Point frequency in uncovered triples
    point_uncovered_count = [0] * v
    for i in range(n_triples):
        if uncovered[i]:
            for p in triples[i]:
                point_uncovered_count[p] += 1
    
    while uncovered_count > 0:
        # Find the uncovered triple with highest demand (max of point frequencies)
        best_seed_idx = -1
        best_demand = -1
        for i in range(n_triples):
            if uncovered[i]:
                demand = max(point_uncovered_count[p] for p in triples[i])
                if demand > best_demand:
                    best_demand = demand
                    best_seed_idx = i
                # Early exit if we find a very high demand
                if demand > best_demand * 1.5:
                    pass  # continue scanning for now
        
        if best_seed_idx < 0:
            break
        seed = triples[best_seed_idx]
        block = list(seed)
        block_set = set(block)
        
        # Greedily extend
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        tri = frozenset((block[i], block[j], x))
                        idx = triple_to_idx.get(tri)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            if best_gain <= 0:
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_x = rng.choice(candidates)
            
            block.append(best_x)
            block_set.add(best_x)
        
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in seen_blocks:
            seen_blocks.add(b_tuple)
            blocks.append(b_tuple)
            # Update uncovered and point counts
            for c in itertools.combinations(block, t):
                idx = triple_to_idx[frozenset(c)]
                if uncovered[idx]:
                    uncovered[idx] = False
                    uncovered_count -= 1
                    for p in c:
                        point_uncovered_count[p] -= 1
    
    return blocks


def _local_search(blocks, v, k, t, time_limit, triples, triple_to_idx):
    """
    Local search: try removing redundant blocks, then try swaps.
    Uses precomputed triple-to-block mapping for fast checks.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    def compute_counts(blk_list):
        """Count how many times each triple is covered."""
        counts = [0] * len(triples)
        for b in blk_list:
            for c in itertools.combinations(b, t):
                idx = triple_to_idx[frozenset(c)]
                counts[idx] += 1
        return counts
    
    def is_valid_counts(counts):
        return all(c >= 1 for c in counts)
    
    def try_remove(blk_list):
        """Try to remove one block. Returns (new_list, removed_idx) or None."""
        counts = compute_counts(blk_list)
        for i, b in enumerate(blk_list):
            removable = True
            for c in itertools.combinations(b, t):
                idx = triple_to_idx[frozenset(c)]
                if counts[idx] <= 1:
                    removable = False
                    break
            if removable:
                return blk_list[:i] + blk_list[i+1:]
        return None
    
    def try_swap(blk_list):
        """Try to swap one block with a random perturbation."""
        counts = compute_counts(blk_list)
        n = len(blk_list)
        if n == 0:
            return None
        
        i = rng.randrange(n)
        old_b = blk_list[i]
        old_set = set(old_b)
        
        # Random perturbation: replace 1-2 elements
        new_set = set(old_set)
        n_replace = rng.choice([1, 2])
        for _ in range(n_replace):
            if not new_set:
                break
            remove = rng.choice(list(new_set))
            new_set.discard(remove)
            candidates = [x for x in range(v) if x not in new_set]
            if candidates:
                add = rng.choice(candidates)
                new_set.add(add)
        
        if new_set == old_set:
            return None
        new_b = tuple(sorted(new_set))
        if new_b in blk_list:
            return None
        
        new_list = list(blk_list)
        new_list[i] = new_b
        
        new_counts = compute_counts(new_list)
        if is_valid_counts(new_counts):
            return new_list
        return None
    
    best = list(blocks)
    best_n = len(best)
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        # Try removal first
        removed = try_remove(blocks)
        if removed is not None:
            blocks = removed
            if len(blocks) < best_n:
                best_n = len(blocks)
                best = list(blocks)
                no_improve = 0
            else:
                no_improve += 1
            continue
        
        # Try swap
        swapped = try_swap(blocks)
        if swapped is not None:
            blocks = swapped
            if len(blocks) < best_n:
                best_n = len(blocks)
                best = list(blocks)
                no_improve = 0
            else:
                no_improve += 1
        
        # Occasionally try a more aggressive move: remove a random block and
        # try to patch by adding a new block that covers the lost triples
        if rng.random() < 0.3 and len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_b = blocks[i]
            # Find triples only covered by this block
            counts = compute_counts(blocks)
            critical = []
            for c in itertools.combinations(old_b, t):
                idx = triple_to_idx[frozenset(c)]
                if counts[idx] == 1:
                    critical.append(idx)
            
            if critical:
                # Try to find a new block that covers all critical triples
                # This is a set cover problem on the critical triples
                # Simple approach: try random blocks containing points from critical triples
                critical_points = set()
                for idx in critical:
                    critical_points.update(triples[idx])
                
                for attempt in range(50):
                    # Build a candidate block from critical points
                    cand_set = set()
                    for p in critical_points:
                        if len(cand_set) >= k:
                            break
                        if rng.random() < 0.7:
                            cand_set.add(p)
                    # Fill remaining
                    while len(cand_set) < k:
                        p = rng.randrange(v)
                        cand_set.add(p)
                    
                    cand_b = tuple(sorted(cand_set))
                    if cand_b in blocks:
                        continue
                    
                    new_list = [b for j, b in enumerate(blocks) if j != i]
                    new_list.append(cand_b)
                    
                    new_counts = compute_counts(new_list)
                    if is_valid_counts(new_counts):
                        blocks = new_list
                        if len(blocks) < best_n:
                            best_n = len(blocks)
                            best = list(blocks)
                            no_improve = 0
                        break
        
        no_improve += 1
        if no_improve > 200:
            break
    
    return best, best_n


def construct(v, k, t):
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for 150s limit
    
    # Precompute
    triples, triple_to_idx = _precompute_triple_indices(v, t)
    
    # Multiple restarts with different strategies
    best_blocks = None
    best_n = float('inf')
    
    strategies = [
        ("bitmask", _greedy_cover_bitmask),
        ("demand", _greedy_cover_demand),
        ("bitmask", _greedy_cover_bitmask),
        ("demand", _greedy_cover_demand),
    ]
    
    for idx, (name, strategy_fn) in enumerate(strategies):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 30:  # Need at least 30s for local search
            break
        
        rng = random.Random(12345 + idx * 7919)
        blocks = strategy_fn(v, k, t, rng, triples, triple_to_idx)
        
        if len(blocks) < best_n:
            best_n = len(blocks)
            best_blocks = blocks
        
        # Local search on this result
        remaining = time_limit - (time.time() - start_time)
        if remaining > 5:
            ls_blocks, ls_n = _local_search(blocks, v, k, t, min(remaining, 25.0), triples, triple_to_idx)
            if ls_n < best_n:
                best_n = ls_n
                best_blocks = ls_blocks
    
    # Final local search on the best result with remaining time
    if best_blocks is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining > 5:
            ls_blocks, ls_n = _local_search(best_blocks, v, k, t, remaining - 2.0, triples, triple_to_idx)
            if ls_n < best_n:
                best_n = ls_n
                best_blocks = ls_blocks
    
    if best_blocks is None:
        # Fallback: should not happen
        return []
    
    return [list(b) for b in best_blocks]


if __name__ == "__main__":
    import time
    start = time.time()
    result = construct(24, 7, 3)
    elapsed = time.time() - start
    print(f"Blocks: {len(result)}, Time: {elapsed:.2f}s")
    # Verify
    covered = set()
    for b in result:
        for c in itertools.combinations(b, 3):
            covered.add(frozenset(c))
    total = math.comb(24, 3)
    print(f"Covered: {len(covered)}/{total}")
    assert len(covered) == total, "Invalid covering!"
