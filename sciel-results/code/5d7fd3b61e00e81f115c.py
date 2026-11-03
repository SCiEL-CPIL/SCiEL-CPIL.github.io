"""Direct structural construction for covering design C(24,7,3)."""
import itertools
import random
import time
import math
import numpy as np


def _precompute_triple_indices(v, t):
    """Map each t-subset to an index, and precompute all triples as frozensets."""
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {frozenset(tr): i for i, tr in enumerate(triples)}
    return triples, triple_to_idx


def _block_triple_indices(block, triple_to_idx, t):
    """Get the set of triple indices covered by a block."""
    return [triple_to_idx[frozenset(c)] for c in itertools.combinations(block, t)]


def _compute_coverage_counts(blocks, v, k, t, triple_to_idx):
    """Compute how many times each triple is covered."""
    n_triples = len(triple_to_idx)
    counts = np.zeros(n_triples, dtype=np.int32)
    for b in blocks:
        for c in itertools.combinations(b, t):
            idx = triple_to_idx[frozenset(c)]
            counts[idx] += 1
    return counts


def _is_valid_covering(counts):
    """Check if all triples are covered at least once."""
    return np.all(counts >= 1)


def _scarcity_weighted_greedy(v, k, t, rng, triples, triple_to_idx, time_limit=None):
    """
    Greedy with dynamic scarcity weighting: prioritize uncovered triples
    whose points appear in fewer remaining potential blocks.
    """
    n_triples = len(triples)
    uncovered = np.ones(n_triples, dtype=bool)
    uncovered_count = n_triples
    blocks = []
    seen_blocks = set()
    
    start_time = time.time()
    
    # Point scarcity: how many uncovered triples contain each point
    point_uncovered = np.zeros(v, dtype=np.int32)
    for i in range(n_triples):
        for p in triples[i]:
            point_uncovered[p] += 1
    
    while uncovered_count > 0:
        if time_limit and time.time() - start_time > time_limit:
            break
        
        # Find the uncovered triple with highest scarcity (max point frequency)
        best_seed_idx = -1
        best_score = -1
        for i in range(n_triples):
            if uncovered[i]:
                score = max(point_uncovered[p] for p in triples[i])
                if score > best_score:
                    best_score = score
                    best_seed_idx = i
        
        if best_seed_idx < 0:
            break
        
        seed = triples[best_seed_idx]
        block = list(seed)
        block_set = set(block)
        
        # Greedily extend with scarcity-aware gain
        while len(block) < k:
            best_x = None
            best_gain = -1
            best_tie = -1
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                scarcity_bonus = 0
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        tri = frozenset((block[i], block[j], x))
                        idx = triple_to_idx.get(tri)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                            # Bonus for covering triples with scarce points
                            scarcity_bonus += point_uncovered[x] * 0.1
                total = gain + scarcity_bonus
                if total > best_gain or (total == best_gain and gain > 0 and best_x is not None and rng.random() < 0.5):
                    best_gain = total
                    best_gain = gain
                    best_tie = scarcity_bonus
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
                        point_uncovered[p] -= 1
    
    return blocks


def _abfkp_cover(v, k, t, rng, triples, triple_to_idx, time_limit=None):
    """
    Alter-Broder-Frieze-Karlin-Peierls algorithm for set cover.
    """
    n_triples = len(triples)
    p = math.comb(k, t) / math.comb(v, t)
    
    uncovered = np.ones(n_triples, dtype=bool)
    uncovered_count = n_triples
    blocks = []
    seen_blocks = set()
    
    start_time = time.time()
    
    while uncovered_count > 0:
        if time_limit and time.time() - start_time > time_limit:
            break
        
        target_remaining = 0.05
        m_needed = int(math.log(target_remaining) / math.log(1 - p)) + 1
        m_needed = min(m_needed, 300)
        
        sampled = []
        for _ in range(m_needed):
            block = tuple(sorted(rng.sample(range(v), k)))
            if block not in seen_blocks:
                sampled.append(block)
        
        if not sampled:
            break
        
        # Compute coverage counts from sampled pool
        triple_sample_count = np.zeros(n_triples, dtype=np.int32)
        block_triples = []
        
        for b in sampled:
            tri_indices = []
            for c in itertools.combinations(b, t):
                idx = triple_to_idx[frozenset(c)]
                if uncovered[idx]:
                    tri_indices.append(idx)
                    triple_sample_count[idx] += 1
            block_triples.append((b, tri_indices))
        
        # Greedy select from sampled pool with ABFKP weighting
        local_uncovered = uncovered.copy()
        local_count = uncovered_count
        selected = []
        selected_set = set()
        
        available = list(range(len(sampled)))
        while available and local_count > 0:
            best_idx = -1
            best_gain = -1
            for ai in available:
                b, tri_indices = block_triples[ai]
                gain = 0.0
                for idx in tri_indices:
                    if local_uncovered[idx]:
                        sc = triple_sample_count[idx]
                        gain += 1.0 / (1.0 + sc)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = ai
            
            if best_idx < 0 or best_gain <= 1e-9:
                break
            
            b, tri_indices = block_triples[best_idx]
            if b not in selected_set:
                selected.append(b)
                selected_set.add(b)
                seen_blocks.add(b)
                for idx in tri_indices:
                    if local_uncovered[idx]:
                        local_uncovered[idx] = False
                        local_count -= 1
            
            available.remove(best_idx)
        
        if selected:
            for b in selected:
                blocks.append(b)
                for c in itertools.combinations(b, t):
                    idx = triple_to_idx[frozenset(c)]
                    if uncovered[idx]:
                        uncovered[idx] = False
                        uncovered_count -= 1
    
    return blocks


def _simulated_annealing(blocks, v, k, t, triples, triple_to_idx, time_limit, rng=None):
    """
    Simulated annealing to minimize block count from a valid covering.
    Moves: remove a redundant block, swap a block with a random perturbation,
    or replace two blocks with one better block.
    """
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    blocks = list(blocks)
    n = len(blocks)
    
    if n == 0:
        return blocks, n
    
    counts = _compute_coverage_counts(blocks, v, k, t, triple_to_idx)
    
    best_blocks = list(blocks)
    best_n = n
    current_n = n
    
    # Temperature schedule
    T_start = 2.0
    T_end = 0.01
    total_steps = 50000
    step = 0
    
    while step < total_steps:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Adaptive temperature
        progress = step / total_steps
        T = T_start * (T_end / T_start) ** progress
        
        move_type = rng.random()
        
        if move_type < 0.3:
            # Try to remove a redundant block
            removable_idx = -1
            for i in range(len(blocks)):
                removable = True
                for c in itertools.combinations(blocks[i], t):
                    idx = triple_to_idx[frozenset(c)]
                    if counts[idx] <= 1:
                        removable = False
                        break
                if removable:
                    removable_idx = i
                    break
            
            if removable_idx >= 0:
                # Remove it
                removed_block = blocks.pop(removable_idx)
                for c in itertools.combinations(removed_block, t):
                    idx = triple_to_idx[frozenset(c)]
                    counts[idx] -= 1
                current_n -= 1
                
                if current_n < best_n:
                    best_n = current_n
                    best_blocks = list(blocks)
                step += 1
                continue
            
        elif move_type < 0.6:
            # Swap: replace one block with a perturbed version
            if not blocks:
                step += 1
                continue
            i = rng.randrange(len(blocks))
            old_b = blocks[i]
            old_set = set(old_b)
            
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
                step += 1
                continue
            
            new_b = tuple(sorted(new_set))
            if new_b in blocks:
                step += 1
                continue
            
            # Check if swap maintains validity
            # First, check if old block has any critically covered triples
            critical = False
            for c in itertools.combinations(old_b, t):
                idx = triple_to_idx[frozenset(c)]
                if counts[idx] == 1:
                    critical = True
                    break
            
            if critical:
                # Need new block to cover all critical triples of old block
                new_triples = set(itertools.combinations(new_b, t))
                for c in itertools.combinations(old_b, t):
                    idx = triple_to_idx[frozenset(c)]
                    if counts[idx] == 1 and frozenset(c) not in new_triples:
                        critical = False
                        break
                if critical:
                    # Also check new block doesn't create issues
                    pass
                else:
                    step += 1
                    continue
            
            # Perform swap
            for c in itertools.combinations(old_b, t):
                idx = triple_to_idx[frozenset(c)]
                counts[idx] -= 1
            for c in itertools.combinations(new_b, t):
                idx = triple_to_idx[frozenset(c)]
                counts[idx] += 1
            
            # Check validity
            if not _is_valid_covering(counts):
                # Reject: revert
                for c in itertools.combinations(new_b, t):
                    idx = triple_to_idx[frozenset(c)]
                    counts[idx] -= 1
                for c in itertools.combinations(old_b, t):
                    idx = triple_to_idx[frozenset(c)]
                    counts[idx] += 1
                step += 1
                continue
            
            blocks[i] = new_b
            step += 1
            
        elif move_type < 0.8:
            # Try to replace two blocks with one
            if len(blocks) < 2:
                step += 1
                continue
            i, j = rng.sample(range(len(blocks)), 2)
            b1, b2 = blocks[i], blocks[j]
            set1, set2 = set(b1), set(b2)
            
            # Try to find a single block that covers all triples uniquely covered by b1 or b2
            critical_triples = []
            for c in itertools.combinations(b1, t):
                idx = triple_to_idx[frozenset(c)]
                if counts[idx] == 1:
                    critical_triples.append(frozenset(c))
            for c in itertools.combinations(b2, t):
                idx = triple_to_idx[frozenset(c)]
                if counts[idx] == 1:
                    critical_triples.append(frozenset(c))
            
            if not critical_triples:
                # Both blocks are redundant; just remove one
                # Remove the one with fewer unique triples
                uniq1 = sum(1 for c in itertools.combinations(b1, t) if triple_to_idx[frozenset(c)] and counts[triple_to_idx[frozenset(c)]] == 1)
                uniq2 = sum(1 for c in itertools.combinations(b2, t) if triple_to_idx[frozenset(c)] and counts[triple_to_idx[frozenset(c)]] == 1)
                if uniq1 <= uniq2:
                    removed = blocks.pop(i)
                    for c in itertools.combinations(removed, t):
                        idx = triple_to_idx[frozenset(c)]
                        counts[idx] -= 1
                    current_n -= 1
                    if current_n < best_n:
                        best_n = current_n
                        best_blocks = list(blocks)
                else:
                    removed = blocks.pop(j)
                    for c in itertools.combinations(removed, t):
                        idx = triple_to_idx[frozenset(c)]
                        counts[idx] -= 1
                    current_n -= 1
                    if current_n < best_n:
                        best_n = current_n
                        best_blocks = list(blocks)
                step += 1
                continue
            
            # Try to build a block from critical triple points
            critical_points = set()
            for ct in critical_triples:
                critical_points.update(ct)
            
            if len(critical_points) > k:
                step += 1
                continue
            
            # Attempt to form a valid replacement block
            success = False
            for attempt in range(20):
                cand_set = set(critical_points)
                while len(cand_set) < k:
                    p = rng.randrange(v)
                    if p not in cand_set:
                        cand_set.add(p)
                
                cand_b = tuple(sorted(cand_set))
                
                # Check if this block covers all critical triples
                cand_triples = set(frozenset(c) for c in itertools.combinations(cand_b, t))
                if all(ct in cand_triples for ct in critical_triples):
                    # Remove both old blocks, add new one
                    for c in itertools.combinations(b1, t):
                        idx = triple_to_idx[frozenset(c)]
                        counts[idx] -= 1
                    for c in itertools.combinations(b2, t):
                        idx = triple_to_idx[frozenset(c)]
                        counts[idx] -= 1
                    for c in itertools.combinations(cand_b, t):
                        idx = triple_to_idx[frozenset(c)]
                        counts[idx] += 1
                    
                    if _is_valid_covering(counts):
                        # Accept
                        del blocks[j]
                        blocks[i] = cand_b
                        current_n -= 1
                        if current_n < best_n:
                            best_n = current_n
                            best_blocks = list(blocks)
                        success = True
                        break
                    else:
                        # Revert
                        for c in itertools.combinations(cand_b, t):
                            idx = triple_to_idx[frozenset(c)]
                            counts[idx] -= 1
                        for c in itertools.combinations(b2, t):
                            idx = triple_to_idx[frozenset(c)]
                            counts[idx] += 1
                        for c in itertools.combinations(b1, t):
                            idx = triple_to_idx[frozenset(c)]
                            counts[idx] += 1
            
            step += 1
            
        else:
            # Random restart perturbation: replace a random block entirely
            if not blocks:
                step += 1
                continue
            i = rng.randrange(len(blocks))
            old_b = blocks[i]
            
            # Check if removable
            removable = True
            for c in itertools.combinations(old_b, t):
                idx = triple_to_idx[frozenset(c)]
                if counts[idx] <= 1:
                    removable = False
                    break
            
            if not removable:
                step += 1
                continue
            
            # Generate a new random block biased toward uncovered triples
            new_set = set()
            # Add some points from old block
            for p in old_b:
                if len(new_set) < k // 2 and rng.random() < 0.5:
                    new_set.add(p)
            while len(new_set) < k:
                p = rng.randrange(v)
                new_set.add(p)
            
            new_b = tuple(sorted(new_set))
            if new_b in blocks:
                step += 1
                continue
            
            # Perform swap
            for c in itertools.combinations(old_b, t):
                idx = triple_to_idx[frozenset(c)]
                counts[idx] -= 1
            for c in itertools.combinations(new_b, t):
                idx = triple_to_idx[frozenset(c)]
                counts[idx] += 1
            
            if not _is_valid_covering(counts):
                for c in itertools.combinations(new_b, t):
                    idx = triple_to_idx[frozenset(c)]
                    counts[idx] -= 1
                for c in itertools.combinations(old_b, t):
                    idx = triple_to_idx[frozenset(c)]
                    counts[idx] += 1
                step += 1
                continue
            
            blocks[i] = new_b
            step += 1
    
    # Final pass: remove any remaining redundant blocks
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            removable = True
            for c in itertools.combinations(blocks[i], t):
                idx = triple_to_idx[frozenset(c)]
                if counts[idx] <= 1:
                    removable = False
                    break
            if removable:
                removed = blocks.pop(i)
                for c in itertools.combinations(removed, t):
                    idx = triple_to_idx[frozenset(c)]
                    counts[idx] -= 1
                current_n -= 1
                changed = True
                if current_n < best_n:
                    best_n = current_n
                    best_blocks = list(blocks)
                break
    
    return best_blocks, best_n


def _local_search(blocks, v, k, t, time_limit, triples, triple_to_idx):
    """
    Local search: try removing redundant blocks, then try swaps.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    def compute_counts(blk_list):
        counts = [0] * len(triples)
        for b in blk_list:
            for c in itertools.combinations(b, t):
                idx = triple_to_idx[frozenset(c)]
                counts[idx] += 1
        return counts
    
    def is_valid_counts(counts):
        return all(c >= 1 for c in counts)
    
    def try_remove(blk_list):
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
        counts = compute_counts(blk_list)
        n = len(blk_list)
        if n == 0:
            return None
        
        i = rng.randrange(n)
        old_b = blk_list[i]
        old_set = set(old_b)
        
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
        
        swapped = try_swap(blocks)
        if swapped is not None:
            blocks = swapped
            if len(blocks) < best_n:
                best_n = len(blocks)
                best = list(blocks)
                no_improve = 0
            else:
                no_improve += 1
        
        no_improve += 1
        if no_improve > 200:
            break
    
    return best, best_n


def construct(v, k, t):
    start_time = time.time()
    time_limit = 140.0
    
    triples, triple_to_idx = _precompute_triple_indices(v, t)
    
    best_blocks = None
    best_n = float('inf')
    
    # Strategy 1: Scarcity-weighted greedy
    for idx in range(2):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 60:
            break
        
        rng = random.Random(77771 + idx * 31337)
        remaining_for_greedy = min(time_limit - 60 - elapsed, 25.0)
        blocks = _scarcity_weighted_greedy(v, k, t, rng, triples, triple_to_idx, time_limit=remaining_for_greedy)
        
        if blocks and len(blocks) < best_n:
            best_n = len(blocks)
            best_blocks = blocks
    
    # Strategy 2: ABFKP
    for idx in range(2):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 50:
            break
        
        rng = random.Random(99991 + idx * 31337)
        remaining_for_abfkp = min(time_limit - 50 - elapsed, 20.0)
        blocks = _abfkp_cover(v, k, t, rng, triples, triple_to_idx, time_limit=remaining_for_abfkp)
        
        if blocks and len(blocks) < best_n:
            best_n = len(blocks)
            best_blocks = blocks
    
    # Strategy 3: Simulated annealing on best result
    if best_blocks is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 10:
            remaining_sa = min(time_limit - 10 - elapsed, 40.0)
            rng_sa = random.Random(123456)
            sa_blocks, sa_n = _simulated_annealing(best_blocks, v, k, t, triples, triple_to_idx, remaining_sa, rng_sa)
            if sa_n < best_n:
                best_n = sa_n
                best_blocks = sa_blocks
    
    # Final local search polish
    if best_blocks is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - 5:
            remaining_ls = min(time_limit - 5 - elapsed, 15.0)
            ls_blocks, ls_n = _local_search(best_blocks, v, k, t, remaining_ls, triples, triple_to_idx)
            if ls_n < best_n:
                best_n = ls_n
                best_blocks = ls_blocks
    
    # Verify and return
    if best_blocks is None:
        # Emergency fallback: greedy without time limit
        rng = random.Random(1)
        best_blocks = _scarcity_weighted_greedy(v, k, t, rng, triples, triple_to_idx, time_limit=None)
    
    # Final verification
    covered = set()
    for b in best_blocks:
        for c in itertools.combinations(b, t):
            covered.add(frozenset(c))
    if len(covered) < len(triples):
        # Patch any missing triples
        missing = [tr for tr in triples if frozenset(tr) not in covered]
        for tr in missing:
            block = list(tr)
            while len(block) < k:
                p = random.Random(hash(tr) + len(block)).randrange(v)
                if p not in block:
                    block.append(p)
            block.sort()
            b_tuple = tuple(block)
            if b_tuple not in best_blocks:
                best_blocks.append(b_tuple)
    
    return [list(b) for b in best_blocks]


if __name__ == "__main__":
    import time
    start = time.time()
    result = construct(24, 7, 3)
    elapsed = time.time() - start
    print(f"Blocks: {len(result)}, Time: {elapsed:.2f}s")
    covered = set()
    for b in result:
        for c in itertools.combinations(b, 3):
            covered.add(frozenset(c))
    total = math.comb(24, 3)
    print(f"Covered: {len(covered)}/{total}")
    assert len(covered) == total, "Invalid covering!"
