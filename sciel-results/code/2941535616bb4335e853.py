import itertools
import random
import time
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # safety margin for 120s wall-clock

    rng = random.Random(12345)

    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    triple_index = {tri: i for i, tri in enumerate(all_triples)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, store the list of triple-indices it covers
    block_triple_indices = []
    for b in all_blocks:
        idxs = [triple_index[tri] for tri in itertools.combinations(b, t)]
        block_triple_indices.append(idxs)

    # For each triple, store the list of block-indices that cover it
    triple_block_indices = [[] for _ in range(n_triples)]
    for bi, b in enumerate(all_blocks):
        for ti in block_triple_indices[bi]:
            triple_block_indices[ti].append(bi)

    def time_ok(margin=2):
        return time.time() < deadline - margin

    # --- Structured seed: round-robin tournament schedule ---
    def round_robin_seed():
        """Generate blocks via round-robin scheduling on v points, padding with random points to size k."""
        n = v
        # If n is odd, add a dummy point for scheduling
        use_dummy = (n % 2 == 1)
        m = n + 1 if use_dummy else n
        points = list(range(n))
        if use_dummy:
            points.append(-1)  # dummy

        # Round-robin: fix point 0, rotate the rest
        half = m // 2
        schedule = []  # list of rounds, each round is list of pairs
        for round_idx in range(m - 1):
            round_pairs = []
            # Points in this round: 0, and 1..m-1 rotated
            # Standard round-robin: for round r, pair i with (m-1-i) in the rotated list
            # Use the "circle method"
            circle = [0] + list(range(1, m))
            # Rotate the inner part by round_idx
            inner = circle[1:]
            inner_rot = inner[round_idx:] + inner[:round_idx]
            rot_circle = [0] + inner_rot
            for i in range(half):
                a = rot_circle[i]
                b = rot_circle[m - 1 - i]
                round_pairs.append((a, b))
            schedule.append(round_pairs)

        # Now build blocks: for each round, each pair is a 2-set.
        # We need 7-sets. Group pairs and add extra points.
        # Strategy: for each round, we have 'half' pairs. 
        # Combine pairs to form larger blocks.
        # Since k=7, and pairs are size 2, we can't directly combine.
        # Instead, use the schedule to define a 1-factorization.
        # Each round is a perfect matching.
        # We can take all pairs in a round and assign them to blocks.
        # But we need 7-element blocks.
        
        # Alternative: use the cyclic group Z_v if v is prime or suitable.
        # For v=24, use cyclic shifts of a base block.
        # Base block: {0, 1, 4, 10, 12, 17, 23} (arbitrary, will be optimized)
        # But cyclic shifts of a 7-set on 24 points: 24 blocks.
        # This is too few to cover all triples likely.
        
        # Let's use a different structured approach: 
        # Take all blocks from a Steiner-like system approximation.
        # For now, just return a random set of blocks as seed, 
        # but structured: pick blocks with high "spread".
        
        # Actually, let's just use a simple deterministic seed:
        # Take every 7th block from the lexicographic list, etc.
        # Or better: use the round-robin pairs to form 7-sets by combining 3 pairs + 1 point.
        # This is complex. Let's use a simpler seed: 
        # Start with a greedy seed using a fixed order.
        
        # For this implementation, I'll use a "cyclic" seed if possible, 
        # otherwise a deterministic greedy.
        
        # Cyclic seed for v=24:
        # Base block B0 = {0, 1, 4, 10, 12, 17, 23}
        # Generate 24 blocks: (B0 + i) mod 24 for i in 0..23
        base = [0, 1, 4, 10, 12, 17, 23]
        blocks = []
        seen = set()
        for i in range(v):
            b = tuple(sorted((x + i) % v for x in base))
            if b not in seen:
                seen.add(b)
                blocks.append(b)
        
        # This gives at most 24 blocks. Likely not a covering.
        # So we use this as a seed and let the greedy fill in the rest.
        return blocks

    def greedy_seed():
        """Deterministic greedy: always pick the block with max uncovered coverage."""
        uncovered = set(range(n_triples))
        blocks = []
        used_blocks = set()
        
        # Precompute block coverage counts for speed
        # We'll use a list of (block_idx, gain) and sort at each step (slow)
        # Or maintain a priority queue.
        
        while uncovered and time_ok(5):
            best_bi = -1
            best_gain = 0
            for bi in range(n_blocks):
                if bi in used_blocks:
                    continue
                gain = 0
                for ti in block_triple_indices[bi]:
                    if ti in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            
            if best_bi == -1 or best_gain == 0:
                break
            
            blocks.append(all_blocks[best_bi])
            used_blocks.add(best_bi)
            for ti in block_triple_indices[best_bi]:
                uncovered.discard(ti)
        
        return blocks

    def remove_redundant(blocks):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return []
        
        # Count coverage
        counts = [0] * n_triples
        block_set = set(blocks)
        for b in blocks:
            bi = all_blocks.index(b)  # Slow, but blocks list is small
            for ti in block_triple_indices[bi]:
                counts[ti] += 1
        
        active = [True] * len(blocks)
        changed = True
        while changed and time_ok(3):
            changed = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                b = blocks[i]
                bi = all_blocks.index(b)
                can_remove = True
                for ti in block_triple_indices[bi]:
                    if counts[ti] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active[i] = False
                    changed = True
                    for ti in block_triple_indices[bi]:
                        counts[ti] -= 1
        
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def local_search(blocks, max_iters=200):
        """Try to replace or remove blocks to reduce count."""
        if not blocks:
            return blocks
        
        # Convert to indices for speed
        block_indices = []
        for b in blocks:
            block_indices.append(all_blocks.index(b))
        
        # Coverage counts
        counts = [0] * n_triples
        for bi in block_indices:
            for ti in block_triple_indices[bi]:
                counts[ti] += 1
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time_ok(5):
            improved = False
            iters += 1
            
            # Try removing a block
            for i in range(len(block_indices)):
                bi = block_indices[i]
                can_remove = True
                for ti in block_triple_indices[bi]:
                    if counts[ti] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove it
                    for ti in block_triple_indices[bi]:
                        counts[ti] -= 1
                    block_indices.pop(i)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping a block with another that allows further reduction
            # This is expensive. Instead, try adding a block and then removing two.
            # Or just try random swaps.
            
            # Pick a random block to replace
            i = rng.randint(0, len(block_indices) - 1)
            old_bi = block_indices[i]
            
            # Find a candidate new block
            # Pick a random block not in the set
            new_bi = rng.randint(0, n_blocks - 1)
            if new_bi in block_indices:
                continue
            
            # Check if swapping is valid (all triples still covered)
            # First, remove old_bi coverage
            temp_counts = counts.copy()
            for ti in block_triple_indices[old_bi]:
                temp_counts[ti] -= 1
            
            # Check if any triple becomes uncovered
            valid = True
            for ti in block_triple_indices[old_bi]:
                if temp_counts[ti] == 0:
                    # This triple is only covered by old_bi
                    # So new_bi must cover it
                    if ti not in block_triple_indices[new_bi]:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Add new_bi coverage
            for ti in block_triple_indices[new_bi]:
                temp_counts[ti] += 1
            
            # Now try to remove another block
            removed_second = False
            for j in range(len(block_indices)):
                if j == i:
                    continue
                bj = block_indices[j]
                can_remove = True
                for ti in block_triple_indices[bj]:
                    if temp_counts[ti] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove bj
                    for ti in block_triple_indices[bj]:
                        temp_counts[ti] -= 1
                    block_indices.pop(j)
                    # Update old_bi to new_bi
                    block_indices[i] = new_bi
                    counts = temp_counts
                    improved = True
                    removed_second = True
                    break
            
            if removed_second:
                break
        
        return [all_blocks[bi] for bi in block_indices]

    # --- Main ---
    # 1. Generate a seed
    seed_blocks = greedy_seed()
    if not seed_blocks:
        seed_blocks = round_robin_seed()
    
    # 2. Remove redundancies
    seed_blocks = remove_redundant(seed_blocks)
    
    best_blocks = seed_blocks
    best_count = len(best_blocks)
    
    # 3. Local search
    if time_ok(10):
        ls_blocks = local_search(seed_blocks, max_iters=100)
        if len(ls_blocks) < best_count:
            best_blocks = ls_blocks
            best_count = len(ls_blocks)
    
    # 4. Try a few random restarts of greedy with different random orders
    for restart in range(5):
        if not time_ok(10):
            break
        
        # Perturb the greedy: start with a random set of blocks
        random_seed = []
        n_init = rng.randint(20, 50)
        random_seed = [all_blocks[rng.randint(0, n_blocks - 1)] for _ in range(n_init)]
        random_seed = list(set(random_seed))
        random_seed = remove_redundant(random_seed)
        
        # Now complete with greedy
        uncovered = set(range(n_triples))
        for b in random_seed:
            bi = all_blocks.index(b)
            for ti in block_triple_indices[bi]:
                uncovered.discard(ti)
        
        blocks = list(random_seed)
        used_blocks = set(all_blocks.index(b) for b in random_seed)
        
        while uncovered and time_ok(5):
            best_bi = -1
            best_gain = 0
            for bi in range(n_blocks):
                if bi in used_blocks:
                    continue
                gain = 0
                for ti in block_triple_indices[bi]:
                    if ti in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            
            if best_bi == -1 or best_gain == 0:
                break
            
            blocks.append(all_blocks[best_bi])
            used_blocks.add(best_bi)
            for ti in block_triple_indices[best_bi]:
                uncovered.discard(ti)
        
        blocks = remove_redundant(blocks)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        
        # Local search on the new best
        if time_ok(5) and len(blocks) < best_count + 2:
            ls_blocks = local_search(blocks, max_iters=50)
            if len(ls_blocks) < best_count:
                best_blocks = ls_blocks
                best_count = len(ls_blocks)
    
    # 5. Final cleanup
    best_blocks = remove_redundant(best_blocks)
    
    # Verify and fill if necessary
    covered = set()
    for b in best_blocks:
        bi = all_blocks.index(b)
        for ti in block_triple_indices[bi]:
            covered.add(ti)
    
    uncovered = set(range(n_triples)) - covered
    while uncovered and time_ok(1):
        ti = next(iter(uncovered))
        for bi in triple_block_indices[ti]:
            if all_blocks[bi] not in best_blocks:
                best_blocks.append(all_blocks[bi])
                for ti2 in block_triple_indices[bi]:
                    uncovered.discard(ti2)
                break
    
    return best_blocks

if __name__ == "__main__":
    blocks = construct(24, 7, 3)
    print(f"Found {len(blocks)} blocks")
    
    # Verify
    covered = set()
    for b in blocks:
        for tri in itertools.combinations(b, 3):
            covered.add(tri)
    
    all_tris = set(itertools.combinations(range(24), 3))
    missing = all_tris - covered
    if missing:
        print(f"Invalid! Missing {len(missing)} triples")
    else:
        print("Valid covering!")
