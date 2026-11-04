import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_blocks = None
    
    def build_counts(blocks):
        counts = {}
        for b in blocks:
            for tri in itertools.combinations(b, t):
                counts[tri] = counts.get(tri, 0) + 1
        return counts

    def try_remove_redundant(blocks, counts):
        """Try to remove redundant blocks. Returns new list of blocks and updated counts."""
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            for i in indices:
                if not active[i]:
                    continue
                b = blocks[i]
                can_remove = True
                for tri in itertools.combinations(b, t):
                    if counts[tri] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active[i] = False
                    changed = True
                    for tri in itertools.combinations(b, t):
                        counts[tri] -= 1
        return [blocks[i] for i in range(len(blocks)) if active[i]], counts

    def try_replace_blocks(blocks, counts):
        """Try to replace a block with a better one that covers more under-covered triples."""
        n = len(blocks)
        if n == 0:
            return blocks, counts
        
        # Build set of all triples for quick lookup
        all_triples = set(itertools.combinations(range(v), t))
        
        for i in range(n):
            if time.time() > deadline - 2:
                break
            old_block = blocks[i]
            old_triples = set(itertools.combinations(old_block, t))
            
            # Count how many triples in old_block are covered only by this block
            only_covered = 0
            for tri in old_triples:
                if counts[tri] == 1:
                    only_covered += 1
            
            if only_covered == 0:
                continue  # Block is redundant, will be handled by removal
            
            # Try to find a replacement block that covers at least as many unique triples
            # A replacement is beneficial if it covers triples that old_block doesn't,
            # while still covering all the triples that only old_block covers.
            
            # Get the set of triples that MUST be covered (only covered by old_block)
            must_cover = set(tri for tri in old_triples if counts[tri] == 1)
            
            if not must_cover:
                continue
            
            # Try random replacement blocks
            best_gain = 0
            best_block = None
            
            for attempt in range(50):
                # Generate a random block that contains all must_cover triples
                # First, find the union of all must_cover triples
                must_points = set()
                for tri in must_cover:
                    must_points.update(tri)
                
                if len(must_points) > k:
                    break  # Can't fit all must-cover points in one block
                
                # Fill remaining spots randomly
                remaining = k - len(must_points)
                available = [x for x in range(v) if x not in must_points]
                
                if remaining > len(available):
                    break
                
                new_block_set = set(must_points)
                for _ in range(remaining):
                    new_block_set.add(random.choice(available))
                
                new_block = sorted(new_block_set)
                
                # Calculate gain: how many new triples does this block cover?
                new_triples = set(itertools.combinations(new_block, t))
                old_triple_set = set(itertools.combinations(old_block, t))
                
                # Gain = triples covered by new but not old, minus triples lost
                # But we must still cover all must_cover triples
                if not must_cover.issubset(new_triples):
                    continue
                
                new_unique = new_triples - old_triple_set
                old_unique = old_triple_set - new_triples
                
                # Net gain in coverage
                gain = len(new_unique) - len(old_unique)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = new_block
            
            if best_block is not None and best_gain > 0:
                # Update counts
                old_triple_set = set(itertools.combinations(old_block, t))
                new_triple_set = set(itertools.combinations(best_block, t))
                
                for tri in old_triple_set:
                    counts[tri] -= 1
                for tri in new_triple_set:
                    counts[tri] = counts.get(tri, 0) + 1
                
                blocks[i] = best_block
        
        return blocks, counts

    def optimize_blocks(blocks):
        if not blocks:
            return []
        
        counts = build_counts(blocks)
        
        # Phase 1: Remove redundant blocks
        blocks, counts = try_remove_redundant(blocks, counts)
        
        # Phase 2: Try to replace blocks with better ones
        blocks, counts = try_replace_blocks(blocks, counts)
        
        # Phase 3: Remove redundant blocks again (replacement may have created redundancy)
        blocks, counts = try_remove_redundant(blocks, counts)
        
        # Phase 4: Try replacement again
        blocks, counts = try_replace_blocks(blocks, counts)
        
        # Phase 5: Final removal
        blocks, counts = try_remove_redundant(blocks, counts)
        
        return blocks

    def greedy_construct():
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            start_triple = random.choice(list(uncovered))
            block = list(start_triple)
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                gains = []
                for x in candidates:
                    gain = 0
                    for pair in itertools.combinations(block, t-1):
                        tri = tuple(sorted(pair + (x,)))
                        if tri in uncovered:
                            gain += 1
                    gains.append((gain, x))
                
                if not gains:
                    break
                    
                max_gain = max(g for g, _ in gains)
                best_candidates = [x for g, x in gains if g == max_gain]
                
                if max_gain > 0:
                    block.append(random.choice(best_candidates))
                else:
                    block.append(random.choice(candidates))
            
            block.sort()
            blocks.append(block)
            
            for tri in itertools.combinations(block, t):
                if tri in uncovered:
                    uncovered.remove(tri)
                    
        return blocks

    # Main Search Loop
    while time.time() < deadline:
        cand = greedy_construct()
        opt_cand = optimize_blocks(cand)
        
        if best_blocks is None or len(opt_cand) < len(best_blocks):
            best_blocks = opt_cand
            
        if len(best_blocks) <= 69:
            break

    return best_blocks
