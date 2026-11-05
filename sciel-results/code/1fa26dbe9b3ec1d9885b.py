import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Precompute for each triple, the list of points that can complete it to a k-block
    # triple -> list of (point_a, point_b) pairs such that triple + {a,b} has k distinct elements
    # Actually, for k=7, t=3, we need 4 more points. That's C(21,4) = 5985 completions per triple.
    # Too many. Instead, let's precompute for each pair of points, which triples they're in.
    
    # Better: For greedy, given an uncovered triple, find a block containing it that covers the most other uncovered triples.
    # We'll use a different strategy: precompute block_bitsets for a SUBSET of blocks (the cyclic orbits).
    
    # Strategy:
    # 1. Generate candidate blocks via cyclic shifts of small seed sets.
    # 2. Greedily add blocks to cover all triples.
    # 3. Local search: 1-swap, 2-swap, redundancy removal.
    
    # Step 1: Generate cyclic orbit blocks
    # A cyclic shift of a block B = {b0, b1, ..., b6} gives {b0+s, b1+s, ..., b6+s} mod 24
    # Try many random seed blocks and take their orbits.
    
    candidate_blocks = set()
    
    # Add some structured seeds
    # Seed 1: {0,1,2,3,4,5,6}
    # Seed 2: {0,1,2,4,8,16,12} (geometric-ish)
    # Seed 3: {0,3,6,9,12,15,18} (arithmetic)
    # etc.
    
    seeds = [
        (0, 1, 2, 3, 4, 5, 6),
        (0, 1, 2, 4, 8, 16, 12),
        (0, 3, 6, 9, 12, 15, 18),
        (0, 1, 3, 7, 12, 18, 21),
        (0, 2, 4, 8, 14, 20, 10),
        (0, 1, 4, 10, 15, 21, 5),
    ]
    
    # Add random seeds
    for _ in range(200):
        seed = tuple(sorted(random.sample(range(v), k)))
        seeds.append(seed)
    
    for seed in seeds:
        for s in range(v):
            block = tuple(sorted((x + s) % v for x in seed))
            candidate_blocks.add(block)
    
    # Also add some random blocks
    for _ in range(5000):
        block = tuple(sorted(random.sample(range(v), k)))
        candidate_blocks.add(block)
    
    # Convert to list
    candidate_blocks = list(candidate_blocks)
    n_candidates = len(candidate_blocks)
    
    # Precompute bitsets for candidate blocks
    block_bitsets = []
    for b in candidate_blocks:
        bitset = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            bitset |= (1 << idx)
        block_bitsets.append(bitset)
    
    # Step 2: Greedy construction
    uncovered = (1 << num_triples) - 1
    selected = []
    selected_set = set()
    
    while uncovered != 0:
        best_gain = -1
        best_idx = -1
        
        # For speed, if we have many candidates, sample a subset
        if n_candidates > 50000 and time.time() > deadline - 30:
            # Sample candidates
            sample_size = min(10000, n_candidates)
            sample_indices = random.sample(range(n_candidates), sample_size)
            candidates_to_check = sample_indices
        else:
            candidates_to_check = range(n_candidates)
        
        for i in candidates_to_check:
            if i in selected_set:
                continue
            gain = bin(block_bitsets[i] & uncovered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain <= 0:
            # No improvement possible from current candidates
            # Add a random block that covers at least one uncovered triple
            # Find an uncovered triple and make a block containing it
            uncovered_triples = [i for i in range(num_triples) if (uncovered >> i) & 1]
            if not uncovered_triples:
                break
            tri_idx = random.choice(uncovered_triples)
            tri = all_triples[tri_idx]
            # Make a random block containing this triple
            remaining = [x for x in range(v) if x not in tri]
            extra = random.sample(remaining, k - t)
            block = tuple(sorted(list(tri) + extra))
            
            # Add to candidates if not present
            if block not in candidate_blocks:
                candidate_blocks.append(block)
                bitset = 0
                for tri2 in itertools.combinations(block, t):
                    idx = triple_to_idx[tri2]
                    bitset |= (1 << idx)
                block_bitsets.append(bitset)
                n_candidates += 1
                best_idx = n_candidates - 1
            else:
                best_idx = candidate_blocks.index(block)
            
            if best_idx in selected_set:
                # Already selected, skip
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_bitsets[best_idx]
        else:
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~block_bitsets[best_idx]
        
        if time.time() > deadline - 20:
            break
    
    # Convert to block list
    blocks = [candidate_blocks[i] for i in selected]
    
    # Step 3: Local search
    # Helper: compute coverage count for each triple
    def compute_counts(block_list):
        counts = {}
        for b in block_list:
            for tri in itertools.combinations(b, t):
                counts[tri] = counts.get(tri, 0) + 1
        return counts
    
    def is_valid_covering(block_list):
        """Check if every triple is covered."""
        covered = set()
        for b in block_list:
            for tri in itertools.combinations(b, t):
                covered.add(tri)
        return len(covered) == num_triples
    
    def try_remove_redundant(block_list):
        """Remove blocks that are redundant (all their triples are covered by other blocks)."""
        if not block_list:
            return block_list
        
        counts = compute_counts(block_list)
        active = [True] * len(block_list)
        changed = True
        max_iterations = 50
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            indices = list(range(len(block_list)))
            random.shuffle(indices)
            for i in indices:
                if not active[i]:
                    continue
                b = block_list[i]
                can_remove = True
                for tri in itertools.combinations(b, t):
                    if counts.get(tri, 0) <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active[i] = False
                    changed = True
                    for tri in itertools.combinations(b, t):
                        counts[tri] -= 1
        
        return [block_list[i] for i in range(len(block_list)) if active[i]]
    
    def try_single_swap(block_list):
        """Try to replace one block with another that covers more uncovered triples."""
        if len(block_list) < 2:
            return block_list
        
        counts = compute_counts(block_list)
        
        # For each block, compute which triples it uniquely covers
        # Try to find a replacement that covers all uniquely covered triples plus more
        
        # Build set of all candidate blocks not in current list
        current_set = set(block_list)
        all_candidates = list(candidate_blocks)
        
        improved = False
        for i in range(len(block_list)):
            if time.time() > deadline - 10:
                break
            b_old = block_list[i]
            
            # Triples uniquely covered by b_old
            unique_triples = []
            for tri in itertools.combinations(b_old, t):
                if counts.get(tri, 0) == 1:
                    unique_triples.append(tri)
            
            if not unique_triples:
                # Can just remove
                continue
            
            # Try to find a replacement block that covers all unique_triples
            # and has more total coverage
            best_new_block = None
            best_gain = -1
            
            # Sample candidates
            sample_size = min(2000, len(all_candidates))
            sample = random.sample(all_candidates, sample_size) if len(all_candidates) > sample_size else all_candidates
            
            for b_new in sample:
                if b_new in current_set and b_new != b_old:
                    continue
                if b_new == b_old:
                    continue
                
                # Check if b_new covers all unique_triples
                new_triples = set(itertools.combinations(b_new, t))
                if not all(tri in new_triples for tri in unique_triples):
                    continue
                
                # Compute gain: how many additional triples does b_new cover that aren't already covered?
                # Actually, we're replacing b_old with b_new. The net change is:
                # Triples in b_new but not in b_old (and not covered by others)
                # vs triples in b_old but not in b_new (and not covered by others)
                
                old_triples = set(itertools.combinations(b_old, t))
                
                # Triples that would become uncovered: in b_old, not in b_new, and count==1
                would_uncover = 0
                for tri in old_triples:
                    if tri not in new_triples and counts.get(tri, 0) == 1:
                        would_uncover += 1
                
                if would_uncover > 0:
                    continue
                
                # Triples that would become covered: in b_new, not in b_old, and count==0
                would_cover = 0
                for tri in new_triples:
                    if tri not in old_triples and counts.get(tri, 0) == 0:
                        would_cover += 1
                
                if would_cover > best_gain:
                    best_gain = would_cover
                    best_new_block = b_new
            
            if best_new_block is not None and best_gain > 0:
                block_list[i] = best_new_block
                current_set.discard(b_old)
                current_set.add(best_new_block)
                # Update counts
                for tri in itertools.combinations(b_old, t):
                    counts[tri] -= 1
                for tri in itertools.combinations(best_new_block, t):
                    counts[tri] = counts.get(tri, 0) + 1
                improved = True
        
        return block_list
    
    def try_two_swap(block_list):
        """Try to replace two blocks with two other blocks."""
        if len(block_list) < 4:
            return block_list
        
        counts = compute_counts(block_list)
        current_set = set(block_list)
        all_candidates = list(candidate_blocks)
        
        # Try random pairs
        n_pairs = min(500, len(block_list) * (len(block_list) - 1) // 2)
        
        for _ in range(n_pairs):
            if time.time() > deadline - 10:
                break
            
            i, j = random.sample(range(len(block_list)), 2)
            b1_old = block_list[i]
            b2_old = block_list[j]
            
            # Unique triples covered by {b1_old, b2_old}
            old_triples = set(itertools.combinations(b1_old, t)) | set(itertools.combinations(b2_old, t))
            
            unique_triples = []
            for tri in old_triples:
                # Count how many of {b1_old, b2_old} cover this triple
                in_b1 = tri in set(itertools.combinations(b1_old, t))
                in_b2 = tri in set(itertools.combinations(b2_old, t))
                local_count = (1 if in_b1 else 0) + (1 if in_b2 else 0)
                total_count = counts.get(tri, 0)
                if total_count == local_count:
                    # This triple is only covered by b1_old and/or b2_old
                    unique_triples.append(tri)
            
            if not unique_triples:
                continue
            
            # Try to find two replacement blocks that cover all unique_triples
            sample_size = min(1000, len(all_candidates))
            sample = random.sample(all_candidates, sample_size) if len(all_candidates) > sample_size else all_candidates
            
            best_pair = None
            best_gain = -1
            
            # Try pairs from sample
            for a in range(len(sample)):
                if time.time() > deadline - 10:
                    break
                b_new1 = sample[a]
                if b_new1 in current_set and b_new1 not in (b1_old, b2_old):
                    continue
                
                triples1 = set(itertools.combinations(b_new1, t))
                
                for b in range(a + 1, len(sample)):
                    b_new2 = sample[b]
                    if b_new2 in current_set and b_new2 not in (b1_old, b2_old):
                        continue
                    if b_new2 == b_new1:
                        continue
                    
                    triples2 = set(itertools.combinations(b_new2, t))
                    new_triples = triples1 | triples2
                    
                    # Check if all unique_triples are covered
                    if not all(tri in new_triples for tri in unique_triples):
                        continue
                    
                    # Compute gain
                    would_cover = 0
                    for tri in new_triples:
                        if tri not in old_triples and counts.get(tri, 0) == 0:
                            would_cover += 1
                    
                    if would_cover > best_gain:
                        best_gain = would_cover
                        best_pair = (b_new1, b_new2)
            
            if best_pair is not None and best_gain > 0:
                block_list[i] = best_pair[0]
                block_list[j] = best_pair[1]
                current_set.discard(b1_old)
                current_set.discard(b2_old)
                current_set.add(best_pair[0])
                current_set.add(best_pair[1])
                
                # Update counts
                for tri in itertools.combinations(b1_old, t):
                    counts[tri] -= 1
                for tri in itertools.combinations(b2_old, t):
                    counts[tri] -= 1
                for tri in itertools.combinations(best_pair[0], t):
                    counts[tri] = counts.get(tri, 0) + 1
                for tri in itertools.combinations(best_pair[1], t):
                    counts[tri] = counts.get(tri, 0) + 1
        
        return block_list
    
    # Run local search passes
    # Pass 1: Redundancy removal
    blocks = try_remove_redundant(blocks)
    
    # Pass 2: Single swaps
    if time.time() < deadline - 15:
        blocks = try_single_swap(blocks)
        blocks = try_remove_redundant(blocks)
    
    # Pass 3: Two swaps
    if time.time() < deadline - 10:
        blocks = try_two_swap(blocks)
        blocks = try_remove_redundant(blocks)
    
    # Pass 4: Another round of single swaps
    if time.time() < deadline - 5:
        blocks = try_single_swap(blocks)
        blocks = try_remove_redundant(blocks)
    
    # Final verification: ensure validity
    # If not valid, add blocks to cover missing triples
    if time.time() < deadline - 2:
        covered = set()
        for b in blocks:
            for tri in itertools.combinations(b, t):
                covered.add(tri)
        
        missing = set(all_triples) - covered
        while missing:
            if time.time() > deadline - 1:
                break
            tri = random.choice(list(missing))
            remaining = [x for x in range(v) if x not in tri]
            extra = random.sample(remaining, k - t)
            block = tuple(sorted(list(tri) + extra))
            blocks.append(block)
            
            # Update covered
            for tri2 in itertools.combinations(block, t):
                covered.add(tri2)
                if tri2 in missing:
                    missing.discard(tri2)
    
    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique_blocks.append(b)
    
    # Return as list of lists
    return [list(b) for b in unique_blocks]
