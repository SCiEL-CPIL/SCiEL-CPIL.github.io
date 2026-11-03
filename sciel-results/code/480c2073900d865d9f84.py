import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    # Parameters: v=24, k=7, t=3
    # But use passed args.
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute triples and indices
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # For fast access: idx_3d[a,b,c] for sorted a,b,c
    idx_3d = np.full((v, v, v), -1, dtype=np.int32)
    for a in range(v):
        for b in range(v):
            if a == b: continue
            for c in range(v):
                if a == c or b == c: continue
                t1, t2, t3 = sorted((a, b, c))
                idx_3d[a, b, c] = triple_to_idx[(t1, t2, t3)]

    def get_triple_idx(a, b, c):
        return idx_3d[a, b, c]

    def greedy_cover(uncovered_mask=None, random_state=0):
        """
        Greedily add blocks to cover all triples.
        If uncovered_mask is provided, only cover those (for repair).
        """
        rng = random.Random(random_state)
        if uncovered_mask is None:
            uncovered = np.ones(num_triples, dtype=np.uint8)
            num_uncovered = num_triples
        else:
            uncovered = uncovered_mask.copy()
            num_uncovered = np.sum(uncovered)
            
        blocks = []
        points = list(range(v))
        
        while num_uncovered > 0:
            # Find first uncovered triple
            start_idx = int(np.argmax(uncovered))
            if uncovered[start_idx] == 0:
                break
                
            target_triple = triples[start_idx]
            block = list(target_triple)
            block_set = set(block)
            
            # Mark as covered
            uncovered[start_idx] = 0
            num_uncovered -= 1
            
            # Add remaining k - t points
            while len(block) < k:
                candidates = [x for x in points if x not in block_set]
                if not candidates:
                    break
                    
                # Shuffle for random tie-breaking
                rng.shuffle(candidates)
                
                best_gain = -1
                best_point = -1
                
                # Precompute pairs in current block
                block_pairs = list(itertools.combinations(block, 2))
                
                for x in candidates:
                    gain = 0
                    for (p1, p2) in block_pairs:
                        idx = get_triple_idx(x, p1, p2)
                        if uncovered[idx]:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        # Early exit if we covered all possible new triples for this step?
                        # Max possible gain is len(block_pairs). If we hit it, it's optimal.
                        if gain == len(block_pairs):
                            break
                            
                if best_point == -1:
                    break
                
                block.append(best_point)
                block_set.add(best_point)
                
                # Update uncovered status
                # New triples are (best_point, p1, p2) for p1,p2 in block before adding best_point?
                # No, for all pairs in the final block that include best_point?
                # Actually, adding best_point creates new triples (best_point, u, v) where u,v are in block_set BEFORE adding best_point?
                # Wait, the block is built incrementally.
                # When we add best_point to block B, the new triples covered are (best_point, u, v) where u,v in B.
                # My loop above calculated gain based on `block_pairs` which was computed BEFORE adding best_point.
                # So I need to mark those specific triples as covered.
                
                for (p1, p2) in block_pairs:
                    idx = get_triple_idx(best_point, p1, p2)
                    if uncovered[idx]:
                        uncovered[idx] = 0
                        num_uncovered -= 1
                        
            blocks.append(tuple(sorted(block)))
            
        return blocks, uncovered, num_uncovered

    def trim_blocks(blocks):
        """
        Remove redundant blocks.
        """
        if not blocks:
            return blocks
            
        # Count frequency of each triple
        counts = np.zeros(num_triples, dtype=np.int32)
        block_triples = []
        
        for block in blocks:
            b_triples = []
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                counts[idx] += 1
                b_triples.append(idx)
            block_triples.append(b_triples)
            
        active = [True] * len(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if not active[i]:
                    continue
                essential = False
                for idx in block_triples[i]:
                    if counts[idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    changed = True
                    for idx in block_triples[i]:
                        counts[idx] -= 1
                        
        return [blocks[i] for i in range(len(blocks)) if active[i]]

    def is_valid(blocks):
        """
        Check if all triples are covered.
        """
        covered = np.zeros(num_triples, dtype=np.uint8)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                idx = triple_to_idx[triple]
                covered[idx] = 1
        return np.all(covered == 1)

    # Main Loop
    best_blocks = None
    best_len = float('inf')
    
    seed_counter = 0
    
    # Initial greedy runs
    while time.time() < deadline - 10:
        blocks, uncovered, num_uncovered = greedy_cover(random_state=seed_counter)
        
        # If greedy didn't cover everything (shouldn't happen with this logic, but just in case)
        if num_uncovered > 0:
             # This shouldn't happen if k>=t and v is large enough, but handle gracefully
             pass

        trimmed = trim_blocks(blocks)
        
        if len(trimmed) < best_len:
            best_len = len(trimmed)
            best_blocks = list(trimmed)
            
        if best_len <= 69:
            break
            
        # Perturbation: Kick
        if time.time() < deadline - 5:
            # Pick a random subset of blocks to remove
            # We want to remove blocks that might be "wasteful"
            # Simply remove 1-3 random blocks
            num_kicks = random.Random(seed_counter).randint(1, 3)
            kick_indices = random.Random(seed_counter + 100).sample(range(len(trimmed)), min(num_kicks, len(trimmed)))
            
            # Create a new list without these blocks
            remaining_blocks = [b for i, b in enumerate(trimmed) if i not in kick_indices]
            
            # Calculate uncovered triples after removal
            # We need to recalculate counts or just mark what is covered by remaining
            covered_mask = np.zeros(num_triples, dtype=np.uint8)
            for block in remaining_blocks:
                for triple in itertools.combinations(block, t):
                    idx = triple_to_idx[triple]
                    covered_mask[idx] = 1
            
            uncovered_mask = 1 - covered_mask
            
            # Repair: Greedily add blocks to cover uncovered_mask
            # We pass the mask to greedy
            repaired_blocks, _, _ = greedy_cover(uncovered_mask=uncovered_mask, random_state=seed_counter + 200)
            
            # Combine
            combined = remaining_blocks + repaired_blocks
            
            # Trim the combined set
            trimmed_combined = trim_blocks(combined)
            
            if len(trimmed_combined) < best_len:
                best_len = len(trimmed_combined)
                best_blocks = list(trimmed_combined)
                
        seed_counter += 1

    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
