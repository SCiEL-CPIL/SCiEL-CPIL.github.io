"""Optimized covering C(24,7,3) using Information-Theoretic (Entropy Reduction) Greedy."""
import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using an Information-Theoretic approach:
    1. Entropy-Reduction Greedy: Prioritize covering 'scarce' or 'isolated' triples.
    2. Fast Local Search: Optimize using count-based redundancy removal and replacement.
    """
    all_triples = list(itertools.combinations(range(v), t))
    all_triples_set = set(all_triples)
    
    # Phase 1: Information-Theoretic Greedy
    blocks = _info_greedy(v, k, t, all_triples_set)
    
    # Phase 2: Local Search Optimization
    blocks = _local_search_fast(v, k, t, all_triples_set, blocks)
    
    return blocks

def _info_greedy(v, k, t, all_triples):
    """
    Greedy covering that maximizes entropy reduction.
    We approximate entropy reduction by weighting uncovered triples based on their
    'scarcity' or 'isolation'.
    """
    uncovered = set(all_triples)
    blocks = []
    
    while uncovered:
        # Calculate point frequencies in uncovered triples to identify "rare" points
        point_freq = [0] * v
        for triple in uncovered:
            for p in triple:
                point_freq[p] += 1
        
        # Score each uncovered triple by the sum of inverses of its points' frequencies.
        # High score = contains rare points = high priority.
        best_triple = None
        best_score = -1
        for triple in uncovered:
            score = 0.0
            for p in triple:
                score += 1.0 / (point_freq[p] + 1)
            if score > best_score:
                best_score = score
                best_triple = triple
        
        # Build a block around best_triple
        block = list(best_triple)
        
        # Extend the block by picking points that maximize the weighted gain
        while len(block) < k:
            best_point = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0.0
                # Calculate weighted gain of adding x
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        triple = tuple(sorted([block[i], block[j], x]))
                        if triple in uncovered:
                            # Weight this triple by rarity of its points
                            w = 0.0
                            for p in triple:
                                w += 1.0 / (point_freq[p] + 1)
                            gain += w
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
    return blocks

def _local_search_fast(v, k, t, all_triples_set, blocks, max_passes=100):
    """
    Fast local search using triple counts.
    1. Remove redundant blocks (all triples in block have count > 1).
    2. Replace blocks if their unique triples can be covered by a new, potentially better block.
    """
    blocks = [tuple(b) for b in blocks]
    n_blocks = len(blocks)
    
    # Pre-compute triples for each block
    block_triples = []
    for b in blocks:
        block_triples.append(list(itertools.combinations(b, t)))
    
    # Initialize counts
    triple_counts = defaultdict(int)
    for triples in block_triples:
        for tr in triples:
            triple_counts[tr] += 1
            
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        # Pass 1: Removal
        i = 0
        while i < len(blocks):
            b_triples = block_triples[i]
            can_remove = True
            for tr in b_triples:
                if triple_counts[tr] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block i
                for tr in b_triples:
                    triple_counts[tr] -= 1
                blocks.pop(i)
                block_triples.pop(i)
                improved = True
                # Do not increment i, check next block at same index
            else:
                i += 1
                
        # Pass 2: Replacement
        # We iterate through blocks and try to replace them
        # Note: Modifying the list while iterating is tricky, so we collect indices to replace
        indices_to_replace = []
        for i in range(len(blocks)):
            b_triples = block_triples[i]
            must_cover = [tr for tr in b_triples if triple_counts[tr] == 1]
            
            if not must_cover:
                continue # Handled in removal pass, or will be removed next pass
                
            # If there are unique triples, we must cover them in the replacement
            # The replacement block must contain the union of points in must_cover
            must_points = set()
            for tr in must_cover:
                must_points.update(tr)
                
            if len(must_points) > k:
                continue # Cannot cover in one block of size k
                
            # Try to find a replacement block
            # The new block must cover all must_cover triples.
            # It should ideally cover more other triples to increase their counts.
            
            # Strategy: Start with must_points. Fill the rest greedily to maximize new coverage.
            new_block_set = set(must_points)
            remaining_slots = k - len(new_block_set)
            
            # If we have no remaining slots, the block is fixed
            if remaining_slots == 0:
                new_block = tuple(sorted(new_block_set))
                # Check if it's valid and covers must_cover (it does by construction)
                # Check if it's different from old block
                if new_block != blocks[i]:
                    # Verify it covers must_cover
                    new_triples_set = set(itertools.combinations(new_block, t))
                    if all(tr in new_triples_set for tr in must_cover):
                        indices_to_replace.append((i, new_block, new_triples_set))
            else:
                # Greedy fill
                # We want to add points that cover triples with count == 1 (other than must_cover) 
                # or simply maximize the number of new triples covered.
                # For speed, we just pick points that maximize the number of *uncovered* or *low count* triples.
                # Actually, since we just need to cover must_cover, any valid completion works.
                # But to improve the solution, we should pick points that help other blocks.
                # Let's try to maximize the number of triples with count == 1 (excluding must_cover) that get covered.
                
                # To keep it fast, we will just try a few random completions or a simple greedy one.
                # Simple greedy:
                current_block = list(new_block_set)
                for _ in range(remaining_slots):
                    best_point = None
                    best_gain = -1
                    for x in range(v):
                        if x in new_block_set:
                            continue
                        gain = 0
                        # Check how many new triples are formed with count 1
                        # This is expensive. Let's just pick the point that maximizes 
                        # the number of triples in the final block that are currently count 1.
                        # Approximation: Pick point that maximizes intersection with must_cover? No, must_cover is already covered.
                        # Let's just pick a random point or the one with lowest frequency in existing blocks?
                        # For now, let's try all points and pick the one that maximizes 
                        # the number of triples (in the potential new block) that have count == 1.
                        
                        # To make this fast, we can just check if adding x creates any triple 
                        # that is currently count 1 (and not in must_cover).
                        # Actually, we just need *a* valid block. The replacement is beneficial 
                        # if it allows us to remove the old block. The new block's quality 
                        # (how many extra triples it covers) determines if we can remove *other* blocks later.
                        # So, let's maximize the number of triples covered by the new block 
                        # that are currently covered by ONLY the old block (count 1).
                        # Wait, must_cover are the count 1 triples of the old block.
                        # So we MUST cover them.
                        # Any other triple covered by the new block that is count 1 (from other blocks) 
                        # would be bad? No, if we cover it, its count becomes 2.
                        # If we cover a triple that is count 1 from *another* block, it doesn't hurt.
                        # Ideally, we want the new block to cover as many *other* count-1 triples as possible 
                        # so that those other blocks might become removable.
                        
                        # Heuristic: Maximize number of triples in new_block that have count == 1 
                        # (globally, before update).
                        
                        # This is O(V * K^2) per step. Acceptable for small K.
                        for i1 in range(len(current_block)):
                            for i2 in range(i1+1, len(current_block)):
                                tr = tuple(sorted([current_block[i1], current_block[i2], x]))
                                if triple_counts[tr] == 1:
                                    gain += 1
                                # We might also care about count 0 (uncovered)? 
                                # No, the design is already a covering, so no count 0.
                                
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                            
                    if best_point is None:
                        break
                    current_block.append(best_point)
                    new_block_set.add(best_point)
                    
                new_block = tuple(sorted(current_block))
                new_triples_set = set(itertools.combinations(new_block, t))
                if all(tr in new_triples_set for tr in must_cover):
                    if new_block != blocks[i]:
                        indices_to_replace.append((i, new_block, new_triples_set))

        # Apply replacements
        # We must be careful if replacing one block affects the "must_cover" status of another.
        # However, replacing a block with another that covers the same unique triples 
        # and *more* other triples generally improves the situation.
        # We will apply them one by one, updating counts.
        for idx, new_block, new_triples_set in indices_to_replace:
            old_block = blocks[idx]
            old_triples = block_triples[idx]
            
            # Check if it's still worth it? 
            # The must_cover might have changed if a previous replacement affected them.
            # For simplicity and robustness, we only replace if the new block still covers 
            # the current set of count-1 triples of the old block.
            current_must = [tr for tr in old_triples if triple_counts[tr] == 1]
            if all(tr in new_triples_set for tr in current_must):
                # Update counts
                for tr in old_triples:
                    triple_counts[tr] -= 1
                for tr in new_triples_set:
                    triple_counts[tr] += 1
                blocks[idx] = new_block
                block_triples[idx] = list(new_triples_set)
                improved = True

    return [list(b) for b in blocks]
