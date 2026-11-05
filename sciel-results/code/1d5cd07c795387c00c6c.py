"""Covering design C(24,7,3) using Bitset Greedy + VNS-style Local Search."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer for safety

    # Precompute all t-subsets as indices for bitset representation
    # C(24,3) = 2024. We can map each triple to an index 0..2023.
    # This allows us to use Python's arbitrary precision integers as bitsets.
    
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Cache for block masks to avoid recomputation
    mask_cache = {}
    
    def get_block_mask(block):
        """Compute the bitset of triples covered by a block. Uses cache."""
        key = tuple(sorted(block))
        if key in mask_cache:
            return mask_cache[key]
        
        mask = 0
        # block is a sorted list of k points
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[tuple(sorted(triple))]
            mask |= (1 << idx)
        
        mask_cache[key] = mask
        return mask

    # Greedy Construction
    # Strategy: 
    # 1. Start with all triples uncovered.
    # 2. While uncovered:
    #    a. Pick a random uncovered triple.
    #    b. Generate candidate blocks containing this triple.
    #       To do this efficiently, we can pick the triple, then choose k-t other points from the remaining v-t points.
    #       Number of candidates: C(v-t, k-t). For v=24, t=3, k=7: C(21, 4) = 5985.
    #       This is manageable.
    #    c. Among candidates, pick the one that covers the most *uncovered* triples.
    #    d. Add to blocks, update uncovered.
    
    uncovered = (1 << num_triples) - 1
    blocks = []
    
    # To avoid re-checking the same candidates or to speed up, we can just iterate.
    # 5985 candidates * 2024 bits check is fast in Python? 
    # 5985 * (bitwise AND + popcount) is very fast.
    
    while uncovered:
        if time.time() - start_time > timeout:
            break
        
        # Pick a random uncovered triple to anchor the new block
        # Find an index of a set bit in 'uncovered'
        # We can use a list of uncovered indices for random selection, 
        # but maintaining that list is costly. 
        # Instead, pick a random index and check if it's set? 
        # Or just pick the lowest set bit? 
        # Random is better for diversity.
        
        # Efficient way to get a random set bit:
        # If uncovered is sparse, this might be slow if we just sample.
        # But 2024 bits is small.
        
        # Let's pick a random triple from the full list and check if it's uncovered.
        # If not, try again. 
        # As coverage progresses, this might take many tries.
        # Better: Maintain a list of uncovered triple indices?
        # Updating the list is O(N).
        
        # Alternative: Just iterate through all triples to find one uncovered? O(2024).
        # Or use the bitset to find a random bit.
        
        # Let's use a helper to get a random set bit.
        # We can convert to int, but random bit selection is tricky.
        # Simple approach: Pick a random index in [0, num_triples). If bit is set, use it.
        # If not, retry.
        
        anchor_idx = -1
        # Try a few times to find an uncovered triple
        for _ in range(100):
            idx = random.randint(0, num_triples - 1)
            if (uncovered >> idx) & 1:
                anchor_idx = idx
                break
        
        if anchor_idx == -1:
            # Fallback: find the lowest set bit
            # This should rarely happen if uncovered is not empty
            # But if it does, we need to handle it.
            # For robustness, let's just find the first set bit.
            temp = uncovered
            while temp:
                # Get lowest set bit index
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                anchor_idx = idx
                break
            else:
                break # No uncovered triples?
        
        anchor_triple = all_triples[anchor_idx]
        
        # Generate candidate blocks containing anchor_triple
        # anchor_triple has t points. We need k-t more points from the remaining v-t points.
        anchor_set = set(anchor_triple)
        available_points = [p for p in range(v) if p not in anchor_set]
        
        best_block = None
        best_gain = -1
        
        # Iterate over combinations of k-t points from available_points
        # C(21, 4) = 5985 combinations.
        for comb in itertools.combinations(available_points, k - t):
            candidate_block = tuple(sorted(anchor_triple + comb))
            
            # Calculate mask
            mask = get_block_mask(candidate_block)
            
            # Calculate gain: number of newly covered triples
            gain = (mask & uncovered).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_block = candidate_block
            
            # Optimization: If we found a block that covers ALL uncovered triples, we can stop early?
            # Unlikely in one step.
            
            # Time check inside loop? 5985 iterations is fast, probably < 1ms.
            # But if we do this 76 times, it's fine.
        
        if best_block is None:
            break
            
        blocks.append(list(best_block))
        
        # Update uncovered
        best_mask = get_block_mask(best_block)
        uncovered &= ~best_mask
        
        # If uncovered becomes 0, we are done
        if not uncovered:
            break

    # If we timed out, we might have an invalid covering.
    # Check if valid.
    if uncovered:
        # Try to finish quickly? Or just return invalid?
        # The task says "Valid when...". Invalid scores 0.
        # We should try to ensure validity if possible.
        # But if we are close to timeout, returning invalid is better than crashing?
        # Actually, the evaluator checks validity.
        # Let's try a fast completion if time permits.
        # For now, if uncovered is not 0, we have a problem.
        # Let's force complete the last few steps if we have time.
        # But the loop above breaks on timeout.
        # Let's re-run a simple greedy without randomization if we fail?
        # Or just return what we have.
        pass

    # Local Search (VNS-style)
    # 1. Remove redundant blocks.
    # 2. Perturb and Re-greedy.
    
    # First, ensure we have a valid covering.
    # If the greedy didn't finish, we need to complete it.
    # Let's run a standard greedy to completion if uncovered is not 0.
    if uncovered:
        # Simple completion
        while uncovered:
            if time.time() - start_time > timeout:
                break
            # Pick first uncovered
            idx = 0
            while idx < num_triples and not ((uncovered >> idx) & 1):
                idx += 1
            if idx == num_triples:
                break
            anchor_triple = all_triples[idx]
            anchor_set = set(anchor_triple)
            available_points = [p for p in range(v) if p not in anchor_set]
            
            best_block = None
            best_gain = -1
            for comb in itertools.combinations(available_points, k - t):
                candidate_block = tuple(sorted(anchor_triple + comb))
                mask = get_block_mask(candidate_block)
                gain = (mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block:
                blocks.append(list(best_block))
                best_mask = get_block_mask(best_block)
                uncovered &= ~best_mask
            else:
                break

    if uncovered:
        # Still invalid. Return what we have.
        return blocks

    # Now we have a valid covering. Minimize it.
    
    # Step 1: Remove redundant blocks.
    # Calculate coverage count for each triple.
    # Or simply, for each block, check if removing it leaves coverage intact.
    # This is O(N * C(v,t)).
    
    def is_valid_covering(b_list):
        covered_mask = 0
        for b in b_list:
            covered_mask |= get_block_mask(b)
        return covered_mask == ((1 << num_triples) - 1)

    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if time.time() - start_time > timeout:
                return blocks
            
            # Try removing blocks[i]
            new_blocks = blocks[:i] + blocks[i+1:]
            if is_valid_covering(new_blocks):
                blocks = new_blocks
                changed = True
                break # Restart loop as indices changed

    # Step 2: VNS Perturbation
    # Try to replace blocks or remove more via perturbation.
    # Strategy: 
    # 1. Randomly remove M blocks (M=1, 2, ...).
    # 2. Re-greedy fill the gaps.
    # 3. If the resulting covering is smaller, accept it.
    
    best_blocks = blocks
    best_size = len(blocks)
    
    # Limit iterations for perturbation
    max_perturbations = 50
    perturb_count = 0
    
    while perturb_count < max_perturbations:
        if time.time() - start_time > timeout:
            break
        
        perturb_count += 1
        
        # Create a copy
        current_blocks = list(best_blocks)
        
        # Randomly remove some blocks
        num_to_remove = random.randint(1, min(3, len(current_blocks)//5))
        if num_to_remove == 0:
            continue
            
        indices_to_remove = random.sample(range(len(current_blocks)), num_to_remove)
        # Sort indices in reverse to remove safely
        for idx in sorted(indices_to_remove, reverse=True):
            current_blocks.pop(idx)
        
        # Calculate uncovered
        covered_mask = 0
        for b in current_blocks:
            covered_mask |= get_block_mask(b)
        uncovered = ((1 << num_triples) - 1) & ~covered_mask
        
        # Re-greedy fill
        while uncovered:
            if time.time() - start_time > timeout:
                uncovered = (1 << num_triples) - 1 # Invalidate
                break
            
            # Pick random uncovered
            anchor_idx = -1
            for _ in range(50):
                idx = random.randint(0, num_triples - 1)
                if (uncovered >> idx) & 1:
                    anchor_idx = idx
                    break
            
            if anchor_idx == -1:
                # Fallback
                temp = uncovered
                while temp:
                    lsb = temp & -temp
                    anchor_idx = lsb.bit_length() - 1
                    break
            
            if anchor_idx == -1:
                break
                
            anchor_triple = all_triples[anchor_idx]
            anchor_set = set(anchor_triple)
            available_points = [p for p in range(v) if p not in anchor_set]
            
            best_block = None
            best_gain = -1
            
            for comb in itertools.combinations(available_points, k - t):
                candidate_block = tuple(sorted(anchor_triple + comb))
                mask = get_block_mask(candidate_block)
                gain = (mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block:
                current_blocks.append(list(best_block))
                best_mask = get_block_mask(best_block)
                uncovered &= ~best_mask
            else:
                break
        
        # Check if valid and smaller
        if not uncovered and len(current_blocks) < best_size:
            # Verify validity just in case
            if is_valid_covering(current_blocks):
                best_blocks = current_blocks
                best_size = len(current_blocks)
        
        # Also try simple removal on the new best
        changed = True
        while changed:
            changed = False
            for i in range(len(best_blocks)):
                if time.time() - start_time > timeout:
                    break
                new_blocks = best_blocks[:i] + best_blocks[i+1:]
                if is_valid_covering(new_blocks):
                    best_blocks = new_blocks
                    best_size = len(best_blocks)
                    changed = True
                    break
            if time.time() - start_time > timeout:
                break

    return best_blocks

# Test locally
if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    
    # Verify
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, 3):
            covered.add(tuple(sorted(c)))
    
    all_subsets = set(itertools.combinations(range(24), 3))
    if covered == all_subsets:
        print("Valid covering")
    else:
        print("Invalid covering")
        print(f"Missing: {len(all_subsets - covered)}")
