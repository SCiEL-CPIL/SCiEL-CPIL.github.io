"""Optimized covering C(24,7,3) using Information-Theoretic Greedy and Aggressive Local Search."""
import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Uses a fast greedy initialization followed by aggressive local search 
    to minimize the number of blocks.
    """
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    # 1. Fast Greedy Initialization
    blocks = _fast_greedy(v, k, t)
    
    # 2. Remove Redundant Blocks (Fast Bitset Check)
    blocks = _remove_redundant_fast(v, k, t, blocks)
    
    # 3. Local Search Optimization
    if time.time() - start_time < time_limit - 5.0:
        blocks = _local_search_aggressive(v, k, t, blocks, time_limit)
        
    # Final cleanup
    blocks = _remove_redundant_fast(v, k, t, blocks)
    
    return blocks

def _fast_greedy(v, k, t):
    """
    High-speed greedy construction.
    Pre-computes pair counts to quickly evaluate the marginal gain of adding a point.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    # Pre-compute all pairs
    all_pairs = list(itertools.combinations(range(v), 2))
    pair_count = {p: 0 for p in all_pairs}
    
    # Initial state: all triples uncovered
    # We don't need to track uncovered triples explicitly if we can compute gain directly?
    # No, we need to know if a triple is already covered to avoid double counting gain.
    # For v=24, t=3, total triples is 2024.
    # We can maintain a set of uncovered triples.
    
    # Optimization: 
    # Gain of a block B is the number of triples in B that are currently uncovered.
    # To speed up, we can maintain a count of how many times each pair is involved in uncovered triples?
    # Actually, for t=3, a triple {a,b,c} is covered if it's in a block.
    # Let's stick to set of uncovered triples, it's small enough (2024 items).
    
    while uncovered:
        best_block = None
        best_gain = -1
        
        # Heuristic: Pick a random uncovered triple as a seed to build the block around.
        # This avoids O(N^3) search for the best block from scratch every time.
        seed_triple = random.choice(list(uncovered))
        
        # Build block greedily around seed_triple
        block = list(seed_triple)
        
        # Calculate initial gain
        current_gain = 1 # The seed triple itself is uncovered
        
        # Extend block
        while len(block) < k:
            best_point = None
            best_point_gain = -1
            
            # Try adding each point not in block
            # We want to maximize new triples covered
            for x in range(v):
                if x in block:
                    continue
                
                gain = 0
                # New triples formed by adding x to block
                # {x, b1, b2} for all pairs (b1, b2) in block
                for b1, b2 in itertools.combinations(block, 2):
                    triple = tuple(sorted((x, b1, b2)))
                    if triple in uncovered:
                        gain += 1
                
                if gain > best_point_gain:
                    best_point_gain = gain
                    best_point = x
            
            if best_point is None or best_point_gain == 0:
                # No point adds new coverage? This shouldn't happen if uncovered is not empty
                # unless the block is already full or we are stuck.
                # If stuck, break.
                break
            
            block.append(best_point)
            current_gain += best_point_gain
        
        block.sort()
        
        # Verify this block is better than any previous best? 
        # The "seed + greedy extend" is usually good. 
        # To be safer, we can check a few random seeds or just accept the first good one.
        # Given the speed requirement, we accept the first constructed block.
        # However, a better heuristic: 
        # Find the triple with the "highest scarcity" (fewest points shared with other uncovered triples).
        # But that's expensive. 
        # Let's try to pick the seed triple that maximizes the potential.
        # Actually, the standard greedy picks the block with max gain.
        # Doing that for all C(24,7) blocks is too slow.
        # The "seed" approach is a standard approximation.
        
        # Update uncovered
        for triple in itertools.combinations(block, t):
            uncovered.discard(triple)
            
        blocks.append(block)
        
    return blocks

def _remove_redundant_fast(v, k, t, blocks):
    """
    Remove redundant blocks using bitsets for coverage tracking.
    """
    if not blocks:
        return blocks
    
    # Map each triple to an index
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Bitset for each block: which triples it covers
    # Python ints can be used as bitsets
    block_bitsets = []
    for b in blocks:
        bits = 0
        for triple in itertools.combinations(b, t):
            idx = triple_to_idx[triple]
            bits |= (1 << idx)
        block_bitsets.append(bits)
    
    # Union of all bitsets
    union_bits = 0
    for bits in block_bitsets:
        union_bits |= bits
    
    # Check if removing block i still covers all
    new_blocks = []
    for i in range(len(blocks)):
        # If union without block i is still full, remove it
        bits_without_i = union_bits & ~block_bitsets[i]
        if bits_without_i == union_bits:
            # Redundant
            continue
        else:
            # Keep it, but we need to update union_bits?
            # No, we are just checking against the *original* union.
            # Wait, if we remove a block, the union changes.
            # This simple check only works if we check against the *current* set of kept blocks?
            # No, the standard way is:
            # Iterate through blocks. For each, check if its unique coverage is 0 given the *other* blocks.
            # "Unique coverage" = (block_bits) & ~(union of all other blocks)
            
            pass
    
    # Let's do it properly
    # We want to find the minimal set.
    # Greedy removal:
    # 1. Compute union of all.
    # 2. For each block i, compute union of all blocks except i.
    #    If union_except_i == union_all, then i is redundant.
    
    # This is O(N^2 * BitOps). N is ~76. 76^2 * (2024 bit ops) is fast.
    
    kept_blocks = list(blocks)
    changed = True
    while changed:
        changed = False
        if not kept_blocks:
            break
        
        # Compute total union
        total_union = 0
        for i, b in enumerate(kept_blocks):
            total_union |= block_bitsets[i] # Note: block_bitsets corresponds to original 'blocks', not 'kept_blocks'
            # We need to recompute bitsets for kept_blocks or map indices.
            # Easier: recompute bitsets for kept_blocks every pass? 
            # Or just map block content to index.
            
        # Let's restructure to avoid index confusion.
        current_bitsets = []
        for b in kept_blocks:
            bits = 0
            for triple in itertools.combinations(b, t):
                idx = triple_to_idx[triple]
                bits |= (1 << idx)
            current_bitsets.append(bits)
            
        total_union = 0
        for bits in current_bitsets:
            total_union |= bits
            
        for i in range(len(kept_blocks)):
            union_except_i = 0
            for j in range(len(kept_blocks)):
                if i != j:
                    union_except_i |= current_bitsets[j]
                    
            if union_except_i == total_union:
                # Redundant
                kept_blocks.pop(i)
                current_bitsets.pop(i)
                changed = True
                break # Restart loop with new list
                
    return kept_blocks

def _local_search_aggressive(v, k, t, blocks, time_limit):
    """
    Aggressive local search.
    1. Try to replace a block with another that covers the same "critical" triples.
    2. Try to add a block that makes an existing one redundant.
    """
    blocks = list(blocks)
    random.seed(42)
    
    # Pre-compute triple indices
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    # Helper to get bitset of a block
    def get_bitset(block):
        bits = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            bits |= (1 << idx)
        return bits
        
    # Helper to check if block A covers all triples in bitset B
    # (A & B) == B
    
    # Helper to get "critical" triples for a block i:
    # Triples covered by i but not by any other block.
    def get_critical_triples(i, current_blocks, current_bitsets):
        union_others = 0
        for j in range(len(current_blocks)):
            if i != j:
                union_others |= current_bitsets[j]
        critical_bits = current_bitsets[i] & ~union_others
        return critical_bits
        
    # Initial bitsets
    current_bitsets = [get_bitset(b) for b in blocks]
    
    # Main Loop
    while time.time() < time_limit:
        n_blocks = len(blocks)
        if n_blocks == 0:
            break
            
        # Strategy 1: Replace block i with a new block that covers its critical triples
        # and potentially covers more, allowing other blocks to be removed later.
        
        # Pick a random block to replace
        i = random.randint(0, n_blocks - 1)
        critical_bits = get_critical_triples(i, blocks, current_bitsets)
        
        if critical_bits == 0:
            # Block i is redundant, remove it
            blocks.pop(i)
            current_bitsets.pop(i)
            continue
            
        # We need to find a new block that covers all triples in critical_bits.
        # critical_bits is a set of triples. Let's extract them.
        critical_triples = []
        b = critical_bits
        idx = 0
        while b:
            if b & 1:
                critical_triples.append(all_triples[idx])
            b >>= 1
            idx += 1
            
        # If critical set is large, it's hard to cover with one block.
        # If the union of points in critical_triples is <= k, we can try to construct a block.
        points_in_critical = set()
        for triple in critical_triples:
            points_in_critical.update(triple)
            
        if len(points_in_critical) > k:
            # Can't cover all critical triples with one block of size k if they span > k points?
            # Wait, a block of size k covers C(k,3) triples.
            # If critical_triples require points {1,2,3,4,5,6,7,8} and k=7, we can't cover all 
            # if they are disjoint enough? 
            # Actually, if the set of points involved is > k, we *cannot* cover all of them in one block.
            # So we can only replace block i if the critical triples can be covered by *some* block.
            # If not, we must keep block i or split its responsibility? No, we just can't replace it with a single block.
            # So skip this block for replacement.
            continue
            
        # Construct a new block covering points_in_critical
        new_block = list(points_in_critical)
        
        # Extend new_block to size k
        # We want to maximize coverage of *other* triples to help other blocks.
        # Greedy extension:
        while len(new_block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in new_block:
                    continue
                gain = 0
                # Count how many triples in new_block + x are covered
                # Specifically, we care about:
                # 1. Triples in critical_bits (must be covered, but we already ensured points are there? 
                #    No, adding x forms new triples. We need to ensure the *critical* triples are covered.
                #    Since we started with all points from critical triples, any triple in critical_bits 
                #    is a subset of points_in_critical. 
                #    Wait. If critical triple is (1,2,3) and points_in_critical={1,2,3}, it is covered.
                #    If we add x=4, we form (1,2,4), (1,3,4), (2,3,4).
                #    The critical triple (1,2,3) is still covered because 1,2,3 are in the block.
                #    So yes, starting with points_in_critical guarantees coverage of critical_triples.
                
                # So we just want to maximize *additional* coverage to help redundancy removal.
                for p1, p2 in itertools.combinations(new_block, 2):
                    triple = tuple(sorted((x, p1, p2)))
                    # Check if this triple is already covered by others?
                    # If so, gain is 0.
                    # If not, gain is 1.
                    # We can check against union_others.
                    # But union_others changes if we keep block i? No, we are *replacing* block i.
                    # So union_others is the union of all blocks except i.
                    
                    # We need to check if triple is in union_others.
                    # This is expensive inside the loop.
                    # Approximation: just count raw new triples?
                    # Let's just count how many *uncovered* triples (by others) we cover.
                    # We can pre-compute union_others for this iteration.
                    
                    pass # Placeholder
                    
            # This inner loop is getting complex. Let's simplify.
            # Just pick random points to extend? Or greedy based on total uncovered count?
            
            # Let's use a simpler heuristic: pick x that maximizes the number of triples 
            # (x, p1, p2) that are NOT in union_others.
            
            # Pre-compute union_others for this candidate replacement
            union_others = 0
            for j in range(len(blocks)):
                if j != i:
                    union_others |= current_bitsets[j]
                    
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in new_block:
                    continue
                gain = 0
                for p1, p2 in itertools.combinations(new_block, 2):
                    triple = tuple(sorted((x, p1, p2)))
                    idx_t = triple_to_idx[triple]
                    if not (union_others & (1 << idx_t)):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point is None or best_gain == 0:
                break
            new_block.append(best_point)
            
        new_block.sort()
        
        # Check if new_block is distinct from existing blocks?
        # If it's the same, no change.
        if new_block == blocks[i]:
            continue
            
        # Verify new_block covers critical_bits
        new_bits = get_bitset(new_block)
        if (new_bits & critical_bits) != critical_bits:
            # Should not happen if logic is correct, but safety check
            continue
            
        # Perform replacement
        blocks[i] = new_block
        current_bitsets[i] = new_bits
        
        # After replacement, check for new redundancies
        # We can call _remove_redundant_fast, but that's expensive.
        # Just do a quick check for the block we replaced and its neighbors?
        # Or just let the next iteration's critical check handle it?
        # The critical check in the next iteration will see if block i (now new) is redundant?
        # No, it checks if block i is redundant given others.
        # It doesn't check if *other* blocks became redundant.
        
        # Let's do a fast redundancy check for the whole set every 10 iterations?
        # Or just rely on the fact that we are maximizing gain.
        
        # To be safe and effective, let's do a quick redundancy scan for the block we just modified 
        # and any block that shares many triples with it.
        
        # Actually, the simplest effective move is:
        # If we replace block i, we might have made block j redundant.
        # Let's just check all blocks for redundancy in a lightweight way.
        
        # Lightweight redundancy check:
        # Recompute union_others for each block? That's O(N^2).
        # Given N~76, it's fine.
        
        changed = True
        while changed:
            changed = False
            for j in range(len(blocks)):
                if j == i:
                    continue # Skip the one we just changed? Or include it?
                # Check if block j is redundant
                union_except_j = 0
                for m in range(len(blocks)):
                    if m != j:
                        union_except_j |= current_bitsets[m]
                
                total_union = 0
                for m in range(len(blocks)):
                    total_union |= current_bitsets[m]
                    
                if union_except_j == total_union:
                    # Block j is redundant
                    blocks.pop(j)
                    current_bitsets.pop(j)
                    changed = True
                    # Indices changed, break inner loop
                    break
    
    return blocks
