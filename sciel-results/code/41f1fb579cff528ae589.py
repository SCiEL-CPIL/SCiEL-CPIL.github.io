import itertools
import random
import time
import numpy as np

# Global precomputation
V = 24
K = 7
T = 3

# Triple index mapping
triple_to_index = {}
index_to_triple = []
idx = 0
for c in itertools.combinations(range(V), T):
    triple_to_index[c] = idx
    index_to_triple.append(c)
    idx += 1

NUM_TRIPLES = len(triple_to_index)

# Precompute block masks as numpy arrays for faster operations
# Each block is a 7-subset. There are C(24,7) = 346104 possible blocks.
# We don't precompute all, but we can precompute the mask for a given block quickly.
# The mask is a set of 35 triple indices.

# For speed in local search, we might want to precompute masks for "common" blocks?
# Or just use the fast block_mask function.

# Let's precompute the list of triples for each point? No, we have masks.
# Let's precompute the list of pairs for each point?

# Helper to compute mask of a block (returns a set of triple indices or a bitmask)
# Using a set of indices is often faster for intersection logic in Python than big integers
# if the set size is small (35). But big integers are also very fast in Python.
# Let's stick to the bitmask approach but optimize the construction.

# Precompute for each pair (a, b) the list of points c such that (a,b,c) is a triple?
# Actually, for a block B, the triples are all C(B,3).
# To calculate the mask of a block B:
# mask = 0
# for c in itertools.combinations(B, 3):
#     mask |= 1 << triple_to_index[c]

# Optimization: Precompute the bit for each triple.
triple_bit = [1 << i for i in range(NUM_TRIPLES)]

def block_mask(block):
    m = 0
    # Block is a tuple or list of 7 points
    for c in itertools.combinations(block, T):
        m |= triple_bit[triple_to_index[c]]
    return m

# Helper to compute gain of adding x to current_block
# We need to count how many NEW triples are covered.
# Triples covered by adding x to block P are (x, u, v) for u, v in P.
def gain_count(x, current_block, uncovered_mask):
    count = 0
    # Pairs in current_block
    for i in range(len(current_block)):
        for j in range(i+1, len(current_block)):
            u = current_block[i]
            v = current_block[j]
            # Triple (x, u, v)
            # Sort to get canonical triple
            t = tuple(sorted((x, u, v)))
            idx = triple_to_index[t]
            if (uncovered_mask >> idx) & 1:
                count += 1
    return count

def popcount(x):
    # Python 3.10+ has int.bit_count()
    if hasattr(int, 'bit_count'):
        return x.bit_count()
    return bin(x).count("1")

def construct(v, k, t):
    if v != V or k != K or t != T:
        # Fallback for other parameters, though task is specific
        # For now, return empty or raise, but we assume standard case
        pass
        
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    best_blocks = None
    best_count = float('inf')
    
    # Initial Greedy with Structural Density
    def run_greedy(seed):
        rng = random.Random(seed)
        uncovered = (1 << NUM_TRIPLES) - 1
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple
            while True:
                r = rng.randint(0, NUM_TRIPLES - 1)
                if (uncovered >> r) & 1:
                    seed_triple = index_to_triple[r]
                    break
            
            block = list(seed_triple)
            
            # Build block
            while len(block) < k:
                best_x = -1
                best_score = -1
                
                # Candidates: all points not in block
                # To speed up, we can filter candidates that have any gain
                # But checking gain is O(K^2) per candidate, K=7, so ~21 checks.
                # 24 candidates * 21 = ~500 ops per step.
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate gain
                    g = gain_count(x, block, uncovered)
                    
                    # Structural Density Heuristic:
                    # Penalize points that create "dense" overlaps?
                    # Or just maximize gain.
                    # The inspiration suggests "marginal coverage density per unit of structural redundancy".
                    # Redundancy could be measured by how much x overlaps with other points in the block?
                    # But x is new.
                    # Maybe penalize if x is "close" to others in some metric?
                    # For now, let's stick to pure gain but add a small random noise for tie-breaking
                    # to encourage diversity.
                    
                    # Let's try a slight penalty for points that are already "well connected"?
                    # No, let's just use gain. The local search will handle the rest.
                    
                    if g > best_score:
                        best_score = g
                        best_x = x
                    elif g == best_score and g > 0:
                        # Tie-breaking: random
                        if rng.random() < 0.5:
                            best_x = x
                            
                if best_x == -1:
                    # Should not happen
                    break
                block.append(best_x)
            
            block.sort()
            bm = block_mask(block)
            uncovered &= ~bm
            blocks.append(block)
            
        return blocks

    # Local Search with Deletion and Swap
    def local_search(blocks, rng):
        blocks = [list(b) for b in blocks]
        block_masks = [block_mask(b) for b in blocks]
        n = len(blocks)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Deletion Pass
            # Find a block that is completely redundant
            # We can speed this up by maintaining a count of coverage for each triple
            # But recalculating is O(N * 35) which is fine for N~70.
            
            # Let's use the count array approach for robustness
            # counts[i] = number of blocks covering triple i
            counts = np.zeros(NUM_TRIPLES, dtype=np.int32)
            for bm in block_masks:
                # Extract indices from bitmask?
                # It's faster to iterate the triples of the block
                pass
            
            # Actually, the bitmask "others_mask" approach in the seed was okay, but O(N^2) in the loop.
            # Let's stick to the simpler bitmask intersection for deletion check.
            
            for i in range(n):
                # Check if block i is redundant
                # Redundant if block_masks[i] is subset of (OR of all other masks)
                # OR of all other masks = (OR of all masks) & ~block_masks[i] ? No.
                # Let total_mask = OR of all block_masks
                # If total_mask == (total_mask & ~block_masks[i]), then block i is redundant?
                # No. If total_mask & block_masks[i] == block_masks[i], it means all triples in i are covered by the union of ALL blocks.
                # But they might be covered ONLY by block i.
                # We need to check if (block_masks[i] & ~OR_others) == 0.
                
                # Efficient way:
                # OR_others = 0
                # for j in range(n):
                #   if j != i: OR_others |= block_masks[j]
                # This is O(N). N is small.
                
                OR_others = 0
                for j in range(n):
                    if j != i:
                        OR_others |= block_masks[j]
                
                if (block_masks[i] & ~OR_others) == 0:
                    # Redundant
                    del blocks[i]
                    del block_masks[i]
                    n -= 1
                    improved = True
                    break # Restart pass
            
            if not improved and n > 0:
                # 2. Swap Pass
                # Try to replace a block with a better one?
                # Or just try random swaps that don't break coverage.
                # The seed did random swaps.
                # Let's try to find a swap that increases "slack" or allows future deletion.
                # Simple random swap:
                for _ in range(20): # More attempts per pass
                    if time.time() >= deadline:
                        break
                    i = rng.randint(0, n - 1)
                    b = blocks[i]
                    x = rng.choice(b)
                    candidates = [p for p in range(v) if p not in b]
                    if not candidates:
                        continue
                    y = rng.choice(candidates)
                    
                    new_b = [p for p in b if p != x]
                    new_b.append(y)
                    new_b.sort()
                    new_mask = block_mask(new_b)
                    
                    OR_others = 0
                    for j in range(n):
                        if j != i:
                            OR_others |= block_masks[j]
                    
                    lost = block_masks[i] & ~OR_others
                    # Check if lost is covered by new_mask
                    if (lost & ~new_mask) == 0:
                        # Safe swap
                        blocks[i] = new_b
                        block_masks[i] = new_mask
                        improved = True
                        break # Restart deletion pass after a swap
            
        return blocks

    # Main Loop
    seed = 0
    while time.time() < deadline:
        blocks = run_greedy(seed)
        # Refine
        blocks = local_search(blocks, random.Random(seed + 1000))
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
            # print(f"Seed {seed}: {best_count} blocks")
            
        seed += 1
        
    return best_blocks
