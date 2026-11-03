import time
import random
import itertools
import sys

# Precomputation for v=24, k=7, t=3
V = 24
K = 7
T = 3

class CoveringCache:
    def __init__(self):
        self.v = V
        self.k = K
        self.t = T
        self.points = list(range(V))
        
        # Generate all triples
        self.triples = list(itertools.combinations(self.points, T))
        self.num_triples = len(self.triples)
        self.triple_to_idx = {t: i for i, t in enumerate(self.triples)}
        
        # Generate all blocks
        self.blocks = list(itertools.combinations(self.points, K))
        self.num_blocks = len(self.blocks)
        
        # Map block index to list of triple indices
        self.block_to_triples = []
        for b in self.blocks:
            ts = []
            for t in itertools.combinations(b, T):
                ts.append(self.triple_to_idx[t])
            self.block_to_triples.append(ts)
            
        # Map triple index to list of block indices
        self.triple_to_blocks = [[] for _ in range(self.num_triples)]
        for b_idx, ts in enumerate(self.block_to_triples):
            for t_idx in ts:
                self.triple_to_blocks[t_idx].append(b_idx)

_cache = None

def get_cache():
    global _cache
    if _cache is None:
        _cache = CoveringCache()
    return _cache

def _greedy_randomized(cache):
    """
    Randomized greedy construction.
    Returns a list of block indices.
    """
    uncovered = set(range(cache.num_triples))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered triple
        # random.choice is O(1) if list, O(N) if set. 
        # Converting set to list every step is O(N). 
        # Better: pick a random element from set.
        t_idx = random.choice(list(uncovered))
        
        # Pick a random block covering this triple
        candidates = cache.triple_to_blocks[t_idx]
        b_idx = random.choice(candidates)
        
        blocks.append(b_idx)
        
        # Remove all triples covered by this block from uncovered
        for t in cache.block_to_triples[b_idx]:
            if t in uncovered:
                uncovered.remove(t)
                
    return blocks

def _local_search(cache, blocks, deadline):
    """
    Local search with deletion and swap.
    Modifies and returns the list of block indices.
    """
    block_set = set(blocks)
    counts = [0] * cache.num_triples
    
    # Initialize counts
    for b in blocks:
        for t in cache.block_to_triples[b]:
            counts[t] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Deletion Phase
        # Shuffle to avoid bias
        random.shuffle(blocks)
        new_blocks = []
        for b in blocks:
            redundant = True
            for t in cache.block_to_triples[b]:
                if counts[t] <= 1:
                    redundant = False
                    break
            if redundant:
                for t in cache.block_to_triples[b]:
                    counts[t] -= 1
                improved = True
            else:
                new_blocks.append(b)
                
        if improved:
            blocks = new_blocks
            block_set = set(blocks)
            continue # Restart to catch new redundancies
            
        # 2. Swap Phase
        # Try to swap a block to create redundancy elsewhere
        random.shuffle(blocks)
        swap_found = False
        for b_old in blocks:
            if time.time() >= deadline:
                break
                
            # Tentatively remove
            for t in cache.block_to_triples[b_old]:
                counts[t] -= 1
                
            lost_set = set()
            for t in cache.block_to_triples[b_old]:
                if counts[t] == 0:
                    lost_set.add(t)
                    
            if not lost_set:
                # Redundant, should have been caught. Restore.
                for t in cache.block_to_triples[b_old]:
                    counts[t] += 1
                break
            
            # Find best replacement
            candidates = set()
            for t in lost_set:
                candidates.update(cache.triple_to_blocks[t])
                
            best_gain = 0
            best_b = -1
            
            # Optimization: If candidates is huge, this loop is slow.
            # But lost_set is small.
            for b_new in candidates:
                if b_new in block_set:
                    continue
                # Calculate gain
                gain = 0
                # Check intersection with lost_set
                for t in cache.block_to_triples[b_new]:
                    if t in lost_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_new
                    if gain == len(lost_set):
                        break # Perfect replacement found
                        
            if best_b != -1 and best_gain > 0:
                # Perform swap
                block_set.remove(b_old)
                block_set.add(best_b)
                
                for t in cache.block_to_triples[best_b]:
                    counts[t] += 1
                    
                # Update blocks list
                # Find index of b_old
                for i, b in enumerate(blocks):
                    if b == b_old:
                        blocks[i] = best_b
                        break
                        
                swap_found = True
                improved = True
                break # Restart deletion phase
                
            else:
                # Restore counts
                for t in cache.block_to_triples[b_old]:
                    counts[t] += 1

    return blocks

def construct(v, k, t):
    if v != 24 or k != 7 or t != 3:
        # Fallback for other parameters (not expected in this task)
        return _generic_greedy(v, k, t)
        
    cache = get_cache()
    deadline = time.time() + 110 # Safety margin
    best_blocks = None
    best_size = float('inf')
    
    while time.time() < deadline:
        # Randomize seed for diversity? 
        # random module uses system time by default.
        # We can explicitly seed if we want reproducibility, 
        # but for search, we want variety.
        
        blocks = _greedy_randomized(cache)
        blocks = _local_search(cache, blocks, deadline)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            if best_size <= 69:
                break
                
    if best_blocks is None:
        best_blocks = _greedy_randomized(cache)
        
    # Convert indices to actual point lists
    result = [list(cache.blocks[b]) for b in best_blocks]
    return result

def _generic_greedy(v, k, t):
    """
    Fallback greedy for non-standard parameters.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        # Find block with max gain
        best_block = None
        max_gain = -1
        # This is slow for large v, but v=24 is small.
        # For generic case, we might need a better heuristic.
        # Given the task is specific to 24,7,3, this is just a safety net.
        for block in itertools.combinations(range(v), k):
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block = block
        if max_gain == 0:
            break
        blocks.append(list(best_block))
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks
