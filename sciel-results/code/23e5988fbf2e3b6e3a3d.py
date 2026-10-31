import time
import itertools
import random
import sys

_DEADLINE = 115.0
_START = None

def _elapsed():
    return time.time() - _START

def _precompute(v, k, t):
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # t_lists[i] is a list of t-subset indices covered by k-subset i
    t_lists = []
    # coverage_bits[i] is a bitmask of t-subset indices covered by k-subset i
    coverage_bits = []
    
    # t_to_blocks[j] is a list of k-subset indices that cover t-subset j
    t_to_blocks = [[] for _ in range(n_t)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        ts_list = []
        for ts in itertools.combinations(ks, t):
            idx = t_index[ts]
            mask |= (1 << idx)
            ts_list.append(idx)
            t_to_blocks[idx].append(i)
        t_lists.append(ts_list)
        coverage_bits.append(mask)
        
    return all_t, t_index, k_subsets, t_lists, coverage_bits, t_to_blocks

def _greedy_construct(v, k, t, coverage_bits, rng):
    n_t = sum(1 for _ in itertools.combinations(range(v), t)) # Just to be safe, or pass n_t
    # Actually n_t is len(all_t). Let's pass it or compute once.
    # We'll pass n_t from outside to avoid re-computation.
    
    uncovered = (1 << n_t) - 1
    blocks = []
    
    # Optimization: Keep a list of available blocks? 
    # No, just iterate all.
    
    while uncovered and _elapsed() < _DEADLINE - 5:
        best_i = -1
        best_gain = 0
        
        # Iterate all k-subsets
        for i, mask in enumerate(coverage_bits):
            g = (mask & uncovered).bit_count()
            if g > best_gain:
                best_gain = g
                best_i = i
                if best_gain == 0:
                    break
        
        if best_i < 0:
            break
            
        blocks.append(best_i)
        uncovered &= ~coverage_bits[best_i]
        
    return blocks

def _local_search(blocks, t_lists, t_to_blocks, n_t, time_budget):
    """
    Local search using count array and swap-to-enable-removal.
    """
    blocks = list(blocks)
    
    # Initialize count array
    count = [0] * n_t
    for b in blocks:
        for t_idx in t_lists[b]:
            count[t_idx] += 1
            
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Removal Pass ---
        i = 0
        while i < len(blocks) and _elapsed() < time_budget:
            b = blocks[i]
            removable = True
            for t_idx in t_lists[b]:
                if count[t_idx] == 1:
                    removable = False
                    break
            if removable:
                # Remove block b
                for t_idx in t_lists[b]:
                    count[t_idx] -= 1
                blocks.pop(i)
                improved = True
                # Do not increment i, check the same index again (next block shifted)
            else:
                i += 1
                
        # --- Swap Pass ---
        # Try to swap blocks to enable further removals
        # We iterate through blocks and try to find a replacement for those with critical t-subsets
        for i in range(len(blocks)):
            if _elapsed() >= time_budget:
                break
                
            b = blocks[i]
            # Find critical t-subsets (count == 1)
            critical_ts = [t_idx for t_idx in t_lists[b] if count[t_idx] == 1]
            
            if not critical_ts:
                continue # Should have been removed, but maybe count changed? No, removal pass is stable.
                
            # To find a replacement, we need a block that covers ALL critical_ts.
            # Pick the critical t with the fewest covering blocks to minimize search space
            # Sort critical_ts by len(t_to_blocks[t])
            critical_ts.sort(key=lambda x: len(t_to_blocks[x]))
            
            # The replacement must be in the intersection of t_to_blocks for all critical_ts
            # Start with the smallest list
            candidates = t_to_blocks[critical_ts[0]]
            
            # Filter candidates: must cover all other critical_ts
            # And must not be the block itself (though if it is, swap is useless)
            # And ideally, the swap should not increase the number of blocks (it's a swap, so count stays same)
            # But it might enable removal of ANOTHER block.
            
            found_swap = False
            # To keep it fast, limit candidates
            for new_b in candidates:
                if new_b == b:
                    continue
                    
                # Check if new_b covers all critical_ts
                covers_all = True
                new_t_set = set(t_lists[new_b]) # Expensive? k=7, t=4, len(t_lists)=35. Set creation is ok.
                for t_idx in critical_ts:
                    if t_idx not in new_t_set:
                        covers_all = False
                        break
                        
                if not covers_all:
                    continue
                    
                # Perform swap
                # Remove b
                for t_idx in t_lists[b]:
                    count[t_idx] -= 1
                # Add new_b
                for t_idx in t_lists[new_b]:
                    count[t_idx] += 1
                    
                blocks[i] = new_b
                found_swap = True
                improved = True
                break # Restart removal pass after a successful swap
            
            if found_swap:
                break # Go back to removal pass
    
    return blocks

def construct(v, k, t):
    global _START
    _START = time.time()
    
    all_t, t_index, k_subsets, t_lists, coverage_bits, t_to_blocks = _precompute(v, k, t)
    n_t = len(all_t)
    
    best_blocks = None
    best_size = float('inf')
    
    # Multiple restarts
    num_restarts = 10 # Adjust based on speed
    
    for r in range(num_restarts):
        if _elapsed() > _DEADLINE - 10:
            break
            
        rng = random.Random(12345 + r)
        
        # Greedy
        blocks = _greedy_construct(v, k, t, coverage_bits, rng)
        
        # Local Search
        # Give local search a portion of the remaining time
        # Or just let it run until global deadline - safety
        ls_budget = _DEADLINE - 2
        blocks = _local_search(blocks, t_lists, t_to_blocks, n_t, ls_budget)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = list(blocks)
            
            # If we hit the lower bound, we can stop
            # Lower bound for C(19,7,4) is 131
            if best_size <= 131:
                break
                
    if best_blocks is None:
        # Fallback: just greedy
        rng = random.Random(0)
        best_blocks = _greedy_construct(v, k, t, coverage_bits, rng)
        
    # Convert to list of lists
    result = [list(k_subsets[b]) for b in best_blocks]
    
    # Deduplicate just in case
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
            
    return dedup
