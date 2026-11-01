import time
import random
import numpy as np
from itertools import combinations

# Precomputation
V = 19
K = 7
T = 4

t_subsets_list = list(combinations(range(V), T))
t_subset_to_id = {ts: i for i, ts in enumerate(t_subsets_list)}
num_t_subsets = len(t_subsets_list)

all_blocks = list(combinations(range(V), K))
num_blocks = len(all_blocks)

block_t_ids = []
t_id_to_blocks = [[] for _ in range(num_t_subsets)]

for i, b in enumerate(all_blocks):
    ids = [t_subset_to_id[ts] for ts in combinations(b, T)]
    block_t_ids.append(ids)
    for tid in ids:
        t_id_to_blocks[tid].append(i)

def greedy_construct(seed):
    random.seed(seed)
    uncovered = set(range(num_t_subsets))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered element to anchor the search
        # This ensures we make progress and diversifies the path
        target = random.choice(list(uncovered))
        
        best_block_idx = -1
        max_gain = 0
        
        # Only consider blocks that cover the target
        for b_idx in t_id_to_blocks[target]:
            # Calculate gain
            # Optimization: If we have a set of uncovered, checking membership is O(1)
            gain = 0
            for tid in block_t_ids[b_idx]:
                if tid in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
                # Early break if we covered all? No, max possible is 35.
                if max_gain == len(block_t_ids[b_idx]):
                    break
        
        if best_block_idx == -1:
            # Should not happen if uncovered is not empty and we picked from it
            break
            
        blocks.append(best_block_idx)
        # Remove covered t-subsets
        for tid in block_t_ids[best_block_idx]:
            if tid in uncovered:
                uncovered.remove(tid)
                
    return blocks

def reduce_blocks(block_indices):
    # Returns a new list of block indices with redundant blocks removed
    if not block_indices:
        return []
        
    counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in block_indices:
        for tid in block_t_ids[b_idx]:
            counts[tid] += 1
            
    # Iterative removal
    changed = True
    current_blocks = list(block_indices)
    
    while changed:
        changed = False
        new_blocks = []
        for b_idx in current_blocks:
            redundant = True
            for tid in block_t_ids[b_idx]:
                if counts[tid] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove this block
                for tid in block_t_ids[b_idx]:
                    counts[tid] -= 1
                changed = True
            else:
                new_blocks.append(b_idx)
        current_blocks = new_blocks
        
    return current_blocks

def construct(v, k, t):
    # The evaluator passes v, k, t. 
    # Our precomputation is hardcoded for 19, 7, 4.
    # The task says "The evaluator passes this task's parameters v, k and t."
    # And "Keep the function names and signatures".
    # If v, k, t are different, we need to recompute.
    # But the task is specifically C(19,7,4). 
    # To be safe, I will check if v,k,t match 19,7,4. 
    # If not, I should probably run a generic solver, but the precomputation is specific.
    # Given the context "Covering design C(19,7,4)", it's highly likely v=19, k=7, t=4.
    # I will add a fallback for generic cases just in case, but optimize for 19,7,4.
    
    if v != 19 or k != 7 or t != 4:
        # Fallback to simple greedy for other sizes (slow but correct)
        # This path is unlikely to be used based on the prompt.
        return simple_greedy(v, k, t)
        
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_blocks = []
    best_len = float('inf')
    
    # Initial run
    blocks = greedy_construct(0)
    blocks = reduce_blocks(blocks)
    if len(blocks) < best_len:
        best_len = len(blocks)
        best_blocks = blocks
        
    # ILS Loop
    seed = 1
    while time.time() < deadline:
        # Perturbation
        if not best_blocks:
            blocks = greedy_construct(seed)
        else:
            # Copy best blocks
            blocks = list(best_blocks)
            
            # Remove a random block
            if blocks:
                rem_idx = random.randint(0, len(blocks) - 1)
                removed_block = blocks.pop(rem_idx)
                
                # Identify uncovered t-subsets
                # We need to know which t-subsets were ONLY covered by the removed block.
                # We can recompute counts for the remaining blocks.
                counts = np.zeros(num_t_subsets, dtype=np.int32)
                for b_idx in blocks:
                    for tid in block_t_ids[b_idx]:
                        counts[tid] += 1
                
                uncovered = set()
                for tid in block_t_ids[removed_block]:
                    if counts[tid] == 0:
                        uncovered.add(tid)
                        
                # Repair: Greedy add to cover uncovered
                # We can reuse the greedy logic but start with existing blocks and uncovered set
                # To save time, just run a mini-greedy on the uncovered set
                
                while uncovered:
                    target = random.choice(list(uncovered))
                    best_block_idx = -1
                    max_gain = 0
                    for b_idx in t_id_to_blocks[target]:
                        gain = 0
                        for tid in block_t_ids[b_idx]:
                            if tid in uncovered:
                                gain += 1
                        if gain > max_gain:
                            max_gain = gain
                            best_block_idx = b_idx
                            if max_gain == len(block_t_ids[b_idx]):
                                break
                    if best_block_idx == -1:
                        break
                    blocks.append(best_block_idx)
                    for tid in block_t_ids[best_block_idx]:
                        if tid in uncovered:
                            uncovered.remove(tid)
                
                # Reduce
                blocks = reduce_blocks(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            # If we found the theoretical lower bound or a very good number, we could stop?
            # Lower bound is 131. Best known 151.
            # If we hit 131, we are done.
            if best_len <= 131:
                break
                
        seed += 1
        
    # Convert block indices back to lists of points
    result = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return result

def simple_greedy(v, k, t):
    # Fallback for non-19,7,4
    uncovered = set(combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(combinations(block, t))
    return blocks
