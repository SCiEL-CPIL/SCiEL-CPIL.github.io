import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tsubs_set = set(all_tsubs)
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast array operations
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute t-subsets for each possible block (k-combination of v points)
    # For v=24, k=7: C(24,7) = 346104 blocks, each with C(7,3)=35 triples
    # That's ~12M triple memberships - too many to precompute all
    # Instead, compute on the fly but cache
    
    def block_to_tris(block):
        """Convert a block (tuple of sorted points) to list of t-subset indices."""
        return [tsub_to_idx[tri] for tri in itertools.combinations(block, t)]
    
    def greedy_construct(seed=None):
        """Greedy construction: pick blocks that cover most uncovered triples."""
        if seed is not None:
            random.seed(seed)
        
        uncovered = np.ones(num_tsubs, dtype=np.int32)  # 1 = uncovered
        blocks = []
        block_tris_list = []
        remaining = num_tsubs
        
        while remaining > 0:
            # Find best block: covers most uncovered triples
            # Sample candidate blocks for speed
            best_block = None
            best_gain = 0
            
            # Generate candidate blocks
            candidates = []
            
            # Strategy 1: blocks containing a random uncovered triple
            for _ in range(5):
                # Pick a random uncovered triple
                uncov_indices = np.where(uncovered > 0)[0]
                if len(uncov_indices) == 0:
                    break
                tri_idx = uncov_indices[random.randint(0, len(uncov_indices)-1)]
                tri = all_tsubs[tri_idx]
                tri_set = set(tri)
                others = [p for p in range(v) if p not in tri_set]
                
                # Sample blocks containing this triple
                for _ in range(10):
                    extra = random.sample(others, k - t)
                    block = tuple(sorted(tri_set | set(extra)))
                    candidates.append(block)
            
            # Strategy 2: random blocks
            for _ in range(10):
                block = tuple(sorted(random.sample(range(v), k)))
                candidates.append(block)
            
            # Evaluate candidates
            for block in candidates:
                tris = block_to_tris(block)
                gain = sum(uncovered[i] for i in tris)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block is None or best_gain == 0:
                # Fallback: exhaustive over blocks containing the first uncovered triple
                uncov_indices = np.where(uncovered > 0)[0]
                if len(uncov_indices) == 0:
                    break
                tri_idx = uncov_indices[0]
                tri = all_tsubs[tri_idx]
                tri_set = set(tri)
                others = [p for p in range(v) if p not in tri_set]
                
                best_gain = 0
                for combo in itertools.combinations(others, k - t):
                    block = tuple(sorted(tri_set | set(combo)))
                    tris = block_to_tris(block)
                    gain = sum(uncovered[i] for i in tris)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
            
            if best_block is not None:
                blocks.append(list(best_block))
                tris = block_to_tris(best_block)
                block_tris_list.append(tris)
                for i in tris:
                    if uncovered[i] > 0:
                        uncovered[i] -= 1
                        remaining -= 1
        
        return blocks, block_tris_list, uncovered
    
    def local_search(blocks, block_tris_list, max_time=30):
        """SA-based local search to minimize block count."""
        if time.time() > start_time + max_time:
            return blocks, block_tris_list
        
        blocks = [list(b) for b in blocks]
        block_tris_list = [list(bt) for bt in block_tris_list]
        n = len(blocks)
        
        # Build coverage count
        cov = np.zeros(num_tsubs, dtype=np.int32)
        for tris in block_tris_list:
            for i in tris:
                cov[i] += 1
        
        # Verify validity
        if np.any(cov == 0):
            return blocks, block_tris_list
        
        T = 100.0
        cooling = 0.97
        best_n = n
        best_blocks = [list(b) for b in blocks]
        best_block_tris = [list(bt) for bt in block_tris_list]
        best_cov = cov.copy()
        
        end_time = start_time + max_time
        steps = 0
        
        while time.time() < end_time and T > 0.01:
            steps += 1
            
            if n < best_n:
                best_n = n
                best_blocks = [list(b) for b in blocks]
                best_block_tris = [list(bt) for bt in block_tris_list]
                best_cov = cov.copy()
            
            move_type = random.random()
            
            if move_type < 0.4 and n > 1:
                # Try removing a block
                i = random.randint(0, n - 1)
                removable = True
                for tri_idx in block_tris_list[i]:
                    if cov[tri_idx] == 1:
                        removable = False
                        break
                
                if removable:
                    for tri_idx in block_tris_list[i]:
                        cov[tri_idx] -= 1
                    blocks.pop(i)
                    block_tris_list.pop(i)
                    n -= 1
                else:
                    # Try swapping: replace block i with one covering its unique triples
                    only_tris = [tri_idx for tri_idx in block_tris_list[i] if cov[tri_idx] == 1]
                    
                    if len(only_tris) <= 15:  # Limit complexity
                        must_include = set()
                        for tri_idx in only_tris:
                            must_include.update(all_tsubs[tri_idx])
                        
                        if len(must_include) <= k:
                            remaining_pts = [p for p in range(v) if p not in must_include]
                            need = k - len(must_include)
                            
                            if 0 <= need <= len(remaining_pts):
                                for _ in range(15):
                                    if need == 0:
                                        candidate = tuple(sorted(must_include))
                                    else:
                                        extra = random.sample(remaining_pts, need)
                                        candidate = tuple(sorted(must_include | set(extra)))
                                    
                                    cand_tris = block_to_tris(candidate)
                                    
                                    # Check covers all only_tris
                                    cand_set = set(cand_tris)
                                    if all(t in cand_set for t in only_tris):
                                        # Compute delta: how many new triples become uniquely covered
                                        new_unique = 0
                                        for tri_idx in cand_tris:
                                            if tri_idx not in set(block_tris_list[i]):
                                                if cov[tri_idx] == 1:
                                                    new_unique += 1
                                        
                                        delta = len(only_tris) - new_unique
                                        # Accept if delta <= 0 or with probability exp(-delta/T)
                                        if delta <= 0 or random.random() < np.exp(-delta / T):
                                            for tri_idx in block_tris_list[i]:
                                                cov[tri_idx] -= 1
                                            for tri_idx in cand_tris:
                                                cov[tri_idx] += 1
                                            blocks[i] = list(candidate)
                                            block_tris_list[i] = cand_tris
                                            break
            
            elif move_type < 0.7:
                # Try replacing a random block with a better one
                i = random.randint(0, n - 1)
                only_tris = [tri_idx for tri_idx in block_tris_list[i] if cov[tri_idx] == 1]
                
                if only_tris and len(only_tris) <= 12:
                    must_include = set()
                    for tri_idx in only_tris:
                        must_include.update(all_tsubs[tri_idx])
                    
                    if len(must_include) <= k:
                        remaining_pts = [p for p in range(v) if p not in must_include]
                        need = k - len(must_include)
                        
                        if 0 <= need <= len(remaining_pts):
                            best_candidate = None
                            best_delta = float('inf')
                            
                            for _ in range(20):
                                if need == 0:
                                    candidate = tuple(sorted(must_include))
                                else:
                                    extra = random.sample(remaining_pts, need)
                                    candidate = tuple(sorted(must_include | set(extra)))
                                
                                cand_tris = block_to_tris(candidate)
                                cand_set = set(cand_tris)
                                
                                if all(t in cand_set for t in only_tris):
                                    new_unique = 0
                                    for tri_idx in cand_tris:
                                        if tri_idx not in set(block_tris_list[i]):
                                            if cov[tri_idx] == 1:
                                                new_unique += 1
                                    
                                    delta = len(only_tris) - new_unique
                                    if delta < best_delta:
                                        best_delta = delta
                                        best_candidate = candidate
                                        best_cand_tris = cand_tris
                            
                            if best_candidate is not None:
                                if best_delta <= 0 or random.random() < np.exp(-best_delta / T):
                                    for tri_idx in block_tris_list[i]:
                                        cov[tri_idx] -= 1
                                    for tri_idx in best_cand_tris:
                                        cov[tri_idx] += 1
                                    blocks[i] = list(best_candidate)
                                    block_tris_list[i] = best_cand_tris
            
            else:
                # Try adding a block then removing a redundant one
                if n < 150:
                    # Find triples with lowest coverage
                    low_cov = np.where(cov <= 2)[0]
                    if len(low_cov) > 0:
                        target_idx = low_cov[random.randint(0, len(low_cov)-1)]
                        target_tri = all_tsubs[target_idx]
                        target_set = set(target_tri)
                        others = [p for p in range(v) if p not in target_set]
                        
                        best_block = None
                        best_gain = -1
                        
                        for _ in range(30):
                            extra = random.sample(others, k - t)
                            block = tuple(sorted(target_set | set(extra)))
                            tris = block_to_tris(block)
                            gain = sum(1 for i in tris if cov[i] <= 1)
                            if gain > best_gain:
                                best_gain = gain
                                best_block = block
                                best_block_tris = tris
                        
                        if best_block is not None and best_gain > 0:
                            for tri_idx in best_block_tris:
                                cov[tri_idx] += 1
                            blocks.append(list(best_block))
                            block_tris_list.append(best_block_tris)
                            n += 1
                            
                            # Try to remove a redundant block
                            for j in range(n - 1, -1, -1):
                                removable = True
                                for tri_idx in block_tris_list[j]:
                                    if cov[tri_idx] == 1:
                                        removable = False
                                        break
                                if removable:
                                    for tri_idx in block_tris_list[j]:
                                        cov[tri_idx] -= 1
                                    blocks.pop(j)
                                    block_tris_list.pop(j)
                                    n -= 1
                                    break
            
            T *= cooling
        
        # Return best found
        return best_blocks, best_block_tris
    
    def remove_redundant(blocks, block_tris_list):
        """Remove blocks whose triples are all covered by other blocks."""
        blocks = [list(b) for b in blocks]
        block_tris_list = [list(bt) for bt in block_tris_list]
        n = len(blocks)
        
        cov = np.zeros(num_tsubs, dtype=np.int32)
        for tris in block_tris_list:
            for i in tris:
                cov[i] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(n):
                removable = True
                for tri_idx in block_tris_list[i]:
                    if cov[tri_idx] == 1:
                        removable = False
                        break
                if removable:
                    for tri_idx in block_tris_list[i]:
                        cov[tri_idx] -= 1
                    blocks.pop(i)
                    block_tris_list.pop(i)
                    n -= 1
                    changed = True
        
        return blocks, block_tris_list
    
    # Main search loop
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple iterations with different seeds
    iteration = 0
    while time.time() < deadline - 20:
        seed = iteration * 12345 + 67890
        blocks, block_tris_list, uncovered = greedy_construct(seed)
        
        # Remove redundant
        blocks, block_tris_list = remove_redundant(blocks, block_tris_list)
        
        # Local search if we have time
        if time.time() < deadline - 15:
            sa_time = min(25, deadline - time.time() - 10)
            if sa_time > 5:
                blocks, block_tris_list = local_search(blocks, block_tris_list, max_time=sa_time)
                blocks, block_tris_list = remove_redundant(blocks, block_tris_list)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        iteration += 1
        
        # If we've found a good solution, spend remaining time polishing
        if best_len <= 76 and time.time() < deadline - 10:
            sa_time = deadline - time.time() - 5
            if sa_time > 5:
                blocks, block_tris_list = local_search(best_blocks, [block_to_tris(tuple(b)) for b in best_blocks], max_time=sa_time)
                blocks, block_tris_list = remove_redundant(blocks, block_tris_list)
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = [list(b) for b in blocks]
            break
    
    # Final fallback if no valid solution found
    if best_blocks is None or len(best_blocks) == 0:
        blocks, block_tris_list, _ = greedy_construct(0)
        best_blocks = blocks
    
    return best_blocks
