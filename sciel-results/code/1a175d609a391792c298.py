import time
import random
import itertools
import numpy as np
from math import comb as math_comb

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: return before 148s to be safe against 150s limit
    deadline = start_time + 148.0 

    def comb(n, r):
        if r > n or r < 0:
            return 0
        r = min(r, n - r)
        if r == 0:
            return 1
        res = 1
        for i in range(1, r + 1):
            res = res * (n - i + 1) // i
        return res

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    block_to_ts_idx = []
    ts_to_block_idx = [[] for _ in range(num_t)]
    for j, blk in enumerate(all_k_blocks):
        idxs = []
        for ts in itertools.combinations(blk, t):
            idx = ts_index[ts]
            idxs.append(idx)
            ts_to_block_idx[idx].append(j)
        block_to_ts_idx.append(idxs)

    # Fast Greedy Implementation
    def greedy(seed_val, alpha=1.0):
        rng = np.random.RandomState(seed_val)
        gain = np.full(num_k, comb(k, t), dtype=np.int32)
        uncovered = np.ones(num_t, dtype=bool)
        selected = np.zeros(num_k, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 15.0:
                return None
            mg = int(np.max(gain))
            if mg <= 0:
                break
            cands = np.where((gain == mg) & (~selected))[0]
            if len(cands) == 0:
                break
            
            # Randomized selection
            if alpha < 1.0 and len(cands) > 1:
                # Simple random pick from top candidates for diversity
                b = cands[rng.randint(0, len(cands))]
            else:
                b = cands[rng.randint(0, len(cands))]
            
            sol.append(int(b))
            selected[b] = True
            gain[b] = -1
            
            # Update gains
            for ti in block_to_ts_idx[b]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    for nb in ts_to_block_idx[ti]:
                        if not selected[nb]:
                            gain[nb] -= 1
        return sol

    def idx_to_blocks(idxs):
        return [list(all_k_blocks[i]) for i in idxs]

    def remove_redundant(blks):
        if not blks:
            return blks
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blks)):
                if time.time() > deadline - 5.0:
                    return blks
                red = True
                for ts in itertools.combinations(blks[i], t):
                    if cc[ts_index[ts]] == 1:
                        red = False
                        break
                if red:
                    rem = blks.pop(i)
                    for ts in itertools.combinations(rem, t):
                        cc[ts_index[ts]] -= 1
                    changed = True
                    break
        return blks

    def swap_search(blks, passes=20):
        if not blks:
            return blks, False
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
        sol_set = set()
        for b in blks:
            sol_set.add(tuple(b))
        improved = False
        for _ in range(passes):
            if time.time() > deadline - 5.0:
                break
            found = False
            for i in range(len(blks)):
                if time.time() > deadline - 5.0:
                    break
                ut = []
                for ts in itertools.combinations(blks[i], t):
                    idx = ts_index[ts]
                    if cc[idx] == 1:
                        ut.append(idx)
                if not ut:
                    continue
                # Find a block that covers all unique t-subsets of blks[i]
                cands = set(ts_to_block_idx[ut[0]])
                for ti in ut[1:]:
                    cands.intersection_update(ts_to_block_idx[ti])
                    if not cands:
                        break
                if not cands:
                    continue
                valid = [c for c in cands if tuple(all_k_blocks[c]) not in sol_set]
                if not valid:
                    continue
                ni = random.choice(valid)
                nb = list(all_k_blocks[ni])
                old = blks[i]
                for ts in itertools.combinations(old, t):
                    cc[ts_index[ts]] -= 1
                for ts in itertools.combinations(nb, t):
                    cc[ts_index[ts]] += 1
                sol_set.discard(tuple(old))
                sol_set.add(tuple(nb))
                blks[i] = nb
                improved = True
                found = True
                break
            if not found:
                break
        return blks, improved

    # 1. Generate initial population using Greedy
    pop = []
    for s in range(20):
        if time.time() > deadline - 30.0:
            break
        sol = greedy(s, alpha=1.0)
        if sol:
            blks = idx_to_blocks(sol)
            blks = remove_redundant(blks)
            # Local search polish
            blks, _ = swap_search(blks, passes=10)
            pop.append(blks)
    
    if not pop:
        # Fallback
        sol = greedy(0, 1.0)
        if sol:
            return idx_to_blocks(sol)
        return []

    # Sort population by size
    pop.sort(key=lambda x: len(x))
    best_sol = pop[0]
    best_len = len(best_sol)

    # 2. Genetic Algorithm / Memetic Phase
    pop_size = 10
    # Trim or duplicate to maintain pop size if needed, but we have 20
    if len(pop) > pop_size:
        pop = pop[:pop_size]
    
    # Helper to repair a set of blocks (ensure coverage)
    def repair(blks):
        # Check coverage
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
        
        uncovered = np.where(cc == 0)[0]
        if len(uncovered) == 0:
            return blks
        
        # Greedily add blocks to cover uncovered
        # We need to find blocks that cover the most uncovered subsets
        # This is expensive if done from scratch every time.
        # Simple heuristic: pick random blocks from universe until covered?
        # Better: Use the greedy gain logic restricted to uncovered.
        
        # For speed in GA, if we remove too many, we might just re-greedy from scratch?
        # Or, we can just return None if invalid, and rely on crossover keeping valid parents.
        # Crossover: Union of two valid sets is valid.
        # Mutation: Swap. If we swap a block with another, validity is not guaranteed.
        # So we need a fast validity check and repair.
        
        # Fast repair:
        # Identify uncovered t-subsets.
        # For each uncovered t-subset, pick a block that covers it.
        # To minimize additions, pick blocks that cover multiple uncovered subsets.
        
        # This is essentially a set cover problem on the uncovered subsets.
        # Approximation:
        # 1. Calculate gain for all blocks w.r.t uncovered.
        # 2. Pick best, add, update uncovered.
        
        if time.time() > deadline - 2.0:
            return None
            
        # Use a simplified greedy for repair
        # We need a list of blocks not in blks
        blks_set = set(map(tuple, blks))
        
        # Calculate initial gains for all blocks not in blks
        # gain[b] = number of uncovered t-subsets covered by b
        gains = np.zeros(num_k, dtype=np.int32)
        for u_idx in uncovered:
            for b_idx in ts_to_block_idx[u_idx]:
                if tuple(all_k_blocks[b_idx]) not in blks_set:
                    gains[b_idx] += 1
        
        while len(uncovered) > 0:
            if time.time() > deadline - 2.0:
                return None
            max_gain = np.max(gains)
            if max_gain == 0:
                return None # Should not happen if universe is complete
            
            cands = np.where(gains == max_gain)[0]
            # Pick one
            b_idx = cands[random.randint(0, len(cands)-1)]
            blks.append(list(all_k_blocks[b_idx]))
            blks_set.add(tuple(all_k_blocks[b_idx]))
            
            # Update uncovered and gains
            for ti in block_to_ts_idx[b_idx]:
                if cc[ti] == 0: # Was uncovered
                    cc[ti] = 1
                    # Remove from uncovered list (lazy deletion or set)
                    # We can just recalculate uncovered at the end of loop? No, too slow.
                    # We can mark it.
                    # Let's just decrement a counter?
                    # Actually, we can just re-scan uncovered list if it's small.
                    pass
            
            # Recalculate uncovered efficiently?
            # Since we added a block, some uncovered became covered.
            # We need to update the `uncovered` array or list.
            # For simplicity in this repair, let's just check coverage again.
            # It's O(num_t) per addition.
            # If we add many, it's slow.
            # Optimization: Only check the t-subsets of the added block.
            
            # Let's refine:
            # We maintain a boolean array `covered`
            # Initially, `covered` is cc > 0.
            # When we add a block, we update `covered` for its t-subsets.
            # Then `uncovered` is `~covered`.
            
            # Let's rewrite repair with this in mind for clarity, though it's called rarely.
        
        # Re-verify validity
        cc_final = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc_final[ts_index[ts]] += 1
        if np.any(cc_final == 0):
            return None
        return blks

    # Improved Repair
    def repair_fast(blks):
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ts in itertools.combinations(b, t):
                cc[ts_index[ts]] += 1
        
        uncovered_mask = (cc == 0)
        num_uncovered = np.sum(uncovered_mask)
        if num_uncovered == 0:
            return blks
        
        blks_set = set(map(tuple, blks))
        
        # Gains for blocks not in blks
        # Only consider blocks that cover at least one uncovered subset
        # We can iterate through uncovered subsets and accumulate gains
        gains = np.zeros(num_k, dtype=np.int32)
        for u_idx in np.where(uncovered_mask)[0]:
            for b_idx in ts_to_block_idx[u_idx]:
                if tuple(all_k_blocks[b_idx]) not in blks_set:
                    gains[b_idx] += 1
        
        while np.sum(uncovered_mask) > 0:
            if time.time() > deadline - 2.0:
                return None
            max_gain = np.max(gains)
            if max_gain == 0:
                return None
            
            cands = np.where(gains == max_gain)[0]
            b_idx = cands[random.randint(0, len(cands)-1)]
            new_block = list(all_k_blocks[b_idx])
            blks.append(new_block)
            blks_set.add(tuple(new_block))
            
            # Update coverage
            for ti in block_to_ts_idx[b_idx]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    # Update gains for other blocks that also cover this ti
                    for nb in ts_to_block_idx[ti]:
                        if tuple(all_k_blocks[nb]) not in blks_set:
                            gains[nb] -= 1
                            
        return blks

    # GA Loop
    num_generations = 50
    for gen in range(num_generations):
        if time.time() > deadline - 10.0:
            break
        
        new_pop = []
        
        # Elitism: keep best
        new_pop.append(best_sol)
        
        for _ in range(pop_size - 1):
            # Selection
            p1 = pop[random.randint(0, len(pop)-1)]
            p2 = pop[random.randint(0, len(pop)-1)]
            
            # Crossover: Union
            child = p1 + p2
            # Remove duplicates
            child_tuples = list(set(map(tuple, child)))
            child = [list(t) for t in child_tuples]
            
            # Remove redundancy
            child = remove_redundant(child)
            
            # Mutation: Swap
            if child and random.random() < 0.5:
                # Pick a random block to swap
                idx = random.randint(0, len(child)-1)
                old_block = child[idx]
                
                # Find a replacement
                # We can pick a random block from the universe
                # Or a block that covers many of the same t-subsets
                # For simplicity, random block from universe
                new_block_idx = random.randint(0, num_k-1)
                new_block = list(all_k_blocks[new_block_idx])
                
                # Ensure it's not already in child
                if tuple(new_block) not in set(map(tuple, child)):
                    child[idx] = new_block
                    # Repair
                    child = repair_fast(child)
                    if child is None:
                        # Repair failed, discard or revert
                        child = p1 # Revert to parent
                    else:
                        child = remove_redundant(child)
            
            # Local Search
            child, _ = swap_search(child, passes=5)
            
            # Evaluate
            if len(child) < len(best_sol):
                best_sol = child
                best_len = len(child)
            
            new_pop.append(child)
        
        # Update population
        # Sort by size
        new_pop.sort(key=lambda x: len(x))
        pop = new_pop[:pop_size]
        
        # Update best
        if len(pop[0]) < best_len:
            best_sol = pop[0]
            best_len = len(best_sol)

    return best_sol
