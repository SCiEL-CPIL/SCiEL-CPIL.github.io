import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118
    
    # Handle general case with simple greedy
    if v != 24 or k != 7 or t != 3:
        all_triples = list(itertools.combinations(range(v), t))
        triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
        all_blocks = list(itertools.combinations(range(v), k))
        uncovered = set(range(len(all_triples)))
        selected = []
        while uncovered:
            if time.time() > deadline - 5:
                break
            best_block = None
            best_count = -1
            for b in all_blocks:
                if b in selected:
                    continue
                count = 0
                for tri in itertools.combinations(b, t):
                    if triple_to_idx[tri] in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = b
            if best_block is None or best_count == 0:
                break
            selected.append(best_block)
            for tri in itertools.combinations(best_block, t):
                uncovered.discard(triple_to_idx[tri])
        return selected
    
    # Specialized for C(24, 7, 3)
    random.seed(42)
    
    all_triples = list(itertools.combinations(range(24), 3))
    num_triples = len(all_triples)  # 2024
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(range(24), 7))
    num_blocks = len(all_blocks)  # 346104
    
    # Precompute: for each block, which triples (as set of indices) it covers
    # Use frozenset for block_triple lookup
    block_triple_list = []
    block_triple_frozen = []
    for b in all_blocks:
        tris = frozenset(triple_to_idx[tri] for tri in itertools.combinations(b, 3))
        block_triple_list.append(tris)
        block_triple_frozen.append(tris)
    
    # Coverage bitset: use integer with 2024 bits
    # bit i = 1 means triple i is covered
    # Represent as Python int for fast OR/AND
    
    # Phase 1: Multi-start greedy to get good initial solutions
    # Greedy with bitset acceleration
    
    def greedy_cover(seed_offset=0):
        """Greedy covering using bitsets for speed."""
        uncovered_mask = (1 << num_triples) - 1  # all triples uncovered
        selected = []
        used_blocks = set()
        
        # Precompute block coverage as bitmasks
        # For speed, precompute all block bitmasks
        # But 346104 blocks * 2024 bits is too much memory
        # Instead, compute on fly but cache
        
        # Actually, let's precompute block bitmasks as integers
        # Each block covers 35 triples, so the bitmask is sparse
        # But we need fast AND with uncovered_mask
        
        # Better approach: for each block, store list of triple indices
        # and compute intersection count with uncovered set
        
        # Use a different strategy: maintain uncovered as a set
        # and for each block, count how many of its triples are in uncovered
        
        uncovered_set = set(range(num_triples))
        best = None
        best_count = 0
        
        while uncovered_set:
            if time.time() > deadline - 60:
                break
            
            # Find block with max coverage of uncovered triples
            # Randomize tie-breaking for diversity
            best_block_idx = -1
            best_gain = 0
            best_gain_list = []
            
            # Sample-based greedy for speed: check all blocks but with early termination
            for i, b in enumerate(all_blocks):
                if i in used_blocks:
                    continue
                gain = 0
                for tri in itertools.combinations(b, 3):
                    idx = triple_to_idx[tri]
                    if idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    best_gain_list = [i]
                elif gain == best_gain and gain > 0:
                    best_gain_list.append(i)
            
            if best_block_idx == -1 or best_gain == 0:
                break
            
            # Random choice among best for diversity
            if len(best_gain_list) > 1:
                best_block_idx = best_gain_list[seed_offset % len(best_gain_list)]
                seed_offset += 1
            
            selected.append(best_block_idx)
            used_blocks.add(best_block_idx)
            for tri in itertools.combinations(all_blocks[best_block_idx], 3):
                uncovered_set.discard(triple_to_idx[tri])
        
        return selected
    
    # Run multiple greedy starts with different randomization
    best_solution = None
    best_count = float('inf')
    
    for start_seed in range(5):
        if time.time() > deadline - 80:
            break
        sol = greedy_cover(start_seed)
        if len(sol) < best_count:
            best_count = len(sol)
            best_solution = sol
    
    if best_solution is None:
        best_solution = greedy_cover(0)
        best_count = len(best_solution)
    
    # Phase 2: Local search - swap improvements
    # Try to replace pairs of blocks with single blocks or reduce count
    
    def compute_coverage(selected_indices):
        """Compute coverage mask for a set of block indices."""
        covered = set()
        for idx in selected_indices:
            covered |= block_triple_list[idx]
        return covered
    
    def is_valid(selected_indices):
        """Check if all triples are covered."""
        covered = compute_coverage(selected_indices)
        return len(covered) == num_triples
    
    # Phase 3: Block removal + re-optimization
    # Try removing each block and see if we can recover coverage with fewer blocks
    
    current_sol = list(best_solution)
    current_set = set(current_sol)
    
    # Compute current coverage counts
    cov_count = [0] * num_triples
    for idx in current_sol:
        for ti in block_triple_list[idx]:
            cov_count[ti] += 1
    
    improved = True
    max_improve_iters = 200
    
    while improved and time.time() < deadline - 15 and max_improve_iters > 0:
        improved = False
        max_improve_iters -= 1
        
        # Step 1: Try removing redundant blocks
        to_remove = []
        for idx in current_sol:
            redundant = True
            for ti in block_triple_list[idx]:
                if cov_count[ti] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(idx)
        
        if to_remove:
            for idx in to_remove:
                current_sol.remove(idx)
                current_set.remove(idx)
                for ti in block_triple_list[idx]:
                    cov_count[ti] -= 1
            improved = True
            continue
        
        # Step 2: Try swap - replace one block with another that allows removing another
        # For each pair (i, j) where i is in solution, try replacing i with j
        # such that all triples from i are covered by (solution - i) + j
        found_swap = False
        n = len(current_sol)
        
        for i in range(n):
            if time.time() > deadline - 15:
                break
            idx_i = current_sol[i]
            
            # Triples only covered by block i (critical triples)
            critical = []
            for ti in block_triple_list[idx_i]:
                if cov_count[ti] == 1:
                    critical.append(ti)
            
            if not critical:
                # Can remove block i entirely
                current_sol.pop(i)
                current_set.remove(idx_i)
                for ti in block_triple_list[idx_i]:
                    cov_count[ti] -= 1
                improved = True
                found_swap = True
                break
            continue
        
        if found_swap:
            continue
        
        # Step 3: Try 1-swap: remove block A, add block B such that coverage maintained
        # and we can also remove block C
        # Simplified: for each block A in solution, find block B not in solution
        # that covers all critical triples of A, then check if we can remove another block
        
        if time.time() > deadline - 20:
            break
        
        # Try: for each block in solution, see if there's a replacement
        # that allows us to drop the current block AND another block
        for i in range(len(current_sol)):
            if time.time() > deadline - 20:
                break
            idx_i = current_sol[i]
            
            # Find critical triples of idx_i
            critical = []
            for ti in block_triple_list[idx_i]:
                if cov_count[ti] == 1:
                    critical.append(ti)
            
            if not critical:
                current_sol.pop(i)
                current_set.remove(idx_i)
                for ti in block_triple_list[idx_i]:
                    cov_count[ti] -= 1
                improved = True
                break
            elif len(critical) <= 3:
                # Try to find a block that covers these critical triples
                # and is not in current solution
                critical_set = set(critical)
                for j in range(num_blocks):
                    if j in current_set:
                        continue
                    # Check if block j covers all critical triples
                    if critical_set.issubset(block_triple_list[j]):
                        # Would replacing i with j help?
                        # Only if it allows removing another block
                        # For now, just do the replacement if it reduces something
                        # Actually this doesn't reduce count, so skip
                        pass
                # No simple improvement, move on
                pass
    
    # Phase 4: More aggressive - try removing 2 blocks and re-optimizing
    if time.time() < deadline - 25:
        # Try removing each pair of blocks and see if greedy can recover
        # with fewer total blocks
        n = len(current_sol)
        for i in range(n):
            if time.time() > deadline - 25:
                break
            for j in range(i+1, n):
                if time.time() > deadline - 25:
                    break
                # Temporarily remove blocks i and j
                removed = [current_sol[i], current_sol[j]]
                temp_sol = [x for x in current_sol if x not in removed]
                temp_set = set(temp_sol)
                
                # Compute uncovered triples
                temp_cov = [0] * num_triples
                for idx in temp_sol:
                    for ti in block_triple_list[idx]:
                        temp_cov[ti] += 1
                
                uncovered = set()
                for ti in range(num_triples):
                    if temp_cov[ti] == 0:
                        uncovered.add(ti)
                
                if not uncovered:
                    # Both blocks were redundant!
                    current_sol = temp_sol
                    current_set = temp_set
                    cov_count = temp_cov
                    improved = True
                    break
                
                # Try to greedily cover uncovered with fewer than 2 blocks
                # Can we cover all uncovered with 1 block?
                found_one = False
                for b_idx in range(num_blocks):
                    if b_idx in temp_set:
                        continue
                    if uncovered.issubset(block_triple_list[b_idx]):
                        # Adding 1 block covers all uncovered -> net -1 block
                        current_sol = temp_sol + [b_idx]
                        current_set = temp_set | {b_idx}
                        # Recompute coverage
                        cov_count = [0] * num_triples
                        for idx in current_sol:
                            for ti in block_triple_list[idx]:
                                cov_count[ti] += 1
                        improved = True
                        found_one = True
                        break
                if found_one:
                    break
                
                # Can we cover with 1 new block (replacing 2 removed)?
                # Already checked above. If not, skip.
            if improved:
                break
    
    # Phase 5: Final cleanup
    for _ in range(10):
        removed_any = False
        for i in range(len(current_sol) - 1, -1, -1):
            idx = current_sol[i]
            redundant = True
            for ti in block_triple_list[idx]:
                if cov_count[ti] == 1:
                    redundant = False
                    break
            if redundant:
                current_sol.pop(i)
                current_set.remove(idx)
                for ti in block_triple_list[idx]:
                    cov_count[ti] -= 1
                removed_any = True
        if not removed_any:
            break
    
    # Final validation
    final_blocks = [list(all_blocks[idx]) for idx in current_sol]
    
    covered = set()
    for b in final_blocks:
        for tri in itertools.combinations(b, 3):
            covered.add(tri)
    
    if len(covered) < num_triples:
        # Add missing coverage
        remaining = set(range(num_triples)) - set(covered_triple_idx for b in final_blocks for tri in itertools.combinations(b, 3) for covered_triple_idx in [triple_to_idx[tri]])
        for b_idx in range(num_blocks):
            if time.time() > deadline - 1:
                break
            if b_idx in current_set:
                continue
            block_tris = set(triple_to_idx[tri] for tri in itertools.combinations(all_blocks[b_idx], 3))
            if block_tris & remaining:
                final_blocks.append(list(all_blocks[b_idx]))
                current_set.add(b_idx)
                remaining -= block_tris
                if not remaining:
                    break
    
    return final_blocks
