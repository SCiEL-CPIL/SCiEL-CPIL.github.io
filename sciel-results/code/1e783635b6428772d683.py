import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Precompute all triples
    triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    # Generate all candidate blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build masks for each block
    block_masks = np.zeros(num_blocks, dtype=np.uint64) if num_triples <= 64 else [0] * num_blocks
    
    # For v=24, t=3, num_triples = 2024, so we need a list of ints or a different approach
    # Let's use a list of Python ints for masks since 2024 bits is too many for uint64
    block_masks = [0] * num_blocks
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[triple]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Precompute: for each triple, which blocks contain it
    # triple_to_blocks: list of block indices
    triple_to_blocks = [[] for _ in range(num_triples)]
    for i in range(num_blocks):
        mask = block_masks[i]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            triple_to_blocks[idx].append(i)
            m ^= lsb
    
    # Greedy set cover with weighted scoring
    def run_weighted_greedy(seed_val):
        random.seed(seed_val)
        uncovered = (1 << num_triples) - 1
        selected_indices = []
        
        # Track coverage count for each triple
        coverage_count = [0] * num_triples
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_score = -1
            best_candidates = []
            
            for i in range(num_blocks):
                # Score = number of uncovered triples covered, weighted by rarity
                mask = block_masks[i]
                # Count uncovered triples in this block
                gain_mask = uncovered & mask
                if gain_mask == 0:
                    continue
                
                # Weighted score: sum of 1/coverage_count for each uncovered triple
                # But coverage_count is 0 for uncovered, so use a proxy
                # Simpler: just count uncovered, with tie-breaking by random
                gain = gain_mask.bit_count()
                
                # Add a small random perturbation for tie-breaking
                score = gain + random.random() * 0.1
                
                if score > best_score + 1e-9:
                    best_score = score
                    best_candidates = [i]
                elif abs(score - best_score) <= 1e-9:
                    best_candidates.append(i)
            
            if not best_candidates:
                return selected_indices
            
            best_idx = random.choice(best_candidates)
            selected_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return selected_indices
    
    # Redundancy removal
    def remove_redundant(indices):
        if not indices:
            return []
        
        sol_set = set(indices)
        
        # Compute coverage counts
        coverage_count = [0] * num_triples
        for idx in sol_set:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_t = lsb.bit_length() - 1
                coverage_count[idx_t] += 1
                m ^= lsb
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for idx in list(sol_set):
                mask = block_masks[idx]
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    if coverage_count[idx_t] <= 1:
                        is_redundant = False
                        break
                    m ^= lsb
                
                if is_redundant:
                    sol_set.remove(idx)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx_t = lsb.bit_length() - 1
                        coverage_count[idx_t] -= 1
                        m ^= lsb
                    changed = True
        
        return list(sol_set)
    
    # Variable Neighborhood Search (VNS)
    def vns_local_search(initial_indices):
        """
        VNS that tries swapping m blocks at a time (m=1,2,3) to reduce total count.
        For m=1: try replacing one block with another that covers its unique triples.
        For m=2: try replacing two blocks with one or two new blocks.
        """
        sol_set = set(initial_indices)
        if not sol_set:
            return []
        
        # Precompute coverage counts
        def compute_coverage(sol):
            cc = [0] * num_triples
            for idx in sol:
                mask = block_masks[idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx_t = lsb.bit_length() - 1
                    cc[idx_t] += 1
                    m ^= lsb
            return cc
        
        coverage_count = compute_coverage(sol_set)
        best_sol = list(sol_set)
        best_size = len(best_sol)
        
        # Neighborhood sizes to try
        neighborhoods = [1, 2]  # m=1 (single swap), m=2 (pair swap)
        
        for m in neighborhoods:
            if time.time() > deadline:
                break
            
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                
                sol_list = list(sol_set)
                coverage_count = compute_coverage(sol_set)
                
                if m == 1:
                    # Try replacing each block with another that covers its unique triples
                    for i in range(len(sol_list)):
                        if time.time() > deadline:
                            break
                        b1 = sol_list[i]
                        mask1 = block_masks[b1]
                        
                        # Find triples uniquely covered by b1
                        unique_triples = []
                        m_mask = mask1
                        while m_mask:
                            lsb = m_mask & -m_mask
                            idx_t = lsb.bit_length() - 1
                            if coverage_count[idx_t] == 1:
                                unique_triples.append(idx_t)
                            m_mask ^= lsb
                        
                        if not unique_triples:
                            # b1 is redundant, remove it
                            sol_set.remove(b1)
                            improved = True
                            break
                        
                        # Find blocks that cover all unique_triples
                        # Use the rarest triple to get candidates
                        unique_triples.sort(key=lambda x: len(triple_to_blocks[x]))
                        candidates = set(triple_to_blocks[unique_triples[0]])
                        
                        for t_idx in unique_triples[1:]:
                            if not candidates:
                                break
                            candidates &= set(triple_to_blocks[t_idx])
                        
                        # Check if any candidate is not in sol_set and improves
                        for b_new in candidates:
                            if b_new not in sol_set:
                                # Check if replacing b1 with b_new is valid
                                # b_new must cover all unique_triples (already ensured)
                                # And the resulting set must still cover everything
                                # Since b_new covers all unique_triples, and other triples
                                # covered by b1 are covered by others, this should be valid
                                sol_set.remove(b1)
                                sol_set.add(b_new)
                                improved = True
                                break
                        
                        if improved:
                            break
                
                elif m == 2:
                    # Try replacing pairs of blocks
                    for i in range(len(sol_list)):
                        if time.time() > deadline:
                            break
                        for j in range(i + 1, len(sol_list)):
                            if time.time() > deadline:
                                break
                            b1 = sol_list[i]
                            b2 = sol_list[j]
                            mask1 = block_masks[b1]
                            mask2 = block_masks[b2]
                            mask_pair = mask1 | mask2
                            
                            # Find critical triples: covered by b1 or b2 but not by others
                            critical = []
                            m_mask = mask_pair
                            while m_mask:
                                lsb = m_mask & -m_mask
                                idx_t = lsb.bit_length() - 1
                                if coverage_count[idx_t] == 1:
                                    # Only covered by one of b1, b2
                                    critical.append(idx_t)
                                elif coverage_count[idx_t] == 2:
                                    # Covered by exactly two blocks; check if both are b1, b2
                                    if (mask1 & lsb) and (mask2 & lsb):
                                        critical.append(idx_t)
                                m_mask ^= lsb
                            
                            if not critical:
                                # Both blocks are redundant
                                sol_set.remove(b1)
                                sol_set.remove(b2)
                                improved = True
                                break
                            
                            # Try to find one block that covers all critical triples
                            critical.sort(key=lambda x: len(triple_to_blocks[x]))
                            candidates = set(triple_to_blocks[critical[0]])
                            
                            for t_idx in critical[1:]:
                                if not candidates:
                                    break
                                candidates &= set(triple_to_blocks[t_idx])
                            
                            for b_new in candidates:
                                if b_new not in sol_set:
                                    sol_set.remove(b1)
                                    sol_set.remove(b2)
                                    sol_set.add(b_new)
                                    improved = True
                                    break
                            
                            if improved:
                                break
                        
                        if improved:
                            break
                
                if len(sol_set) < best_size:
                    best_size = len(sol_set)
                    best_sol = list(sol_set)
        
        return best_sol
    
    # Run multiple restarts of greedy + VNS
    best_solution = None
    best_size = float('inf')
    
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        greedy_result = run_weighted_greedy(i)
        if greedy_result is None:
            continue
        
        # Remove redundant blocks
        refined = remove_redundant(greedy_result)
        
        if len(refined) < best_size:
            best_size = len(refined)
            best_solution = refined
        
        # Run VNS local search
        vns_result = vns_local_search(refined)
        
        if len(vns_result) < best_size:
            best_size = len(vns_result)
            best_solution = vns_result
    
    if best_solution is None:
        # Fallback
        greedy_result = run_weighted_greedy(0)
        best_solution = remove_redundant(greedy_result) if greedy_result else []
    
    # Convert indices to blocks
    result = [list(all_blocks[i]) for i in best_solution]
    result.sort()
    return result
