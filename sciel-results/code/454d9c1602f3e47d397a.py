import time
from itertools import combinations

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) using deterministic backtracking 
    with symmetry breaking and MRV heuristic.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_points = 12
    block_size = 6
    sub_size = 5
    target_blocks = 132
    total_subs = 792 # C(12, 5)
    
    # 1. Precomputation
    # Generate all 6-subsets
    all_blocks = list(combinations(range(n_points), block_size))
    
    # Map each block to the set of 5-subsets it covers
    block_to_subs = {}
    for b in all_blocks:
        subs = tuple(combinations(b, sub_size))
        block_to_subs[b] = subs
        
    # Inverted index: for each 5-subset, which blocks contain it?
    # subs_to_blocks[sub] = list of blocks containing sub
    subs_to_blocks = {sub: [] for sub in combinations(range(n_points), sub_size)}
    for b, subs in block_to_subs.items():
        for s in subs:
            subs_to_blocks[s].append(b)

    # Symmetry Breaking:
    # The automorphism group M12 is 5-transitive. 
    # We can fix the first block to be (0, 1, 2, 3, 4, 5).
    # Any solution can be transformed to include this block.
    first_block = (0, 1, 2, 3, 4, 5)
    
    # Initial state
    covered = set(block_to_subs[first_block])
    chosen_blocks = [first_block]
    
    # We need to find 131 more blocks.
    # To avoid permutations, we enforce lexicographical order on the chosen blocks.
    # last_block is the lexicographically largest block chosen so far.
    last_block = first_block
    
    def backtrack(covered, chosen_blocks, last_block, depth):
        # Check time limit
        if time.time() > deadline:
            return None
            
        if len(chosen_blocks) == target_blocks:
            if len(covered) == total_subs:
                return list(chosen_blocks)
            else:
                return None
        
        # Pruning: 
        # The number of blocks we can still add is limited by the uncovered subsets.
        # Each new block covers 6 subsets.
        # If (target_blocks - len(chosen_blocks)) * 6 > len(uncovered), it's impossible?
        # No, because blocks can share subsets? No, in a Steiner system, no 5-subset is repeated.
        # So every new block must cover 6 *new* subsets.
        # Thus, we need exactly (target_blocks - len(chosen_blocks)) * 6 uncovered subsets remaining.
        # If len(covered) + (target_blocks - len(chosen_blocks)) * 6 != total_subs, it's invalid.
        # Actually, since we start with valid blocks and check conflicts, len(covered) grows by 6 each step.
        # So len(covered) should be 6 * len(chosen_blocks).
        # If this invariant holds, we just need to reach depth 132.
        
        # MRV Heuristic:
        # Find the uncovered 5-subset with the fewest candidate blocks.
        # Candidate blocks for an uncovered subset S are blocks B such that:
        # 1. S in B
        # 2. B is lexicographically greater than last_block (to maintain order)
        # 3. B does not conflict with current covered set (i.e., no 5-subset in B is in covered)
        
        uncovered_subs = [s for s in subs_to_blocks if s not in covered]
        
        if not uncovered_subs:
            return None
            
        # Sort uncovered subs by the number of valid candidate blocks
        # We compute the count of valid candidates for each uncovered sub
        best_sub = None
        best_candidates = None
        min_count = float('inf')
        
        # Optimization: We can iterate through uncovered subs and count candidates.
        # There are at most 792 subs.
        for s in uncovered_subs:
            candidates = []
            for b in subs_to_blocks[s]:
                # Check lexicographical order
                if b <= last_block:
                    continue
                # Check conflict
                # If any subset of b is in covered, conflict
                conflict = False
                for sub in block_to_subs[b]:
                    if sub in covered:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(b)
            
            count = len(candidates)
            if count == 0:
                return None # Dead end
            
            if count < min_count:
                min_count = count
                best_sub = s
                best_candidates = candidates
                if count == 1:
                    break # Can't get better than 1
        
        if best_candidates is None:
            return None
            
        # Try each candidate block
        # Sort candidates to maintain some deterministic order if needed, 
        # but lexicographical order of blocks is already enforced by the filter b > last_block.
        # However, the order in which we try them might affect speed. 
        # Let's just iterate.
        
        for b in best_candidates:
            new_covered = set(covered)
            # Add subsets of b to new_covered
            # We know they are disjoint from covered, so we can just update
            # But for safety and simplicity in backtracking, we create a new set or use a flag.
            # Since we checked conflict, they are disjoint.
            for sub in block_to_subs[b]:
                new_covered.add(sub)
                
            result = backtrack(new_covered, chosen_blocks + [b], b, depth + 1)
            if result is not None:
                return result
                
        return None

    # Start search
    solution = backtrack(covered, chosen_blocks, last_block, 1)
    
    if solution:
        return solution
    else:
        # Fallback: Return empty or a partial result if it fails (should not happen)
        # If it fails, it means the symmetry breaking or logic is flawed.
        # But for S(5,6,12), this is a known unique solution up to isomorphism.
        # If we hit the time limit, we return whatever we have? No, must be valid.
        # If no solution found, return empty list to avoid invalid output?
        # The task says "Maximize the number of blocks". 
        # If we fail to find 132, we should return a valid packing of fewer blocks.
        # But the backtracking is exact. If it returns None, it means no solution with that prefix.
        # Given the symmetry breaking is correct, it should find one.
        
        # Greedy fallback for safety (should not be reached)
        # Simple greedy to get *some* valid packing
        covered = set()
        greedy_blocks = []
        for b in all_blocks:
            subs = block_to_subs[b]
            if not any(s in covered for s in subs):
                greedy_blocks.append(list(b))
                covered.update(subs)
                if len(greedy_blocks) >= target_blocks:
                    break
        return greedy_blocks
