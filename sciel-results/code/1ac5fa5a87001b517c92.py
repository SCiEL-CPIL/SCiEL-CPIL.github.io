import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute all t-sets and their indices
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, compute which t-sets it covers (as a set of tset indices)
    block_tsets = []
    for block in all_blocks:
        tset_indices = set()
        for ts in itertools.combinations(block, t):
            tset_indices.add(tset_to_idx[ts])
        block_tsets.append(tset_indices)
    
    # Build tset_to_block_indices for fast lookup
    tset_to_block_indices = [[] for _ in range(n_tsets)]
    for b_idx in range(n_blocks):
        for ts_idx in block_tsets[b_idx]:
            tset_to_block_indices[ts_idx].append(b_idx)
    
    # Convert block_tsets to numpy arrays for fast boolean operations
    block_tsets_np = []
    for bs in block_tsets:
        arr = np.zeros(n_tsets, dtype=bool)
        for idx in bs:
            arr[idx] = True
        block_tsets_np.append(arr)
    
    # ============================================================
    # PHASE 1: Structured initialization using Latin square / affine geometry
    # ============================================================
    # For v=24, k=7, t=3: Use a structured approach based on partitioning
    # into 3 groups of 8 points, using Fano-plane-like structures.
    # The Fano plane has 7 points and 7 lines (3-subsets). We can use
    # affine geometry AG(3,2) which has 8 points.
    
    def _structured_initialization(v, k, t, rng):
        """Generate a structured initial covering using algebraic symmetry."""
        if v != 24 or k != 7 or t != 3:
            # Fallback to random greedy for non-standard parameters
            return _random_greedy(v, k, t, all_blocks, block_tsets_np, n_tsets, n_blocks, tset_to_block_indices, rng)
        
        # Partition 24 points into 3 groups of 8
        # Group 0: points 0-7, Group 1: points 8-15, Group 2: points 16-23
        groups = [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]
        
        # For each group of 8, generate 7 blocks of size 7 using the Fano plane structure
        # The Fano plane on 7 points {0,...,6} has lines:
        # {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
        # Each point is in 3 lines.
        # We create 7 blocks per group by taking 7 of the 8 points (omitting one).
        # But we need blocks of size 7 from 24 points that cover all 3-subsets.
        
        # Better approach: Use a 3x8 grid structure
        # Create blocks that span across groups to cover cross-group 3-subsets
        
        blocks_chosen = []
        
        # Step 1: Cover all 3-subsets within each group of 8
        # For 8 points, we need to cover C(8,3)=56 3-subsets with blocks of size 7
        # A block of size 7 from 8 points misses exactly 1 point, so it covers all 3-subsets
        # except those containing the missing point: C(7,2)=21 3-subsets are covered.
        # Actually, a 7-subset of 8 points covers C(7,3)=35 of the 56 3-subsets.
        # To cover all 56, we need at least ceil(56/35) = 2 blocks per group.
        # Let's use 4 blocks per group (omitting each of 4 different points)
        for g_idx, group in enumerate(groups):
            # Use 4 blocks, each omitting a different point
            for omit_idx in range(4):
                block = [p for i, p in enumerate(group) if i != omit_idx]
                if len(block) == 7:
                    blocks_chosen.append(block)
        
        # Step 2: Cover cross-group 3-subsets
        # 3-subsets can be: (3 from same group) - already covered above
        #                   (2 from group A, 1 from group B)
        #                   (2 from group A, 1 from group C)
        #                   (1 from group A, 2 from group B)
        #                   (1 from group A, 2 from group C)
        #                   (1 from group A, 1 from group B, 1 from group C)
        
        # For cross-group coverage, create blocks with mixed points
        # Strategy: For each pair of groups, create blocks with 4 from one, 3 from other
        # Or: 3 from group A, 2 from group B, 2 from group C, etc.
        
        # Let's use a systematic approach:
        # Create blocks with composition (4,2,1), (3,3,1), (3,2,2), (2,3,1), (2,2,3), (1,4,1), etc.
        
        # Simpler: Use random mixing but ensure good coverage
        # Generate additional random blocks to cover remaining 3-subsets
        
        # First, check what's covered
        covered = np.zeros(n_tsets, dtype=bool)
        for block in blocks_chosen:
            b_idx = all_blocks.index(tuple(block)) if tuple(block) in [tuple(b) for b in all_blocks] else -1
            if b_idx >= 0:
                covered |= block_tsets_np[b_idx]
        
        # Add random blocks to cover remaining
        uncovered_count = np.sum(~covered)
        max_additional = 60  # Cap on additional blocks
        added = 0
        for _ in range(max_additional):
            if time.time() > deadline - 20:
                break
            if uncovered_count == 0:
                break
            
            # Find an uncovered 3-set
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            
            rand_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            target_tset = all_tsets[rand_idx]
            
            # Find best block covering this 3-set
            best_block = -1
            best_gain = -1
            for b_idx in tset_to_block_indices[rand_idx]:
                gain = np.sum(block_tsets_np[b_idx] & ~covered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block >= 0 and best_gain > 0:
                blocks_chosen.append(all_blocks[best_block])
                covered |= block_tsets_np[best_block]
                uncovered_count = np.sum(~covered)
                added += 1
        
        return blocks_chosen
    
    def _random_greedy(v, k, t, all_blocks, block_tsets_np, n_tsets, n_blocks, tset_to_block_indices, rng):
        """Standard greedy covering as fallback."""
        covered = np.zeros(n_tsets, dtype=bool)
        blocks_chosen = []
        
        while np.any(~covered):
            if time.time() > deadline - 20:
                break
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            
            rand_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            
            best_block = -1
            best_gain = -1
            for b_idx in tset_to_block_indices[rand_idx]:
                gain = np.sum(block_tsets_np[b_idx] & ~covered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            
            if best_block >= 0 and best_gain > 0:
                blocks_chosen.append(all_blocks[best_block])
                covered |= block_tsets_np[best_block]
        
        return blocks_chosen
    
    # Run structured initialization with multiple seeds
    best_blocks = None
    best_size = float('inf')
    
    for seed in range(5):
        if time.time() > deadline - 60:
            break
        rng = random.Random(seed)
        blocks = _structured_initialization(v, k, t, rng)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
    
    # Also try pure random greedy for comparison
    for seed in range(3):
        if time.time() > deadline - 60:
            break
        rng = random.Random(100 + seed)
        blocks = _random_greedy(v, k, t, all_blocks, block_tsets_np, n_tsets, n_blocks, tset_to_block_indices, rng)
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
    
    if best_blocks is None:
        rng = random.Random(42)
        best_blocks = _random_greedy(v, k, t, all_blocks, block_tsets_np, n_tsets, n_blocks, tset_to_block_indices, rng)
        best_size = len(best_blocks)
    
    # ============================================================
    # PHASE 2: Fast local search with bitset acceleration
    # ============================================================
    def _fast_local_search(blocks, all_blocks, block_tsets_np, n_tsets, n_blocks, tset_to_block_indices, rng, time_limit):
        """Fast local search using numpy boolean vectors."""
        deadline_ls = time.time() + time_limit
        
        # Convert blocks to indices
        block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
        current_idx = []
        for b in blocks:
            t = tuple(b)
            if t in block_to_idx:
                current_idx.append(block_to_idx[t])
            else:
                # Should not happen
                current_idx.append(0)
        
        # Compute coverage
        covered = np.zeros(n_tsets, dtype=np.int8)
        for b_idx in current_idx:
            covered += block_tsets_np[b_idx].astype(np.int8)
        
        best_idx = current_idx[:]
        best_uncovered = int(np.sum(covered == 0))
        
        while time.time() < deadline_ls:
            improved = False
            
            # Try removing redundant blocks
            for i in range(len(current_idx)):
                if time.time() > deadline_ls:
                    break
                b_idx = current_idx[i]
                # Check if any t-set is only covered by this block
                only_covered = (covered == 1) & block_tsets_np[b_idx]
                if not np.any(only_covered):
                    # Redundant - remove it
                    del current_idx[i]
                    covered -= block_tsets_np[b_idx].astype(np.int8)
                    uncovered_count = int(np.sum(covered == 0))
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best_idx = current_idx[:]
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: replace one block with another
            for i in range(len(current_idx)):
                if time.time() > deadline_ls:
                    break
                old_b = current_idx[i]
                
                # Find t-sets that become uncovered if we remove old_b
                newly_uncovered = (covered == 1) & block_tsets_np[old_b]
                newly_uncovered_indices = np.where(newly_uncovered)[0]
                
                if len(newly_uncovered_indices) == 0:
                    # Just remove it
                    del current_idx[i]
                    covered -= block_tsets_np[old_b].astype(np.int8)
                    uncovered_count = int(np.sum(covered == 0))
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best_idx = current_idx[:]
                    improved = True
                    break
                
                # Try to find a replacement block that covers all newly_uncovered
                # and maximizes additional coverage
                best_new_b = -1
                best_gain = -1
                
                # Use the first newly uncovered t-set to find candidate blocks
                first_uncovered = newly_uncovered_indices[0]
                for b_idx in tset_to_block_indices[first_uncovered]:
                    if b_idx == old_b or b_idx in current_idx:
                        continue
                    
                    # Check if b_idx covers all newly_uncovered
                    if not np.all(block_tsets_np[b_idx][newly_uncovered_indices]):
                        continue
                    
                    # Count additional coverage
                    gain = int(np.sum(block_tsets_np[b_idx] & (covered == 0)))
                    if gain > best_gain:
                        best_gain = gain
                        best_new_b = b_idx
                
                if best_new_b != -1:
                    # Perform swap
                    covered -= block_tsets_np[old_b].astype(np.int8)
                    covered += block_tsets_np[best_new_b].astype(np.int8)
                    current_idx[i] = best_new_b
                    uncovered_count = int(np.sum(covered == 0))
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best_idx = current_idx[:]
                    improved = True
                    break
            
            if not improved:
                # Try random perturbation to escape local optima
                if time.time() > deadline_ls:
                    break
                for _ in range(20):
                    if time.time() > deadline_ls:
                        break
                    if len(current_idx) < 2:
                        break
                    
                    i, j = rng.sample(range(len(current_idx)), 2)
                    old_i, old_j = current_idx[i], current_idx[j]
                    
                    # Remove both
                    covered -= block_tsets_np[old_i].astype(np.int8)
                    covered -= block_tsets_np[old_j].astype(np.int8)
                    
                    # Pick two random blocks
                    new_i, new_j = rng.sample(range(n_blocks), 2)
                    current_idx[i] = new_i
                    current_idx[j] = new_j
                    covered += block_tsets_np[new_i].astype(np.int8)
                    covered += block_tsets_np[new_j].astype(np.int8)
                    
                    uncovered_count = int(np.sum(covered == 0))
                    if uncovered_count < best_uncovered:
                        best_uncovered = uncovered_count
                        best_idx = current_idx[:]
        
        return [all_blocks[idx] for idx in best_idx]
    
    # Run local search on best solution
    if time.time() < deadline - 10:
        remaining_time = min(80.0, deadline - time.time() - 10)
        rng = random.Random(123)
        improved = _fast_local_search(best_blocks, all_blocks, block_tsets_np, n_tsets, n_blocks, tset_to_block_indices, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(best_size)
    
    # Try more aggressive local search if still above 70
    if time.time() < deadline - 5 and best_size > 70:
        remaining_time = min(30.0, deadline - time.time() - 5)
        rng = random.Random(456)
        improved = _fast_local_search(best_blocks, all_blocks, block_tsets_np, n_tsets, n_blocks, tset_to_block_indices, rng, remaining_time)
        if len(improved) < best_size:
            best_blocks = improved
            best_size = len(improved)
    
    # Final verification
    covered = np.zeros(n_tsets, dtype=bool)
    for block in best_blocks:
        b_idx = all_blocks.index(tuple(block))
        covered |= block_tsets_np[b_idx]
    
    if not np.all(covered):
        # Something went wrong; rebuild with simple greedy
        rng = random.Random(789)
        best_blocks = _random_greedy(v, k, t, all_blocks, block_tsets_np, n_tsets, n_blocks, tset_to_block_indices, rng)
    
    return best_blocks
