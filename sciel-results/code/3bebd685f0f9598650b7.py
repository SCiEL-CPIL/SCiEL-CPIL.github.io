import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    # Parameters: v=24, k=7, t=3
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute all triples and mapping
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # Precompute for each point, the list of triple indices that include it
    # triple_idx_by_point[p] = list of (triple_idx, other_two_points)
    # Actually, let's precompute for each point p, the triples containing p
    triples_by_point = [[] for _ in range(v)]
    for idx, triple in enumerate(triples):
        for p in triple:
            triples_by_point[p].append((idx, triple))
    
    # Precompute for each pair (a,b), the list of triples containing both
    # pair_to_triples[(a,b)] = list of (triple_idx, third_point)
    pair_to_triples = {}
    for idx, triple in enumerate(triples):
        for i in range(t):
            for j in range(i+1, t):
                pair = (triple[i], triple[j])
                if pair not in pair_to_triples:
                    pair_to_triples[pair] = []
                third = triple[0] if i==0 and j==1 else (triple[1] if i==0 and j==2 else triple[2])
                # Actually let me redo this
                pair_set = {triple[i], triple[j]}
                for x in triple:
                    if x not in pair_set:
                        third = x
                        break
                pair_to_triples.setdefault(pair, []).append((idx, third))
    
    # Precompute for each point p and each pair (a,b), the triple index
    # triple_idx_lookup[p][a][b] = triple index for {p, a, b}
    # Use a 3D array for O(1) lookup
    triple_idx_3d = np.full((v, v, v), -1, dtype=np.int32)
    for idx, triple in enumerate(triples):
        a, b, c = triple
        triple_idx_3d[a][b][c] = idx
        triple_idx_3d[a][c][b] = idx
        triple_idx_3d[b][a][c] = idx
        triple_idx_3d[b][c][a] = idx
        triple_idx_3d[c][a][b] = idx
        triple_idx_3d[c][b][a] = idx
    
    # Precompute for each block (7-subset), the list of triple indices it covers
    # And for each block, the set of points
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks_all = len(all_blocks)
    
    # For each block, precompute the triples it covers
    block_triples = []
    for block in all_blocks:
        tris = []
        for triple in itertools.combinations(block, t):
            tris.append(triple_to_idx[triple])
        block_triples.append(tris)
    
    # For each block, precompute the set of points as a frozenset for fast lookup
    block_sets = [frozenset(b) for b in all_blocks]
    
    # For each triple, precompute which blocks cover it
    triple_to_blocks = [[] for _ in range(num_triples)]
    for bi, tris in enumerate(block_triples):
        for tri_idx in tris:
            triple_to_blocks[tri_idx].append(bi)
    
    def greedy_cover(random_state, time_limit=None):
        rng = random.Random(random_state)
        uncovered = np.ones(num_triples, dtype=np.uint8)
        num_uncovered = num_triples
        blocks = []
        points = list(range(v))
        block_indices = list(range(num_blocks_all))
        
        if time_limit is None:
            time_limit = 10.0
        local_deadline = time.time() + time_limit
        
        while num_uncovered > 0 and time.time() < local_deadline:
            # Find first uncovered triple
            start_idx = int(np.argmax(uncovered))
            if uncovered[start_idx] == 0:
                break
            
            target_triple = triples[start_idx]
            target_idx = start_idx
            
            # Find all blocks that cover this triple
            candidate_block_indices = triple_to_blocks[target_idx]
            
            if not candidate_block_indices:
                # This shouldn't happen, but skip
                break
            
            # Score each candidate block by how many new triples it covers
            best_block_idx = -1
            best_gain = -1
            
            for bi in candidate_block_indices:
                gain = 0
                for tri_idx in block_triples[bi]:
                    if uncovered[tri_idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
            
            if best_block_idx == -1:
                break
            
            # Add this block
            block = all_blocks[best_block_idx]
            blocks.append(best_block_idx)
            
            # Update uncovered
            for tri_idx in block_triples[best_block_idx]:
                if uncovered[tri_idx]:
                    uncovered[tri_idx] = 0
                    num_uncovered -= 1
        
        return [all_blocks[bi] for bi in blocks]
    
    def trim_blocks(blocks, time_limit=None):
        if time_limit is None:
            time_limit = 5.0
        local_deadline = time.time() + time_limit
        
        # Convert to block indices
        block_to_bi = {frozenset(b): i for i, b in enumerate(all_blocks)}
        block_indices = []
        for b in blocks:
            bi = block_to_bi.get(frozenset(b))
            if bi is not None:
                block_indices.append(bi)
        
        if not block_indices:
            return []
        
        # Count frequency of each triple
        counts = np.zeros(num_triples, dtype=np.int32)
        for bi in block_indices:
            for tri_idx in block_triples[bi]:
                counts[tri_idx] += 1
        
        active = [True] * len(block_indices)
        changed = True
        while changed and time.time() < local_deadline:
            changed = False
            for i, bi in enumerate(block_indices):
                if not active[i]:
                    continue
                essential = False
                for tri_idx in block_triples[bi]:
                    if counts[tri_idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    changed = True
                    for tri_idx in block_triples[bi]:
                        counts[tri_idx] -= 1
        
        return [all_blocks[block_indices[i]] for i in range(len(block_indices)) if active[i]]
    
    def local_search_trim(blocks, random_state, time_limit=3.0):
        rng = random.Random(random_state)
        block_to_bi = {frozenset(b): i for i, b in enumerate(all_blocks)}
        block_indices = []
        for b in blocks:
            bi = block_to_bi.get(frozenset(b))
            if bi is not None:
                block_indices.append(bi)
        
        if not block_indices:
            return []
        
        counts = np.zeros(num_triples, dtype=np.int32)
        for bi in block_indices:
            for tri_idx in block_triples[bi]:
                counts[tri_idx] += 1
        
        active = [True] * len(block_indices)
        local_deadline = time.time() + time_limit
        
        while time.time() < local_deadline:
            active_indices = [i for i in range(len(block_indices)) if active[i]]
            if not active_indices:
                break
            rng.shuffle(active_indices)
            
            found_removal = False
            for i in active_indices:
                if not active[i]:
                    continue
                bi = block_indices[i]
                essential = False
                for tri_idx in block_triples[bi]:
                    if counts[tri_idx] == 1:
                        essential = True
                        break
                if not essential:
                    active[i] = False
                    found_removal = True
                    for tri_idx in block_triples[bi]:
                        counts[tri_idx] -= 1
            
            if not found_removal:
                break
        
        return [all_blocks[block_indices[i]] for i in range(len(block_indices)) if active[i]]
    
    def swap_local_search(blocks, random_state, time_limit=5.0):
        """
        Try to replace blocks with better alternatives and remove redundant blocks.
        """
        rng = random.Random(random_state)
        block_to_bi = {frozenset(b): i for i, b in enumerate(all_blocks)}
        block_indices = []
        for b in blocks:
            bi = block_to_bi.get(frozenset(b))
            if bi is not None:
                block_indices.append(bi)
        
        if not block_indices:
            return []
        
        n_blocks = len(block_indices)
        counts = np.zeros(num_triples, dtype=np.int32)
        for bi in block_indices:
            for tri_idx in block_triples[bi]:
                counts[tri_idx] += 1
        
        active = [True] * n_blocks
        local_deadline = time.time() + time_limit
        
        while time.time() < local_deadline:
            active_indices = [i for i in range(n_blocks) if active[i]]
            if not active_indices:
                break
            
            rng.shuffle(active_indices)
            made_change = False
            
            for i in active_indices:
                if not active[i]:
                    continue
                
                bi = block_indices[i]
                
                # Check if block i is removable
                essential_triples = []
                for tri_idx in block_triples[bi]:
                    if counts[tri_idx] == 1:
                        essential_triples.append(tri_idx)
                
                if not essential_triples:
                    # Block is removable
                    active[i] = False
                    made_change = True
                    for tri_idx in block_triples[bi]:
                        counts[tri_idx] -= 1
                    continue
                
                # Block has essential triples; try to find a replacement
                # that covers all essential triples
                best_replacement = None
                best_score = float('inf')
                
                # Sample candidate blocks that cover the first essential triple
                first_essential = essential_triples[0]
                candidates = triple_to_blocks[first_essential]
                
                if len(candidates) > 30:
                    candidates = rng.sample(candidates, 30)
                
                for cand_bi in candidates:
                    # Check if cand covers all essential triples
                    covers_all = True
                    cand_tris_set = set(block_triples[cand_bi])
                    for tri_idx in essential_triples:
                        if tri_idx not in cand_tris_set:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Score: number of new triples covered
                    new_triples = 0
                    for tri_idx in block_triples[cand_bi]:
                        if counts[tri_idx] == 0:
                            new_triples += 1
                    
                    if new_triples < best_score:
                        best_score = new_triples
                        best_replacement = cand_bi
                
                if best_replacement is not None and best_replacement != bi:
                    # Replace block i with best_replacement
                    old_bi = bi
                    block_indices[i] = best_replacement
                    
                    # Update counts
                    for tri_idx in block_triples[old_bi]:
                        counts[tri_idx] -= 1
                    for tri_idx in block_triples[best_replacement]:
                        counts[tri_idx] += 1
                    
                    made_change = True
                    break
            
            if not made_change:
                break
        
        return [all_blocks[block_indices[i]] for i in range(n_blocks) if active[i]]
    
    def verify_covering(blocks):
        """Verify that every triple is covered."""
        covered = np.zeros(num_triples, dtype=np.uint8)
        for block in blocks:
            for triple in itertools.combinations(block, t):
                covered[triple_to_idx[triple]] = 1
        return np.all(covered == 1)
    
    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    seed_counter = 0
    while time.time() < deadline:
        # Phase 1: Greedy construction
        blocks = greedy_cover(seed_counter, time_limit=5.0)
        
        # Phase 2: Trim redundant blocks
        blocks = trim_blocks(blocks, time_limit=2.0)
        
        # Phase 3: Local search for more trimming
        blocks = local_search_trim(blocks, seed_counter, time_limit=2.0)
        
        # Phase 4: Swap local search
        blocks = swap_local_search(blocks, seed_counter, time_limit=3.0)
        
        # Verify
        if verify_covering(blocks):
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
                if best_len <= 70:
                    break
        
        seed_counter += 1
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback: return a valid covering (should not happen)
        return greedy_cover(0, time_limit=10.0)

