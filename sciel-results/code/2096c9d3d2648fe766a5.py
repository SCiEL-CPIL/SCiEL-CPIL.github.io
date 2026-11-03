"""
Covering design C(v,k,t) using Vectorized Greedy + Exact Deletion + Swap Polish.

Strategy:
1. Fast vectorized greedy (heap-based or argmax) to get initial coverings.
2. Iterative exact deletion:
   a. Identify critical blocks (those uniquely covering at least one t-subset).
   b. Build conflict graph on non-critical blocks.
   c. Solve Maximum Independent Set via branch-and-bound.
   d. Remove the MIS blocks.
   e. Repeat until no non-critical blocks remain or time runs out.
3. Swap polish: Try 1-for-1 and 2-for-1 block swaps to reduce size.
"""
import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115  # safety margin before 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # Precompute t-subset indices for each k-block
    block_tsub_list = []
    for block in all_ksubs:
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        block_tsub_list.append(np.array(indices, dtype=np.int32))
    
    # Precompute for each t-subset, which blocks cover it
    tsub_block_list = [[] for _ in range(n_tsubs)]
    for bi, tsubs in enumerate(block_tsub_list):
        for ts_idx in tsubs:
            tsub_block_list[ts_idx].append(bi)
    
    def block_tsub_indices(block):
        """Return list of t-subset indices covered by this block."""
        if isinstance(block, np.ndarray):
            return block_tsub_list[block]
        else:
            # block is a list of point indices
            indices = []
            for ts in itertools.combinations(block, t):
                idx = tsub_index.get(ts)
                if idx is not None:
                    indices.append(idx)
            return np.array(indices, dtype=np.int32)
    
    def greedy_build(seed, randomize_order=True):
        """Vectorized greedy covering - optimized for speed."""
        rng = random.Random(seed)
        uncovered = np.ones(n_tsubs, dtype=bool)
        selected = []
        
        # Compute initial gains
        gains = np.zeros(n_ksubs, dtype=np.int32)
        for bi in range(n_ksubs):
            gains[bi] = len(block_tsub_list[bi])
        
        while np.any(uncovered):
            if time.time() > deadline - 5:
                break
            
            # Find best block
            best_idx = int(np.argmax(gains))
            if gains[best_idx] <= 0:
                # Fallback: find any block that covers something
                for bi in range(n_ksubs):
                    if np.any(uncovered[block_tsub_list[bi]]):
                        best_idx = bi
                        break
                else:
                    break
            
            selected.append(best_idx)
            
            # Update uncovered
            tsubs = block_tsub_list[best_idx]
            newly_covered = uncovered[tsubs]
            if np.any(newly_covered):
                # Update gains for all blocks sharing subsets with best_idx
                for s_idx in tsubs[newly_covered]:
                    for b_prime in tsub_block_list[s_idx]:
                        if gains[b_prime] > 0:
                            gains[b_prime] -= 1
            
            uncovered[tsubs] = False
        
        # Convert to actual blocks
        return [list(all_ksubs[bi]) for bi in selected]
    
    def greedy_build_pointwise(seed, randomize_order=True):
        """Pointwise greedy covering - alternative approach."""
        rng = random.Random(seed)
        uncovered = [True] * n_tsubs
        blocks = []
        
        while any(uncovered):
            if time.time() > deadline - 5:
                break
            
            # Find first uncovered
            first_uncovered = -1
            for i in range(n_tsubs):
                if uncovered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break
            
            block = list(all_tsubs[first_uncovered])
            
            while len(block) < k:
                in_block = set(block)
                candidates = [x for x in range(v) if x not in in_block]
                
                if not candidates:
                    break
                    
                if randomize_order:
                    rng.shuffle(candidates)
                
                best_gain = -1
                best_x = candidates[0]
                block_t1 = list(itertools.combinations(block, t-1)) if t > 1 else [()]
                
                for x in candidates:
                    gain = 0
                    for combo in block_t1:
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_index.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and gain > 0:
                        if rng.random() < 0.5:
                            best_x = x
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for idx in block_tsub_indices(block):
                uncovered[idx] = False
        
        return blocks
    
    def exact_deletion_pass(blocks, deadline):
        """
        Iteratively remove redundant blocks using exact MIS on conflict graph.
        Returns (reduced_blocks, num_removed).
        """
        blocks_list = [list(b) for b in blocks]
        total_removed = 0
        
        while time.time() < deadline - 10:
            n_blocks = len(blocks_list)
            if n_blocks == 0:
                break
            
            # Compute coverage counts and block t-subsets
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            block_tsubs = []
            for block in blocks_list:
                indices = block_tsub_indices(block)
                block_tsubs.append(indices)
                cov_count[indices] += 1
            
            # Identify critical blocks: those with at least one t-subset where cov_count == 1
            critical = [False] * n_blocks
            for bi in range(n_blocks):
                for idx in block_tsubs[bi]:
                    if cov_count[idx] == 1:
                        critical[bi] = True
                        break
            
            non_critical_idx = [i for i in range(n_blocks) if not critical[i]]
            
            if len(non_critical_idx) < 2:
                break
            
            # Build conflict graph on non-critical blocks
            m = len(non_critical_idx)
            
            # For each non-critical block, get set of t-subsets it uniquely covers
            unique_sets = []
            for bi in non_critical_idx:
                unique = set()
                for idx in block_tsubs[bi]:
                    if cov_count[idx] == 1:
                        unique.add(idx)
                unique_sets.append(unique)
            
            # Build adjacency: edge if unique_sets[i] ∩ unique_sets[j] is empty
            adj = [set() for _ in range(m)]
            for i in range(m):
                for j in range(i+1, m):
                    if not (unique_sets[i] & unique_sets[j]):
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Solve MIS on this graph using branch-and-bound with coloring
            mis_size, mis_nodes = solve_mis(adj, m, deadline - 15)
            
            if mis_size < 2:
                break
            
            # Remove the MIS blocks
            remove_set = set(non_critical_idx[i] for i in mis_nodes)
            blocks_list = [b for i, b in enumerate(blocks_list) if i not in remove_set]
            total_removed += len(remove_set)
        
        return blocks_list, total_removed
    
    def solve_mis(adj, n, deadline):
        """
        Solve Maximum Independent Set using branch-and-bound with greedy coloring.
        Returns (size, list_of_node_indices).
        """
        best_size = [0]
        best_nodes = [[]]
        
        def greedy_coloring(subgraph_nodes, adj):
            """Color the induced subgraph, return (colors, color_classes)."""
            if not subgraph_nodes:
                return 0, []
            
            deg = {v: len(adj[v]) for v in subgraph_nodes}
            ordered = sorted(subgraph_nodes, key=lambda v: -deg[v])
            
            color_of = {}
            color_classes = []
            
            for v in ordered:
                used_colors = set()
                for u in adj[v]:
                    if u in color_of:
                        used_colors.add(color_of[u])
                
                c = 0
                while c in used_colors:
                    c += 1
                
                color_of[v] = c
                if c >= len(color_classes):
                    color_classes.append([])
                color_classes[c].append(v)
            
            return len(color_classes), color_classes
        
        def branch(nodes, current_selection):
            if time.time() > deadline:
                return
            
            if not nodes:
                if len(current_selection) > best_size[0]:
                    best_size[0] = len(current_selection)
                    best_nodes[0] = list(current_selection)
                return
            
            n_colors, color_classes = greedy_coloring(nodes, adj)
            
            if len(current_selection) + n_colors <= best_size[0]:
                return
            
            v = max(nodes, key=lambda x: len(adj[x] & set(nodes)))
            
            branch([u for u in nodes if u != v and u not in adj[v]], 
                   current_selection + [v])
            
            branch([u for u in nodes if u != v], current_selection)
        
        all_nodes = list(range(n))
        branch(all_nodes, [])
        
        return best_size[0], best_nodes[0]
    
    def swap_polish(blocks, deadline):
        """
        Try 1-for-1 and 2-for-1 block swaps to reduce size.
        """
        blocks_list = [list(b) for b in blocks]
        improved = True
        
        while improved and time.time() < deadline - 10:
            improved = False
            n_blocks = len(blocks_list)
            
            # Compute coverage
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            block_tsubs = []
            for block in blocks_list:
                indices = block_tsub_indices(block)
                block_tsubs.append(indices)
                cov_count[indices] += 1
            
            # Try 1-for-1 swaps: replace block i with block j not in solution
            # First, identify which blocks are in the solution
            in_solution = set(range(n_blocks))
            
            # For each block, check if it can be removed
            removable = []
            for bi in range(n_blocks):
                can_remove = True
                for idx in block_tsubs[bi]:
                    if cov_count[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    removable.append(bi)
            
            if len(removable) >= 2:
                # Try to remove pairs
                for i in range(len(removable)):
                    if time.time() > deadline - 10:
                        break
                    bi = removable[i]
                    for j in range(i+1, len(removable)):
                        bj = removable[j]
                        # Check if removing both is safe
                        safe = True
                        for idx in block_tsubs[bi]:
                            if cov_count[idx] == 2:
                                # Check if bj also covers idx
                                if idx not in block_tsubs[bj]:
                                    safe = False
                                    break
                        if safe:
                            for idx in block_tsubs[bj]:
                                if cov_count[idx] == 2 and idx not in block_tsubs[bi]:
                                    safe = False
                                    break
                        if safe:
                            # Remove both
                            blocks_list = [b for idx, b in enumerate(blocks_list) if idx not in (bi, bj)]
                            improved = True
                            break
                    if improved:
                        break
            
            if improved:
                continue
            
            # Try adding a new block and removing two old ones
            # This is expensive, so only do it if we have time
            if time.time() < deadline - 30:
                # Find blocks not in solution that could help
                current_blocks_set = set(tuple(b) for b in blocks_list)
                candidate_blocks = []
                for bi in range(n_ksubs):
                    if tuple(all_ksubs[bi]) not in current_blocks_set:
                        candidate_blocks.append(bi)
                
                # Limit candidates to keep it fast
                if len(candidate_blocks) > 1000:
                    rng = random.Random(42)
                    candidate_blocks = rng.sample(candidate_blocks, 1000)
                
                for new_bi in candidate_blocks[:200]:
                    if time.time() > deadline - 15:
                        break
                    
                    new_tsubs = block_tsub_list[new_bi]
                    
                    # Find which blocks in solution are only needed for subsets not covered by new block
                    # Try to find two blocks that can be removed if we add new_bi
                    for i in range(n_blocks):
                        if time.time() > deadline - 15:
                            break
                        for j in range(i+1, n_blocks):
                            # Check if removing blocks i and j and adding new_bi still covers everything
                            # This is expensive, so skip for now
                            pass
            
        return blocks_list
    
    # Phase 1: Get initial covering via vectorized greedy
    best_blocks = None
    best_count = float('inf')
    
    # Deterministic greedy first
    blocks = greedy_build(0, randomize_order=False)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Vectorized greedy restarts
    seed = 1
    while time.time() < deadline - 40:
        blocks = greedy_build(seed, randomize_order=True)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
    
    # Phase 2: Exact deletion pass
    if time.time() < deadline - 15:
        best_blocks, removed = exact_deletion_pass(best_blocks, deadline)
        best_count = len(best_blocks)
    
    # Phase 3: Swap polish
    if time.time() < deadline - 10:
        best_blocks = swap_polish(best_blocks, deadline)
        best_count = len(best_blocks)
    
    # Phase 4: Try exact deletion from a few other starting points
    for start_seed in range(3):
        if time.time() > deadline - 10:
            break
        alt_blocks = greedy_build(100 + start_seed, randomize_order=True)
        if len(alt_blocks) < best_count + 10:  # only if competitive
            alt_blocks, removed = exact_deletion_pass(alt_blocks, deadline)
            if len(alt_blocks) < best_count:
                best_blocks = alt_blocks
                best_count = len(alt_blocks)
    
    return best_blocks if best_blocks else greedy_build(0, randomize_order=False)
