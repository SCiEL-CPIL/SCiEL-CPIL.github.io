import itertools
import random
import time
import numpy as np
from collections import defaultdict, Counter

def construct(v, k, t):
    deadline = time.time() + 115
    start_time = time.time()
    
    # Precompute all t-subsets and their index mapping
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    def block_tsub_indices(block):
        """Return list of t-subset indices covered by this block."""
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                indices.append(idx)
        return indices
    
    def greedy_build(seed, randomize_order=True, alpha=0.0):
        """
        Randomized greedy covering with optional alpha-randomization.
        alpha in [0, 1]: 0 = deterministic best, 1 = fully random
        """
        rng = random.Random(seed)
        uncovered = [True] * n_tsubs
        blocks = []
        
        # Precompute t-1 subsets for gain calculation
        while any(uncovered):
            # Find first uncovered t-subset
            first_uncovered = -1
            for i in range(n_tsubs):
                if uncovered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break
            
            block = list(all_tsubs[first_uncovered])
            in_block = set(block)
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in in_block]
                if not candidates:
                    break
                
                if randomize_order:
                    rng.shuffle(candidates)
                
                # Compute gain for each candidate
                block_t1 = list(itertools.combinations(block, t-1)) if t > 1 else [()]
                gains = []
                for x in candidates:
                    gain = 0
                    for combo in block_t1:
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    gains.append(gain)
                
                max_gain = max(gains)
                
                if alpha == 0.0:
                    # Deterministic: pick first candidate with max gain
                    for i, g in enumerate(gains):
                        if g == max_gain:
                            block.append(candidates[i])
                            in_block.add(candidates[i])
                            break
                elif alpha == 1.0:
                    # Fully random
                    chosen = rng.choice(candidates)
                    block.append(chosen)
                    in_block.add(chosen)
                else:
                    # Alpha-randomized: with probability alpha, pick randomly among
                    # top candidates; otherwise pick best
                    if rng.random() < alpha:
                        # Pick randomly from top candidates (within 1 of max)
                        top = [i for i, g in enumerate(gains) if g >= max_gain - 1]
                        chosen_idx = rng.choice(top)
                        block.append(candidates[chosen_idx])
                        in_block.add(candidates[chosen_idx])
                    else:
                        for i, g in enumerate(gains):
                            if g == max_gain:
                                block.append(candidates[i])
                                in_block.add(candidates[i])
                                break
            
            block.sort()
            blocks.append(block)
            
            for idx in block_tsub_indices(block):
                uncovered[idx] = False
        
        return blocks
    
    def deletion_pass(blocks, deadline):
        """
        Iterative deletion of redundant blocks using exact MIS on conflict graph.
        """
        blocks_list = [list(b) for b in blocks]
        total_removed = 0
        
        while time.time() < deadline - 10:
            n_blocks = len(blocks_list)
            if n_blocks < 2:
                break
            
            # Compute coverage counts
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            block_tsubs = []
            for block in blocks_list:
                indices = block_tsub_indices(block)
                block_tsubs.append(indices)
                for idx in indices:
                    cov_count[idx] += 1
            
            # Identify critical blocks
            critical = [False] * n_blocks
            for bi in range(n_blocks):
                for idx in block_tsubs[bi]:
                    if cov_count[idx] == 1:
                        critical[bi] = True
                        break
            
            non_critical_idx = [i for i in range(n_blocks) if not critical[i]]
            
            if len(non_critical_idx) < 2:
                break
            
            # Build conflict graph: edge if removing both is safe
            # (they don't share any t-subset with cov_count == 1)
            m = len(non_critical_idx)
            unique_sets = []
            for bi in non_critical_idx:
                unique = set()
                for idx in block_tsubs[bi]:
                    if cov_count[idx] == 1:
                        unique.add(idx)
                unique_sets.append(unique)
            
            adj = [set() for _ in range(m)]
            for i in range(m):
                for j in range(i+1, m):
                    if not (unique_sets[i] & unique_sets[j]):
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Solve MIS
            mis_size, mis_nodes = solve_mis(adj, m, deadline - 15)
            
            if mis_size < 2:
                break
            
            remove_set = set(non_critical_idx[i] for i in mis_nodes)
            blocks_list = [b for i, b in enumerate(blocks_list) if i not in remove_set]
            total_removed += len(remove_set)
        
        return blocks_list, total_removed
    
    def solve_mis(adj, n, deadline):
        """Maximum Independent Set via branch-and-bound with greedy coloring."""
        best_size = [0]
        best_nodes = [[]]
        
        def greedy_coloring(subgraph_nodes, adj):
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
        
        branch(list(range(n)), [])
        return best_size[0], best_nodes[0]
    
    # Phase 1: Multi-start greedy with varied alpha values
    best_blocks = None
    best_count = float('inf')
    
    # Deterministic greedy
    blocks = greedy_build(0, randomize_order=False)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Randomized greedy restarts with different seeds and alpha values
    alpha_values = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    seed = 1
    while time.time() < deadline - 40:
        alpha = alpha_values[(seed - 1) % len(alpha_values)]
        blocks = greedy_build(seed, randomize_order=True, alpha=alpha)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
    
    # Phase 2: Exact deletion on best solution
    if time.time() < deadline - 15:
        best_blocks, removed = deletion_pass(best_blocks, deadline)
        best_count = len(best_blocks)
    
    # Phase 3: Try deletion from other competitive starting points
    for start_seed in [100, 101, 102, 103, 104]:
        if time.time() > deadline - 10:
            break
        alt_blocks = greedy_build(start_seed, randomize_order=True, alpha=0.3)
        if len(alt_blocks) < best_count + 10:
            alt_blocks, removed = deletion_pass(alt_blocks, deadline)
            if len(alt_blocks) < best_count:
                best_blocks = alt_blocks
                best_count = len(alt_blocks)
    
    return best_blocks if best_blocks else greedy_build(0, randomize_order=False)
