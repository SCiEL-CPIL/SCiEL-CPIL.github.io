"""Combined program: Structured initialisation + Optimized Local Search for colour reduction."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict
import numpy as np

def _build_k4_structure(n=16):
    """
    Precompute:
    1. idx: 2D array mapping (i, j) to edge index.
    2. k4_edges: List of lists, each containing 6 edge indices for a K4.
    3. edge_k4s: List of lists, each containing indices of K4s that include that edge.
    """
    idx = [[-1]*n for _ in range(n)]
    edge_list = []
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            idx[i][j] = k
            idx[j][i] = k
            edge_list.append((i, j))
            k += 1
    
    k4_edges = []
    # We will store the vertex tuple and the edge indices
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        edges = [idx[a][b], idx[a][c], idx[a][d], idx[b][c], idx[b][d], idx[c][d]]
        k4_edges.append(edges)
        
    # Map edge -> list of k4 indices
    edge_k4s = [[] for _ in range(k)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[e].append(k4_idx)
            
    return idx, k4_edges, edge_k4s, edge_list

def _count_violations(colours, k4_edges, min_colours=5):
    """Count K4s with fewer than min_colours distinct colours."""
    violations = 0
    for edges in k4_edges:
        cs = set(colours[e] for e in edges)
        if len(cs) < min_colours:
            violations += 1
    return violations

def _eval_state(colours, k4_edges, min_colours=5):
    """Return (violations, num_colours_used)."""
    viol = _count_violations(colours, k4_edges, min_colours)
    used = len(set(colours))
    return viol, used

def _try_structured(k, n=16, a=1, b=0, c=0, mode='affine'):
    """
    Generate a colouring based on modular arithmetic.
    mode 'affine': (a*(i+j) + b*(j-i) + c) % k
    mode 'sum': (i+j) % k
    mode 'diff': (j-i) % k
    mode 'prod': (i*j) % k
    """
    colours = [0] * (n*(n-1)//2)
    idx_counter = 0
    for i in range(n):
        for j in range(i+1, n):
            if mode == 'affine':
                colours[idx_counter] = (a*(i+j) + b*(j-i) + c) % k
            elif mode == 'sum':
                colours[idx_counter] = (i+j) % k
            elif mode == 'diff':
                colours[idx_counter] = (j-i) % k
            elif mode == 'prod':
                colours[idx_counter] = (i*j) % k
            else:
                colours[idx_counter] = (i+j+i*j) % k
            idx_counter += 1
    return colours

def _local_search_optimized(colours, n=16, k4_edges=None, edge_k4s=None, min_colours=5, max_iter=20000, deadline=None, seed=42):
    """
    Local search to minimize (violations, colours_used).
    Uses edge_k4s to quickly identify affected K4s.
    """
    if k4_edges is None or edge_k4s is None:
        _, k4_edges, edge_k4s, _ = _build_k4_structure(n)
        
    rng = random.Random(seed)
    current_colours = colours[:]
    current_viol, current_used = _eval_state(current_colours, k4_edges, min_colours)
    
    best_colours = current_colours[:]
    best_viol, best_used = current_viol, current_used
    best_key = (best_viol, best_used)
    
    # Cache distinct colour counts for each K4 to speed up updates? 
    # It's complex to maintain. Let's rely on the fact that n=16 is small (1820 K4s).
    # But checking all K4s for every move is 1820 * 6 set ops. ~11k ops per move.
    # 20k moves -> 220M ops. Might be too slow in Python.
    # Optimization: Only check K4s affected by the changed edge.
    
    # Precompute distinct colour count for each K4 initially
    k4_distinct_counts = [0] * len(k4_edges)
    for i, edges in enumerate(k4_edges):
        k4_distinct_counts[i] = len(set(current_colours[e] for e in edges))
        
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if current_viol == 0 and current_used <= 13:
            break
            
        # Select a random edge to modify
        # Bias towards edges in violating K4s if violations exist
        if current_viol > 0:
            # Pick a random violating K4
            viol_k4s = [i for i in range(len(k4_edges)) if k4_distinct_counts[i] < min_colours]
            if not viol_k4s:
                # Inconsistency, recalculate
                current_viol, current_used = _eval_state(current_colours, k4_edges, min_colours)
                for i, edges in enumerate(k4_edges):
                    k4_distinct_counts[i] = len(set(current_colours[e] for e in edges))
                viol_k4s = [i for i in range(len(k4_edges)) if k4_distinct_counts[i] < min_colours]
                if not viol_k4s:
                    break
            target_k4_idx = rng.choice(viol_k4s)
            target_edges = k4_edges[target_k4_idx]
            edge_idx = rng.choice(target_edges)
        else:
            # No violations, try to reduce colours
            # Pick a random edge
            edge_idx = rng.randrange(len(current_colours))
            
        old_colour = current_colours[edge_idx]
        
        # Determine candidate new colours
        # If we have violations, we want to introduce a new colour or change to an existing one that breaks the mono/di/tri/quad color pattern
        # If we have 0 violations, we want to merge colours.
        
        if current_viol > 0:
            # Try all colours 0..max_possible
            # max_possible should be at least current_used
            max_c = max(current_colours) + 1
            # To save time, don't try too many. Try existing colours and a few new ones
            candidates = list(range(max_c))
            rng.shuffle(candidates)
            # Limit candidates to speed up
            candidates = candidates[:20]
        else:
            # Try to change to an existing colour to reduce count
            existing_colours = list(set(current_colours))
            candidates = [c for c in existing_colours if c != old_colour]
            rng.shuffle(candidates)
            candidates = candidates[:10]
            # Also try a new colour? No, we want to reduce.
            
        best_move_colour = None
        best_move_viol = current_viol
        best_move_used = current_used
        best_move_key = (best_move_viol, best_move_used)
        
        for new_c in candidates:
            if new_c == old_colour:
                continue
                
            current_colours[edge_idx] = new_c
            
            # Update affected K4s
            affected_k4s = edge_k4s[edge_idx]
            
            new_viol = current_viol
            # We need to recalculate distinct counts for affected K4s
            # And update global violation count
            # This is tricky because a K4 might be affected by multiple edges, 
            # but we only change one edge at a time.
            
            # Let's just recalculate the distinct count for affected K4s
            # and adjust current_viol accordingly.
            
            # Save old distinct counts for affected K4s
            old_counts = {}
            for k4_idx in affected_k4s:
                old_counts[k4_idx] = k4_distinct_counts[k4_idx]
                
            # Calculate new distinct counts
            for k4_idx in affected_k4s:
                edges = k4_edges[k4_idx]
                cs = set(current_colours[e] for e in edges)
                new_count = len(cs)
                k4_distinct_counts[k4_idx] = new_count
                
                # Update violation count
                if old_counts[k4_idx] < min_colours and new_count >= min_colours:
                    new_viol -= 1
                elif old_counts[k4_idx] >= min_colours and new_count < min_colours:
                    new_viol += 1
            
            # Calculate new used colours
            # This is expensive to do exactly. Approximate or recalculate?
            # Since we only change one edge, the set of colours changes slightly.
            # If new_c is not in colours (excluding this edge), used increases.
            # If old_colour is not in colours (excluding this edge), used decreases.
            
            # Let's just recalculate used for simplicity, it's O(N) = 120
            new_used = len(set(current_colours))
            
            new_key = (new_viol, new_used)
            
            if new_key < best_move_key:
                best_move_key = new_key
                best_move_colour = new_c
                best_move_viol = new_viol
                best_move_used = new_used
                
            # Restore for next candidate? 
            # No, we are testing candidates sequentially. 
            # But we modified current_colours.
            # We must restore current_colours[edge_idx] = old_colour before next candidate
            # And restore k4_distinct_counts
            current_colours[edge_idx] = old_colour
            for k4_idx in affected_k4s:
                k4_distinct_counts[k4_idx] = old_counts[k4_idx]
                
        if best_move_colour is not None:
            # Apply best move
            current_colours[edge_idx] = best_move_colour
            affected_k4s = edge_k4s[edge_idx]
            for k4_idx in affected_k4s:
                edges = k4_edges[k4_idx]
                k4_distinct_counts[k4_idx] = len(set(current_colours[e] for e in edges))
                
            current_viol = best_move_viol
            current_used = best_move_used
            
            # Update best if improved
            if (current_viol, current_used) < best_key:
                best_key = (current_viol, current_used)
                best_viol = current_viol
                best_used = current_used
                best_colours = current_colours[:]
                
        else:
            # No improvement found with current candidates.
            # Try a random perturbation (simulated annealing style)
            if rng.random() < 0.1:
                # Pick a random edge and random colour
                pert_edge = rng.randrange(len(current_colours))
                old_c = current_colours[pert_edge]
                new_c = rng.randint(0, max(current_colours) + 2)
                
                current_colours[pert_edge] = new_c
                affected_k4s = edge_k4s[pert_edge]
                
                old_counts = {}
                new_viol = current_viol
                for k4_idx in affected_k4s:
                    old_counts[k4_idx] = k4_distinct_counts[k4_idx]
                    edges = k4_edges[k4_idx]
                    new_count = len(set(current_colours[e] for e in edges))
                    k4_distinct_counts[k4_idx] = new_count
                    if old_counts[k4_idx] < min_colours and new_count >= min_colours:
                        new_viol -= 1
                    elif old_counts[k4_idx] >= min_colours and new_count < min_colours:
                        new_viol += 1
                        
                new_used = len(set(current_colours))
                
                # Accept if not much worse, or with probability based on "temperature"
                # Simple acceptance: accept if violations don't increase
                if new_viol <= current_viol:
                    current_viol = new_viol
                    current_used = new_used
                    if (current_viol, current_used) < best_key:
                        best_key = (current_viol, current_used)
                        best_viol = current_viol
                        best_used = current_used
                        best_colours = current_colours[:]
                else:
                    # Reject, revert
                    current_colours[pert_edge] = old_c
                    for k4_idx in affected_k4s:
                        k4_distinct_counts[k4_idx] = old_counts[k4_idx]
                        
    return best_colours, best_viol, best_used

def _compress_colours(colours):
    """Remap colours to compact range 0..k-1."""
    colour_map = {}
    next_c = 0
    for c in colours:
        if c not in colour_map:
            colour_map[c] = next_c
            next_c += 1
    return [colour_map[c] for c in colours], len(colour_map)

# EVOLVE-BLOCK-END

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 16
    idx, k4_edges, edge_k4s, edge_list = _build_k4_structure(n)
    
    best_colours = None
    best_viol = float('inf')
    best_used = float('inf')
    best_key = (float('inf'), float('inf'))
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # 1. Known good 17-colouring
    candidates.append(_try_structured(17, n, mode='sum'))
    
    # 2. Try 15, 14, 13 with various affine parameters
    for k in [13, 14, 15, 16]:
        # Random affine parameters
        for _ in range(5):
            a = random.randint(1, k-1)
            b = random.randint(0, k-1)
            c = random.randint(0, k-1)
            candidates.append(_try_structured(k, n, a=a, b=b, c=c, mode='affine'))
            
        # Specific modes
        candidates.append(_try_structured(k, n, mode='sum'))
        candidates.append(_try_structured(k, n, mode='diff'))
        candidates.append(_try_structured(k, n, mode='prod'))
        
    # 3. Random initialisations with low colour counts
    for k in [13, 14, 15]:
        for _ in range(10):
            colours = [random.randint(0, k-1) for _ in range(n*(n-1)//2)]
            candidates.append(colours)
            
    # Evaluate and select best starting points
    # We will run local search on the top few candidates with lowest violations
    evals = []
    for c in candidates:
        v, u = _eval_state(c, k4_edges)
        evals.append((v, u, c))
        
    # Sort by (violations, used)
    evals.sort(key=lambda x: (x[0], x[1]))
    
    # Run local search on the best 5 candidates
    for v, u, c in evals[:5]:
        if time.time() > deadline - 10:
            break
            
        # If we already have a valid solution with <= 13 colours, stop
        if best_viol == 0 and best_used <= 13:
            break
            
        # Run local search
        # Use a seed based on time or index to vary
        seed = int(time.time() * 1000) % 10000
        res_colours, res_viol, res_used = _local_search_optimized(
            c, n, k4_edges, edge_k4s, min_colours=5, 
            max_iter=10000, deadline=deadline - 5, seed=seed
        )
        
        if (res_viol, res_used) < best_key:
            best_key = (res_viol, res_used)
            best_viol = res_viol
            best_used = res_used
            best_colours = res_colours
            
        # If we found a valid solution, try to reduce colours further
        if res_viol == 0:
            # Compress
            comp_colours, comp_used = _compress_colours(res_colours)
            if comp_used < best_used:
                best_used = comp_used
                best_colours = comp_colours
                best_viol = 0
                best_key = (0, best_used)
                
            # Try further local search to reduce colours
            # This is already done in _local_search_optimized when violations are 0
            # But we can run another pass specifically for colour reduction
            if time.time() < deadline - 10 and best_used > 13:
                res_colours2, res_viol2, res_used2 = _local_search_optimized(
                    comp_colours, n, k4_edges, edge_k4s, min_colours=5,
                    max_iter=5000, deadline=deadline - 5, seed=seed+1
                )
                if res_viol2 == 0 and res_used2 < best_used:
                    best_used = res_used2
                    best_colours = res_colours2
                    best_key = (0, best_used)
                    best_colours, best_used = _compress_colours(best_colours)
                    best_key = (0, best_used)

    # Fallback: If no solution found, return the 17-colouring
    if best_colours is None or best_viol > 0:
        # Use the known 17-colouring
        best_colours = _try_structured(17, n, mode='sum')
        best_viol = 0
        best_used = 17
        best_colours, best_used = _compress_colours(best_colours)
        
    return best_colours
# EVOLVE-BLOCK-END
