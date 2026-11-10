"""Combined program: Structured initialisation + Optimized Local Search with Color Merging."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
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
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        edges = [idx[a][b], idx[a][c], idx[a][d], idx[b][c], idx[b][d], idx[c][d]]
        k4_edges.append(edges)
        
    edge_k4s = [[] for _ in range(k)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_k4s[e].append(k4_idx)
            
    return idx, k4_edges, edge_k4s, edge_list

def _eval_state(colours, k4_edges, min_colours=5):
    """Return (violations, num_colours_used)."""
    violations = 0
    for edges in k4_edges:
        cs = set(colours[e] for e in edges)
        if len(cs) < min_colours:
            violations += 1
    used = len(set(colours))
    return violations, used

def _try_structured(k, n=16, a=1, b=0, c=0, mode='affine'):
    """
    Generate a colouring based on modular arithmetic.
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

def _local_search_optimized(colours, n=16, k4_edges=None, edge_k4s=None, min_colours=5, max_iter=50000, deadline=None, seed=42):
    """
    Local search to minimize (violations, colours_used).
    Key improvements:
    1. Maintains colour_counts for O(1) used_colours update.
    2. Uses 'Color Merging' heuristic when violations are 0.
    3. Uses targeted recoloring for violations.
    """
    if k4_edges is None or edge_k4s is None:
        _, k4_edges, edge_k4s, _ = _build_k4_structure(n)
        
    rng = random.Random(seed)
    current_colours = colours[:]
    
    # Initial state
    k4_distinct_counts = [0] * len(k4_edges)
    for i, edges in enumerate(k4_edges):
        k4_distinct_counts[i] = len(set(current_colours[e] for e in edges))
        
    current_viol = sum(1 for c in k4_distinct_counts if c < min_colours)
    colour_counts = np.zeros(max(current_colours) + 1, dtype=int)
    for c in current_colours:
        colour_counts[c] += 1
    current_used = np.sum(colour_counts > 0)
    
    best_colours = current_colours[:]
    best_viol = current_viol
    best_used = current_used
    best_key = (best_viol, best_used)
    
    # Helper to apply a move and return new state metrics
    def apply_move(edge_idx, new_c, old_c):
        # Update colour counts
        if colour_counts[old_c] == 1:
            current_used_delta = -1
        else:
            current_used_delta = 0
            
        if colour_counts[new_c] == 0:
            current_used_delta += 1
            
        colour_counts[old_c] -= 1
        colour_counts[new_c] += 1
        
        # Update K4 distinct counts
        affected_k4s = edge_k4s[edge_idx]
        viol_delta = 0
        for k4_idx in affected_k4s:
            old_count = k4_distinct_counts[k4_idx]
            edges = k4_edges[k4_idx]
            # Recalculate distinct count for this K4
            # Optimization: We know only edge_idx changed.
            # We can check if old_c is still in the K4 (other than edge_idx)
            # and if new_c is already in the K4.
            
            # Simple recalc is fast enough for 6 edges
            cs = set(current_colours[e] for e in edges)
            new_count = len(cs)
            k4_distinct_counts[k4_idx] = new_count
            
            if old_count < min_colours and new_count >= min_colours:
                viol_delta -= 1
            elif old_count >= min_colours and new_count < min_colours:
                viol_delta += 1
                
        return viol_delta, current_used_delta

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        if current_viol == 0 and current_used <= 13:
            break
            
        if current_viol > 0:
            # Phase 1: Fix Violations
            # Find a violating K4
            viol_k4_indices = [i for i in range(len(k4_edges)) if k4_distinct_counts[i] < min_colours]
            if not viol_k4_indices:
                # Resync if inconsistency
                current_viol, current_used = _eval_state(current_colours, k4_edges, min_colours)
                for i, edges in enumerate(k4_edges):
                    k4_distinct_counts[i] = len(set(current_colours[e] for e in edges))
                colour_counts = np.zeros(max(current_colours) + 1, dtype=int)
                for c in current_colours:
                    colour_counts[c] += 1
                current_used = np.sum(colour_counts > 0)
                viol_k4_indices = [i for i in range(len(k4_edges)) if k4_distinct_counts[i] < min_colours]
                if not viol_k4_indices:
                    break
                    
            target_k4_idx = rng.choice(viol_k4_indices)
            target_edges = k4_edges[target_k4_idx]
            
            # Try to change one edge in this K4 to a color that increases distinctness
            # Prioritize colors NOT in this K4
            k4_colors = set(current_colours[e] for e in target_edges)
            candidate_colors = [c for c in range(max(current_colours)+2) if c not in k4_colors]
            # Also allow existing colors if they might help (though unlikely to increase distinct count if already present)
            # Actually, changing to a color already present in K4 doesn't increase distinct count.
            # So we MUST pick a color not in K4 to increase distinct count by 1.
            # If we pick a color in K4, distinct count stays same or drops.
            
            if not candidate_colors:
                # If all colors are in K4, we need to introduce a new one
                candidate_colors = [max(current_colours) + 1]
                
            rng.shuffle(candidate_colors)
            
            best_move = None
            best_move_key = (current_viol, current_used)
            
            for new_c in candidate_colors[:5]: # Limit search
                # Find an edge in target K4 to change
                # Ideally, change an edge whose old color appears multiple times in K4?
                # Or just any edge.
                for edge_idx in target_edges:
                    old_c = current_colours[edge_idx]
                    if old_c == new_c: continue
                    
                    # Simulate
                    current_colours[edge_idx] = new_c
                    viol_delta, used_delta = apply_move(edge_idx, new_c, old_c)
                    
                    new_viol = current_viol + viol_delta
                    new_used = current_used + used_delta
                    new_key = (new_viol, new_used)
                    
                    if new_key < best_move_key:
                        best_move_key = new_key
                        best_move = (edge_idx, new_c, old_c, viol_delta, used_delta)
                    
                    # Revert
                    current_colours[edge_idx] = old_c
                    # Undo apply_move effects
                    colour_counts[old_c] += 1
                    colour_counts[new_c] -= 1
                    # Recalculate k4_distinct_counts for affected
                    affected_k4s = edge_k4s[edge_idx]
                    for k4_idx in affected_k4s:
                        edges = k4_edges[k4_idx]
                        k4_distinct_counts[k4_idx] = len(set(current_colours[e] for e in edges))
                    
                    if best_move:
                        break
                if best_move:
                    break
            
            if best_move:
                edge_idx, new_c, old_c, viol_delta, used_delta = best_move
                current_colours[edge_idx] = new_c
                # Apply permanently
                colour_counts[old_c] -= 1
                colour_counts[new_c] += 1
                current_viol += viol_delta
                current_used += used_delta
                
                if (current_viol, current_used) < best_key:
                    best_key = (current_viol, current_used)
                    best_viol = current_viol
                    best_used = current_used
                    best_colours = current_colours[:]
                    
        else:
            # Phase 2: Reduce Colours (Valid State)
            # Strategy: Try to merge two colors.
            # Pick a color c2 with low frequency, try to recolor all its edges to another color c1.
            # If successful (0 new violations), we reduce used colors by 1.
            
            # Get list of colors currently used
            used_colors = [c for c in range(len(colour_counts)) if colour_counts[c] > 0]
            if len(used_colors) <= 1:
                break
                
            # Sort by frequency, try to eliminate rare colors first
            used_colors.sort(key=lambda c: colour_counts[c])
            
            merge_success = False
            # Try to merge the rarest color
            c_to_merge = used_colors[0]
            edges_of_c = [i for i, c in enumerate(current_colours) if c == c_to_merge]
            
            if not edges_of_c:
                continue
                
            # Try to merge c_to_merge into another color c_target
            # Which c_target? One that is "safe".
            # For each edge in c_to_merge, it is in some K4s.
            # If we change edge e to c_target, we must ensure no K4 containing e drops below 5 colors.
            
            # Heuristic: Try c_targets that are frequent (stable) or random.
            # Let's try a few random c_targets.
            candidates_target = [c for c in used_colors if c != c_to_merge]
            rng.shuffle(candidates_target)
            
            for c_target in candidates_target[:3]:
                # Simulate merging ALL edges of c_to_merge to c_target
                # This is a batch move.
                # We need to check if the batch move is valid.
                
                # Save state
                old_colours = current_colours[:]
                old_k4_counts = k4_distinct_counts[:]
                old_colour_counts = colour_counts.copy()
                
                # Apply batch
                for e in edges_of_c:
                    current_colours[e] = c_target
                    colour_counts[c_to_merge] -= 1
                    colour_counts[c_target] += 1
                    
                # Recalculate violations for affected K4s
                # Affected K4s are those containing any edge in edges_of_c
                affected_k4_set = set()
                for e in edges_of_c:
                    for k4_idx in edge_k4s[e]:
                        affected_k4_set.add(k4_idx)
                        
                new_viol = 0
                for k4_idx in affected_k4_set:
                    edges = k4_edges[k4_idx]
                    new_count = len(set(current_colours[e] for e in edges))
                    k4_distinct_counts[k4_idx] = new_count
                    if new_count < min_colours:
                        new_viol += 1
                # Add violations from unaffected K4s (which haven't changed)
                for k4_idx in range(len(k4_edges)):
                    if k4_idx not in affected_k4_set:
                        if k4_distinct_counts[k4_idx] < min_colours:
                            new_viol += 1
                            
                new_used = np.sum(colour_counts > 0)
                
                if new_viol == 0:
                    # Merge successful!
                    merge_success = True
                    current_viol = 0
                    current_used = new_used
                    if (0, current_used) < best_key:
                        best_key = (0, current_used)
                        best_viol = 0
                        best_used = current_used
                        best_colours = current_colours[:]
                    break
                else:
                    # Revert
                    current_colours = old_colours
                    k4_distinct_counts = old_k4_counts
                    colour_counts = old_colour_counts
                    
            if not merge_success:
                # If simple merge fails, try single edge recoloring to reduce "risk"
                # or just accept a random perturbation to escape local minimum
                if rng.random() < 0.05:
                    e = rng.randrange(len(current_colours))
                    old_c = current_colours[e]
                    new_c = rng.randint(0, max(current_colours) + 1)
                    
                    current_colours[e] = new_c
                    colour_counts[old_c] -= 1
                    colour_counts[new_c] += 1
                    
                    affected_k4s = edge_k4s[e]
                    new_viol = 0
                    for k4_idx in affected_k4s:
                        edges = k4_edges[k4_idx]
                        new_count = len(set(current_colours[ee] for ee in edges))
                        k4_distinct_counts[k4_idx] = new_count
                        if new_count < min_colours:
                            new_viol += 1
                    # Check rest
                    for k4_idx in range(len(k4_edges)):
                        if k4_idx not in affected_k4s:
                            if k4_distinct_counts[k4_idx] < min_colours:
                                new_viol += 1
                                
                    new_used = np.sum(colour_counts > 0)
                    
                    # Accept if not worse in violations, or if it reduces used
                    if new_viol == 0 and new_used < current_used:
                        current_viol = 0
                        current_used = new_used
                        if (0, current_used) < best_key:
                            best_key = (0, current_used)
                            best_viol = 0
                            best_used = current_used
                            best_colours = current_colours[:]
                    else:
                        # Revert
                        current_colours[e] = old_c
                        colour_counts[old_c] += 1
                        colour_counts[new_c] -= 1
                        for k4_idx in affected_k4s:
                            edges = k4_edges[k4_idx]
                            k4_distinct_counts[k4_idx] = len(set(current_colours[ee] for ee in edges))

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
    deadline = start_time + 130  # Strict safety margin
    
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
        for _ in range(5):
            a = random.randint(1, k-1)
            b = random.randint(0, k-1)
            c = random.randint(0, k-1)
            candidates.append(_try_structured(k, n, a=a, b=b, c=c, mode='affine'))
            
        candidates.append(_try_structured(k, n, mode='sum'))
        candidates.append(_try_structured(k, n, mode='diff'))
        candidates.append(_try_structured(k, n, mode='prod'))
        
    # 3. Random initialisations
    for k in [13, 14, 15]:
        for _ in range(5):
            colours = [random.randint(0, k-1) for _ in range(n*(n-1)//2)]
            candidates.append(colours)
            
    # Evaluate and select best starting points
    evals = []
    for c in candidates:
        v, u = _eval_state(c, k4_edges)
        evals.append((v, u, c))
        
    evals.sort(key=lambda x: (x[0], x[1]))
    
    # Run local search on the best few candidates
    for v, u, c in evals[:5]:
        if time.time() > deadline - 10:
            break
            
        if best_viol == 0 and best_used <= 13:
            break
            
        seed = int(time.time() * 1000) % 10000
        res_colours, res_viol, res_used = _local_search_optimized(
            c, n, k4_edges, edge_k4s, min_colours=5, 
            max_iter=20000, deadline=deadline - 5, seed=seed
        )
        
        if (res_viol, res_used) < best_key:
            best_key = (res_viol, res_used)
            best_viol = res_viol
            best_used = res_used
            best_colours = res_colours
            
        if res_viol == 0:
            comp_colours, comp_used = _compress_colours(res_colours)
            if comp_used < best_used:
                best_used = comp_used
                best_colours = comp_colours
                best_viol = 0
                best_key = (0, best_used)
                
            # Try further local search to reduce colours
            if time.time() < deadline - 10 and best_used > 13:
                res_colours2, res_viol2, res_used2 = _local_search_optimized(
                    comp_colours, n, k4_edges, edge_k4s, min_colours=5,
                    max_iter=10000, deadline=deadline - 5, seed=seed+1
                )
                if res_viol2 == 0 and res_used2 < best_used:
                    best_used = res_used2
                    best_colours = res_colours2
                    best_key = (0, best_used)
                    best_colours, best_used = _compress_colours(best_colours)
                    best_key = (0, best_used)

    # Fallback
    if best_colours is None or best_viol > 0:
        best_colours = _try_structured(17, n, mode='sum')
        best_colours, best_used = _compress_colours(best_colours)
        best_viol = 0
        best_key = (0, best_used)
        
    return best_colours
# EVOLVE-BLOCK-END
