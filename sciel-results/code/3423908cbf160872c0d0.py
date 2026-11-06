import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Known valid coloring for n=9
    best_result = {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
    
    # Try to find valid colorings for larger n
    # 17 is the current best known. 18/19 are unknown but possible.
    # We prioritize 17 as it is most likely to be found quickly.
    target_ns = [17, 18, 19, 16, 15]
    
    for n in target_ns:
        if time.time() > deadline:
            break
            
        # Precompute cycles for K_n
        vertices = list(range(n))
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        edge_index = {e: i for i, e in enumerate(edges)}
        num_edges = len(edges)
        
        cycles = []
        edge_to_cycles = [[] for _ in range(num_edges)]
        
        # Generate all 6-cycles
        # For n=19, binom(19,6) = 27132. 27132 * 60 = 1.6M cycles.
        # This might be too slow to precompute in Python within time limit if we do it naively.
        # binom(17,6) = 12376. 12376 * 60 = 742k cycles.
        # Precomputation time for 17 is acceptable. For 18/19, it might be risky.
        # We'll only precompute if n <= 17 to be safe, or check time.
        
        if n > 17:
             # Skip precomputation for large n if time is tight, or use a different strategy?
             # For now, let's stick to 17 and below for robustness.
             # If we don't find 17, we fall back to 16.
             continue

        for v_set in itertools.combinations(vertices, 6):
            v0 = v_set[0]
            for perm in itertools.permutations(v_set[1:]):
                if perm[0] > perm[4]:
                    continue
                edge_set = [
                    (min(v0, perm[0]), max(v0, perm[0])),
                    (min(perm[0], perm[1]), max(perm[0], perm[1])),
                    (min(perm[1], perm[2]), max(perm[1], perm[2])),
                    (min(perm[2], perm[3]), max(perm[2], perm[3])),
                    (min(perm[3], perm[4]), max(perm[3], perm[4])),
                    (min(perm[4], v0), max(perm[4], v0))
                ]
                edge_idxs = [edge_index[e] for e in edge_set]
                cycles.append(edge_idxs)
                for idx in edge_idxs:
                    edge_to_cycles[idx].append(len(cycles) - 1)
        
        num_cycles = len(cycles)
        
        def solve_n(n):
            if time.time() > deadline:
                return None
            
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            cycle_counts = [[0]*4 for _ in range(num_cycles)]
            
            for c_idx, c_edges in enumerate(cycles):
                for e_idx in c_edges:
                    cycle_counts[c_idx][colors[e_idx]] += 1
            
            violations = 0
            for c_idx in range(num_cycles):
                if any(count == 6 for count in cycle_counts[c_idx]):
                    violations += 1
            
            if violations == 0:
                return colors
            
            # Local search with restarts
            max_restarts = 50
            for r in range(max_restarts):
                if time.time() > deadline:
                    return None
                
                # Reset if stuck? No, just run search.
                # We'll do a fixed number of steps or until no improvement.
                steps_without_improvement = 0
                max_steps_no_improve = 5000
                
                while steps_without_improvement < max_steps_no_improve:
                    if time.time() > deadline:
                        return None
                    
                    if violations == 0:
                        return colors
                    
                    # Pick random edge
                    e_idx = random.randint(0, num_edges - 1)
                    old_color = colors[e_idx]
                    c_indices = edge_to_cycles[e_idx]
                    
                    best_new_color = -1
                    best_delta = 0
                    
                    for new_color in range(4):
                        if new_color == old_color:
                            continue
                        
                        delta_v = 0
                        for c_idx in c_indices:
                            cc = cycle_counts[c_idx]
                            was_violation = (cc[old_color] == 6)
                            
                            cc[old_color] -= 1
                            cc[new_color] += 1
                            is_violation = (cc[new_color] == 6)
                            
                            cc[old_color] += 1
                            cc[new_color] -= 1
                            
                            if was_violation and not is_violation:
                                delta_v -= 1
                            elif not was_violation and is_violation:
                                delta_v += 1
                            
                        if delta_v < best_delta:
                            best_delta = delta_v
                            best_new_color = new_color
                            if best_delta < 0:
                                # Found an improvement, can break early?
                                # No, there might be a better one.
                                # But if we find a move that reduces V, it's good.
                                pass
                    
                    if best_new_color != -1 and best_delta < 0:
                        colors[e_idx] = best_new_color
                        for c_idx in c_indices:
                            cycle_counts[c_idx][old_color] -= 1
                            cycle_counts[c_idx][best_new_color] += 1
                        violations += best_delta
                        steps_without_improvement = 0
                    else:
                        steps_without_improvement += 1
                        # If stuck, restart
                        if steps_without_improvement == max_steps_no_improve:
                            break
                
                # Restart with new random init
                colors = [random.randint(0, 3) for _ in range(num_edges)]
                for c_idx in range(num_cycles):
                    cycle_counts[c_idx] = [0]*4
                for c_idx, c_edges in enumerate(cycles):
                    for e_idx in c_edges:
                        cycle_counts[c_idx][colors[e_idx]] += 1
                violations = 0
                for c_idx in range(num_cycles):
                    if any(count == 6 for count in cycle_counts[c_idx]):
                        violations += 1
                if violations == 0:
                    return colors

        solution = solve_n(n)
        if solution:
            return {'n': n, 'colors': solution}
    
    return best_result
