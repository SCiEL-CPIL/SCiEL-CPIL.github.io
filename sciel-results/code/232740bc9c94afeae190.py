import time
import random
from math import gcd, exp
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate Pool
    # Use a grid to ensure many collinearities are possible.
    # Coords: -50, -40, ..., 50 (11 values)
    coords = list(range(-50, 51, 10))
    pool = []
    point_to_idx = {}
    for x in coords:
        for y in coords:
            idx = len(pool)
            pool.append((x, y))
            point_to_idx[(x, y)] = idx
    
    n_pool = len(pool)
    
    # 2. Precompute Lines
    # A line is defined by a normalized (a, b, c) such that ax + by + c = 0.
    # We only care about lines that have at least 3 points in the pool? 
    # Actually, for the constraint "no 4 collinear", we need to know all lines 
    # that could potentially have 4 points from the selected subset.
    # Since the max line size in a 11x11 grid is 11 (diagonals/rows/cols), 
    # we must track all lines that have >= 3 points in the pool to detect violations 
    # if we select 4 of them.
    # However, for scoring "triples", we only count lines with exactly 3 points in the subset.
    
    line_map = {} # key: (a,b,c) -> set of point indices
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            x1, y1 = pool[i]
            x2, y2 = pool[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0: continue
            dx //= g
            dy //= g
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy
            a, b = dy, -dx
            c = a * x1 + b * y1
            key = (a, b, c)
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
            
    # Filter lines to only those with >= 3 points in the pool.
    # Lines with < 3 points in the pool can never form a triple or a 4-tuple in the subset.
    relevant_lines = {}
    line_id_counter = 0
    for key, pts in line_map.items():
        if len(pts) >= 3:
            line_id = line_id_counter
            line_id_counter += 1
            relevant_lines[line_id] = pts
            
    n_lines = line_id_counter
    
    # point_to_lines: list of line IDs for each point
    point_to_lines = [[] for _ in range(n_pool)]
    for lid, pts in relevant_lines.items():
        for p in pts:
            point_to_lines[p].append(lid)
            
    # 3. Objective Function
    # State: subset (list of 20 indices)
    # We maintain line_counts: array of size n_lines
    # triple_count: number of lines with count == 3
    # covered_count: number of points that are in at least one line with count >= 2? 
    # Actually, "covered" usually means part of a triple. Let's stick to maximizing triples.
    # But to minimize the max no-3-collinear subset, covering all points in triples is ideal.
    # If a point is not in any triple, it can always be added to a no-3-collinear set 
    # unless it creates a collinearity with two others that are already in the set.
    # Max triples is a good proxy.
    
    def calculate_score(subset, line_counts):
        # line_counts is updated externally or calculated here?
        # For local search, it's better to maintain state.
        # Here we just compute from scratch for initialization.
        counts = [0] * n_lines
        for p in subset:
            for lid in point_to_lines[p]:
                counts[lid] += 1
        
        triples = 0
        invalid = False
        for c in counts:
            if c > 3:
                invalid = True
                break
            if c == 3:
                triples += 1
        return triples, invalid, counts

    # 4. Initialization
    # Greedy: Pick points one by one to maximize triples.
    # This is O(20 * n_pool * (cost to eval)).
    
    def greedy_init():
        subset = []
        line_counts = [0] * n_lines
        best_triples = 0
        
        for _ in range(20):
            best_point = -1
            best_score = -1
            best_new_counts = None
            
            for p in range(n_pool):
                if p in subset:
                    continue
                    
                # Tentatively add p
                new_counts = line_counts[:] # Copy is expensive, but n_lines is small (~100-200?)
                # Let's check size of relevant_lines.
                # 11x11 grid. Lines:
                # Rows: 11. Cols: 11. Diags: ~20. Slopes 1/2, 2/1, etc.
                # Total lines with >=3 pts is probably < 100.
                
                valid = True
                for lid in point_to_lines[p]:
                    new_counts[lid] += 1
                    if new_counts[lid] > 3:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                triples = sum(1 for c in new_counts if c == 3)
                if triples > best_score:
                    best_score = triples
                    best_point = p
                    best_new_counts = new_counts
            
            if best_point != -1:
                subset.append(best_point)
                line_counts = best_new_counts
                best_triples = best_score
            else:
                # If no point can be added without violating constraints or improving score?
                # Just pick any valid point.
                for p in range(n_pool):
                    if p in subset: continue
                    valid = True
                    for lid in point_to_lines[p]:
                        if line_counts[lid] + 1 > 3:
                            valid = False
                            break
                    if valid:
                        subset.append(p)
                        for lid in point_to_lines[p]:
                            line_counts[lid] += 1
                        break
                if len(subset) < 20:
                    # Force fill if stuck (shouldn't happen with this pool)
                    for p in range(n_pool):
                        if p not in subset:
                            subset.append(p)
                            break
                            
        return subset, line_counts

    subset, line_counts = greedy_init()
    
    # 5. Local Search with Simulated Annealing
    # Move: Swap p_in with p_out.
    # Delta calculation:
    # Removing p_in decreases counts of its lines.
    # Adding p_out increases counts of its lines.
    
    def get_score_delta(p_in, p_out, line_counts):
        # Returns (delta_triples, is_valid, new_line_counts)
        # We need to check validity first.
        
        # Check if adding p_out causes any line to exceed 3, considering p_in is removed.
        # Actually, we should simulate the change.
        
        # Lines affected: union of lines(p_in) and lines(p_out)
        lines_p_in = set(point_to_lines[p_in])
        lines_p_out = set(point_to_lines[p_out])
        affected_lines = lines_p_in | lines_p_out
        
        new_counts = line_counts[:]
        valid = True
        old_triples = 0
        new_triples = 0
        
        # Calculate old triples for affected lines
        for lid in affected_lines:
            if line_counts[lid] == 3:
                old_triples += 1
                
        # Apply changes
        for lid in lines_p_in:
            new_counts[lid] -= 1
        for lid in lines_p_out:
            new_counts[lid] += 1
            if new_counts[lid] > 3:
                valid = False
                break
        
        if not valid:
            return 0, False, None
            
        # Calculate new triples for affected lines
        for lid in affected_lines:
            if new_counts[lid] == 3:
                new_triples += 1
                
        delta = new_triples - old_triples
        return delta, True, new_counts

    current_triples, _, _ = calculate_score(subset, line_counts)
    best_subset = subset[:]
    best_line_counts = line_counts[:]
    best_score = current_triples
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    # Local Search Loop
    while time.time() < deadline - 5:
        if T < min_T:
            T = 1.0 # Restart with high temp if stuck too long? Or just break?
            # Let's try to restart with a perturbed best solution
            subset = best_subset[:]
            line_counts = best_line_counts[:]
            current_triples = best_score
            
        improved = False
        best_swap = None
        best_delta = 0
        best_new_counts = None
        
        p_in_list = subset[:]
        
        for p_in in p_in_list:
            if time.time() > deadline - 5:
                break
            
            # Try swapping p_in with every p_out not in subset
            # Optimization: Only consider p_out that shares lines with p_in or adds new triples?
            # For now, try all. 20 * (121-20) = 2020 swaps.
            # Each swap check is O(num_affected_lines) ~ 20-50.
            # Total ops ~ 100k. Very fast.
            
            for p_out in range(n_pool):
                if p_out in subset:
                    continue
                    
                delta, valid, new_counts = get_score_delta(p_in, p_out, line_counts)
                
                if not valid:
                    continue
                
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (p_in, p_out)
                    best_new_counts = new_counts
                    improved = True
                else:
                    # Simulated Annealing: Accept worse states with probability exp(-delta/T)
                    # delta is negative here
                    prob = exp(delta / T)
                    if random.random() < prob:
                        # Accept this move immediately? 
                        # Or just mark it as a candidate? 
                        # Standard SA accepts the first move that satisfies the criterion.
                        # But we want to find the best move in the neighborhood? 
                        # No, SA typically picks a random neighbor. 
                        # Let's stick to picking the best improving move, but also allow 
                        # random acceptance of a random move if no improving move is found?
                        # Actually, the loop above finds the BEST improving move. 
                        # If we find an improving move, we should take it. 
                        # The SA part should be: if no improving move, accept a random 
                        # non-improving move with probability exp(-delta/T).
                        # My current logic updates best_delta only if delta > best_delta.
                        # So best_delta will be the max delta found.
                        # If best_delta > 0, we take it.
                        # If best_delta <= 0, we might accept a random one?
                        # Let's refine: 
                        # 1. Find best improving move.
                        # 2. If found, take it.
                        # 3. If not found, pick a random valid move and accept if exp(-delta/T) > rand.
                        pass

        if improved and best_delta > 0:
            p_in, p_out = best_swap
            subset.remove(p_in)
            subset.append(p_out)
            line_counts = best_new_counts
            current_triples += best_delta
            
            if current_triples > best_score:
                best_score = current_triples
                best_subset = subset[:]
                best_line_counts = line_counts[:]
        else:
            # No improving move found. Try to accept a random valid move.
            # Pick a random p_in and p_out.
            p_in = random.choice(subset)
            candidates = [p for p in range(n_pool) if p not in subset]
            if candidates:
                p_out = random.choice(candidates)
                delta, valid, new_counts = get_score_delta(p_in, p_out, line_counts)
                if valid:
                    if delta > 0:
                        # This shouldn't happen if 'improved' was False, but just in case
                        subset.remove(p_in)
                        subset.append(p_out)
                        line_counts = new_counts
                        current_triples += delta
                        if current_triples > best_score:
                            best_score = current_triples
                            best_subset = subset[:]
                            best_line_counts = line_counts[:]
                    else:
                        # Accept with probability exp(-delta/T)
                        if random.random() < exp(-delta / T):
                            subset.remove(p_in)
                            subset.append(p_out)
                            line_counts = new_counts
                            current_triples += delta
        
        T *= cooling

    # Final Result
    final_points = [pool[i] for i in best_subset]
    return [[p[0], p[1]] for p in final_points]
