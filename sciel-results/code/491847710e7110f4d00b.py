import time
import random
import numpy as np

def construct():
    deadline = time.time() + 112.0
    d = 7
    nverts = 1 << d
    total_edges = d * nverts // 2  # 448
    
    # 1. Build edges
    edges = []
    edge_to_idx = {}
    for u in range(nverts):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx

    # 2. Build squares (faces)
    squares = []
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for x in range(nverts):
                if (x & mask) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ mask
                    
                    e1 = edge_to_idx[(min(v1, v2), max(v1, v2))]
                    e2 = edge_to_idx[(min(v2, v4), max(v2, v4))]
                    e3 = edge_to_idx[(min(v4, v3), max(v4, v3))]
                    e4 = edge_to_idx[(min(v3, v1), max(v3, v1))]
                    
                    squares.append([e1, e2, e3, e4])
    
    num_squares = len(squares) # 672
    
    # 3. Precompute edge_to_squares and square_edges
    edge_to_squares = [[] for _ in range(total_edges)]
    square_edges = [set(sq) for sq in squares]
    
    for s_idx, sq in enumerate(squares):
        for e_idx in sq:
            edge_to_squares[e_idx].append(s_idx)
            
    def greedy_hitting_set(rng):
        """
        Greedy Set Cover to find a hitting set R for all squares.
        """
        uncovered = set(range(num_squares))
        # uncovered_count[e] = number of uncovered squares containing e
        uncovered_count = [0] * total_edges
        
        for e in range(total_edges):
            uncovered_count[e] = len(edge_to_squares[e]) # Initially all squares are uncovered
            
        R = set()
        
        while uncovered:
            # Find edge with max uncovered_count
            max_count = -1
            best_edges = []
            for e in range(total_edges):
                if e in R:
                    continue
                if uncovered_count[e] > max_count:
                    max_count = uncovered_count[e]
                    best_edges = [e]
                elif uncovered_count[e] == max_count:
                    best_edges.append(e)
            
            if not best_edges:
                break
                
            e = rng.choice(best_edges)
            R.add(e)
            
            # Update uncovered counts
            for f in edge_to_squares[e]:
                if f in uncovered:
                    uncovered.remove(f)
                    # Decrement count for all other edges in this square
                    for e_other in square_edges[f]:
                        if e_other != e:
                            uncovered_count[e_other] -= 1
                            
        return R

    def compute_cover_count(R):
        cover_count = [0] * num_squares
        for e in R:
            for f in edge_to_squares[e]:
                cover_count[f] += 1
        return cover_count

    def remove_redundant(R, cover_count):
        """
        Remove edges from R that are redundant (all their squares are covered by other edges in R).
        """
        changed = True
        while changed:
            changed = False
            for e in list(R):
                redundant = True
                for f in edge_to_squares[e]:
                    if cover_count[f] <= 1:
                        redundant = False
                        break
                if redundant:
                    R.remove(e)
                    for f in edge_to_squares[e]:
                        cover_count[f] -= 1
                    changed = True
        return R, cover_count

    def local_search(R, cover_count, rng, time_budget=5.0):
        """
        Local search: try swaps to reduce |R|.
        """
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            # Try removing edges (redundancy check is already done, but maybe new redundancies appeared?)
            # Actually, local search should focus on swaps that enable further deletions.
            
            # Iterate over edges in R
            R_list = list(R)
            rng.shuffle(R_list)
            
            for e_out in R_list:
                if time.time() > deadline - 5:
                    return R, cover_count
                
                # Simulate removal of e_out
                # Find squares that would become uncovered
                uncovered_squares = []
                for f in edge_to_squares[e_out]:
                    if cover_count[f] == 1:
                        uncovered_squares.append(f)
                
                if not uncovered_squares:
                    # Redundant, should have been removed, but just in case
                    R.remove(e_out)
                    for f in edge_to_squares[e_out]:
                        cover_count[f] -= 1
                    improved = True
                    break # Restart loop with new state
                
                # Find candidate e_in that covers all uncovered_squares
                # Candidates must be in the union of edges of the uncovered squares
                candidates = set()
                for f in uncovered_squares:
                    candidates.update(square_edges[f])
                candidates.discard(e_out)
                
                found_swap = False
                for e_in in candidates:
                    if e_in in R:
                        continue
                    
                    # Check if e_in covers all uncovered_squares
                    valid = True
                    for f in uncovered_squares:
                        if f not in edge_to_squares[e_in]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        R.remove(e_out)
                        R.add(e_in)
                        
                        # Update cover_count
                        for f in edge_to_squares[e_out]:
                            cover_count[f] -= 1
                        for f in edge_to_squares[e_in]:
                            cover_count[f] += 1
                        
                        # Now try to remove redundant edges
                        R, cover_count = remove_redundant(R, cover_count)
                        
                        # If we ended up with smaller R, it's an improvement
                        # Note: Swap itself keeps size same, but redundancy removal might shrink it.
                        # We accept the state if it's valid (it is, by construction) 
                        # and we hope it leads to a global minimum.
                        # To ensure we don't cycle, we can track visited states or just rely on time.
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break # Restart loop with new state
        
        return R, cover_count

    def local_search_2(R, cover_count, rng, time_budget=5.0):
        """
        Alternative local search: try adding an edge and then removing two.
        This is a (1,2) exchange.
        """
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            
            # Try adding an edge not in R
            # We want to add e_in such that it allows removing 2 edges e_out1, e_out2
            # e_in must cover squares that are only covered by e_out1 or e_out2?
            # Actually, if we add e_in, it covers some squares.
            # If e_in covers all squares of e_out1 that are NOT covered by e_out2 or others, 
            # and all squares of e_out2 that are NOT covered by e_out1 or others...
            # This is complex.
            
            # Simpler: Try to replace a pair of edges with a single edge?
            # If e1 and e2 are in R, and there exists e3 not in R such that
            # edge_to_squares[e1] U edge_to_squares[e2] is subset of edge_to_squares[e3] U (squares covered by others)
            # This is hard to check.
            
            # Let's stick to the (1,1) swap which is faster and effective.
            # But we can try a different heuristic for candidate selection.
            
            # For now, just call the previous one? No, let's implement a specific "2-for-1" reduction.
            # If we have e1, e2 in R.
            # Let S1 = squares only covered by e1 (count == 1 and e1 in it).
            # Let S2 = squares only covered by e2.
            # If there is e3 not in R such that S1 U S2 is subset of squares(e3), then we can remove e1, e2 and add e3.
            
            R_list = list(R)
            rng.shuffle(R_list)
            
            for i in range(len(R_list)):
                if time.time() > deadline - 5:
                    return R, cover_count
                e1 = R_list[i]
                
                # Find squares only covered by e1
                only_e1 = []
                for f in edge_to_squares[e1]:
                    if cover_count[f] == 1:
                        only_e1.append(f)
                
                if not only_e1:
                    continue
                
                for j in range(i + 1, len(R_list)):
                    if time.time() > deadline - 5:
                        return R, cover_count
                    e2 = R_list[j]
                    
                    # Find squares only covered by e2
                    only_e2 = []
                    for f in edge_to_squares[e2]:
                        if cover_count[f] == 1:
                            only_e2.append(f)
                    
                    if not only_e2:
                        continue
                    
                    # Union of only_e1 and only_e2
                    target_squares = set(only_e1) | set(only_e2)
                    
                    # Find e3 not in R that covers all target_squares
                    # e3 must be in the intersection of edge_to_squares[f] for all f in target_squares
                    # Or just check candidates from the first square
                    candidates = set(square_edges[target_squares.pop()])
                    candidates.discard(e1)
                    candidates.discard(e2)
                    
                    for e3 in candidates:
                        if e3 in R:
                            continue
                        
                        valid = True
                        for f in target_squares:
                            if f not in edge_to_squares[e3]:
                                valid = False
                                break
                        
                        if valid:
                            # Check if e3 covers all target squares
                            # It does by construction (intersection logic above was weak, let's verify)
                            # Actually, candidates were from one square. We verified all target_squares.
                            
                            # Perform swap: remove e1, e2, add e3
                            R.remove(e1)
                            R.remove(e2)
                            R.add(e3)
                            
                            # Update cover_count
                            for f in edge_to_squares[e1]:
                                cover_count[f] -= 1
                            for f in edge_to_squares[e2]:
                                cover_count[f] -= 1
                            for f in edge_to_squares[e3]:
                                cover_count[f] += 1
                            
                            # Remove redundant
                            R, cover_count = remove_redundant(R, cover_count)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return R, cover_count

    best_R = None
    best_size = total_edges + 1
    rng = random.Random(42)
    
    num_restarts = 200 # Increased restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        R = greedy_hitting_set(rng)
        cover_count = compute_cover_count(R)
        R, cover_count = remove_redundant(R, cover_count)
        
        # Local search
        R, cover_count = local_search(R, cover_count, rng, time_budget=1.0)
        R, cover_count = local_search_2(R, cover_count, rng, time_budget=1.0)
        
        size = len(R)
        if size < best_size:
            best_size = size
            best_R = set(R)
            
        # If we hit the theoretical lower bound, stop
        # Max edges = 448 - best_size.
        # If 448 - best_size >= 308, we are done.
        if 448 - best_size >= 308:
            break

    # Final polish on best
    if best_R is not None and time.time() < deadline - 5:
        cover_count = compute_cover_count(best_R)
        best_R, cover_count = remove_redundant(best_R, cover_count)
        # More aggressive local search on the best solution
        best_R, cover_count = local_search(best_R, cover_count, rng, time_budget=15.0)
        best_R, cover_count = local_search_2(best_R, cover_count, rng, time_budget=15.0)
        best_size = len(best_R)

    # Convert best_R (removed edges) to selected edges
    if best_R is None:
        # Fallback: empty set of removed edges (all edges selected) - invalid but returns something
        best_R = set()
        
    selected_edges = []
    for e_idx in range(total_edges):
        if e_idx not in best_R:
            u, v = edges[e_idx]
            selected_edges.append([u, v])
            
    return selected_edges
