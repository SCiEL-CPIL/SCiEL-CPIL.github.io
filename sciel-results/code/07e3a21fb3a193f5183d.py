import random
import time
import math

# Precomputation
N_DIMS = 7
N_VERTS = 1 << N_DIMS
N_EDGES = N_DIMS * (N_VERTS // 2) # 7 * 64 = 448

def precompute():
    edges = []
    edge_map = {} # (u, v) with u<v -> index
    for u in range(N_VERTS):
        for i in range(N_DIMS):
            v = u ^ (1 << i)
            if u < v:
                edge_map[(u, v)] = len(edges)
                edges.append((u, v))
    
    edge_squares = [[] for _ in range(len(edges))]
    square_count = 0
    
    for i in range(N_DIMS):
        for j in range(i + 1, N_DIMS):
            other_bits = [k for k in range(N_DIMS) if k != i and k != j]
            for mask in range(1 << len(other_bits)):
                base = 0
                for k in range(len(other_bits)):
                    if (mask >> k) & 1:
                        base |= (1 << other_bits[k])
                
                v00 = base
                v10 = base | (1 << i)
                v01 = base | (1 << j)
                v11 = base | (1 << i) | (1 << j)
                
                e1 = (min(v00, v10), max(v00, v10))
                e2 = (min(v00, v01), max(v00, v01))
                e3 = (min(v10, v11), max(v10, v11))
                e4 = (min(v01, v11), max(v01, v11))
                
                idx1 = edge_map[e1]
                idx2 = edge_map[e2]
                idx3 = edge_map[e3]
                idx4 = edge_map[e4]
                
                edge_squares[idx1].append(square_count)
                edge_squares[idx2].append(square_count)
                edge_squares[idx3].append(square_count)
                edge_squares[idx4].append(square_count)
                
                square_count += 1
                
    return edges, edge_squares, square_count

EDGES, EDGE_SQUARES, NUM_SQUARES = precompute()

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = []
    best_count = 0
    
    # Precompute for each square, the list of 4 edge indices
    square_edges = [[] for _ in range(NUM_SQUARES)]
    for e in range(len(EDGES)):
        for s in EDGE_SQUARES[e]:
            square_edges[s].append(e)
    
    # Precompute edge-to-edge adjacency via shared squares
    # For each edge, which other edges share a square with it
    edge_neighbors = [set() for _ in range(len(EDGES))]
    for s in range(NUM_SQUARES):
        es = square_edges[s]
        for a in range(len(es)):
            for b in range(a+1, len(es)):
                edge_neighbors[es[a]].add(es[b])
                edge_neighbors[es[b]].add(es[a])
    
    def greedy_random():
        """Randomized greedy to get a starting solution."""
        indices = list(range(len(EDGES)))
        random.shuffle(indices)
        selected = [False] * len(EDGES)
        square_counts = [0] * NUM_SQUARES
        count = 0
        
        for orig_idx in indices:
            sqs = EDGE_SQUARES[orig_idx]
            valid = True
            for s in sqs:
                if square_counts[s] >= 3:
                    valid = False
                    break
            if valid:
                selected[orig_idx] = True
                count += 1
                for s in sqs:
                    square_counts[s] += 1
        return selected, count
    
    def greedy_degree():
        """Greedy with degree-based ordering: prefer edges that block fewer squares."""
        selected = [False] * len(EDGES)
        square_counts = [0] * NUM_SQUARES
        count = 0
        
        remaining = set(range(len(EDGES)))
        
        while remaining and time.time() < deadline - 5.0:
            best_edge = None
            best_score = -1
            
            if len(remaining) > 50:
                sample = random.sample(list(remaining), 50)
            else:
                sample = list(remaining)
            
            for e in sample:
                sqs = EDGE_SQUARES[e]
                score = sum(square_counts[s] for s in sqs)
                can_add = True
                for s in sqs:
                    if square_counts[s] >= 3:
                        can_add = False
                        break
                if not can_add:
                    continue
                if score < best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                for e in remaining:
                    sqs = EDGE_SQUARES[e]
                    can_add = True
                    for s in sqs:
                        if square_counts[s] >= 3:
                            can_add = False
                            break
                    if can_add:
                        best_edge = e
                        break
            
            if best_edge is None:
                break
            
            selected[best_edge] = True
            count += 1
            for s in EDGE_SQUARES[best_edge]:
                square_counts[s] += 1
            remaining.remove(best_edge)
        
        return selected, count
    
    def local_search(selected, count):
        """
        Max-SAT style local search with 1-swap and 2-swap moves.
        """
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        improved = True
        no_improve_rounds = 0
        max_rounds = 200
        
        while improved and no_improve_rounds < max_rounds and time.time() < deadline - 3.0:
            improved = False
            no_improve_rounds += 1
            
            # Try to add each unselected edge
            candidates = [e for e in range(len(EDGES)) if not selected[e]]
            random.shuffle(candidates)
            
            for e in candidates:
                if time.time() >= deadline - 3.0:
                    break
                
                sqs = EDGE_SQUARES[e]
                blocking_squares = []
                for s in sqs:
                    if square_counts[s] == 3:
                        blocking_squares.append(s)
                
                if not blocking_squares:
                    selected[e] = True
                    count += 1
                    for s in sqs:
                        square_counts[s] += 1
                    improved = True
                else:
                    # Try 1-swap: remove one edge from a blocking square, add e
                    found_swap = False
                    for s in blocking_squares:
                        if time.time() >= deadline - 3.0:
                            break
                        
                        for other_e in square_edges[s]:
                            if other_e == e or not selected[other_e]:
                                continue
                            
                            other_sqs = EDGE_SQUARES[other_e]
                            valid = True
                            
                            for es in sqs:
                                if es != s:
                                    if square_counts[es] >= 3:
                                        valid = False
                                        break
                            
                            if not valid:
                                continue
                            
                            # Perform 1-swap
                            selected[other_e] = False
                            selected[e] = True
                            for os in other_sqs:
                                square_counts[os] -= 1
                            for es in sqs:
                                square_counts[es] += 1
                            improved = True
                            found_swap = True
                            break
                        
                        if found_swap:
                            break
                    
                    if found_swap:
                        continue
                    
                    # Try 2-swap: remove two edges, add e and another edge
                    # Find pairs of edges to remove that would unblock e and allow adding another
                    for s1 in blocking_squares:
                        if time.time() >= deadline - 3.0:
                            break
                        
                        # Try removing one edge from s1
                        for other_e1 in square_edges[s1]:
                            if other_e1 == e or not selected[other_e1]:
                                continue
                            
                            # After removing other_e1, check if e can be added
                            # and if some other edge e2 can also be added
                            other_sqs1 = EDGE_SQUARES[other_e1]
                            
                            # Temporarily remove other_e1
                            for os in other_sqs1:
                                square_counts[os] -= 1
                            
                            # Now check if e can be added
                            e_can_add = True
                            for es in sqs:
                                if square_counts[es] >= 3:
                                    e_can_add = False
                                    break
                            
                            if e_can_add:
                                # Try to also add another edge e2
                                # Look at edges that were blocked by other_e1's squares
                                candidate_e2s = []
                                for os in other_sqs1:
                                    for e2 in square_edges[os]:
                                        if e2 != e and e2 != other_e1 and not selected[e2]:
                                            candidate_e2s.append(e2)
                                
                                random.shuffle(candidate_e2s)
                                best_e2 = None
                                for e2 in candidate_e2s[:20]:
                                    sqs2 = EDGE_SQUARES[e2]
                                    can_add2 = True
                                    for s2 in sqs2:
                                        if square_counts[s2] >= 3:
                                            can_add2 = False
                                            break
                                    if can_add2:
                                        best_e2 = e2
                                        break
                                
                                if best_e2 is not None:
                                    # Perform 2-swap: remove other_e1, add e and best_e2
                                    selected[other_e1] = False
                                    selected[e] = True
                                    selected[best_e2] = True
                                    for es in sqs:
                                        square_counts[es] += 1
                                    sqs2 = EDGE_SQUARES[best_e2]
                                    for s2 in sqs2:
                                        square_counts[s2] += 1
                                    count += 2
                                    improved = True
                                    # No need to restore square_counts since we already decremented other_e1
                                    found_swap = True
                                    break
                                else:
                                    # Just do 1-swap: remove other_e1, add e
                                    selected[other_e1] = False
                                    selected[e] = True
                                    for es in sqs:
                                        square_counts[es] += 1
                                    count += 1
                                    improved = True
                                    found_swap = True
                                    break
                            else:
                                # Restore square_counts
                                for os in other_sqs1:
                                    square_counts[os] += 1
                            
                            if found_swap:
                                break
                        
                        if found_swap:
                            break
                    
                    if found_swap:
                        # After a swap, try to add more edges that are now unblocked
                        for e2 in candidates:
                            if e2 == e or selected[e2]:
                                continue
                            sqs2 = EDGE_SQUARES[e2]
                            can_add = True
                            for s2 in sqs2:
                                if square_counts[s2] >= 3:
                                    can_add = False
                                    break
                            if can_add:
                                selected[e2] = True
                                count += 1
                                for s2 in sqs2:
                                    square_counts[s2] += 1
        
        return selected, count
    
    def kick_and_optimize(selected, count, kick_size=5):
        """Randomly remove some edges targeting saturated squares, then re-optimize."""
        selected = list(selected)
        
        # Compute square counts to identify saturated squares
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        # Find saturated squares (count == 3)
        saturated_squares = [s for s in range(NUM_SQUARES) if square_counts[s] == 3]
        
        selected_indices = [i for i in range(len(EDGES)) if selected[i]]
        
        if saturated_squares:
            # Target edges in saturated squares
            candidate_removals = set()
            for s in saturated_squares:
                for e in square_edges[s]:
                    if selected[e]:
                        candidate_removals.add(e)
            
            if candidate_removals and len(candidate_removals) > kick_size:
                to_remove = random.sample(list(candidate_removals), kick_size)
            elif candidate_removals:
                to_remove = list(candidate_removals)
            else:
                to_remove = random.sample(selected_indices, min(kick_size, len(selected_indices)))
        else:
            to_remove = random.sample(selected_indices, min(kick_size, len(selected_indices)))
        
        for i in to_remove:
            selected[i] = False
        count -= len(to_remove)
        
        # Re-optimize
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        # Try to add edges
        candidates = [e for e in range(len(EDGES)) if not selected[e]]
        random.shuffle(candidates)
        
        for e in candidates:
            if time.time() >= deadline - 3.0:
                break
            sqs = EDGE_SQUARES[e]
            can_add = True
            for s in sqs:
                if square_counts[s] >= 3:
                    can_add = False
                    break
            if can_add:
                selected[e] = True
                count += 1
                for s in sqs:
                    square_counts[s] += 1
        
        # Local search
        selected, count = local_search(selected, count)
        return selected, count
    
    # Phase 1: Multiple random greedy starts
    num_greedy = 0
    while time.time() < deadline - 30.0:
        if num_greedy % 3 == 0:
            selected, count = greedy_random()
        elif num_greedy % 3 == 1:
            selected, count = greedy_degree()
        else:
            selected, count = greedy_random()
        
        if count > best_count:
            best_count = count
            best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
        num_greedy += 1
        if num_greedy >= 60:
            break
    
    # Build edge_map for later use
    edge_map = {}
    for idx, (u, v) in enumerate(EDGES):
        edge_map[(min(u, v), max(u, v))] = idx
    
    # Phase 2: Local search on best solution
    if best_count > 0 and time.time() < deadline - 10.0:
        selected = [False] * len(EDGES)
        for edge in best_edges:
            u, v = edge
            key = (min(u, v), max(u, v))
            selected[edge_map[key]] = True
        
        selected, count = local_search(selected, best_count)
        if count > best_count:
            best_count = count
            best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
    
    # Phase 3: Kick and optimize to escape local optima
    while time.time() < deadline - 5.0:
        selected = [False] * len(EDGES)
        for edge in best_edges:
            u, v = edge
            key = (min(u, v), max(u, v))
            selected[edge_map[key]] = True
        
        # Try different kick sizes
        kick_size = random.choice([3, 5, 7, 10])
        selected, count = kick_and_optimize(selected, best_count, kick_size)
        
        if count > best_count:
            best_count = count
            best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
        else:
            # Try fresh greedy + local search
            if time.time() < deadline - 10.0:
                if random.random() < 0.5:
                    selected, count = greedy_random()
                else:
                    selected, count = greedy_degree()
                selected, count = local_search(selected, count)
                if count > best_count:
                    best_count = count
                    best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
    
    # Final polish: multiple rounds of trying to add remaining edges
    if time.time() < deadline - 2.0:
        selected = [False] * len(EDGES)
        for edge in best_edges:
            u, v = edge
            key = (min(u, v), max(u, v))
            selected[edge_map[key]] = True
        
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        # Multiple polish rounds
        for polish_round in range(3):
            if time.time() >= deadline - 2.0:
                break
            
            candidates = [e for e in range(len(EDGES)) if not selected[e]]
            random.shuffle(candidates)
            added = False
            for e in candidates:
                sqs = EDGE_SQUARES[e]
                can_add = True
                for s in sqs:
                    if square_counts[s] >= 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    best_count += 1
                    for s in sqs:
                        square_counts[s] += 1
                    added = True
            
            if not added:
                break
        
        best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
    
    return [[u, v] for u, v in best_edges]
