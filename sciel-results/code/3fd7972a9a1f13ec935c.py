import random
import time
import math
from collections import defaultdict

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
        # Count how many squares each edge is in (always 2 for hypercube edges)
        # Instead, use a dynamic heuristic: prefer edges that are in squares with fewer already-selected edges
        selected = [False] * len(EDGES)
        square_counts = [0] * NUM_SQUARES
        count = 0
        
        # Precompute for each edge, the squares it's in
        # Use a priority queue approach: at each step, pick the edge that can be added with least "damage"
        remaining = set(range(len(EDGES)))
        
        while remaining and time.time() < deadline - 5.0:
            best_edge = None
            best_score = -1
            
            # Sample a subset to keep it fast
            if len(remaining) > 50:
                sample = random.sample(list(remaining), 50)
            else:
                sample = list(remaining)
            
            for e in sample:
                sqs = EDGE_SQUARES[e]
                # Score: prefer edges where squares have lower counts
                score = sum(square_counts[s] for s in sqs)
                # Check if we can add it
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
                # Try any remaining edge
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
        
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            # Pass 1: Try to add each unselected edge (simple add)
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
                    improved = True
            
            # Pass 2: Try 1-swap moves (remove one selected edge, add one unselected)
            if time.time() >= deadline - 3.0:
                break
            
            selected_list = [e for e in range(len(EDGES)) if selected[e]]
            random.shuffle(selected_list)
            
            for e_out in selected_list:
                if time.time() >= deadline - 3.0:
                    break
                
                out_sqs = EDGE_SQUARES[e_out]
                
                # Find unselected edges that share a square with e_out
                candidate_e_in = []
                for s in out_sqs:
                    for e_in in square_edges[s]:
                        if e_in != e_out and not selected[e_in]:
                            candidate_e_in.append(e_in)
                
                random.shuffle(candidate_e_in)
                
                for e_in in candidate_e_in:
                    if time.time() >= deadline - 3.0:
                        break
                    
                    in_sqs = EDGE_SQUARES[e_in]
                    
                    # Check if swap is valid: after removing e_out and adding e_in,
                    # no square should have count > 3
                    valid = True
                    # Squares that lose an edge (from e_out)
                    lost_sqs = set(out_sqs)
                    # Squares that gain an edge (from e_in)
                    gained_sqs = set(in_sqs)
                    
                    for s in gained_sqs:
                        if s not in lost_sqs:
                            # This square gains 1, must have count < 3
                            if square_counts[s] >= 3:
                                valid = False
                                break
                    
                    if not valid:
                        continue
                    
                    # Perform swap
                    selected[e_out] = False
                    selected[e_in] = True
                    for s in out_sqs:
                        square_counts[s] -= 1
                    for s in in_sqs:
                        square_counts[s] += 1
                    improved = True
                    break
                
                if improved:
                    break
            
            # Pass 3: Try 2-swap moves (remove two selected edges, add two unselected)
            if time.time() >= deadline - 3.0:
                break
            
            selected_list = [e for e in range(len(EDGES)) if selected[e]]
            if len(selected_list) < 2:
                continue
            
            # Sample pairs to keep it tractable
            num_pairs = min(200, len(selected_list) * (len(selected_list) - 1) // 2)
            tried_pairs = set()
            
            for _ in range(num_pairs):
                if time.time() >= deadline - 3.0:
                    break
                
                e_out1 = random.choice(selected_list)
                e_out2 = random.choice(selected_list)
                if e_out1 == e_out2:
                    continue
                pair_key = (min(e_out1, e_out2), max(e_out1, e_out2))
                if pair_key in tried_pairs:
                    continue
                tried_pairs.add(pair_key)
                
                out_sqs1 = EDGE_SQUARES[e_out1]
                out_sqs2 = EDGE_SQUARES[e_out2]
                
                # Find candidate e_in1, e_in2 that share squares with the removed edges
                candidate_in1 = []
                for s in out_sqs1:
                    for e_in in square_edges[s]:
                        if e_in != e_out1 and e_in != e_out2 and not selected[e_in]:
                            candidate_in1.append(e_in)
                
                if not candidate_in1:
                    continue
                
                random.shuffle(candidate_in1)
                
                for e_in1 in candidate_in1[:10]:  # Limit candidates
                    if time.time() >= deadline - 3.0:
                        break
                    
                    in_sqs1 = EDGE_SQUARES[e_in1]
                    
                    # Now find e_in2 that shares a square with e_out2 (and not e_out1)
                    candidate_in2 = []
                    for s in out_sqs2:
                        for e_in in square_edges[s]:
                            if e_in != e_out1 and e_in != e_out2 and e_in != e_in1 and not selected[e_in]:
                                candidate_in2.append(e_in)
                    
                    if not candidate_in2:
                        continue
                    
                    random.shuffle(candidate_in2)
                    
                    for e_in2 in candidate_in2[:10]:
                        if time.time() >= deadline - 3.0:
                            break
                        
                        in_sqs2 = EDGE_SQUARES[e_in2]
                        
                        # Check validity: simulate the 2-swap
                        # Net change per square:
                        # -1 for each square in out_sqs1, -1 for each in out_sqs2
                        # +1 for each square in in_sqs1, +1 for each in in_sqs2
                        
                        delta = defaultdict(int)
                        for s in out_sqs1:
                            delta[s] -= 1
                        for s in out_sqs2:
                            delta[s] -= 1
                        for s in in_sqs1:
                            delta[s] += 1
                        for s in in_sqs2:
                            delta[s] += 1
                        
                        valid = True
                        for s, d in delta.items():
                            if square_counts[s] + d > 3:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        # Perform 2-swap
                        selected[e_out1] = False
                        selected[e_out2] = False
                        selected[e_in1] = True
                        selected[e_in2] = True
                        for s, d in delta.items():
                            square_counts[s] += d
                        improved = True
                        break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        return selected, count
    
    def kick_and_optimize(selected, count, kick_size=5):
        """Remove edges from saturated squares, then re-optimize."""
        selected = list(selected)
        
        # Compute square counts
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        # Find squares at capacity (count == 3)
        saturated_squares = [s for s in range(NUM_SQUARES) if square_counts[s] == 3]
        
        if saturated_squares:
            # Remove one random edge from each of kick_size random saturated squares
            random.shuffle(saturated_squares)
            removed = set()
            for s in saturated_squares[:kick_size]:
                if s in removed:
                    continue
                # Pick a random selected edge in this square
                edges_in_square = [e for e in square_edges[s] if selected[e]]
                if edges_in_square:
                    e_to_remove = random.choice(edges_in_square)
                    if e_to_remove not in removed:
                        selected[e_to_remove] = False
                        count -= 1
                        removed.add(e_to_remove)
                        # Update square counts
                        for s2 in EDGE_SQUARES[e_to_remove]:
                            square_counts[s2] -= 1
        else:
            # No saturated squares, just remove random edges
            selected_indices = [i for i in range(len(EDGES)) if selected[i]]
            if len(selected_indices) > kick_size:
                to_remove = random.sample(selected_indices, kick_size)
                for i in to_remove:
                    selected[i] = False
                    count -= 1
                    for s in EDGE_SQUARES[i]:
                        square_counts[s] -= 1
        
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
    kick_attempts = 0
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
            kick_attempts = 0  # Reset on improvement
        else:
            kick_attempts += 1
            # After several failed kicks, try fresh greedy + local search
            if kick_attempts >= 3 and time.time() < deadline - 10.0:
                kick_attempts = 0
                if random.random() < 0.5:
                    selected, count = greedy_random()
                else:
                    selected, count = greedy_degree()
                selected, count = local_search(selected, count)
                if count > best_count:
                    best_count = count
                    best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
    
    # Final polish: multiple rounds of add + 1-swap
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
        for _ in range(3):
            if time.time() >= deadline - 2.0:
                break
            
            # Add pass
            candidates = [e for e in range(len(EDGES)) if not selected[e]]
            random.shuffle(candidates)
            for e in candidates:
                if time.time() >= deadline - 2.0:
                    break
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
            
            # 1-swap pass
            selected_list = [e for e in range(len(EDGES)) if selected[e]]
            random.shuffle(selected_list)
            for e_out in selected_list:
                if time.time() >= deadline - 2.0:
                    break
                out_sqs = EDGE_SQUARES[e_out]
                candidate_e_in = []
                for s in out_sqs:
                    for e_in in square_edges[s]:
                        if e_in != e_out and not selected[e_in]:
                            candidate_e_in.append(e_in)
                if not candidate_e_in:
                    continue
                random.shuffle(candidate_e_in)
                for e_in in candidate_e_in:
                    in_sqs = EDGE_SQUARES[e_in]
                    valid = True
                    for s in in_sqs:
                        if s not in out_sqs and square_counts[s] >= 3:
                            valid = False
                            break
                    if valid:
                        selected[e_out] = False
                        selected[e_in] = True
                        for s in out_sqs:
                            square_counts[s] -= 1
                        for s in in_sqs:
                            square_counts[s] += 1
                        break
        
        best_edges = [EDGES[i] for i in range(len(EDGES)) if selected[i]]
    
    return [[u, v] for u, v in best_edges]
