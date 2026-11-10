import random
import time
import numpy as np

N_DIMS = 7
N_VERTS = 1 << N_DIMS  # 128
N_EDGES = N_DIMS * (N_VERTS // 2)  # 448

def precompute():
    edges = []
    edge_map = {}
    for u in range(N_VERTS):
        for i in range(N_DIMS):
            v = u ^ (1 << i)
            if u < v:
                edge_map[(u, v)] = len(edges)
                edges.append((u, v))
    
    # Build square-to-edge mapping
    # Each square is defined by 2 dimensions (i,j) and 5 fixed bits
    square_edges = []  # list of lists of 4 edge indices
    edge_squares = [[] for _ in range(len(edges))]
    
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
                
                e1 = edge_map[(min(v00, v10), max(v00, v10))]
                e2 = edge_map[(min(v00, v01), max(v00, v01))]
                e3 = edge_map[(min(v10, v11), max(v10, v11))]
                e4 = edge_map[(min(v01, v11), max(v01, v11))]
                
                sq_idx = len(square_edges)
                square_edges.append([e1, e2, e3, e4])
                edge_squares[e1].append(sq_idx)
                edge_squares[e2].append(sq_idx)
                edge_squares[e3].append(sq_idx)
                edge_squares[e4].append(sq_idx)
    
    return edges, edge_squares, square_edges

EDGES, EDGE_SQUARES, SQUARE_EDGES = precompute()
NUM_SQUARES = len(SQUARE_EDGES)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_count = 0
    best_selected = None
    
    # Build edge_map for quick lookup
    edge_map = {}
    for idx, (u, v) in enumerate(EDGES):
        edge_map[(min(u, v), max(u, v))] = idx
    
    def greedy_random(seed=None):
        """Randomized greedy with edge ordering by square conflict."""
        if seed is not None:
            random.seed(seed)
        
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
    
    def greedy_weighted(seed=None):
        """Greedy that prioritizes edges with fewer conflicts."""
        if seed is not None:
            random.seed(seed)
        
        selected = [False] * len(EDGES)
        square_counts = [0] * NUM_SQUARES
        count = 0
        
        # Sort by number of squares (all edges have 4 squares in Q7, so use random tiebreak)
        # Actually, each edge belongs to exactly 4 squares in Q7
        # So we use a different heuristic: process edges in an order that minimizes future conflicts
        # Use a simple approach: repeatedly pick the edge with minimum "cost"
        
        remaining = set(range(len(EDGES)))
        while remaining:
            best_edge = None
            best_score = -1
            # Sample a few edges to speed up
            candidates = list(remaining)
            random.shuffle(candidates)
            for e in candidates[:50]:
                sqs = EDGE_SQUARES[e]
                # Score: prefer edges where squares have lower counts
                score = sum(square_counts[s] for s in sqs)
                # Add small random perturbation
                score += random.random() * 0.1
                if best_edge is None or score < best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                break
            
            sqs = EDGE_SQUARES[best_edge]
            valid = True
            for s in sqs:
                if square_counts[s] >= 3:
                    valid = False
                    break
            
            if valid:
                selected[best_edge] = True
                count += 1
                for s in sqs:
                    square_counts[s] += 1
            
            remaining.discard(best_edge)
        
        return selected, count
    
    def local_search(selected, count, max_iters=10000):
        """
        Local search with add/swap operations.
        """
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters and time.time() < deadline - 3.0:
            improved = False
            iter_count += 1
            
            # Try to add unselected edges
            candidates = [e for e in range(len(EDGES)) if not selected[e]]
            random.shuffle(candidates)
            
            for e in candidates:
                if time.time() >= deadline - 3.0:
                    break
                
                sqs = EDGE_SQUARES[e]
                blocking_squares = [s for s in sqs if square_counts[s] == 3]
                
                if not blocking_squares:
                    # Can add freely
                    selected[e] = True
                    count += 1
                    for s in sqs:
                        square_counts[s] += 1
                    improved = True
                else:
                    # Try swap: remove one edge from a blocking square, add e
                    best_swap = None
                    best_gain = -100
                    
                    for s in blocking_squares:
                        for other_e in SQUARE_EDGES[s]:
                            if other_e == e or not selected[other_e]:
                                continue
                            
                            # Check if swap is valid
                            # After removing other_e and adding e:
                            # - Square s: count stays 3 (remove 1, add 1)
                            # - Other squares of other_e: count decreases by 1 (good)
                            # - Other squares of e: count increases by 1 (check they don't exceed 3)
                            
                            valid = True
                            for es in sqs:
                                if es != s:
                                    if square_counts[es] >= 3:
                                        valid = False
                                        break
                            
                            if valid:
                                # Estimate gain: we can potentially add more edges after this
                                # For now, just do the swap
                                best_swap = (other_e, e)
                                best_gain = 1
                                break
                        
                        if best_swap:
                            break
                    
                    if best_swap:
                        other_e, new_e = best_swap
                        selected[other_e] = False
                        selected[new_e] = True
                        # Update counts
                        for os in EDGE_SQUARES[other_e]:
                            square_counts[os] -= 1
                        for es in EDGE_SQUARES[new_e]:
                            square_counts[es] += 1
                        improved = True
        
        return selected, count
    
    def kick(selected, count, kick_size=5):
        """
        Random kick: remove some edges and re-add them to escape local optima.
        """
        selected_copy = selected.copy()
        square_counts = [0] * NUM_SQUARES
        for e in range(len(EDGES)):
            if selected_copy[e]:
                for s in EDGE_SQUARES[e]:
                    square_counts[s] += 1
        
        # Remove kick_size random selected edges
        selected_indices = [e for e in range(len(EDGES)) if selected_copy[e]]
        if len(selected_indices) < kick_size:
            kick_size = max(1, len(selected_indices) // 2)
        
        to_remove = random.sample(selected_indices, min(kick_size, len(selected_indices)))
        for e in to_remove:
            selected_copy[e] = False
            for s in EDGE_SQUARES[e]:
                square_counts[s] -= 1
        
        count -= len(to_remove)
        
        # Try to re-add edges
        candidates = [e for e in range(len(EDGES)) if not selected_copy[e]]
        random.shuffle(candidates)
        
        for e in candidates:
            if time.time() >= deadline - 3.0:
                break
            sqs = EDGE_SQUARES[e]
            valid = True
            for s in sqs:
                if square_counts[s] >= 3:
                    valid = False
                    break
            if valid:
                selected_copy[e] = True
                count += 1
                for s in sqs:
                    square_counts[s] += 1
        
        return selected_copy, count
    
    # Phase 1: Multiple greedy starts
    num_greedy = 0
    while time.time() < deadline - 40.0:
        if num_greedy % 2 == 0:
            selected, count = greedy_random()
        else:
            selected, count = greedy_weighted()
        
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
        
        num_greedy += 1
        if num_greedy >= 30:
            break
    
    # Phase 2: Local search on best solution
    if best_count > 0 and time.time() < deadline - 15.0:
        selected, count = local_search(best_selected.copy(), best_count)
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Phase 3: Kick and local search
    while time.time() < deadline - 5.0:
        # Try kick
        selected, count = kick(best_selected.copy(), best_count, kick_size=random.randint(3, 8))
        
        # Local search after kick
        selected, count = local_search(selected, count, max_iters=5000)
        
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
        
        # Occasionally try fresh greedy + local search
        if time.time() < deadline - 10.0 and random.random() < 0.3:
            selected, count = greedy_random()
            selected, count = local_search(selected, count, max_iters=3000)
            if count > best_count:
                best_count = count
                best_selected = selected.copy()
    
    # Convert to output format
    result = []
    for i in range(len(EDGES)):
        if best_selected[i]:
            u, v = EDGES[i]
            result.append([u, v])
    
    return result
