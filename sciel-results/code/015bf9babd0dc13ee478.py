import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    d = 7
    nverts = 1 << d
    total_edges = d * nverts // 2
    
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
    
    num_squares = len(squares)
    
    # 3. For each edge, find all squares containing it, and for each square, the other 3 edges
    edge_to_squares = [[] for _ in range(total_edges)]
    edge_other_masks = [[] for _ in range(total_edges)]  # bitmask of other 3 edges in each square
    
    for s_idx, sq in enumerate(squares):
        for k, e_idx in enumerate(sq):
            edge_to_squares[e_idx].append(s_idx)
            others_mask = 0
            for m in range(4):
                if m != k:
                    others_mask |= (1 << sq[m])
            edge_other_masks[e_idx].append(others_mask)
    
    # 4. Precompute: for each edge, the list of (square_idx, other_edges_list) for quick access
    # Also precompute: for each square, the 4 edges
    square_edges = [set(sq) for sq in squares]
    
    def can_add_fast(e_idx, selected_set):
        """Check if edge e_idx can be added without creating a C4."""
        for s_idx in edge_to_squares[e_idx]:
            sq = square_edges[s_idx]
            # Count how many of the other 3 edges are in selected_set
            count = 0
            for e in sq:
                if e != e_idx and e in selected_set:
                    count += 1
                    if count >= 3:
                        return False
        return True
    
    def get_conflict_count(e_idx, selected_set):
        """Return number of squares that would be completed if e_idx is added."""
        conflicts = 0
        for s_idx in edge_to_squares[e_idx]:
            sq = square_edges[s_idx]
            count = 0
            for e in sq:
                if e != e_idx and e in selected_set:
                    count += 1
            if count >= 3:
                conflicts += 1
        return conflicts
    
    def greedy_init(seed=None):
        """Random greedy initialization."""
        if seed is not None:
            random.seed(seed)
        selected = set()
        order = list(range(total_edges))
        random.shuffle(order)
        for e_idx in order:
            if can_add_fast(e_idx, selected):
                selected.add(e_idx)
        return selected
    
    def local_search(selected, time_budget=5.0):
        """Local search: try to add edges, and do swaps when blocked."""
        start_time = time.time()
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            candidates = [e for e in range(total_edges) if e not in selected]
            random.shuffle(candidates)
            
            for e_idx in candidates:
                if can_add_fast(e_idx, selected):
                    selected.add(e_idx)
                    improved = True
                else:
                    # Try swaps: find a blocking square and try removing one of the other edges
                    for s_idx in edge_to_squares[e_idx]:
                        sq = square_edges[s_idx]
                        others = [e for e in sq if e != e_idx and e in selected]
                        if len(others) >= 3:
                            # Try removing each other edge
                            for f_idx in others:
                                selected.discard(f_idx)
                                if can_add_fast(e_idx, selected):
                                    selected.add(e_idx)
                                    improved = True
                                    break
                                else:
                                    selected.add(f_idx)
                            if improved:
                                break
                        if improved:
                            break
            
            # After a pass, try adding any newly addable edges
            for e_idx in range(total_edges):
                if e_idx not in selected and can_add_fast(e_idx, selected):
                    selected.add(e_idx)
                    improved = True
        
        return selected
    
    best_count = 0
    best_set = set()
    
    # Try multiple random seeds
    rng = random.Random(42)
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random greedy init
        selected = greedy_init()
        
        # Local search
        selected = local_search(selected, time_budget=2.0)
        
        count = len(selected)
        if count > best_count:
            best_count = count
            best_set = set(selected)
        
        # Occasionally do a more aggressive local search on the best
        if restart % 5 == 0 and count >= best_count - 2:
            selected = local_search(set(best_set), time_budget=3.0)
            if len(selected) > best_count:
                best_count = len(selected)
                best_set = set(selected)
    
    # Final polish on the best solution
    if time.time() < deadline - 2:
        best_set = local_search(best_set, time_budget=8.0)
        best_count = len(best_set)
    
    # Convert best_set to list of [u, v]
    result = []
    for e_idx in best_set:
        u, v = edges[e_idx]
        result.append([u, v])
        
    return result
