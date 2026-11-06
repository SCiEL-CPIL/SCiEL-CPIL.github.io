import time
import random

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))  # 448
    
    # 1. Generate Edges
    edges = []
    edge_index = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
                edge_index[(v, u)] = idx
    
    # 2. Generate Squares (2-faces)
    squares = []
    edge_to_squares = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n_vertices):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_index[(v1, v2)]
                    e2 = edge_index[(v3, v4)]
                    e3 = edge_index[(v1, v3)]
                    e4 = edge_index[(v2, v4)]
                    
                    sq_idx = len(squares)
                    squares.append((e1, e2, e3, e4))
                    
                    for e in (e1, e2, e3, e4):
                        edge_to_squares[e].append(sq_idx)
                        
    n_squares = len(squares)
    
    # 3. State Variables
    selected = [True] * n_edges
    sq_count = [4] * n_squares
    blocked_edges = [0] * n_edges
    
    bad_squares = set(range(n_squares))
    candidates = set()
    
    # 4. Phase 1: Greedy Removal
    while bad_squares:
        q = next(iter(bad_squares))
        
        max_impact = -1
        best_edge = -1
        
        for e in squares[q]:
            impact = 0
            for q2 in edge_to_squares[e]:
                if q2 in bad_squares:
                    impact += 1
            if impact > max_impact:
                max_impact = impact
                best_edge = e
                
        e = best_edge
        selected[e] = False
        if e in candidates:
            candidates.remove(e)
            
        for q2 in edge_to_squares[e]:
            old_count = sq_count[q2]
            sq_count[q2] = old_count - 1
            new_count = sq_count[q2]
            
            if old_count == 4 and new_count == 3:
                bad_squares.remove(q2)
                for f in squares[q2]:
                    if f != e:
                        blocked_edges[f] += 1
                        if not selected[f] and f in candidates:
                            candidates.remove(f)
                            
            elif old_count == 3 and new_count == 2:
                for f in squares[q2]:
                    if f != e:
                        blocked_edges[f] -= 1
                        if blocked_edges[f] == 0 and not selected[f]:
                            candidates.add(f)
                            
    for e in range(n_edges):
        if not selected[e] and blocked_edges[e] == 0:
            candidates.add(e)
            
    current_size = sum(selected)
    best_size = current_size
    best_selected = selected[:]
    
    # 5. Phase 2: Local Search
    deadline = time.time() + 140
    
    while time.time() < deadline:
        if candidates:
            e = next(iter(candidates))
            candidates.remove(e)
            
            selected[e] = True
            current_size += 1
            
            for q in edge_to_squares[e]:
                old_count = sq_count[q]
                sq_count[q] = old_count + 1
                new_count = sq_count[q]
                
                if old_count == 2 and new_count == 3:
                    for f in squares[q]:
                        if f != e:
                            blocked_edges[f] += 1
                            if not selected[f] and f in candidates:
                                candidates.remove(f)
                                
            if current_size > best_size:
                best_size = current_size
                best_selected = selected[:]
                
        else:
            # Kick
            while True:
                e_out = random.randrange(n_edges)
                if selected[e_out]:
                    break
            
            selected[e_out] = False
            current_size -= 1
            if e_out in candidates:
                candidates.remove(e_out)
                
            for q in edge_to_squares[e_out]:
                old_count = sq_count[q]
                sq_count[q] = old_count - 1
                new_count = sq_count[q]
                
                if old_count == 4 and new_count == 3:
                    bad_squares.discard(q)
                    for f in squares[q]:
                        if f != e_out:
                            blocked_edges[f] += 1
                            if not selected[f] and f in candidates:
                                candidates.remove(f)
                            
                elif old_count == 3 and new_count == 2:
                    for f in squares[q]:
                        if f != e_out:
                            blocked_edges[f] -= 1
                            if blocked_edges[f] == 0 and not selected[f]:
                                candidates.add(f)
                                
            while candidates:
                e = next(iter(candidates))
                candidates.remove(e)
                
                selected[e] = True
                current_size += 1
                
                for q in edge_to_squares[e]:
                    old_count = sq_count[q]
                    sq_count[q] = old_count + 1
                    new_count = sq_count[q]
                    
                    if old_count == 2 and new_count == 3:
                        for f in squares[q]:
                            if f != e:
                                blocked_edges[f] += 1
                                if not selected[f] and f in candidates:
                                    candidates.remove(f)
                                    
            if current_size > best_size:
                best_size = current_size
                best_selected = selected[:]
            elif current_size < best_size:
                selected = best_selected[:]
                current_size = best_size
                
                sq_count = [0] * n_squares
                for e in range(n_edges):
                    if selected[e]:
                        for q in edge_to_squares[e]:
                            sq_count[q] += 1
                            
                blocked_edges = [0] * n_edges
                candidates = set()
                for q in range(n_squares):
                    if sq_count[q] == 3:
                        for f in squares[q]:
                            blocked_edges[f] += 1
                            
                for e in range(n_edges):
                    if not selected[e] and blocked_edges[e] == 0:
                        candidates.add(e)
                        
    out = []
    for i, is_sel in enumerate(best_selected):
        if is_sel:
            u, v = edges[i]
            out.append([u, v])
            
    return out
