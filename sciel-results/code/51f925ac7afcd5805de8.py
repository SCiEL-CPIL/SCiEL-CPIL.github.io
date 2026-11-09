import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    def propagate_state(state, v, c):
        stack = [(v, c)]
        while stack:
            cv, cc = stack.pop()
            if state[cv] != -1:
                if state[cv] != cc:
                    return False
                else:
                    continue
            state[cv] = cc
            for e_idx in vertex_edges[cv]:
                e = edges[e_idx]
                c0 = 0
                c1 = 0
                uncolored = []
                for u in e:
                    if state[u] == 0:
                        c0 += 1
                    elif state[u] == 1:
                        c1 += 1
                    else:
                        uncolored.append(u)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if c0 == 4 and len(uncolored) == 1:
                    stack.append((uncolored[0], 1))
                elif c1 == 4 and len(uncolored) == 1:
                    stack.append((uncolored[0], 0))
        return True

    def solve(color_state):
        # Find first uncolored vertex
        v = -1
        for i in range(n):
            if color_state[i] == -1:
                v = i
                break
        if v == -1:
            return True  # All colored, valid
        
        # Try 0
        new_state_0 = color_state[:]
        if propagate_state(new_state_0, v, 0):
            if solve(new_state_0):
                return True
        
        # Try 1
        new_state_1 = color_state[:]
        if propagate_state(new_state_1, v, 1):
            if solve(new_state_1):
                return True
        
        return False

    initial_state = [-1] * n
    return solve(initial_state)

def _biased_random_edges(n, m, seed):
    """
    Generate biased random edges that are more likely to create conflicts.
    """
    rng = random.Random(seed)
    all_edges = list(combinations(range(n), 5))
    
    # Bias: prefer edges that share vertices with previously chosen edges
    edges = []
    if m <= 0:
        return edges
    
    # Start with a random edge
    first = rng.choice(all_edges)
    edges.append(list(first))
    
    # Build a vertex-edge map for biasing
    vertex_count = [0] * n
    for v in first:
        vertex_count[v] += 1
    
    remaining = list(all_edges)
    remaining.remove(first)
    
    while len(edges) < m and remaining:
        # Score each remaining edge by how many vertices it shares with existing edges
        best_edge = None
        best_score = -1
        for e in remaining:
            score = sum(vertex_count[v] for v in e)
            if score > best_score:
                best_score = score
                best_edge = e
        
        if best_edge is None:
            break
        
        edges.append(list(best_edge))
        for v in best_edge:
            vertex_count[v] += 1
        remaining.remove(best_edge)
    
    # If we didn't get enough, fill with random
    if len(edges) < m:
        rng.shuffle(remaining)
        for e in remaining:
            if len(edges) >= m:
                break
            if e not in [tuple(x) for x in edges]:
                edges.append(list(e))
    
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    
    n = 15
    best_result = None
    best_m = float('inf')
    
    # Try m from 49 upward
    for m in range(49, 65):
        if time.time() > deadline:
            break
        
        found = False
        for seed in range(50):
            if time.time() > deadline:
                break
            
            edges = _biased_random_edges(n, m, seed)
            if len(edges) != m:
                continue
            
            if not _is_2_colorable(n, edges):
                if m < best_m:
                    best_result = {'n': n, 'edges': edges}
                    best_m = m
                found = True
                break
        
        if found:
            break
    
    # Fallback
    if not best_result:
        n = 15
        all_edges = list(combinations(range(n), 5))
        edges = [list(e) for e in all_edges[:120]]
        best_result = {'n': n, 'edges': edges}
    
    return best_result
