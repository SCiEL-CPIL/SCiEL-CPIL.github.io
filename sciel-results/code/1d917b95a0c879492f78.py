import random
import numpy as np

def construct():
    d = 7
    n = 2 ** d
    total_edges = d * (n // 2)  # 448
    
    # Generate all edges
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Generate all squares and map edges to squares
    # Each square is a set of 4 edge indices
    squares = []  # list of (e0, e1, e2, e3)
    edge_to_squares = [[] for _ in range(total_edges)]
    
    for i in range(d):
        for j in range(i+1, d):
            mask = (1 << i) | (1 << j)
            for x in range(n):
                if x & mask == 0:  # canonical base
                    v0 = x
                    v1 = x | (1 << i)
                    v2 = x | (1 << j)
                    v3 = x | (1 << i) | (1 << j)
                    
                    e01 = (min(v0,v1), max(v0,v1))
                    e13 = (min(v1,v3), max(v1,v3))
                    e32 = (min(v3,v2), max(v3,v2))
                    e20 = (min(v2,v0), max(v2,v0))
                    
                    sq = [edge_to_idx[e01], edge_to_idx[e13], 
                          edge_to_idx[e32], edge_to_idx[e20]]
                    sq_idx = len(squares)
                    squares.append(sq)
                    for e in sq:
                        edge_to_squares[e].append(sq_idx)
    
    num_squares = len(squares)
    
    # Square counts: how many edges of each square are in chosen
    sq_count = [0] * num_squares
    
    def can_add(e_idx):
        for sq_idx in edge_to_squares[e_idx]:
            if sq_count[sq_idx] >= 3:
                return False
        return True
    
    def add_edge(e_idx):
        for sq_idx in edge_to_squares[e_idx]:
            sq_count[sq_idx] += 1
    
    def remove_edge(e_idx):
        for sq_idx in edge_to_squares[e_idx]:
            sq_count[sq_idx] -= 1
    
    # Greedy initialization
    chosen = [False] * total_edges
    for e_idx in range(total_edges):
        if can_add(e_idx):
            chosen[e_idx] = True
            add_edge(e_idx)
    
    current_size = sum(chosen)
    best_size = current_size
    best_chosen = chosen[:]
    
    random.seed(42)
    
    # Simulated annealing
    T = 1.0
    cooling = 0.9995
    iterations = 2000000
    
    for it in range(iterations):
        T *= cooling
        if T < 0.0001:
            break
        
        e_idx = random.randint(0, total_edges - 1)
        
        if chosen[e_idx]:
            # Try to swap: remove this edge, add another
            e2_idx = random.randint(0, total_edges - 1)
            if e2_idx == e_idx or chosen[e2_idx]:
                continue
            
            # Check if e2 can be added after removing e_idx
            # Temporarily remove e_idx
            for sq_idx in edge_to_squares[e_idx]:
                sq_count[sq_idx] -= 1
            
            ok = can_add(e2_idx)
            
            if ok:
                add_edge(e2_idx)
                chosen[e_idx] = False
                chosen[e2_idx] = True
                # Size unchanged, but maybe better structure
                if random.random() < 0.5:  # Accept with some probability
                    pass
                else:
                    # Revert
                    chosen[e2_idx] = False
                    remove_edge(e2_idx)
                    chosen[e_idx] = True
                    add_edge(e_idx)
            else:
                # Revert
                for sq_idx in edge_to_squares[e_idx]:
                    sq_count[sq_idx] += 1
        else:
            # Try to add
            if can_add(e_idx):
                chosen[e_idx] = True
                add_edge(e_idx)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_chosen = chosen[:]
            else:
                # Try swap: remove a conflicting edge, add this one
                # Find a square that blocks us
                for sq_idx in edge_to_squares[e_idx]:
                    if sq_count[sq_idx] >= 3:
                        # Find an edge in this square that is chosen, remove it, add e_idx
                        sq = squares[sq_idx]
                        # Pick a random edge from the square that is chosen
                        candidates = [e for e in sq if chosen[e]]
                        if candidates:
                            victim = random.choice(candidates)
                            # Remove victim, add e_idx
                            chosen[victim] = False
                            remove_edge(victim)
                            chosen[e_idx] = True
                            add_edge(e_idx)
                            # Size unchanged
                        break
    
    return [list(all_edges[i]) for i in range(total_edges) if best_chosen[i]]
