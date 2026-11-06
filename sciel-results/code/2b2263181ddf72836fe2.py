import random
import time
import math

def construct():
    d = 7
    n = 2 ** d
    total_edges = n * d // 2
    
    # Generate all edges
    edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_index[(x, y)] = idx
                edge_index[(y, x)] = idx
    
    # Precompute for each edge, the list of edges that form a C4 with it.
    # This is the "conflict graph" of edges.
    # If edge e1 and e2 are in conflict, they cannot both be in a C4-free set?
    # No, they can be in the set, as long as the other two edges of the C4 are NOT in the set.
    # So simple conflict graph is not enough. We need the C4 structure.
    
    # Let's stick to the `can_add` logic but optimize it.
    
    def can_add(u, v, chosen_set):
        for i in range(d):
            w = u ^ (1 << i)
            if w == v: continue
            if (min(u, w), max(u, w)) in chosen_set:
                for j in range(d):
                    z = v ^ (1 << j)
                    if z == u: continue
                    if (min(v, z), max(v, z)) in chosen_set:
                        if (min(w, z), max(w, z)) in chosen_set:
                            return False
        return True

    # Helper to get a greedy solution given an order of edges
    def greedy_solve(order, initial_set=None):
        if initial_set is None:
            current_set = set()
        else:
            current_set = set(initial_set)
        
        for e in order:
            u, v = e
            if (u, v) not in current_set and (v, u) not in current_set:
                if can_add(u, v, current_set):
                    current_set.add((u, v))
        return current_set

    best_size = 0
    best_set = set()
    
    start_time = time.time()
    max_time = 110.0
    
    # 1. Try several deterministic/random orders for initialization
    orders = []
    # Random order
    e1 = edges[:]
    random.shuffle(e1)
    orders.append(e1)
    
    # Order by dimension
    # We need to know the dimension of each edge
    edge_dims = []
    for e in edges:
        u, v = e
        diff = u ^ v
        # diff is a power of 2
        dim = diff.bit_length() - 1
        edge_dims.append(dim)
        
    # Sort by dim
    sorted_by_dim = sorted(range(len(edges)), key=lambda i: edge_dims[i])
    e2 = [edges[i] for i in sorted_by_dim]
    orders.append(e2)
    
    # Sort by vertex
    sorted_by_vert = sorted(range(len(edges)), key=lambda i: edges[i][0])
    e3 = [edges[i] for i in sorted_by_vert]
    orders.append(e3)
    
    # Random weights
    weights = [random.random() for _ in range(len(edges))]
    sorted_by_weight = sorted(range(len(edges)), key=lambda i: weights[i])
    e4 = [edges[i] for i in sorted_by_weight]
    orders.append(e4)
    
    for order in orders:
        sol = greedy_solve(order)
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = set(sol)
            
    # 2. Simulated Annealing / Local Search
    # We will perform local search on the best_set found so far.
    # Move: Randomly pick an edge.
    # If not in set, try to add.
    # If in set, try to remove (to allow re-adding different edges).
    
    # To make it faster, we maintain a list of "candidate" edges to try adding.
    # Initially, all edges not in best_set.
    
    current_set = set(best_set)
    current_size = len(current_set)
    
    # Temperature schedule
    T = 1.0
    cooling = 0.99
    
    while time.time() - start_time < max_time:
        # Pick a random move
        # 80% chance to try adding, 20% chance to try removing (kick)
        if random.random() < 0.8:
            # Try to add a random edge
            e = random.choice(edges)
            u, v = e
            if (u, v) not in current_set:
                if can_add(u, v, current_set):
                    current_set.add((u, v))
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = set(current_set)
        else:
            # Kick: Remove a random edge from current_set
            if current_size > 0:
                e_to_remove = random.choice(list(current_set))
                current_set.remove(e_to_remove)
                current_size -= 1
                
                # After removing, try to greedily add back edges
                # Focus on edges that were removed or edges that are "near" the removed edge
                # For simplicity, do a few greedy passes on a subset or full list
                # To be efficient, try to add the removed edge back (maybe it fits now? No, we removed it because we wanted to kick, but maybe it fits with others?)
                # Actually, just run a quick greedy on all edges not in set.
                # This is expensive.
                # Instead, try to add random edges.
                for _ in range(10):
                    e_try = random.choice(edges)
                    u, v = e_try
                    if (u, v) not in current_set:
                        if can_add(u, v, current_set):
                            current_set.add((u, v))
                            current_size += 1
                            if current_size > best_size:
                                best_size = current_size
                                best_set = set(current_set)
        
        T *= cooling
        if T < 0.01:
            T = 1.0 # Reset or restart?
            # Let's just keep going, the 80/20 balance handles exploration.
            
    return [list(e) for e in best_set]
