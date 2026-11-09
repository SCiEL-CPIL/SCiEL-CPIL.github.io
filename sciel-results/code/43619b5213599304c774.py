import random
import itertools
import time
import math

# Precompute Q3 properties
# Q3 vertices are 0..7. Edge (i,j) exists if popcount(i^j)==1.
q3_edges = []
q3_adj = [0]*8
for i in range(8):
    adj = 0
    for j in range(8):
        if i != j and bin(i ^ j).count('1') == 1:
            adj |= (1 << j)
            if i < j:
                q3_edges.append((i, j))
    q3_adj[i] = adj

def count_q3s(adj_bits, n=16):
    """
    Count the number of induced Q3 subgraphs in the graph defined by adj_bits.
    adj_bits[v] is a bitmask of neighbors of vertex v.
    """
    count = 0
    # Iterate over all 8-vertex subsets
    for subset in itertools.combinations(range(n), 8):
        # Map local indices 0-7 to global vertices in subset
        # local_adj[i] will be the bitmask of local neighbors for vertex subset[i]
        local_adj = [0]*8
        
        # Build local adjacency bitmasks
        for i in range(8):
            vi = subset[i]
            # For each local vertex i, check which other local vertices are neighbors
            # We can do this by checking adj_bits[vi] against the subset members
            # But it's faster to just iterate j
            for j in range(8):
                if i != j:
                    if (adj_bits[vi] >> subset[j]) & 1:
                        local_adj[i] |= (1 << j)
        
        # Check if this induced subgraph is exactly Q3
        # Q3 has 12 edges.
        # We verify that for every pair (i,j) in q3_edges, local_adj[i] has bit j set,
        # and for every pair (i,j) NOT in q3_edges (i!=j), local_adj[i] does NOT have bit j set.
        is_q3 = True
        for i in range(8):
            # Check required edges
            for j in range(8):
                if i == j: continue
                should_be_edge = (q3_adj[i] >> j) & 1
                is_edge = (local_adj[i] >> j) & 1
                if should_be_edge != is_edge:
                    is_q3 = False
                    break
            if not is_q3:
                break
        
        if is_q3:
            count += 1
            
    return count

def construct():
    n = 16
    total_edges_possible = n * (n-1) // 2  # 120
    
    start_time = time.time()
    time_limit = 115.0
    
    # Initialize with a dense random graph
    # We want to start high and remove edges if necessary, or start medium and optimize.
    # Let's start with p=0.8
    adj_bits = [0]*n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.8:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    current_cost = count_q3s(adj_bits)
    current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    
    best_adj_bits = adj_bits[:]
    best_edges = current_edges
    best_cost = current_cost
    
    # Simulated Annealing
    # We want to maximize edges, subject to cost == 0.
    # Objective: Minimize (Cost * Weight - Edges) ? 
    # Actually, we just want to reach Cost=0 with max Edges.
    # Let's define a score function: 
    # If cost > 0, we prioritize reducing cost.
    # If cost == 0, we prioritize increasing edges.
    
    # Strategy:
    # 1. Use SA to drive cost to 0.
    # 2. Once cost is 0, use a local search (hill climbing) to add edges without creating Q3s.
    
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.001
    
    # Precompute all possible edges for faster flipping
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start_time < time_limit - 5:
        if T < min_T:
            break
            
        # Pick a random edge to flip
        u, v = random.choice(all_edges)
        
        # Flip the edge
        adj_bits[u] ^= (1 << v)
        adj_bits[v] ^= (1 << u)
        
        new_cost = count_q3s(adj_bits)
        new_edges = current_edges + (1 if (adj_bits[u] >> v) & 1 else -1)
        
        # Determine acceptance
        # We define a "score" that we want to maximize.
        # Score = Edges - Penalty * Cost
        # But we strictly want Cost=0 eventually.
        # Let's use a simple heuristic:
        # If new_cost < current_cost: Accept (improvement in constraint satisfaction)
        # If new_cost == current_cost: Accept if new_edges > current_edges (improvement in objective)
        # If new_cost > current_cost: Accept with probability exp(-(new_cost - current_cost) / T)
        
        accept = False
        if new_cost < current_cost:
            accept = True
        elif new_cost == current_cost:
            if new_edges > current_edges:
                accept = True
            elif new_edges == current_edges and random.random() < 0.5:
                accept = True
        else:
            # Penalty for increasing cost
            delta_cost = new_cost - current_cost
            # We also care about edges. If we add edges but increase cost, it's usually bad.
            # If we remove edges and increase cost? Unlikely.
            # Let's penalize cost increase heavily.
            # Effective delta for annealing:
            # We want to minimize Cost.
            # Let's just use Cost for the annealing step to ensure we get to 0.
            prob = math.exp(-delta_cost / T)
            if random.random() < prob:
                accept = True
        
        if accept:
            current_cost = new_cost
            current_edges = new_edges
            # Update best if we have a valid solution (cost=0) with more edges
            if current_cost == 0 and current_edges > best_edges:
                best_edges = current_edges
                best_adj_bits = adj_bits[:]
                best_cost = 0
            # Also track best non-zero cost solution? No, we only care about valid ones.
            # But if we haven't found any valid one, we might want to keep the "least bad" one?
            # The task requires a Q3-free graph. If we don't find one, score is 0?
            # "A Q3-free graph with m edges certifies...". If we return a graph with Q3, it's invalid?
            # "The evaluator measures the raw value... invalid result... scores 0."
            # So we MUST return a Q3-free graph.
            # If we end up with cost > 0, we must clean it up in the final phase.
        else:
            # Revert
            adj_bits[u] ^= (1 << v)
            adj_bits[v] ^= (1 << u)
            
        T *= cooling_rate
        
        # If we found a perfect solution (cost=0), we can switch to a pure hill-climbing mode
        # to try to add more edges.
        if current_cost == 0:
             # Try to add edges greedily
             # Pick random missing edge
             missing = [e for e in all_edges if not ((adj_bits[e[0]] >> e[1]) & 1)]
             if missing:
                 u, v = random.choice(missing)
                 adj_bits[u] |= (1 << v)
                 adj_bits[v] |= (1 << u)
                 new_cost = count_q3s(adj_bits)
                 if new_cost == 0:
                     current_edges += 1
                     current_cost = 0
                     if current_edges > best_edges:
                         best_edges = current_edges
                         best_adj_bits = adj_bits[:]
                 else:
                     # Revert
                     adj_bits[u] &= ~(1 << v)
                     adj_bits[v] &= ~(1 << u)
                     # Also try removing an edge to see if we can get stuck out? 
                     # No, just revert.
             # If graph is complete, stop?
             if current_edges == total_edges_possible:
                 break

    # Final Phase: Ensure the result is valid (Cost=0)
    # If best_cost > 0, we need to fix best_adj_bits
    # We will use the best_adj_bits found. If it has Q3s, we remove edges to destroy them.
    final_adj_bits = best_adj_bits[:]
    
    # Check if final_adj_bits is valid
    # If not, we must make it valid.
    # We can use the same "remove random edge from Q3" strategy on the best solution found.
    # But wait, if we never found a cost=0 solution, best_adj_bits might have Q3s.
    # Let's ensure we have a valid solution.
    # If best_cost > 0, we run a cleanup loop on best_adj_bits.
    
    if count_q3s(final_adj_bits) > 0:
        # Fallback: Start from a known valid structure or clean up.
        # Let's clean up the current best_adj_bits.
        while count_q3s(final_adj_bits) > 0 and time.time() - start_time < time_limit - 1:
            # Find a Q3
            # We need a function to find ONE q3 subset
            # Re-use logic from count_q3s but return first found
            q3_found = None
            for subset in itertools.combinations(range(n), 8):
                local_adj = [0]*8
                for i in range(8):
                    vi = subset[i]
                    for j in range(8):
                        if i != j:
                            if (final_adj_bits[vi] >> subset[j]) & 1:
                                local_adj[i] |= (1 << j)
                is_q3 = True
                for i in range(8):
                    for j in range(8):
                        if i == j: continue
                        should_be_edge = (q3_adj[i] >> j) & 1
                        is_edge = (local_adj[i] >> j) & 1
                        if should_be_edge != is_edge:
                            is_q3 = False
                            break
                    if not is_q3:
                        break
                if is_q3:
                    q3_found = subset
                    break
            
            if q3_found is None:
                break # Should not happen if count > 0
                
            # Remove a random edge from this Q3
            edges_in_q3 = []
            for i in range(8):
                for j in range(i+1, 8):
                    if (final_adj_bits[q3_found[i]] >> q3_found[j]) & 1:
                        edges_in_q3.append((q3_found[i], q3_found[j]))
                        
            if edges_in_q3:
                u, v = random.choice(edges_in_q3)
                final_adj_bits[u] &= ~(1 << v)
                final_adj_bits[v] &= ~(1 << u)
            else:
                break # Safety break

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (final_adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
