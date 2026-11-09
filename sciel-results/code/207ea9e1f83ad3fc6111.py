"""Dense K3,3-free graphs via deterministic greedy + simulated annealing."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    n = 20
    total_possible_edges = n * (n - 1) // 2  # 190

    # Bitset representation: adj[i] is an integer where bit j is set if edge (i,j) exists
    def is_k33_free(adj):
        """Check if the graph is K3,3-free.
        K3,3 exists iff there are 3 vertices i,j,k with at least 3 common neighbors.
        """
        for i in range(n):
            for j in range(i + 1, n):
                common_ij = adj[i] & adj[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (common_ij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_all_edges():
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

    def get_missing_edges(adj):
        return [(i, j) for i in range(n) for j in range(i + 1, n)
                if not ((adj[i] >> j) & 1)]

    def get_existing_edges(adj):
        return [(i, j) for i in range(n) for j in range(i + 1, n)
                if (adj[i] >> j) & 1]

    # Phase 1: Deterministic Greedy Construction
    # Sort edges by a heuristic: prioritize edges that connect vertices with low degree
    # or edges that are "less likely" to create K3,3.
    # A simple heuristic: iterate through all edges in a fixed order (e.g., lexicographic)
    # and add if it doesn't violate K3,3-free property.
    # To make it more effective, we can try multiple deterministic orders.
    
    def greedy_construct(order_seed):
        """Greedy construction with a specific edge order."""
        adj = [0] * n
        edges = get_all_edges()
        # Use order_seed to shuffle edges deterministically
        random.seed(order_seed)
        random.shuffle(edges)
        
        for u, v in edges:
            add_edge(adj, u, v)
            if not is_k33_free(adj):
                remove_edge(adj, u, v)
        return adj

    # Run a few greedy constructions with different seeds to find a good starting point
    best_adj = None
    best_edges = -1
    
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        adj = greedy_construct(seed)
        e = count_edges(adj)
        if e > best_edges:
            best_edges = e
            best_adj = adj[:]

    if best_adj is None:
        # Fallback: empty graph
        best_adj = [0] * n
        best_edges = 0

    # Phase 2: Simulated Annealing
    # We perform edge swaps: remove one edge, add another.
    # Objective: maximize edge count.
    # Move: remove edge (u1, v1), add edge (u2, v2).
    # If new graph is K3,3-free, accept.
    # If new graph is NOT K3,3-free, reject (or accept with prob exp(-delta/T) if we allowed invalid states,
    # but here we strictly require K3,3-free, so we only consider moves that result in valid graphs).
    # Since we must maintain validity, we can only accept moves that keep the graph valid.
    # This is essentially a random walk on the space of valid K3,3-free graphs.
    # To escape local optima, we might need to allow temporary decreases in edge count?
    # No, because we want to MAXIMIZE edges. A local optimum is a maximal K3,3-free graph.
    # If we are at a maximal graph, no single edge can be added.
    # We need swaps: remove an edge, add a different edge.
    # If the new graph has the same number of edges (swap) and is valid, we can move.
    # This allows us to traverse the space of maximal graphs.
    # Sometimes, we might need to remove an edge without adding one (decrease count) to enable adding two later?
    # That's complex. Let's stick to swaps first.
    
    # Improvement: Allow "remove then add" where the add might be blocked, but we can try multiple adds.
    # Better: Randomly pick an existing edge to remove, then try to add a missing edge.
    # If the result is valid, accept.
    # To escape local optima, we can use a "temperature" that allows accepting moves that
    # temporarily reduce the edge count? No, that violates the goal.
    # Instead, we can use a more sophisticated move:
    # 1. Remove edge e1.
    # 2. Try to add a random missing edge e2.
    # 3. If valid, accept (edge count unchanged).
    # 4. If invalid, try another e2.
    # 5. If no e2 works, restore e1.
    
    # To actually improve, we need to find a sequence of moves that increases edge count.
    # This is hard with single swaps if we are at a maximal graph.
    # However, different maximal graphs can have different sizes.
    # So, random walks on the space of maximal graphs can find larger ones.
    
    def simulated_annealing(adj, edges_count, deadline):
        current = adj[:]
        current_edges = edges_count
        best = current[:]
        best_edges = current_edges
        
        # Precompute missing and existing
        missing = get_missing_edges(current)
        existing = get_existing_edges(current)
        
        if not missing or not existing:
            return best, best_edges

        # Temperature for simulated annealing?
        # Since we only accept valid moves, and valid moves preserve edge count (swaps),
        # we are doing a random walk.
        # To bias towards higher edge counts, we don't need temperature if we only accept
        # moves that keep the graph valid and maximal?
        # Actually, if we are at a maximal graph, any swap that results in a valid graph
        # will also be maximal? Not necessarily.
        # If we remove an edge and add another, and the result is valid, it might not be maximal.
        # But we can then try to add more edges.
        
        # Strategy:
        # 1. Pick a random existing edge to remove.
        # 2. Pick a random missing edge to add.
        # 3. If the resulting graph is K3,3-free, accept the swap.
        # 4. After accepting a swap, try to greedily add any missing edges that can be added.
        # 5. Keep track of the best graph found.
        
        steps = 0
        while time.time() < deadline - 2:
            steps += 1
            if steps % 1000 == 0:
                # Periodic greedy fill
                for u, v in get_missing_edges(current):
                    add_edge(current, u, v)
                    if is_k33_free(current):
                        current_edges += 1
                    else:
                        remove_edge(current, u, v)
                if current_edges > best_edges:
                    best_edges = current_edges
                    best = current[:]
                missing = get_missing_edges(current)
                existing = get_existing_edges(current)
                if not missing or not existing:
                    break

            if not missing or not existing:
                break

            e1 = random.choice(existing)
            e2 = random.choice(missing)
            
            u1, v1 = e1
            u2, v2 = e2
            
            # Perform swap
            remove_edge(current, u1, v1)
            add_edge(current, u2, v2)
            
            if is_k33_free(current):
                # Accept swap
                # Update lists
                existing.remove(e1)
                existing.append(e2)
                missing.remove(e2)
                missing.append(e1)
                
                # Try to greedily add edges
                added_any = False
                for u, v in get_missing_edges(current):
                    add_edge(current, u, v)
                    if is_k33_free(current):
                        current_edges += 1
                        added_any = True
                        missing.remove((u, v))
                        existing.append((u, v))
                    else:
                        remove_edge(current, u, v)
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best = current[:]
            else:
                # Reject swap, restore
                remove_edge(current, u2, v2)
                add_edge(current, u1, v1)
        
        return best, best_edges

    # Run simulated annealing
    best_adj, best_edges = simulated_annealing(best_adj, best_edges, deadline)

    # Final greedy fill
    for u, v in get_missing_edges(best_adj):
        add_edge(best_adj, u, v)
        if is_k33_free(best_adj):
            best_edges += 1
        else:
            remove_edge(best_adj, u, v)

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-END
