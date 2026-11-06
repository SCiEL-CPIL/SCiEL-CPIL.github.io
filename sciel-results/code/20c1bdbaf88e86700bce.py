"""Global 2-Opt improvement loop with nearest-neighbor initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # 5s safety margin before 150s kill


def _nearest_neighbor_tour(start, distance_matrix):
    """Construct a tour using nearest-neighbor heuristic."""
    n = distance_matrix.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    
    for _ in range(n - 1):
        # Find nearest unvisited node
        d = distance_matrix[current].copy()
        d[visited] = np.inf
        next_node = int(np.argmin(d))
        visited[next_node] = True
        tour.append(next_node)
        current = next_node
    
    return tour


def _two_opt_improve(tour, distance_matrix, max_iterations=10000):
    """Apply 2-Opt improvement to a tour. Returns improved tour."""
    n = len(tour)
    if n <= 3:
        return tour
    
    # Precompute tour as list for easy access
    tour = list(tour)
    improved = True
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Check time budget
        if time.time() - _START_TIME > _DEADLINE - 2.0:
            break
        
        for i in range(n - 1):
            if time.time() - _START_TIME > _DEADLINE - 2.0:
                break
            
            for j in range(i + 2, n):
                # Skip if i and j are adjacent in the cycle (i, i+1) or (0, n-1)
                if i == 0 and j == n - 1:
                    continue
                
                a = tour[i]
                b = tour[i + 1]
                c = tour[j]
                d = tour[j + 1] if j + 1 < n else tour[0]
                
                # Calculate improvement
                current_cost = distance_matrix[a, b] + distance_matrix[c, d]
                new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                
                if new_cost < current_cost - 1e-10:
                    # Reverse the segment between i+1 and j
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    
    return tour


def _compute_tour_length(tour, distance_matrix):
    """Compute total length of a closed tour."""
    n = len(tour)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        total += distance_matrix[tour[i], tour[(i + 1) % n]]
    return total


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a global 2-Opt improvement strategy.
    
    Since we need to return a single next node but the tour is built incrementally,
    we use a different approach: simulate the remaining tour construction using
    2-Opt, and return the first node in the optimal remaining path.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        if n == 0:
            return int(destination_node)
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    if n == 2:
        u1, u2 = int(unv[0]), int(unv[1])
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        return u1 if cost1 <= cost2 else u2
    
    # For small n, use exact DP
    if n <= 12 and elapsed < _DEADLINE - 3.0:
        return _held_karp_next(current_node, destination_node, unv, distance_matrix)
    
    # For larger n, use 2-Opt based selection
    # Strategy: Build the remaining tour using nearest-neighbor + 2-Opt,
    # then return the first node in that optimized tour.
    
    # Create a local distance matrix for the remaining nodes + current + destination
    # We'll work with the full distance matrix but only consider the relevant nodes
    
    # Build initial tour for the remaining nodes starting from current_node
    # The tour should be: current_node -> ... -> destination_node (closed)
    
    # Nearest neighbor construction for remaining nodes
    remaining = list(unv)
    visited_local = set()
    current = int(current_node)
    tour = [current]
    
    for _ in range(n):
        if time.time() - _START_TIME > _DEADLINE - 2.0:
            # Fallback to nearest neighbor
            d = distance_matrix[current]
            d[visited_local] = np.inf
            d[destination_node] = np.inf if n > 0 else 0
            if np.all(np.isinf(d)):
                next_node = int(destination_node)
            else:
                next_node = int(np.argmin(d))
            break
        
        # Find nearest unvisited node
        d = distance_matrix[current].copy()
        for v in visited_local:
            d[v] = np.inf
        d[destination_node] = np.inf  # Don't go to destination until the end
        
        if np.all(np.isinf(d)):
            next_node = int(destination_node)
        else:
            next_node = int(np.argmin(d))
        
        visited_local.add(next_node)
        tour.append(next_node)
        current = next_node
    
    # Ensure destination is at the end
    if tour[-1] != int(destination_node):
        tour.append(int(destination_node))
    
    # Remove duplicates and ensure valid tour
    # The tour should be: current_node, node1, node2, ..., destination_node
    
    # Now apply 2-Opt to improve the tour (excluding the start and end)
    # We can only permute the middle nodes
    
    if len(tour) > 4:
        # Extract middle nodes
        middle_start = 1
        middle_end = len(tour) - 1  # exclusive
        
        # Apply 2-Opt on the middle section
        middle_tour = tour[middle_start:middle_end]
        
        if len(middle_tour) > 2:
            # Create a local distance matrix for the middle nodes
            # We need to consider connections to current_node and destination_node
            
            # For 2-Opt, we need to check if reversing a segment improves the tour
            # The cost change when reversing segment [i, j] in the middle:
            # Original: ... a -> b ... c -> d ...
            # New: ... a -> c ... b -> d ...
            # where a, b, c, d are in the middle, or a is current_node, or d is destination_node
            
            improved = True
            max_iter = 5000
            iter_count = 0
            
            while improved and iter_count < max_iter:
                improved = False
                iter_count += 1
                
                if time.time() - _START_TIME > _DEADLINE - 2.0:
                    break
                
                m = len(middle_tour)
                for i in range(m - 1):
                    if time.time() - _START_TIME > _DEADLINE - 2.0:
                        break
                    
                    for j in range(i + 1, m):
                        # Get the four nodes involved
                        if i == 0:
                            a = int(current_node)
                        else:
                            a = middle_tour[i - 1]
                        
                        b = middle_tour[i]
                        
                        if j == m - 1:
                            c = int(destination_node)
                        else:
                            c = middle_tour[j + 1]
                        
                        d = middle_tour[j]
                        
                        # Calculate improvement
                        current_cost = distance_matrix[a, b] + distance_matrix[c, d]
                        new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                        
                        if new_cost < current_cost - 1e-10:
                            # Reverse the segment from i to j
                            middle_tour[i:j + 1] = reversed(middle_tour[i:j + 1])
                            improved = True
            
            # Reconstruct the full tour
            tour = [int(current_node)] + middle_tour + [int(destination_node)]
    
    # The first node after current_node is the answer
    if len(tour) > 1:
        return int(tour[1])
    else:
        return int(destination_node)


def _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Finds the optimal next node for the remaining set using Held-Karp DP.
    Returns the first node in the optimal sequence.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Map unvisited nodes to local indices 0..n-1
    full_mask = (1 << n) - 1
    
    INF = float('inf')
    # dp[mask][i] = min cost to start at current_node, visit nodes in mask, end at unv[i]
    dp = np.full((1 << n, n), INF, dtype=float)
    parent = np.full((1 << n, n), -1, dtype=int)
    
    # Base cases: start at current_node, go to unv[i]
    for i in range(n):
        dp[1 << i, i] = distance_matrix[current_node, unv[i]]
        
    # Fill DP table
    for mask in range(1 << n):
        if mask == 0:
            continue
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            current_cost = dp[mask, i]
            if current_cost == INF:
                continue
            
            # Try extending to j not in mask
            for j in range(n):
                if mask & (1 << j):
                    continue
                
                new_mask = mask | (1 << j)
                step_cost = distance_matrix[unv[i], unv[j]]
                new_cost = current_cost + step_cost
                
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost
                    parent[new_mask, j] = i

    # Find the best ending node i in full_mask
    best_cost = INF
    best_last_i = -1
    for i in range(n):
        total_cost = dp[full_mask, i] + distance_matrix[unv[i], destination_node]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last_i = i
            
    if best_last_i == -1:
        return int(unv[0])
        
    # Backtrack to find the first node
    current_mask = full_mask
    current_i = best_last_i
    first_i = -1
    
    path = []
    while current_mask != 0:
        path.append(current_i)
        if current_mask & (current_mask - 1) == 0:
            first_i = current_i
            break
        
        prev_i = parent[current_mask, current_i]
        if prev_i == -1:
            break
        current_mask = current_mask ^ (1 << current_i)
        current_i = prev_i
        
    if first_i == -1:
        first_i = 0
        
    return int(unv[first_i])
# EVOLVE-BLOCK-END
