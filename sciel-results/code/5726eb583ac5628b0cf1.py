import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety

    def bitmask_to_edges(n, adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_k4(n, adj):
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        count += (adj[w] & c).bit_count()
        return count // 6

    def has_ind6(n, adj):
        mask = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask & ~(adj[i] | (1 << i))
        
        # Quick check: if any vertex in comp has degree < 5, it can't be in a K6?
        # No, a vertex in a K6 must have degree >= 5 within the K6, but global degree can be lower?
        # No, if it's in a K6, it is connected to 5 other vertices in the K6.
        # So its degree in the induced subgraph must be 5.
        # So its global degree in comp_adj must be at least 5.
        for i in range(n):
            if comp_adj[i].bit_count() < 5:
                return False

        def find_clique(candidates, size):
            if size == 0:
                return True
            if len(candidates) < size:
                return False
            
            # Heuristic: try vertices with higher degree in the candidate set first?
            # Or just iterate.
            # To speed up, we can pick a pivot.
            # Simple recursion:
            # Pick first vertex v
            v = candidates[0]
            # New candidates are neighbors of v in comp_adj that are in candidates[1:]
            # We can filter candidates[1:] by checking bit in comp_adj[v]
            new_cands = [u for u in candidates[1:] if (comp_adj[v] >> u) & 1]
            
            if find_clique(new_cands, size - 1):
                return True
            
            # Exclude v
            return find_clique(candidates[1:], size)

        return find_clique(list(range(n)), 6)

    def is_valid(n, adj):
        if count_k4(n, adj) > 0:
            return False
        if has_ind6(n, adj):
            return False
        return True

    # Local Search / Simulated Annealing
    def local_search(n, adj, time_limit):
        current_adj = adj[:]
        current_k4 = count_k4(n, current_adj)
        current_ind6 = has_ind6(n, current_adj)
        
        # Penalty: Weighted sum
        # If K4 exists, it's a major violation.
        # If Ind6 exists, it's a major violation.
        def get_penalty(adj):
            k4 = count_k4(n, adj)
            if k4 > 0:
                return k4 * 10 # Scale up K4
            ind6 = has_ind6(n, adj)
            if ind6:
                return 100 # Scale up Ind6
            return 0

        current_penalty = get_penalty(current_adj)
        
        if current_penalty == 0:
            return current_adj

        temp = 10.0
        cooling = 0.999
        
        while time.time() < time_limit:
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Flip
            if (current_adj[u] >> v) & 1:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            
            new_penalty = get_penalty(current_adj)
            
            # Acceptance criterion
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if current_penalty == 0:
                    return current_adj
            else:
                # Metropolis criterion
                delta = new_penalty - current_penalty
                if random.random() < pow(2.71828, -delta / temp):
                    current_penalty = new_penalty
                else:
                    # Revert
                    if (current_adj[u] >> v) & 1:
                        current_adj[u] &= ~(1 << v)
                        current_adj[v] &= ~(1 << u)
                    else:
                        current_adj[u] |= (1 << v)
                        current_adj[v] |= (1 << u)
            
            temp *= cooling

        return current_adj

    best_n = 17
    best_adj = None
    
    # Seed: Paley graph on 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj17 = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj17[i] |= (1 << j)
                adj17[j] |= (1 << i)
    best_adj = adj17

    # Try to find larger graphs
    # We allocate time based on remaining budget
    # Start with larger n, as they are harder but yield higher score
    targets = [35, 34, 33, 32, 31, 30, 28, 25, 22, 20]
    
    for target_n in targets:
        if time.time() > deadline - 15:
            break
        
        # Initialize random graph
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Run local search for a slice of time
        # Larger n gets less time per attempt? Or more?
        # Let's give 10s per target, but break if total time is low.
        slice_end = time.time() + 10
        if slice_end > deadline - 5:
            slice_end = deadline - 5
            
        result_adj = local_search(target_n, adj, slice_end)
        
        if is_valid(target_n, result_adj):
            best_n = target_n
            best_adj = result_adj
            # If we found a valid graph, we can stop searching for smaller ones?
            # No, we want the MAX n. So we stop here.
            break

    # Fallback to seed if no improvement
    if best_n == 17:
        return {'n': 17, 'edges': bitmask_to_edges(17, best_adj)}
    
    return {'n': best_n, 'edges': bitmask_to_edges(best_n, best_adj)}
