import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for output

    def find_k4(adj, n):
        # Returns a tuple (u, v, w, x) forming a K4, or None
        # Iterate over edges
        for u in range(n):
            # Get neighbors of u
            neighbors_u = adj[u]
            # Only consider v > u to avoid duplicates and self
            # We can just iterate v
            v_mask = neighbors_u & (0xFFFFFFFF ^ ((1 << (u+1)) - 1)) # v > u
            # Actually, simpler:
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    # common contains w, x such that u-w, u-x, v-w, v-x exist
                    # We need an edge w-x
                    # Iterate w in common
                    w_mask = common
                    while w_mask:
                        # Extract lowest set bit
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        w_mask ^= lsb
                        
                        # Check if w has a neighbor x in common, x > w
                        # x must be in common and adj[w]
                        x_mask = common & adj[w] & (0xFFFFFFFF ^ ((1 << (w+1)) - 1))
                        if x_mask:
                            # Find x
                            lsb_x = x_mask & -x_mask
                            x = lsb_x.bit_length() - 1
                            return (u, v, w, x)
        return None

    def find_ind_6(adj, n):
        # Returns a list of 6 vertices that are independent, or None
        # Backtracking
        # candidates: bitmask of available vertices
        # current: list of vertices in the independent set so far
        
        # Heuristic: Start with all vertices
        # To speed up, we can sort vertices by degree (ascending) to pick "loose" vertices first?
        # Actually, for independent set, we want vertices with few neighbors.
        # Let's just do standard backtracking with bitmasks.
        
        def backtrack(current, candidates):
            if len(current) == 6:
                return current[:]
            
            # Pruning: if not enough candidates left
            if candidates.bit_count() < 6 - len(current):
                return None
                
            # Pick a vertex from candidates
            # Heuristic: pick the one with the smallest degree in the induced subgraph of candidates?
            # That's expensive.
            # Let's just pick the lowest index for determinism, or random for diversity.
            # Random might escape local minima better in the outer loop, 
            # but inside the search, we just need to find *any* one.
            
            # To make it fast, we can try vertices in a specific order.
            # Let's try vertices with the fewest connections to other candidates.
            # This is O(n^2) per step.
            
            # Simpler: Just iterate through candidates.
            # To avoid getting stuck, we can shuffle the order of candidates initially?
            # No, the candidates set changes.
            
            # Let's just pick the first available one.
            # This is fast but might miss solutions if the first branch is deep and dead.
            # But since we are in a local search, we just need *one* violation.
            
            cand_list = []
            mask = candidates
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                cand_list.append(v)
                mask ^= lsb
                
            # Try each candidate
            for v in cand_list:
                # v is independent from current?
                # Since we maintain candidates as non-neighbors of all in current, 
                # any v in candidates is independent from current.
                
                new_current = current + [v]
                # New candidates: 
                # Must be in candidates
                # Must not be v
                # Must not be neighbor of v
                new_candidates = candidates & ~(1 << v) & ~adj[v]
                
                res = backtrack(new_current, new_candidates)
                if res:
                    return res
            return None

        return backtrack([], (1 << n) - 1)

    def is_valid(adj, n):
        if find_k4(adj, n):
            return False
        if find_ind_6(adj, n):
            return False
        return True

    # Fallback: Paley(17)
    squares = {x * x % 17 for x in range(1, 17)}
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17

    for target_n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 10.0:
            break
            
        found = False
        attempts = 0
        max_attempts = 50 # Increase attempts for lower n
        
        while not found and attempts < max_attempts and time.time() < deadline - 10.0:
            attempts += 1
            
            # Initialize random graph
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            steps = 0
            max_steps = 2000
            
            while steps < max_steps and time.time() < deadline - 5.0:
                steps += 1
                
                # Check K4
                k4 = find_k4(adj, target_n)
                if k4:
                    u, v, w, x = k4
                    # Remove a random edge from the K4
                    edges_in_k4 = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                    ru, rv = random.choice(edges_in_k4)
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    continue
                
                # Check Ind 6
                ind6 = find_ind_6(adj, target_n)
                if ind6:
                    # Add a random edge within the independent set
                    u, v = random.sample(ind6, 2)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    continue
                
                # Valid!
                found = True
                best_n = target_n
                # Convert adj to edge list
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                best_edges = edges
                break
                
            if found:
                break

    return {'n': best_n, 'edges': best_edges}
