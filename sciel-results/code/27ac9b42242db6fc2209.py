"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin for 120s limit
    
    n = 40
    
    def creates_c6(adj_bits, u, v):
        """Check if adding edge (u,v) creates a C6 using bit operations."""
        nu_mask = adj_bits[u] & ~(1 << v)
        nv_mask = adj_bits[v] & ~(1 << u)
        
        temp_u = nu_mask
        while temp_u:
            lsb_u = temp_u & -temp_u
            x1 = lsb_u.bit_length() - 1
            temp_u ^= lsb_u
            
            mask_x1 = adj_bits[x1] & ~(1 << u) & ~(1 << v)
            
            temp_v = nv_mask
            while temp_v:
                lsb_v = temp_v & -temp_v
                x4 = lsb_v.bit_length() - 1
                temp_v ^= lsb_v
                
                if x1 == x4:
                    continue
                
                mask_x4 = adj_bits[x4] & ~(1 << u) & ~(1 << v)
                
                mask_x1_eff = mask_x1 & ~(1 << x4)
                mask_x4_eff = mask_x4 & ~(1 << x1)
                
                if mask_x1_eff.bit_count() < mask_x4_eff.bit_count():
                    temp_a = mask_x1_eff
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        x2 = lsb_a.bit_length() - 1
                        if (adj_bits[x2] & mask_x4_eff) != 0:
                            return True
                        temp_a ^= lsb_a
                else:
                    temp_a = mask_x4_eff
                    while temp_a:
                        lsb_a = temp_a & -temp_a
                        x3 = lsb_a.bit_length() - 1
                        if (adj_bits[x3] & mask_x1_eff) != 0:
                            return True
                        temp_a ^= lsb_a
        return False

    def bqp_init(adj_bits, current_edges, current_count):
        """Use BQP-inspired smooth optimization to find a better initial solution.
        
        The C6-free constraint can be expressed as: for any non-adjacent pair (u,v),
        the number of length-5 paths between u and v must be zero. This gives a
        quadratic penalty structure. We use gradient descent on a relaxed version
        to find a good initial point, then refine with discrete methods.
        """
        if time.time() > deadline - 30.0:
            return current_edges, current_count
        
        # Build adjacency matrix from current solution
        A = np.zeros((n, n), dtype=np.float64)
        for u, v in current_edges:
            A[u, v] = 1.0
            A[v, u] = 1.0
        
        # Compute length-5 path counts using matrix multiplication
        # P5[u,v] = (A^5)[u,v] gives number of length-5 walks from u to v
        A2 = A @ A
        A3 = A2 @ A
        A4 = A3 @ A
        A5 = A4 @ A
        
        # For C6-free graphs, non-adjacent pairs should have zero length-5 paths
        # (actually, we need to be more careful: we need to exclude paths that
        # repeat vertices, but as a surrogate, this is a reasonable approximation)
        
        # Build penalty: for each non-adjacent pair, penalize length-5 paths
        # This creates a quadratic form in the edge variables
        I = np.eye(n)
        non_adj = 1.0 - A - I  # Non-adjacent pairs (excluding self-loops)
        
        # Penalty = sum over non-adjacent (u,v) of (A^5)[u,v]^2
        # This is smooth and differentiable
        penalty = np.sum(non_adj * A5 * A5)
        
        # Use gradient-based optimization to improve the solution
        # We'll do a few steps of projected gradient descent on the edge weights
        lr = 0.1
        max_steps = 50
        
        X = A.copy()  # Edge weight matrix (symmetric, zero diagonal)
        
        for step in range(max_steps):
            if time.time() > deadline - 25.0:
                break
            
            # Recompute A5 with current X
            X2 = X @ X
            X3 = X2 @ X
            X4 = X3 @ X
            X5 = X4 @ X
            
            # Objective: maximize edges - lambda * penalty
            # gradient of penalty w.r.t. X
            # d/dX [sum(non_adj * X5 * X5)] = 2 * X5 * (d X5/dX)^T * non_adj
            # This is complex, so we use a simpler heuristic:
            # For each edge (u,v), compute how much it contributes to C6 violations
            
            # Simplified gradient: for each edge, estimate its contribution to penalty
            grad = np.zeros_like(X)
            
            # For each edge, compute local contribution to length-5 paths
            for u in range(n):
                for v in range(u + 1, n):
                    if X[u, v] > 0.01:  # Only consider existing edges
                        # Estimate contribution: paths through this edge
                        # This is approximate but fast
                        contrib = 0.0
                        # Length-5 paths using edge (u,v): 
                        # u-v-x-y-z-w where x,y,z,w are other vertices
                        # Approximate by counting paths from u and v
                        for x in range(n):
                            if x != u and x != v and X[v, x] > 0.01:
                                for y in range(n):
                                    if y != u and y != v and y != x and X[x, y] > 0.01:
                                        for z in range(n):
                                            if z != u and z != v and z != x and z != y and X[y, z] > 0.01:
                                                for w in range(n):
                                                    if w != u and w != v and w != x and w != y and w != z and X[z, w] > 0.01:
                                                        # Check if (w, u) would complete a C6
                                                        # This is getting too expensive, so we bail
                                                        pass
                        # Too expensive, use a simpler heuristic
                        break
                    else:
                        break
                break  # Bail out of expensive computation
            
            # Use a simpler approach: random perturbation with acceptance based on penalty
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Try flipping this edge
            old_val = X[u, v]
            X[u, v] = 1.0 - old_val
            X[v, u] = 1.0 - old_val
            
            # Recompute penalty
            X2 = X @ X
            X3 = X2 @ X
            X4 = X3 @ X
            X5 = X4 @ X
            new_penalty = np.sum(non_adj * X5 * X5)
            
            # Accept if penalty decreased or randomly
            if new_penalty < penalty or random.random() < 0.1:
                penalty = new_penalty
            else:
                # Revert
                X[u, v] = old_val
                X[v, u] = old_val
        
        # Convert back to discrete solution
        new_edges = []
        new_adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if X[u, v] > 0.5:
                    new_edges.append([u, v])
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
        
        # Verify C6-freeness and keep only valid edges
        valid_edges = []
        valid_adj = [0] * n
        for u, v in new_edges:
            if not creates_c6(valid_adj, u, v):
                valid_adj[u] |= (1 << v)
                valid_adj[v] |= (1 << u)
                valid_edges.append([u, v])
        
        return valid_edges, len(valid_edges)

    def greedy_add(adj_bits, current_edges, current_count):
        """Try to add all possible non-C6-creating edges in random order."""
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_bits[u] >> v) & 1:
                    candidates.append((u, v))
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() > deadline - 5.0:
                break
            if not creates_c6(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                current_edges.append([u, v])
                current_count += 1
        return current_edges, current_count

    def local_search(adj_bits, current_edges, current_count):
        """Local search: remove an edge, try to add multiple new edges."""
        if not current_edges:
            return current_edges, current_count
            
        improved = True
        max_rounds = 50
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            
            random.shuffle(current_edges)
            for i in range(len(current_edges)):
                if time.time() > deadline - 5.0:
                    break
                u, v = current_edges[i]
                
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                
                added = []
                for a in range(n):
                    for b in range(a + 1, n):
                        if not (adj_bits[a] >> b) & 1:
                            if not creates_c6(adj_bits, a, b):
                                adj_bits[a] |= (1 << b)
                                adj_bits[b] |= (1 << a)
                                added.append((a, b))
                
                new_count = current_count - 1 + len(added)
                
                if len(added) >= 1:
                    current_edges.pop(i)
                    for a, b in added:
                        current_edges.append([a, b])
                    current_count = new_count
                    improved = True
                    break
                else:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        
        return current_edges, current_count

    def simulated_annealing(adj_bits, current_edges, current_count):
        """Simulated Annealing with structured moves."""
        if not current_edges:
            return current_edges, current_count
            
        T = 2.0
        cooling = 0.998
        min_T = 0.1
        max_steps = 3000
        
        for step in range(max_steps):
            if time.time() > deadline - 5.0:
                break
            if T < min_T:
                break
                
            i = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[i]
            
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            current_edges.pop(i)
            new_count = current_count - 1
            
            added_edges = []
            attempts = 0
            max_add_attempts = 30
            
            while attempts < max_add_attempts and len(added_edges) < 3:
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    attempts += 1
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                    
                if (adj_bits[ru] >> rv) & 1:
                    attempts += 1
                    continue
                    
                if not creates_c6(adj_bits, ru, rv):
                    adj_bits[ru] |= (1 << rv)
                    adj_bits[rv] |= (1 << ru)
                    current_edges.append([ru, rv])
                    added_edges.append((ru, rv))
                    new_count += 1
                attempts += 1
            
            delta = new_count - current_count
            if delta >= 0 or random.random() < pow(2.718, delta / T):
                current_count = new_count
            else:
                for a, b in added_edges:
                    adj_bits[a] &= ~(1 << b)
                    adj_bits[b] &= ~(1 << a)
                    for j in range(len(current_edges) - 1, -1, -1):
                        if current_edges[j] == [a, b]:
                            current_edges.pop(j)
                            break
                current_edges.append([u, v])
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
            
            T *= cooling
            
        return current_edges, current_count

    # Main loop
    best_edges = []
    best_count = 0
    
    for restart in range(30):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with 8 K5s
        adj_bits = [0] * n
        current_edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    current_edges.append([u, v])
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        
        current_count = len(current_edges)
        
        # Greedy phase
        current_edges, current_count = greedy_add(adj_bits, current_edges, current_count)
        
        # BQP-inspired initialization
        if time.time() < deadline - 20.0:
            bqp_edges, bqp_count = bqp_init(adj_bits, current_edges, current_count)
            if bqp_count > current_count:
                current_edges = bqp_edges
                current_count = bqp_count
                # Rebuild adj_bits
                adj_bits = [0] * n
                for u, v in current_edges:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        
        # Local search phase
        if time.time() < deadline - 8.0:
            current_edges, current_count = local_search(adj_bits, current_edges, current_count)
        
        # SA phase
        if time.time() < deadline - 5.0:
            current_edges, current_count = simulated_annealing(adj_bits, current_edges, current_count)
        
        # Final greedy pass
        if time.time() < deadline - 3.0:
            current_edges, current_count = greedy_add(adj_bits, current_edges, current_count)
        
        if current_count > best_count:
            best_count = current_count
            best_edges = list(current_edges)
    
    return best_edges
# EVOLVE-BLOCK-END
