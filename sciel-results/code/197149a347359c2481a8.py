import itertools
import time
import random

def construct():
    start_time = time.time()
    timeout = 290  # Safety margin for 300s limit
    
    # Helper to generate Paley graph
    def get_paley(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Fallback: Paley 37
    n_seed, edges_seed = get_paley(37)
    
    # Function to compute all K4s and I4s given n and edges
    def get_k4_i4(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        k4s = []
        i4s = []
        for combo in itertools.combinations(range(n), 4):
            is_k4 = True
            is_i4 = True
            for i in range(4):
                u = combo[i]
                for j in range(i + 1, 4):
                    v = combo[j]
                    if adj[u] & (1 << v):
                        is_i4 = False
                    else:
                        is_k4 = False
                    if not is_k4 and not is_i4:
                        break
                if not is_k4 and not is_i4:
                    break
            if is_k4:
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                k4s.append(mask)
            if is_i4:
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                i4s.append(mask)
        return k4s, i4s

    # Try to extend graph by adding one vertex
    # Returns True and updates edges if successful
    def try_extend(n, edges, k4s, i4s):
        # k4s and i4s are lists of bitmasks
        # We want to find S (bitmask of size n) such that:
        # 1. No k4 in k4s is a subset of S
        # 2. No i4 in i4s is a subset of ~S (i.e., S & i4 == 0)
        
        # Heuristic: Start with S = random subset
        # Or start with S = 0 and add, or S = all and remove.
        # Let's try a few random restarts of local search on S
        
        all_mask = (1 << n) - 1
        
        for restart in range(50):
            if time.time() - start_time > timeout:
                return False, edges, k4s, i4s
            
            random.seed(12345 + restart)
            # Initialize S
            # Bias towards 0.5 density
            S = 0
            for i in range(n):
                if random.random() < 0.5:
                    S |= (1 << i)
            
            # Local Search
            # Objective: Minimize violations
            # Violations = count(k4 in k4s where k4 subset S) + count(i4 in i4s where i4 subset ~S)
            
            # Precompute counts? No, just check all each time? 
            # n is small (37-41), k4s and i4s size is manageable (C(n,4) ~ 65k for n=37).
            # Checking 65k masks is fast.
            
            best_violations = float('inf')
            current_S = S
            
            for step in range(2000):
                if time.time() - start_time > timeout:
                    return False, edges, k4s, i4s
                
                # Calculate violations
                viol_k = 0
                for mask in k4s:
                    if (current_S & mask) == mask:
                        viol_k += 1
                        if viol_k > 0: # Early break? No, need count for optimization
                            pass
                
                viol_i = 0
                for mask in i4s:
                    if (current_S & mask) == 0:
                        viol_i += 1
                
                total_viol = viol_k + viol_i
                
                if total_viol == 0:
                    # Success
                    # Add new vertex n connected to all i in S
                    new_edges = list(edges)
                    for i in range(n):
                        if current_S & (1 << i):
                            new_edges.append((i, n))
                    return True, new_edges, k4s, i4s # Note: k4s/i4s for new graph will be recalculated by caller if needed, but we return success
                
                if total_viol < best_violations:
                    best_violations = total_viol
                    # We could accept this as new current, but let's just do hill climbing on flips
                    
                # Pick a random vertex to flip
                v = random.randint(0, n - 1)
                
                # Calculate delta violations if we flip v
                # This is complex. Let's just flip and re-evaluate in next loop? 
                # Or do a simple 1-flip hill climb.
                
                new_S = current_S ^ (1 << v)
                
                # Estimate change
                # If v was in S, it was contributing to K4 violations. Now it might break them.
                # If v was NOT in S, it was contributing to I4 violations (by being absent). Now it might break them.
                
                # Just re-evaluate is safer and n is small enough.
                # But 2000 steps * 65k checks = 130M ops. Might be too slow.
                # Let's optimize the check.
                
                # Actually, let's just do a greedy repair.
                # If viol_k > 0, pick a random K4 subset of S, remove a random vertex from S.
                # If viol_i > 0, pick a random I4 subset of ~S, add a random vertex to S.
                
                if viol_k > 0:
                    # Find a violating K4
                    # Randomly sample K4s until we find one in S
                    for _ in range(10):
                        idx = random.randint(0, len(k4s) - 1)
                        mask = k4s[idx]
                        if (current_S & mask) == mask:
                            # Pick a random vertex in this mask to remove from S
                            bits = [i for i in range(n) if mask & (1 << i)]
                            v_to_remove = random.choice(bits)
                            current_S &= ~(1 << v_to_remove)
                            break
                elif viol_i > 0:
                    # Find a violating I4
                    for _ in range(10):
                        idx = random.randint(0, len(i4s) - 1)
                        mask = i4s[idx]
                        if (current_S & mask) == 0:
                            # Pick a random vertex in this mask to add to S
                            bits = [i for i in range(n) if mask & (1 << i)]
                            v_to_add = random.choice(bits)
                            current_S |= (1 << v_to_add)
                            break
                else:
                    # No violations? Should have returned above.
                    break

        return False, edges, k4s, i4s

    # Main Logic
    current_n = n_seed
    current_edges = edges_seed
    current_k4s, current_i4s = get_k4_i4(current_n, current_edges)
    
    target_n = 42
    
    while current_n < target_n:
        if time.time() - start_time > timeout - 30: # Leave time for final check
            break
            
        success, new_edges, _, _ = try_extend(current_n, current_edges, current_k4s, current_i4s)
        
        if success:
            current_edges = new_edges
            current_n += 1
            # Recalculate K4s and I4s for the new graph
            current_k4s, current_i4s = get_k4_i4(current_n, current_edges)
        else:
            # If we fail to extend, we can't just break, we might try different strategies?
            # For now, if we can't extend 37->38, we likely can't reach 42 via this method.
            # But maybe we can extend 37->38 with a different random seed?
            # try_extend already does restarts.
            # If it fails, we stop extending.
            break

    # If we reached target_n, verify and return
    if current_n == target_n:
        # Final verification
        adj = [0] * current_n
        for u, v in current_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        valid = True
        for combo in itertools.combinations(range(current_n), 5):
            is_k5 = True
            is_i5 = True
            for i in range(5):
                u = combo[i]
                for j in range(i + 1, 5):
                    v = combo[j]
                    if adj[u] & (1 << v):
                        is_i5 = False
                    else:
                        is_k5 = False
                    if not is_k5 and not is_i5:
                        break
                if not is_k5 and not is_i5:
                    break
            if is_k5 or is_i5:
                valid = False
                break
        
        if valid:
            return current_n, current_edges
        else:
            # Should not happen if extension logic is correct, but fallback
            return n_seed, edges_seed

    # If we didn't reach 42, try the local search from the seed program on 42
    # But initialize with the best graph we found (current_n vertices) + random vertices?
    # Or just use the seed 37 and try to jump to 42?
    # The local search in the seed program is slow.
    # Let's try to extend the current graph (maybe 38 or 39) to 42 using a different method?
    # Or just return the best we have.
    
    # If current_n > n_seed, return it.
    if current_n > n_seed:
        return current_n, current_edges
        
    return n_seed, edges_seed

# Test runner
if __name__ == "__main__":
    n, edges = construct()
    print(f"Vertices: {n}, Edges: {len(edges)}")
