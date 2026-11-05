import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    N = 256
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Filter out vertices where 2*v is a square
    valid_vertices = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_vertices.append(v)
    
    M = len(valid_vertices)
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}

    # Build adjacency bitsets
    adj = [0] * M
    for i in range(M):
        vi = valid_vertices[i]
        for j in range(i + 1, M):
            vj = valid_vertices[j]
            if vi + vj in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
    def can_add(mask, v):
        return (mask & adj[v]) == 0

    def mask_to_set(mask):
        return [valid_vertices[i] for i in range(M) if (mask >> i) & 1]

    def greedy_complete(mask):
        """Greedily add any non-conflicting vertex."""
        changed = True
        while changed:
            changed = False
            for v in range(M):
                if not (mask >> v) & 1:
                    if can_add(mask, v):
                        mask |= (1 << v)
                        changed = True
        return mask

    def local_search(initial_mask, budget):
        """
        Perform local search with 1-2 and 2-3 exchanges.
        """
        current_mask = greedy_complete(initial_mask)
        
        while time.time() - start_time < budget:
            improved = False
            
            sel_list = [v for v in range(M) if (current_mask >> v) & 1]
            unselected = [u for u in range(M) if not (current_mask >> u) & 1]
            
            if not sel_list or not unselected:
                break

            # --- Try 1-2 Swaps ---
            # To speed up, we don't try all v in sel_list if it's too slow.
            # But for M~248, it's manageable if we break early.
            
            # Randomize the order of sel_list to avoid bias
            random.shuffle(sel_list)
            
            for v in sel_list:
                if time.time() - start_time > budget:
                    break
                mask_without_v = current_mask & ~(1 << v)
                
                # Find U_v: vertices in unselected that don't conflict with mask_without_v
                U_v = []
                for u in unselected:
                    if (adj[u] & mask_without_v) == 0:
                        U_v.append(u)
                        
                if len(U_v) < 2:
                    continue
                    
                # Find pair in U_v
                found_pair = False
                # Shuffle U_v to find diverse swaps
                random.shuffle(U_v)
                for i in range(len(U_v)):
                    u = U_v[i]
                    # Check w in U_v[i+1:]
                    for j in range(i + 1, len(U_v)):
                        w = U_v[j]
                        if (adj[u] & (1 << w)) == 0:
                            # Valid swap found
                            current_mask = mask_without_v | (1 << u) | (1 << w)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if improved:
                    break # Restart outer while loop

            if improved:
                continue

            # --- Try 2-3 Swaps ---
            # This is more expensive. We limit the search.
            # Pick 2 random vertices from sel_list to remove
            if len(sel_list) >= 2:
                v1, v2 = random.sample(sel_list, 2)
                mask_without_v1v2 = current_mask & ~(1 << v1) & ~(1 << v2)
                
                # Find U_v1v2: vertices in unselected that don't conflict with mask_without_v1v2
                U_v1v2 = []
                for u in unselected:
                    if (adj[u] & mask_without_v1v2) == 0:
                        U_v1v2.append(u)
                
                if len(U_v1v2) >= 3:
                    # Find 3 mutually non-conflicting vertices in U_v1v2
                    random.shuffle(U_v1v2)
                    found_triplet = False
                    # Greedy selection of 3
                    # Try to pick u1, then u2, then u3
                    # This is not exhaustive, but a heuristic.
                    # We can try a few random subsets or just the first valid ones found in a randomized order.
                    
                    # Simple heuristic: Try to add u1, then u2, then u3 greedily from the shuffled list
                    # But we need them to be mutually non-conflicting.
                    
                    # Let's try a small random search for the triplet
                    # Since U_v1v2 might be large, we only check the first K candidates
                    K = min(50, len(U_v1v2))
                    candidates = U_v1v2[:K]
                    
                    for i in range(len(candidates)):
                        u1 = candidates[i]
                        for j in range(i+1, len(candidates)):
                            u2 = candidates[j]
                            if (adj[u1] & (1 << u2)) != 0:
                                continue
                            for k in range(j+1, len(candidates)):
                                u3 = candidates[k]
                                if (adj[u1] & (1 << u3)) != 0 or (adj[u2] & (1 << u3)) != 0:
                                    continue
                                # Valid triplet found
                                current_mask = mask_without_v1v2 | (1 << u1) | (1 << u2) | (1 << u3)
                                improved = True
                                found_triplet = True
                                break
                            if found_triplet:
                                break
                        if found_triplet:
                            break
                    
                    if improved:
                        break # Restart outer while loop

            if not improved:
                # Stuck in local optimum for 1-2 and 2-3 (heuristic)
                # Try a random perturbation? Or break?
                # If we break, we lose the budget.
                # Let's try a random 1-1 swap (size same) to escape?
                # Or just break and rely on restarts.
                break

        return current_mask

    best_mask = 0
    best_size = 0

    def consider(mask):
        nonlocal best_mask, best_size
        sz = bin(mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = mask

    # Seed 1: Min-Degree Greedy
    deg = [bin(adj[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda i: deg[i])
    mask = 0
    for v in order:
        if can_add(mask, v):
            mask |= (1 << v)
    mask = greedy_complete(mask)
    consider(mask)
    
    # Allocate time for local search
    # We will run multiple restarts.
    
    num_restarts = 20
    time_per_restart = (DEADLINE - 5.0) / num_restarts
    
    # First, run local search on the initial greedy result
    mask = local_search(mask, time_per_restart * 2) # Give it double time initially
    consider(mask)

    # Random Restarts
    for r in range(1, num_restarts):
        if elapsed() > DEADLINE - 2:
            break
        
        random.seed(r * 98765 + 43210)
        order = list(range(M))
        random.shuffle(order)
        
        # Randomized Min-Degree Greedy
        # Sort by degree + random noise
        noisy_deg = [deg[i] + random.uniform(-0.1, 0.1) for i in range(M)]
        order = sorted(range(M), key=lambda i: noisy_deg[i])
        
        mask = 0
        for v in order:
            if can_add(mask, v):
                mask |= (1 << v)
        mask = greedy_complete(mask)
        consider(mask)
        
        # Local search
        mask = local_search(mask, time_per_restart)
        consider(mask)

    # Final Polish
    if elapsed() < DEADLINE - 1:
        mask = local_search(best_mask, DEADLINE - 1)
        consider(mask)

    result = mask_to_set(best_mask)
    result.sort()
    
    if not result:
        return [1]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found set size: {len(res)}")
