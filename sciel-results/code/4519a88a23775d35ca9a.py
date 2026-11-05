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
    # Map index in valid_vertices to the actual value
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}

    # Build adjacency bitsets
    # adj[i] is a bitmask of vertices conflicting with valid_vertices[i]
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
        Perform local search with 1-1, 1-2, and 2-3 exchanges.
        Returns the best mask found.
        """
        current_mask = greedy_complete(initial_mask)
        current_size = bin(current_mask).count('1')
        
        while time.time() - start_time < budget:
            improved = False
            
            # 1-1 Exchange: Remove one, add one
            # Try removing each selected vertex and see if we can add a better one
            # Actually, simpler: try to add a vertex by removing one that conflicts with it.
            # Or: For each unselected vertex u, check if removing one selected vertex v allows adding u.
            # This is O(N^2) per iteration if not careful.
            
            # Let's use a standard heuristic:
            # 1. Try to add any available vertex (should be none if greedy_complete was run).
            # 2. Try 1-for-2 swap: Remove v, add u, w.
            
            sel_list = [v for v in range(M) if (current_mask >> v) & 1]
            
            # Optimization: Limit the search space for swaps if time is tight
            # But for N=248, O(N^3) is too slow. We need efficient checks.
            
            # Strategy:
            # Iterate through selected vertices v.
            # For each v, consider removing it.
            # Then look for pairs (u, w) that can be added.
            
            for v in sel_list:
                if time.time() - start_time > budget:
                    break
                without_v = current_mask & ~(1 << v)
                
                # Find candidates u, w that can be added to without_v
                # and don't conflict with each other.
                # We can precompute potential additions for 'without_v'? No, it changes.
                
                # Efficient check:
                # Candidates for u: all unselected vertices that don't conflict with without_v.
                # Let avail = ~current_mask & ~adj[?] ... no, adj depends on the set.
                
                # Simpler approach for 1-2 swap:
                # For each v in sel_list:
                #   temp_mask = current_mask ^ (1 << v)
                #   Find all u not in temp_mask such that can_add(temp_mask, u)
                #   For each such u, find w not in temp_mask|{u} such that can_add(temp_mask|{u}, w)
                #   This is expensive.
                
                # Let's try a different angle:
                # Randomized hill climbing with restarts might be more effective and easier to bound time.
                # However, the prompt asks for a stronger algorithm.
                
                # Let's implement a specific 1-2 swap check:
                # We want to find u, w such that:
                # 1. u, w not in current_mask
                # 2. u, w not conflicting with current_mask \ {v}
                # 3. u, w not conflicting with each other
                # 4. u, v and w, v might conflict, but they are being removed/added.
                #    Actually, u and w must not conflict with any vertex in current_mask EXCEPT v.
                #    So (adj[u] | adj[w]) & (current_mask ^ (1 << v)) == 0
                #    And (adj[u] & (1 << w)) == 0 and (adj[w] & (1 << u)) == 0
                
                # Let's iterate u, w from the set of vertices NOT in current_mask.
                # This is still O(N^3).
                
                # Alternative:
                # Use the fact that we just want to maximize size.
                # A common effective heuristic is:
                # 1. Greedy Min-Degree
                # 2. Local Search:
                #    - Try to add any vertex.
                #    - If stuck, try removing a vertex and see if it allows adding 2.
                
                # To make this fast, we can limit the number of swaps tried per iteration.
                
                found_swap = False
                # Try a limited number of random v's or all v's?
                # For N=248, trying all v (248) and then finding u,w is the bottleneck.
                
                # Let's try a simpler 1-1 swap first?
                # If we remove v, can we add u?
                # If yes, and u is "better" (e.g., lower degree), maybe?
                # But we just want size. 1-1 swap doesn't increase size.
                # So we MUST do 1-2 or 2-3.
                
                # Let's try to find u, w for a specific v.
                # We can precompute the list of vertices not in current_mask.
                unselected = [u for u in range(M) if not (current_mask >> u) & 1]
                
                # For each v in sel_list:
                #   mask_without_v = current_mask ^ (1 << v)
                #   valid_u = [u for u in unselected if (adj[u] & mask_without_v) == 0]
                #   For each u in valid_u:
                #       mask_with_u = mask_without_v | (1 << u)
                #       valid_w = [w for w in valid_u if w != u and (adj[w] & mask_with_u) == 0]
                #       If valid_w is not empty:
                #           new_mask = mask_with_u | (1 << valid_w[0])
                #           size increases by 1.
                #           Update current_mask.
                #           improved = True
                #           break
                
                # This is roughly O(N * (N/2)^2) in worst case, which is ~ 248 * 124^2 ~ 3.8M ops.
                # In Python, this might be too slow for a tight loop.
                # We need to be smarter or limit the search.
                
                # Let's restrict the search for u, w to a small random subset if time is tight?
                # Or just try the first few valid u's.
                
                # Let's try to optimize the inner loop.
                # If we find a swap, we restart the search.
                
                # To keep it simple and robust within time:
                # We will perform a limited number of such checks per "epoch".
                
                # For now, let's just do a randomized local search with 1-2 swaps.
                # Pick a random v from sel_list.
                # Pick random u, w from unselected.
                # Check if swap is valid.
                
                # Actually, let's stick to the deterministic 1-2 swap but limit the inner loops.
                
                # If we find a valid u, w, we take the first one found.
                
                # To speed up, we can precompute `conflicts_with_mask`? No.
                
                # Let's just run a few iterations of this.
                
                # If N=248, 3.8M operations in Python takes ~1-2 seconds.
                # We have 85 seconds. We can do this.
                
                # Optimization: Use bit operations where possible.
                
                # Let's implement the 1-2 swap check efficiently.
                
                # We'll iterate v in sel_list.
                # For each v, we compute `mask_without_v`.
                # We iterate u in unselected.
                # If (adj[u] & mask_without_v) == 0:
                #     We have a candidate u.
                #     Now we need w in unselected, w != u, such that:
                #     (adj[w] & (mask_without_v | (1 << u))) == 0
                #     This is equivalent to:
                #     (adj[w] & mask_without_v) == 0 AND (adj[w] & (1 << u)) == 0
                #     So w must be in `valid_u` (excluding u) and not conflict with u.
                
                # So, for a fixed v:
                # 1. Find all u in unselected such that (adj[u] & mask_without_v) == 0. Let this set be U_v.
                # 2. For each u in U_v, find w in U_v \ {u} such that (adj[u] & (1 << w)) == 0.
                # 3. If such w exists, we have a valid swap.
                
                # This is much faster!
                # Finding U_v is O(N).
                # Finding w for each u is O(|U_v|).
                # Total per v is O(N + |U_v|^2).
                # Worst case |U_v| ~ N/2. So O(N^2) per v.
                # Total O(N^3) ~ 15M ops. Still a bit high for Python in a tight loop, but doable if we break early.
                
                # Let's implement this.
                
                mask_without_v = current_mask & ~(1 << v)
                
                # Find U_v
                U_v = []
                for u in unselected:
                    if (adj[u] & mask_without_v) == 0:
                        U_v.append(u)
                        
                if len(U_v) < 2:
                    continue
                    
                # Find pair in U_v
                found_pair = False
                for i in range(len(U_v)):
                    u = U_v[i]
                    # Check w in U_v[i+1:]
                    for j in range(i + 1, len(U_v)):
                        w = U_v[j]
                        if (adj[u] & (1 << w)) == 0:
                            # Valid swap found
                            current_mask = mask_without_v | (1 << u) | (1 << w)
                            current_size += 1
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if improved:
                    # Restart the outer loop with new current_mask
                    break # Break out of v loop to restart while loop with new state
                    
            if not improved:
                # Try 2-3 swap? Or just stop?
                # 1-2 swaps are usually sufficient for good results.
                # Let's add a simple 2-3 swap check if time permits.
                # For now, if no 1-2 swap found, we are in a local optimum for 1-2.
                # We can try a random restart or break.
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
    mask = local_search(mask, DEADLINE - 5.0)
    consider(mask)

    # Seed 2: Random Restarts
    for r in range(50):
        if elapsed() > DEADLINE - 10:
            break
        random.seed(r * 12345 + 67890)
        order = list(range(M))
        random.shuffle(order)
        mask = 0
        for v in order:
            if can_add(mask, v):
                mask |= (1 << v)
        mask = greedy_complete(mask)
        consider(mask)
        
        # Allocate remaining time to local search, decreasing for later iterations
        # or fixed small budget for many iterations?
        # Let's give 1.0s to each random restart's local search.
        mask = local_search(mask, 1.0)
        consider(mask)

    # Final Polish
    if elapsed() < DEADLINE - 2:
        mask = local_search(best_mask, DEADLINE - 2)
        consider(mask)

    result = mask_to_set(best_mask)
    result.sort()
    
    if not result:
        return [1] # Fallback, though 1 is likely valid (2*1=2 not square)
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found set size: {len(res)}")
