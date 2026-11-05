import math
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safety margin

    # 1. Build compatibility graph
    # adj[i] is a bitmask of j such that lcm(i,j) <= N
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            if i * j // g <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
    # Add self-loops for convenience in some algorithms, but for clique we usually work with neighbors
    # Let's keep adj as neighbors only.

    # 2. Dominance Reduction
    # If N(u) \subseteq N(v) \cup {v}, remove u.
    # We represent the graph as a list of bitmasks.
    # active[i] = 1 if i is in the graph.
    active = [1] * (N + 1)
    
    # Iterative removal
    changed = True
    while changed:
        changed = False
        # Sort vertices by degree (or just iterate)
        # For efficiency, we can check pairs.
        # O(N^2) check per iteration.
        for u in range(1, N + 1):
            if not active[u]:
                continue
            for v in range(1, N + 1):
                if u == v or not active[v]:
                    continue
                # Check if N(u) \subseteq N(v) \cup {v}
                # N(u) is adj[u]
                # N(v) \cup {v} is adj[v] | (1 << v)
                # Condition: (adj[u] & active_mask) == (adj[u] & (adj[v] | (1 << v)) & active_mask)
                # Actually, we only care about active vertices.
                # Let's just check if (adj[u] & ~(adj[v] | (1 << v))) == 0
                # But we must ignore inactive vertices.
                # So: (adj[u] & active_mask & ~(adj[v] | (1 << v))) == 0
                # We don't have a global active_mask readily updated in this loop structure easily without pre-computation.
                # Let's use a simpler check:
                # For all w in N(u), w must be in N(v) or w==v.
                # This is equivalent to: adj[u] is a subset of (adj[v] | (1 << v))
                # We can do this with bitmasks if we maintain active_mask.
                
                # Let's maintain active_mask.
                # This inner loop is O(N^2) which is 512^2 = 262k.
                # With bitmasks, it's fast.
                
                # We need to know which vertices are active.
                # Let's just do the bitmask check.
                # If (adj[u] & ~(adj[v] | (1 << v))) == 0, then u is dominated by v.
                # BUT this assumes all vertices in adj[u] are active.
                # If some are inactive, they don't matter.
                # So we need (adj[u] & active_mask & ~(adj[v] | (1 << v))) == 0.
                
                # To avoid calculating active_mask every time, we can just check:
                # Is there any w such that active[w] and (adj[u] has w) and NOT (adj[v] has w or w==v)?
                # If yes, u is not dominated.
                
                # For speed, let's just do the bitmask op with a precomputed active_mask.
                # We'll update active_mask after the inner loop.
                
                # Actually, let's just do the simple check in Python, it might be slow.
                # Better: use bitmasks.
                
                # Let's refactor to use a list of active vertices and bitmasks.
                pass

    # The simple O(N^3) or O(N^2 * bit_ops) might be okay for N=512.
    # Let's implement the bitmask version properly.
    
    active_mask = 0
    for i in range(1, N + 1):
        active_mask |= (1 << i)
        
    # Initial adj is for all vertices.
    # We need to mask adj with active_mask when checking.
    
    changed = True
    while changed:
        changed = False
        for u in range(1, N + 1):
            if not (active_mask & (1 << u)):
                continue
            for v in range(1, N + 1):
                if u == v or not (active_mask & (1 << v)):
                    continue
                # Check if N(u) \subseteq N(v) \cup {v}
                # N(u) masked: adj[u] & active_mask
                # N(v) \cup {v} masked: (adj[v] | (1 << v)) & active_mask
                # If (adj[u] & active_mask) & ~((adj[v] | (1 << v)) & active_mask) == 0
                # Then u is dominated by v.
                
                # Optimization: if degree(u) > degree(v), u cannot be dominated by v?
                # Not necessarily, but if N(u) \subseteq N(v) \cup {v}, then |N(u)| <= |N(v)| + 1.
                # So if bit_count(adj[u] & active_mask) > bit_count((adj[v] | (1 << v)) & active_mask), skip.
                
                deg_u = (adj[u] & active_mask).bit_count()
                deg_v_plus_1 = ((adj[v] | (1 << v)) & active_mask).bit_count()
                
                if deg_u > deg_v_plus_1:
                    continue
                    
                if (adj[u] & active_mask) & ~((adj[v] | (1 << v)) & active_mask) == 0:
                    # u is dominated by v
                    active_mask &= ~(1 << u)
                    changed = True
                    break # restart u loop? or just continue?
                    # If we remove u, it affects other checks.
                    # We should restart the outer loop or at least the inner loop.
                    # Breaking inner loop and restarting outer is safer.
                    # But breaking outer loop is expensive.
                    # Let's just break inner and let the while loop handle it.
                    # Actually, if we remove u, we should stop checking u against other v's.
                    # And we should re-evaluate other pairs?
                    # Removing u might make another vertex dominated.
                    # So we set changed=True and break.
                    
    # Now we have the reduced graph.
    # We need to run Max Clique on the active vertices.
    
    # Extract active vertices
    vertices = [i for i in range(1, N + 1) if (active_mask & (1 << i))]
    M = len(vertices)
    
    if M == 0:
        return [1] # Fallback, though 1 should be active
        
    # Map old indices to new indices 0..M-1
    # Or just work with the original indices and bitmasks.
    # Working with original indices is easier for the B&B.
    
    # B&B Solver
    best_size = 0
    best_clique = []
    
    # Precompute degrees in the reduced graph
    # deg[i] = bit_count(adj[i] & active_mask)
    
    # Coloring bound function
    def color_bound(mask, limit):
        # mask is a bitmask of active vertices
        colors = 0
        uncolored = mask
        while uncolored:
            colors += 1
            if colors >= limit:
                return colors
            # Greedy coloring: pick vertex with max degree in uncolored
            best_v = -1
            max_deg = -1
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj[v] & uncolored).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            # Color class: best_v and its non-neighbors in uncolored
            color_class = uncolored & ~adj[best_v]
            color_class |= (1 << best_v)
            uncolored &= ~color_class
        return colors

    def solve(mask, current_size, current_clique):
        nonlocal best_size, best_clique
        
        if time.time() - start_time > time_limit:
            return
            
        count = mask.bit_count()
        if current_size + count <= best_size:
            return
            
        needed = best_size - current_size + 1
        ub = color_bound(mask, needed)
        if current_size + ub <= best_size:
            return
            
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_clique = list(current_clique)
            return
            
        # Order vertices by degree in mask (descending)
        # We process them in reverse order (lowest degree first? or highest?)
        # Standard: process in order of increasing color?
        # Let's just iterate through bits.
        
        # To improve, we can sort vertices in mask by degree.
        # But sorting every time is expensive.
        # We can just use the bit iteration.
        
        temp = mask
        while temp:
            if time.time() - start_time > time_limit:
                return
                
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            # Check bound for v
            # If current_size + 1 + color_bound(mask & adj[v]) <= best_size, skip v?
            # This is expensive.
            
            # Standard B&B:
            # Include v
            new_mask = mask & adj[v]
            current_clique.append(v)
            solve(new_mask, current_size + 1, current_clique)
            current_clique.pop()
            
            # Exclude v
            mask &= ~lsb
            
            # Check if we can prune the rest
            # If current_size + mask.bit_count() <= best_size: break
            
    # Initial solution: greedy
    def greedy_clique():
        chosen = []
        remaining = active_mask
        while remaining:
            best_v = -1
            max_deg = -1
            temp = remaining
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            chosen.append(best_v)
            remaining &= ~(1 << best_v)
            remaining &= adj[best_v]
        return chosen

    init_sol = greedy_clique()
    best_size = len(init_sol)
    best_clique = list(init_sol)
    
    # Start B&B
    solve(active_mask, 0, [])
    
    return sorted(best_clique)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                print(f"Invalid: {a}, {b}, lcm={l}")
                break
