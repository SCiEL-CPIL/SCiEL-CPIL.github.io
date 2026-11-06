"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random

def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    Uses a greedy MIS heuristic with optimized degree calculation and multiple restarts.
    """
    best_result = None
    
    # Number of restarts to try different tie-breaks
    num_restarts = 50
    
    for _ in range(num_restarts):
        selected = []
        sums = set()
        available = set(range(1, N + 1))
        
        # Precomputed sets that depend on 'selected'
        # D_S: set of differences s' - s for s, s' in selected
        D_S = set()
        # We will update D_S incrementally or recompute if small enough.
        # |S| is small, so recomputing D_S each iteration is fine.
        
        while available:
            best_v = None
            max_degree = -1
            candidates = []
            
            # Evaluate each candidate in 'available'
            for v in available:
                # 1. Check if v is compatible with current 'selected'
                new_sums_v = {2 * v}
                for s in selected:
                    new_sums_v.add(v + s)
                
                # If any new sum collides with existing sums, v is invalid
                if sums.intersection(new_sums_v):
                    continue
                
                # v is a valid candidate to add.
                # Calculate 'degree': how many other available elements u become incompatible?
                
                # B_v = { u | u is blocked by adding v }
                # u is blocked if:
                # 1. 2u in sums_temp (sums U new_sums_v)
                #    2u in sums OR 2u in new_sums_v
                #    2u in new_sums_v => 2u=2v (u=v, skip) or 2u=v+s => u=(v+s)/2
                # 2. u+s in sums_temp for some s in selected U {v}
                #    (u+s) in sums OR (u+s) in new_sums_v
                #    For s in selected:
                #       u+s in sums => u in sums - s
                #       u+s in new_sums_v => u+s=2v (u=2v-s) or u+s=v+s' (u=v+s'-s)
                #    For s=v:
                #       u+v in sums => u in sums - v
                #       u+v in new_sums_v => u+v=2v (u=v, skip) or u+v=v+s' (u=s', skip)
                
                # Let's construct the set of blocked u's
                blocked_u = set()
                
                # Part 1: 2u in new_sums_v
                for s in selected:
                    val = v + s
                    if val % 2 == 0:
                        u_val = val // 2
                        if 1 <= u_val <= N:
                            blocked_u.add(u_val)
                
                # Part 2: u+s in sums for s in selected
                # This is union of (sums - s) for s in selected
                # Optimized: We can just iterate available u and check?
                # No, constructing the set is better for intersection.
                # But sums is a set. sums - s is a set comprehension.
                # |sums| is ~ |S|^2/2. |S| is small.
                for s in selected:
                    # u + s in sums => u = x - s for x in sums
                    for x in sums:
                        u_val = x - s
                        if 1 <= u_val <= N:
                            blocked_u.add(u_val)
                            
                # Part 3: u+s in new_sums_v for s in selected
                # u+s = 2v => u = 2v - s
                for s in selected:
                    u_val = 2 * v - s
                    if 1 <= u_val <= N:
                        blocked_u.add(u_val)
                    
                # u+s = v + s' => u = v + s' - s
                for s in selected:
                    for sp in selected:
                        u_val = v + sp - s
                        if 1 <= u_val <= N:
                            blocked_u.add(u_val)
                        
                # Part 4: u+v in sums (s=v case)
                for x in sums:
                    u_val = x - v
                    if 1 <= u_val <= N:
                        blocked_u.add(u_val)
                
                # Calculate degree: count of u in available \ {v} that are in blocked_u
                # Optimization: iterate over available and check membership in blocked_u
                # Or iterate over blocked_u and check membership in available.
                # blocked_u might be larger than available?
                # |blocked_u| <= |sums|*|S| + ... ~ 136*16 = 2000.
                # |available| <= 256.
                # So iterate over available.
                
                degree = 0
                for u in available:
                    if u == v:
                        continue
                    if u in blocked_u:
                        degree += 1
                
                if degree > max_degree:
                    max_degree = degree
                    best_v = v
                    candidates = [v]
                elif degree == max_degree:
                    candidates.append(v)
            
            if best_v is None:
                # No valid vertex, set is maximal
                break
            
            # Random tie-breaking
            if len(candidates) > 1:
                best_v = random.choice(candidates)
            
            # Update state with best_v
            selected.append(best_v)
            
            # Update sums
            sums.add(2 * best_v)
            for s in selected[:-1]:
                sums.add(best_v + s)
            
            # Update available: remove best_v and any u incompatible with new selected
            # We can reuse the logic: u is available if it is NOT blocked by the new set.
            # But we already computed blocked_u for the *hypothetical* addition of best_v.
            # So new_available = available - {best_v} - blocked_u
            # Wait, blocked_u was calculated based on S_temp = S U {best_v}.
            # Yes, u is blocked if S_temp U {u} is not Sidon.
            # So we just remove all u in blocked_u from available.
            
            available.discard(best_v)
            for u in blocked_u:
                available.discard(u)
        
        if best_result is None or len(selected) < len(best_result):
            best_result = selected[:]
            
        # Early termination if we found a very small set?
        # Minimum size is roughly sqrt(N) ~ 16.
        # If we find 16 or 17, it's likely optimal.
        if best_result and len(best_result) <= 16:
            break

    return best_result
# EVOLVE-BLOCK-END
