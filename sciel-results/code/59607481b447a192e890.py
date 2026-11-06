import time
import random
import numpy as np
from collections import defaultdict, Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 128
    
    # Pre-compute all 4-term APs
    aps = []
    ap_indices = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            ap_idx = len(aps)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(ap_idx)
                
    num_aps = len(aps)
    aps_np = np.array(aps, dtype=np.int32)

    # Run seed to get initial best
    def run_seed():
        colors=[]
        for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
        return colors

    seed_colors = run_seed()
    best_colors = seed_colors[:]
    best_k = len(set(seed_colors))

    # CSP Solver with Forward Checking and MRV
    def csp_solve(k, time_limit):
        """
        Attempts to find a valid coloring with exactly k colors.
        Returns list of colors or None.
        """
        if time.time() > time_limit:
            return None
            
        # Domains: list of sets
        # Initially all {0, ..., k-1}
        domains = [set(range(k)) for _ in range(N)]
        assigned = [-1] * N
        
        # Precompute AP neighbors for each variable
        # For each AP, we need to know the other 3 variables.
        # We'll maintain a state for each AP: how many are assigned, and which values.
        
        # To make MRV efficient, we need to pick the variable with smallest domain.
        # Since domains change, we use a simple loop or a heap.
        # N=128 is small, so a linear scan for MRV is O(N) which is fine.
        
        def get_ap_state(ap_idx):
            ap = aps_np[ap_idx]
            vals = [assigned[i] for i in ap]
            assigned_count = sum(1 for v in vals if v != -1)
            return assigned_count, vals

        def propagate(idx, val):
            """
            Forward checking: assign idx=val, update domains of neighbors.
            Returns False if contradiction (domain empty), True otherwise.
            """
            assigned[idx] = val
            domains[idx] = {val}
            
            for ap_idx in ap_indices[idx]:
                ap = aps_np[ap_idx]
                # Get current values in this AP
                vals = [assigned[i] for i in ap]
                unassigned_idx_in_ap = [i for i in ap if assigned[i] == -1]
                
                if len(unassigned_idx_in_ap) == 0:
                    # All assigned, check validity
                    if len(set(vals)) < 3:
                        return False
                elif len(unassigned_idx_in_ap) == 1:
                    # 3 assigned, 1 unassigned. Prune the unassigned one.
                    u_idx = unassigned_idx_in_ap[0]
                    # The assigned values are the other 3
                    assigned_vals = [v for v in vals if v != -1]
                    
                    # Determine which values for u_idx are valid
                    # Condition: len(set(assigned_vals + [u_val])) >= 3
                    
                    # If all 3 assigned are same (e.g., 0,0,0):
                    # u_val must be != 0.
                    if len(set(assigned_vals)) == 1:
                        # u_val cannot be the single value
                        bad_val = assigned_vals[0]
                        new_domain = domains[u_idx] - {bad_val}
                        if not new_domain:
                            return False
                        domains[u_idx] = new_domain
                    # If 2 assigned are same, 1 different (e.g., 0,0,1):
                    # u_val cannot be 0 or 1?
                    # If u_val=0: {0,0,1,0} -> {0,1} (2) -> Bad
                    # If u_val=1: {0,0,1,1} -> {0,1} (2) -> Bad
                    # If u_val=2: {0,0,1,2} -> {0,1,2} (3) -> Good
                    # So u_val must be different from both distinct values in assigned_vals.
                    elif len(set(assigned_vals)) == 2:
                        distinct_vals = set(assigned_vals)
                        # u_val must not be in distinct_vals
                        new_domain = domains[u_idx] - distinct_vals
                        if not new_domain:
                            return False
                        domains[u_idx] = new_domain
                    # If 3 assigned are all distinct (e.g., 0,1,2):
                    # u_val can be anything?
                    # If u_val=0: {0,1,2,0} -> {0,1,2} (3) -> Good
                    # If u_val=3: {0,1,2,3} -> {0,1,2,3} (4) -> Good
                    # So no pruning needed.
                    else:
                        pass
                        
                # If 2 or fewer assigned, no immediate pruning possible with simple FC
                # (We could do more complex propagation, but this is a good start)
                
            return True

        def backtrack(depth):
            if time.time() > time_limit:
                return False
            
            if depth == N:
                return True
            
            # MRV: Pick unassigned variable with smallest domain
            min_size = k + 1
            var = -1
            for i in range(N):
                if assigned[i] == -1:
                    if len(domains[i]) < min_size:
                        min_size = len(domains[i])
                        var = i
                        if min_size == 1: # Optimization
                            break
            
            if var == -1:
                return True # Should not happen if depth < N
            
            # Try values in var's domain
            # Heuristic: Try values that appear in neighbors? Or just iterate.
            # For speed, we can randomize order or use a fixed order.
            # Fixed order 0..k-1 is often fine.
            for val in sorted(domains[var]):
                # Save state for backtracking
                old_assigned_var = assigned[var]
                old_domains = [set(d) for d in domains] # Copy all domains? Expensive.
                
                # Better: Only copy affected domains?
                # But FC updates domains of neighbors.
                # Let's copy the whole domains list. N=128, sets of size <=10.
                # 128 * 10 * 8 bytes ~ 10KB. Copying 10KB per step is okay.
                
                if propagate(var, val):
                    if backtrack(depth + 1):
                        return True
                
                # Restore state
                assigned[var] = old_assigned_var
                for i in range(N):
                    domains[i] = old_domains[i]
                
            return False

        # Start backtracking
        # We can add a random restart or random value ordering to escape symmetry
        # But let's try deterministic first.
        
        # To help with symmetry breaking, we can fix the first variable to 0.
        # And maybe the second to 0 or 1?
        # Let's just fix assigned[0] = 0.
        
        assigned[0] = 0
        domains[0] = {0}
        
        if backtrack(1):
            return assigned
        return None

    # Main loop: Try to reduce k
    current_k = best_k
    
    # Try to find solution with fewer colors
    # We iterate downwards from current_k - 1
    for k in range(current_k - 1, 2, -1):
        if time.time() > deadline - 5:
            break
            
        # Allow some time for CSP
        time_limit = min(deadline - 2, time.time() + 10.0)
        
        solution = csp_solve(k, time_limit)
        
        if solution is not None:
            best_colors = solution
            best_k = k
            # Continue trying to reduce further?
            # The loop continues, so yes.
            
    return best_colors
# EVOLVE-BLOCK-END
