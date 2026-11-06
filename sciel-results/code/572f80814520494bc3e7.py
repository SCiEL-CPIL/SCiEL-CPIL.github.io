import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct():
    start_time = time.time()
    # Hard kill is 90s. We target finishing within 85s to be safe.
    limit = 85.0
    
    n = 256
    # Weights: 1/i
    weights = np.array([1.0 / i for i in range(1, n + 1)])
    
    # Precompute all 3-term arithmetic progressions
    # (a, b, c) such that b-a = c-b
    ap_constraints = []
    for a in range(1, n + 1):
        max_d = (n - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            ap_constraints.append((a - 1, b - 1, c - 1))  # 0-indexed
            
    num_constraints = len(ap_constraints)
    
    # Build sparse constraint matrix A
    # Constraint: x_a + x_b + x_c <= 2
    rows = []
    cols = []
    data = []
    
    for i, (a, b, c) in enumerate(ap_constraints):
        rows.extend([i, i, i])
        cols.extend([a, b, c])
        data.extend([1, 1, 1])
        
    A = csr_matrix((data, (rows, cols)), shape=(num_constraints, n))
    b_ub = np.full(num_constraints, 2.0)
    
    # Objective: Maximize sum(weights * x) -> Minimize -sum(weights * x)
    c_obj = -weights
    
    integrality = np.ones(n)
    lin_cons = LinearConstraint(A, lb=None, ub=b_ub)
    var_bounds = Bounds(lb=0, ub=1)
    
    best_S = set()
    best_score = 0.0
    
    # 1. Try MILP first to get a strong initial solution
    # We allocate 25s for MILP. If it fails or times out, we fall back.
    try:
        # Use a tight gap to try for near-optimality, but time-limit protects us
        res = milp(c=c_obj, constraints=lin_cons, integrality=integrality, bounds=var_bounds, 
                   options={'time_limit': 25, 'mip_gap': 0.01})
        
        if res.x is not None:
            solution_mask = (res.x > 0.5).astype(int)
            selected_indices = [i + 1 for i in range(n) if solution_mask[i] == 1]
            
            if selected_indices:
                # Verify validity
                S_set = set(selected_indices)
                valid = True
                for a, b, c in ap_constraints:
                    count = sum(1 for x in [a+1, b+1, c+1] if x in S_set)
                    if count >= 3:
                        valid = False
                        break
                
                if valid:
                    score = sum(weights[x] for x in S_set)
                    if score > best_score:
                        best_score = score
                        best_S = S_set
    except Exception:
        pass
    
    # 2. Fallback / Initial Solution: Greedy
    # If MILP didn't provide a solution or we want a faster baseline to start SA
    if not best_S:
        S = set()
        # Sort by weight descending? Or just 1..n?
        # Greedy by weight (smallest numbers) is usually good for reciprocal sum.
        for x in range(1, n + 1):
            valid = True
            for y in S:
                # Check if adding x creates a 3-AP with any y in S
                # 1. x is middle: z = 2x - y
                z = 2 * x - y
                if 1 <= z <= n and z in S:
                    valid = False
                    break
                # 2. y is middle: z = 2y - x
                z = 2 * y - x
                if 1 <= z <= n and z in S:
                    valid = False
                    break
                # 3. x and y are ends: z = (x+y)/2
                if (x + y) % 2 == 0:
                    z = (x + y) // 2
                    if 1 <= z <= n and z in S:
                        valid = False
                        break
            if valid:
                S.add(x)
        
        score = sum(weights[x] for x in S)
        if score > best_score:
            best_score = score
            best_S = S
    
    # 3. Simulated Annealing to refine
    # We have ~60s left. Use it to polish the solution.
    if time.time() - start_time < limit - 2: # Leave 2s buffer
        S = set(best_S)
        current_score = best_score
        
        # Helper for fast validity check
        def is_valid_addition(v, S_set):
            """Check if v can be added to S_set without creating a 3-AP."""
            # We need to ensure no {a,b,c} in S_set U {v} is a 3-AP.
            # Since S_set is valid, any new 3-AP must involve v.
            # So we check all pairs (u, w) in S_set such that {u, v, w} is AP?
            # No, we check if v forms an AP with any two elements in S_set.
            # Actually, it's faster to iterate over elements in S_set and see if they form AP with v.
            # For each u in S_set:
            #   If v is middle: u, v, w => w = 2v - u. Check if w in S_set.
            #   If u is middle: u, v, w? No, u is in S_set.
            #   Let's just iterate through S_set and check the 3 potential 3rd elements.
            
            for u in S_set:
                # Case 1: v is the middle term. u, v, z. z = 2v - u.
                z = 2 * v - u
                if 1 <= z <= n and z in S_set:
                    return False
                
                # Case 2: u is the middle term. v, u, z or z, u, v.
                # If v, u, z: z = 2u - v.
                z = 2 * u - v
                if 1 <= z <= n and z in S_set:
                    return False
                
                # Case 3: v and u are the ends. m = (v+u)/2.
                if (v + u) % 2 == 0:
                    m = (v + u) // 2
                    if 1 <= m <= n and m in S_set:
                        return False
            return True

        # Helper to calculate score change for a move
        def get_score_delta(add_v, remove_u, S_set):
            delta = 0.0
            if add_v is not None:
                delta += weights[add_v - 1]
            if remove_u is not None:
                delta -= weights[remove_u - 1]
            return delta

        T = 0.05
        cooling_rate = 0.9995
        min_T = 0.001
        
        all_nodes = list(range(1, n + 1))
        
        # Main SA Loop
        while time.time() - start_time < limit:
            if T < min_T:
                T = min_T
                
            # Choose a move type:
            # 1. Add random element (if valid)
            # 2. Remove random element
            # 3. Swap (remove one, add one)
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add
                v = random.choice(all_nodes)
                if v not in S:
                    if is_valid_addition(v, S):
                        # Always accept valid additions (they increase score)
                        S.add(v)
                        current_score += weights[v - 1]
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
                        T *= 0.999 # Cool down slightly on success? Or keep same.
                        continue
            elif move_type < 0.6:
                # Try to remove
                if S:
                    u = random.choice(list(S))
                    # Calculate delta
                    delta = -weights[u - 1]
                    # Metropolis criterion
                    if delta > 0 or random.random() < np.exp(delta / T):
                        S.remove(u)
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
            else:
                # Try swap
                if S:
                    u = random.choice(list(S))
                    v = random.choice(all_nodes)
                    if v == u:
                        continue
                    
                    # Check if we can remove u and add v
                    # First remove u from consideration for validity of v
                    # Actually, we need to check if v is valid in (S \ {u})
                    # And we don't need to check if u is valid in (S \ {u} U {v}) because we are removing u.
                    # But we must ensure S \ {u} U {v} is a valid set.
                    # Since S is valid, S \ {u} is valid.
                    # So we just need to check if v is valid in S \ {u}.
                    
                    # Optimization: If v is in S, it's not a swap, it's a no-op or invalid move type.
                    # We picked v from all_nodes. If v in S, skip?
                    # If v in S and v != u, then we are swapping u for v which is already there? No, that's just removing u.
                    # Let's ensure v not in S for a proper swap.
                    if v in S:
                        continue
                    
                    # Check validity of v in S without u
                    # We can't just use is_valid_addition(v, S) because u is in S and might conflict with v.
                    # But if u conflicts with v, removing u might make v valid.
                    # So we must check against S \ {u}.
                    
                    # Efficient check:
                    valid = True
                    for w in S:
                        if w == u:
                            continue
                        # Check APs involving v and w
                        # 1. v middle: 2v - w in S\{u}?
                        z = 2 * v - w
                        if 1 <= z <= n and z in S and z != u:
                            valid = False
                            break
                        # 2. w middle: 2w - v in S\{u}?
                        z = 2 * w - v
                        if 1 <= z <= n and z in S and z != u:
                            valid = False
                            break
                        # 3. v, w ends: (v+w)/2 in S\{u}?
                        if (v + w) % 2 == 0:
                            m = (v + w) // 2
                            if 1 <= m <= n and m in S and m != u:
                                valid = False
                                break
                    
                    if valid:
                        delta = weights[v - 1] - weights[u - 1]
                        if delta > 0 or random.random() < np.exp(delta / T):
                            S.remove(u)
                            S.add(v)
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_S = set(S)
            
            T *= cooling_rate
            
    return list(best_S)

