import time
import random
import sys
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy:
    1. Precompute edges and faces.
    2. Create a high-quality initial feasible solution using random greedy repair.
    3. Formulate an Integer Linear Programming (ILP) problem:
       Maximize sum(x_e) subject to sum(x_e for e in face) <= 3 for all faces.
    4. Solve using scipy.optimize.milp (HiGHS) with a time limit.
    5. If the solver times out, use the best incumbent found.
    6. Perform a final greedy local improvement on the result.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s kill
    
    d = 7
    n = 2 ** d  # 128 vertices
    
    # 1. Precompute edges and faces
    edge_list = []
    edge_to_idx = {}
    
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edge_list)
                edge_list.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    E = len(edge_list)  # 448
    
    # Precompute faces (C4s)
    faces = []
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v1 = u
                v2 = u ^ (1 << i)
                v3 = u ^ (1 << j)
                v4 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v2, v4), max(v2, v4))
                e4 = (min(v3, v4), max(v3, v4))
                
                idx1 = edge_to_idx[e1]
                idx2 = edge_to_idx[e2]
                idx3 = edge_to_idx[e3]
                idx4 = edge_to_idx[e4]
                
                face_tuple = tuple(sorted([idx1, idx2, idx3, idx4]))
                
                if face_tuple not in face_set:
                    face_set.add(face_tuple)
                    faces.append([idx1, idx2, idx3, idx4])
    
    F = len(faces)  # 672
    
    # Precompute edge-to-face mapping for local search/repair
    edge_faces = [[] for _ in range(E)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_faces[e_idx].append(f_idx)
    
    # 2. Initialize with a valid dense state
    # Random start with high density, then repair
    def initialize_valid():
        state = bytearray(E)
        p = 0.75
        for i in range(E):
            state[i] = 1 if random.random() < p else 0
        
        full_count = [0] * F
        for f_idx in range(F):
            f = faces[f_idx]
            count = state[f[0]] + state[f[1]] + state[f[2]] + state[f[3]]
            full_count[f_idx] = count
        
        # Repair loop
        while True:
            found_conflict = False
            for f_idx in range(F):
                if full_count[f_idx] == 4:
                    f = faces[f_idx]
                    # Remove edge with highest degree in conflict graph? 
                    # For speed, just remove first one or random.
                    # Random is fine for initialization.
                    e_to_remove = random.choice(f)
                    if state[e_to_remove]:
                        state[e_to_remove] = 0
                        for f_idx2 in edge_faces[e_to_remove]:
                            full_count[f_idx2] -= 1
                    found_conflict = True
                    break
            if not found_conflict:
                break
        
        score = sum(state)
        return state, full_count, score
    
    random.seed(42)
    init_state, init_full_count, init_score = initialize_valid()
    
    # 3. Formulate ILP
    # Variables: x_e for e in 0..E-1
    # Objective: Maximize sum(x_e) -> Minimize -sum(x_e)
    c = -np.ones(E)
    
    # Constraints: For each face f, sum_{e in f} x_e <= 3
    # We can build the A matrix directly or use sparse.
    # Since F=672, E=448, a dense matrix is 672x448 ~ 300k floats, which is small.
    A = np.zeros((F, E))
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            A[f_idx, e_idx] = 1.0
    
    # Bounds: 0 <= x_e <= 1
    bounds = Bounds(lb=0.0, ub=1.0)
    
    # Integrality: all variables are integer
    integrality = np.ones(E)
    
    # Linear constraints: A x <= 3
    # scipy.optimize.milp expects LinearConstraint(A, lb, ub)
    # Here A is the constraint matrix.
    linear_constraint = LinearConstraint(A, lb=-np.inf, ub=3.0)
    
    # Initial guess to help the solver
    # Note: x0 must be feasible. init_state is feasible.
    x0 = np.array(init_state, dtype=float)
    
    # Time limit for the solver
    # We reserve 10s for post-processing and overhead
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time < 1:
        # Not enough time for ILP, just return init
        best_state = init_state
        best_score = init_score
    else:
        try:
            # options: time_limit, mip_rel_gap, etc.
            # We want to find a good solution quickly. 
            # mip_rel_gap=0.0 asks for optimality, but might take too long.
            # Let's allow a small gap or just rely on time limit.
            # HiGHS default gap is 1%. 
            # We can set 'time_limit' in options.
            options = {
                'time_limit': remaining_time,
                'mip_rel_gap': 0.0, # Try for optimal, but stop at time limit
                'presolve': 'on',
                'parallel': 'on'
            }
            
            res = milp(c=c, constraints=linear_constraint, integrality=integrality, 
                       bounds=bounds, x0=x0, options=options)
            
            if res.success or (res.status == 2 and res.x is not None): # status 2 = optimal, but also check incumbent
                # res.x might be None if no feasible solution found, but we have x0
                if res.x is not None:
                    # Round to nearest integer? ILP should return integers.
                    # But due to floating point, might be 0.9999.
                    sol = np.round(res.x).astype(int)
                    # Ensure binary
                    sol = (sol > 0.5).astype(int)
                    
                    # Validate
                    is_valid = True
                    for f_idx in range(F):
                        count = 0
                        for e_idx in faces[f_idx]:
                            count += sol[e_idx]
                        if count > 3:
                            is_valid = False
                            break
                    
                    if is_valid:
                        best_score = int(sum(sol))
                        best_state = bytearray(sol)
                    else:
                        # Fallback to init
                        best_state = init_state
                        best_score = init_score
                else:
                    best_state = init_state
                    best_score = init_score
            else:
                # Solver failed or timed out without a better solution?
                # HiGHS usually returns the best incumbent in res.x even if not optimal, 
                # but sometimes res.x is None if it didn't find anything better than x0?
                # Actually, if x0 is provided, it's the incumbent.
                # Let's check res.x again.
                if res.x is not None:
                    sol = np.round(res.x).astype(int)
                    sol = (sol > 0.5).astype(int)
                    is_valid = True
                    for f_idx in range(F):
                        count = 0
                        for e_idx in faces[f_idx]:
                            count += sol[e_idx]
                        if count > 3:
                            is_valid = False
                            break
                    if is_valid and sum(sol) > init_score:
                        best_state = bytearray(sol)
                        best_score = int(sum(sol))
                    else:
                        best_state = init_state
                        best_score = init_score
                else:
                    best_state = init_state
                    best_score = init_score
        except Exception:
            best_state = init_state
            best_score = init_score
    
    # 4. Final Greedy Local Improvement
    # Rebuild full_count for best_state
    best_full_count = [0] * F
    for f_idx in range(F):
        f = faces[f_idx]
        best_full_count[f_idx] = best_state[f[0]] + best_state[f[1]] + best_state[f[2]] + best_state[f[3]]
    
    improved = True
    while improved:
        improved = False
        # Try adding edges
        for e in range(E):
            if best_state[e]:
                continue
            can_add = True
            for f_idx in edge_faces[e]:
                if best_full_count[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                best_state[e] = 1
                for f_idx in edge_faces[e]:
                    best_full_count[f_idx] += 1
                best_score += 1
                improved = True
        
        # Try swapping? (Remove one, add two)
        # This is more complex. Given the time, simple add is often sufficient if ILP is strong.
        # But let's do a simple 1-1 swap if time permits.
        if time.time() - start_time < time_limit - 5:
            # Simple swap: for each missing edge e1, for each present edge e2, 
            # if removing e2 allows adding e1 and maybe e3?
            # Too slow for O(E^2). 
            # Stick to greedy add.
            pass

    # 5. Construct result
    result = []
    for i in range(E):
        if best_state[i]:
            u, v = edge_list[i]
            result.append([u, v])
    
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
