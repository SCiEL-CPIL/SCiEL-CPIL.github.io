import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Constructs a C4-free subgraph of Q7 maximizing edge count.
    Strategy:
    1. Model the problem as an Integer Linear Program (ILP).
       Variables: x_e in {0, 1} for each edge e.
       Objective: Maximize sum(x_e).
       Constraints: For each 2D face (C4) with edges e1, e2, e3, e4:
                    x_e1 + x_e2 + x_e3 + x_e4 <= 3
    2. Use scipy.optimize.milp to solve the ILP.
       This provides a globally optimal (or near-optimal within solver limits) solution,
       unlike heuristic methods which may get stuck in local optima.
    3. Fallback to a deterministic heuristic if the solver fails or times out.
    """
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
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
    
    faces = []
    face_set = set()
    
    for u in range(n):
        for i in range(d):
            for j in range(i + 1, d):
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
    
    # 2. Formulate ILP
    # Objective: Maximize sum(x_e) -> Minimize -sum(x_e)
    c = np.full(E, -1.0)
    
    # Constraints: For each face, sum of its 4 edges <= 3
    # A is a sparse matrix of shape (F, E)
    # scipy.optimize.milp accepts sparse matrices
    import scipy.sparse as sp
    
    row_indices = []
    col_indices = []
    data = []
    
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            row_indices.append(f_idx)
            col_indices.append(e_idx)
            data.append(1.0)
    
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(F, E))
    
    # Constraints: A @ x <= 3
    # In scipy.optimize.milp, constraints are defined as lb <= A @ x <= ub
    # We only have upper bounds here.
    lb = np.full(F, -np.inf)
    ub = np.full(F, 3.0)
    
    constraints = LinearConstraint(A, lb, ub)
    
    # Bounds: x_e in [0, 1]
    integrality = np.ones(E)  # All variables are integers
    bounds = Bounds(lb=np.zeros(E), ub=np.ones(E))
    
    best_state = None
    best_score = 0
    
    # Try to solve with ILP
    try:
        # Set time limit for the solver
        # scipy.optimize.milp options: 
        # 'time_limit' is supported in recent versions of scipy using HiGHS backend
        # If 'time_limit' is not supported or fails, we rely on the wall-clock check
        # But we must ensure we don't hang.
        
        # Note: HiGHS is the default backend for milp in scipy >= 1.8.
        # We pass options to the solver.
        options = {
            'time_limit': time_limit - (time.time() - start_time) - 5.0,
            'mip_rel_gap': 0.0  # Try for optimal, but gap helps if time is short
        }
        
        # Some versions might not like 'time_limit' in options if not passed correctly.
        # Let's try a standard call first. If it takes too long, we can't easily interrupt milp directly.
        # However, the problem size (448 vars, 672 constraints) is small for a modern MILP solver.
        # It should solve in seconds.
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
        
        if res.status == 0:  # Optimal
            x = res.x
            # Round to nearest integer to be safe
            x = np.rint(x).astype(int)
            
            # Validate
            valid = True
            for f in faces:
                s = x[f[0]] + x[f[1]] + x[f[2]] + x[f[3]]
                if s > 3:
                    valid = False
                    break
            
            if valid:
                best_state = x
                best_score = int(np.sum(x))
                
        elif res.status == 2:  # Infeasible (should not happen)
            pass
        else:
            # Check if we have a feasible solution even if not optimal
            if res.x is not None and len(res.x) == E:
                x = np.rint(res.x).astype(int)
                # Clip to 0/1
                x = np.clip(x, 0, 1)
                
                valid = True
                for f in faces:
                    s = x[f[0]] + x[f[1]] + x[f[2]] + x[f[3]]
                    if s > 3:
                        valid = False
                        break
                
                if valid:
                    best_state = x
                    best_score = int(np.sum(x))
                    
    except Exception as e:
        # Fallback if ILP fails
        pass
    
    # Fallback Heuristic if ILP didn't find a good solution or failed
    if best_score < 300:
        # Simple greedy repair from a dense random start
        import random
        random.seed(42)
        
        state = bytearray(E)
        p = 0.7
        for i in range(E):
            state[i] = 1 if random.random() < p else 0
        
        full_count = [0] * F
        for f_idx in range(F):
            f = faces[f_idx]
            full_count[f_idx] = state[f[0]] + state[f[1]] + state[f[2]] + state[f[3]]
        
        # Repair
        while True:
            found_conflict = False
            for f_idx in range(F):
                if full_count[f_idx] == 4:
                    f = faces[f_idx]
                    # Remove edge with highest degree in conflict graph? Just random for speed
                    e_to_remove = random.choice(f)
                    if state[e_to_remove]:
                        state[e_to_remove] = 0
                        for f_idx2 in range(F):
                            if e_to_remove in faces[f_idx2]:
                                full_count[f_idx2] -= 1
                    found_conflict = True
                    break
            if not found_conflict:
                break
        
        # Greedy Add
        improved = True
        while improved:
            improved = False
            for e in range(E):
                if state[e]:
                    continue
                can_add = True
                for f_idx in range(F):
                    if e in faces[f_idx] and full_count[f_idx] == 3:
                        can_add = False
                        break
                if can_add:
                    state[e] = 1
                    for f_idx in range(F):
                        if e in faces[f_idx]:
                            full_count[f_idx] += 1
                    improved = True
        
        score = sum(state)
        if score > best_score:
            best_score = score
            best_state = np.array(state)
    
    if best_state is None:
        return []
        
    result = []
    for i in range(E):
        if best_state[i] == 1:
            u, v = edge_list[i]
            result.append([u, v])
            
    return result

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}, Time: {end-start:.2f}s")
