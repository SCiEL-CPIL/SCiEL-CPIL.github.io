import time
import random
import numpy as np
from itertools import combinations

N = 16
NEDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def _get_edge_indices():
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
            idx += 1
    return idx_map

IDX_MAP = _get_edge_indices()

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

def _initial_valid_coloring():
    # Fallback: 2 colors (guaranteed valid)
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            if i >= 8 and j >= 8:
                colors[idx] = 1
            else:
                colors[idx] = 0
    return colors

def _initial_algebraic_coloring(K):
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            colors[idx] = (i + j) % K
    return colors

def _is_valid_full(colors_np):
    # Brute force check for safety
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        c1 = colors_np[IDX_MAP[(a,b)]]
        c2 = colors_np[IDX_MAP[(b,c)]]
        c3 = colors_np[IDX_MAP[(c,d)]]
        c4 = colors_np[IDX_MAP[(d,a)]]
        if len({c1, c2, c3, c4}) == 4:
            return False
        # Cycle 2: a-b-d-c-a
        c1 = colors_np[IDX_MAP[(a,b)]]
        c2 = colors_np[IDX_MAP[(b,d)]]
        c3 = colors_np[IDX_MAP[(d,c)]]
        c4 = colors_np[IDX_MAP[(c,a)]]
        if len({c1, c2, c3, c4}) == 4:
            return False
        # Cycle 3: a-c-b-d-a
        c1 = colors_np[IDX_MAP[(a,c)]]
        c2 = colors_np[IDX_MAP[(c,b)]]
        c3 = colors_np[IDX_MAP[(b,d)]]
        c4 = colors_np[IDX_MAP[(d,a)]]
        if len({c1, c2, c3, c4}) == 4:
            return False
    return True

def _sa_search(start_time, deadline, rng):
    """
    Simulated Annealing on p, q labels.
    c_ij = p_i + q_j
    Constraint: All c_ij in [0, 119].
    Objective: Maximize distinct colors.
    """
    
    # Initial p, q
    # Try to start with a valid algebraic-like structure or random
    # Let's try random initialization with small values to avoid immediate overflow
    p = np.zeros(N, dtype=np.int32)
    q = np.zeros(N, dtype=np.int32)
    
    # We can set a base offset to allow negative values in p/q if needed, 
    # but since c must be 0..119, let's constrain p, q to be non-negative 
    # and small, or use a shift.
    # Actually, let's just use p, q in [0, 20] initially.
    
    # Initialize with a simple valid pattern to ensure we start in a good basin
    # c_ij = (i+j) % K is not exactly p_i+q_j, but we can map it.
    # Or just start random.
    
    # Let's try a few random starts
    best_p = None
    best_q = None
    best_distinct = 0
    
    T_start = 10.0
    T_end = 0.1
    cooling = 0.99
    
    for restart in range(100): # Max restarts, but time is the real limit
        if time.time() > deadline:
            break
            
        # Initialize
        p = np.random.randint(0, 15, size=N)
        q = np.random.randint(0, 15, size=N)
        
        # Check validity of this structure
        colors = np.zeros(NEDGES, dtype=np.int32)
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                c = p[i] + q[j]
                if c >= MAX_COLOR:
                    valid = False
                    break
                colors[IDX_MAP[(i,j)]] = c
            if not valid:
                break
        
        if not valid:
            continue
            
        # Check if it's a valid coloring (it should be by construction, but let's be sure)
        # The additive property ensures no 4 distinct colors in a cycle.
        # c_ab + c_cd = p_a+q_b + p_c+q_d
        # c_ac + c_bd = p_a+q_c + p_b+q_d
        # Sum is same. If all 4 distinct, sums would have to be different? 
        # Wait. The condition is "No four-cycle may have four distinct edge colors".
        # If c_ab, c_bc, c_cd, c_da are distinct, is that possible?
        # c_ab = p_a+q_b
        # c_bc = p_b+q_c
        # c_cd = p_c+q_d
        # c_da = p_d+q_a
        # If these are distinct, it's a violation.
        # Does additive structure prevent this?
        # Consider p=[0,1,2,3], q=[0,1,2,3].
        # c_01=1, c_12=3, c_23=5, c_30=3. (3 appears twice).
        # It is generally true that additive colorings avoid 4 distinct colors in cycles?
        # Let's verify with a counter-example search or just trust the math.
        # Actually, if p and q are arbitrary, it IS possible to have 4 distinct.
        # Example: p=[0,10,20,30], q=[0,1,2,3].
        # c_01=1, c_12=12, c_23=23, c_30=30. All distinct!
        # So we MUST check validity.
        
        if not _is_valid_full(colors):
            continue
            
        distinct = _count_distinct(colors)
        if distinct > best_distinct:
            best_distinct = distinct
            best_p = p.copy()
            best_q = q.copy()
            
        # Annealing
        T = T_start
        current_p = p.copy()
        current_q = q.copy()
        current_colors = colors.copy()
        current_distinct = distinct
        
        while time.time() < deadline and T > T_end:
            # Move: change one label
            if rng.random() < 0.5:
                v = rng.randint(0, N-1)
                delta = rng.randint(-5, 5)
                if delta == 0: continue
                new_val = current_p[v] + delta
                if new_val < 0 or new_val > 100: # Keep bounds reasonable
                    continue
                old_val = current_p[v]
                current_p[v] = new_val
                
                # Update colors and check validity
                # Only edges involving v change
                valid = True
                new_colors = current_colors.copy()
                for u in range(N):
                    if u == v: continue
                    idx = IDX_MAP[(min(u,v), max(u,v))]
                    new_c = new_val + current_q[u]
                    if new_c >= MAX_COLOR:
                        valid = False
                        break
                    new_colors[idx] = new_c
                if not valid:
                    current_p[v] = old_val
                    continue
                    
                # Check validity of new coloring
                # We only need to check cycles involving v
                # But _is_valid_full is fast enough for N=16? 1820 quads.
                # It might be a bit slow if called every step.
                # Let's optimize: check only cycles containing v.
                
                # Actually, let's just use _is_valid_full for simplicity first, 
                # but it might be too slow for 90s if we do thousands of steps.
                # Let's implement a faster check for local moves.
                
                # For now, let's estimate speed. 
                # If we do 1000 steps/sec, and each step takes 1ms, we do 1000 steps.
                # _is_valid_full takes ~1ms? 
                # Let's assume it's okay for a few hundred restarts/annealings.
                
                if not _is_valid_full(new_colors):
                    # Reject
                    current_p[v] = old_val
                    continue
                else:
                    # Accept or reject based on SA
                    new_distinct = _count_distinct(new_colors)
                    delta_obj = new_distinct - current_distinct
                    if delta_obj > 0 or rng.random() < np.exp(delta_obj / T):
                        # Accept
                        current_colors = new_colors
                        current_distinct = new_distinct
                        if current_distinct > best_distinct:
                            best_distinct = current_distinct
                            best_p = current_p.copy()
                            best_q = current_q.copy()
                    else:
                        # Reject
                        current_p[v] = old_val
            else:
                v = rng.randint(0, N-1)
                delta = rng.randint(-5, 5)
                if delta == 0: continue
                new_val = current_q[v] + delta
                if new_val < 0 or new_val > 100:
                    continue
                old_val = current_q[v]
                current_q[v] = new_val
                
                valid = True
                new_colors = current_colors.copy()
                for u in range(N):
                    if u == v: continue
                    idx = IDX_MAP[(min(u,v), max(u,v))]
                    new_c = current_p[u] + new_val
                    if new_c >= MAX_COLOR:
                        valid = False
                        break
                    new_colors[idx] = new_c
                if not valid:
                    current_q[v] = old_val
                    continue
                    
                if not _is_valid_full(new_colors):
                    current_q[v] = old_val
                    continue
                else:
                    new_distinct = _count_distinct(new_colors)
                    delta_obj = new_distinct - current_distinct
                    if delta_obj > 0 or rng.random() < np.exp(delta_obj / T):
                        current_colors = new_colors
                        current_distinct = new_distinct
                        if current_distinct > best_distinct:
                            best_distinct = current_distinct
                            best_p = current_p.copy()
                            best_q = current_q.copy()
                    else:
                        current_q[v] = old_val
                        
            T *= cooling
            
    return best_p, best_q, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    rng = random.Random(42)
    
    best_overall_colors = None
    best_overall_distinct = 0
    
    # 1. Try simple algebraic colorings (fast)
    for K in range(2, 25):
        colors = _initial_algebraic_coloring(K)
        if _is_valid_full(colors):
            distinct = _count_distinct(colors)
            if distinct > best_overall_distinct:
                best_overall_distinct = distinct
                best_overall_colors = colors.copy()
                
    # 2. Run SA on additive structure
    p, q, distinct_sa = _sa_search(start_time, deadline, rng)
    
    if p is not None and distinct_sa > best_overall_distinct:
        # Reconstruct colors
        colors = np.zeros(NEDGES, dtype=np.int32)
        for i in range(N):
            for j in range(i+1, N):
                colors[IDX_MAP[(i,j)]] = p[i] + q[j]
        
        # Final verification
        if _is_valid_full(colors):
            best_overall_distinct = distinct_sa
            best_overall_colors = colors.copy()
            
    if best_overall_colors is None:
        best_overall_colors = _initial_valid_coloring()
        
    return best_overall_colors.tolist()
