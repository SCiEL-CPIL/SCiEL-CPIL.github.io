import numpy as np
import time
import random
import heapq

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 10.0

def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    
    # Precompute field operations
    mul_table = np.zeros((q, q), dtype=np.int32)
    add_table = np.zeros((q, q), dtype=np.int32)
    sub_table = np.zeros((q, q), dtype=np.int32)
    inv_table = np.zeros(q, dtype=np.int32)
    
    for i in range(q):
        for j in range(q):
            a1, b1 = i // p, i % p
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + 3 * b1 * b2) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
            
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add_table[i, j] = ra * p + rb
            
            ra = (a1 - a2) % p
            rb = (b1 - b2) % p
            sub_table[i, j] = ra * p + rb
            
    for i in range(1, q):
        a, b = i // p, i % p
        denom = (a * a - 3 * b * b) % p
        denom_inv = pow(denom, 5, 7)
        ia = (a * denom_inv) % p
        ib = (-b * denom_inv) % p
        inv_table[i] = ia * p + ib
    inv_table[0] = 0
    
    # Build lines
    # Line ID: m*49 + c for non-vertical, 2401 + x0 for vertical
    point_lines = [[] for _ in range(q2)]
    line_points = [set() for _ in range(2450)]
    
    for xi in range(q):
        for yi in range(q):
            pi = xi * q + yi
            vl = 2401 + xi
            point_lines[pi].append(vl)
            line_points[vl].add(pi)
            
            for m in range(q):
                mc = sub_table[yi, mul_table[m, xi]]
                nl = m * q + mc
                point_lines[pi].append(nl)
                line_points[nl].add(pi)
    
    line_points_list = [list(s) for s in line_points]
    # Convert line_points to tuples for faster iteration if needed, but lists are fine

    def init_C_parabola():
        in_C = np.zeros(q2, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                ya = (xa * xa + 3 * xb * xb) % p
                yb = (2 * xa * xb) % p
                x_idx = xa * p + xb
                y_idx = ya * p + yb
                pi = x_idx * q + y_idx
                in_C[pi] = True
        return in_C

    def compute_W_and_wcount(in_C):
        W = [set() for _ in range(q2)]
        wcount = np.zeros(q2, dtype=np.int32)
        for i in range(q2):
            for lid in point_lines[i]:
                lp = line_points_list[lid]
                valid = True
                for q_idx in lp:
                    if q_idx != i and in_C[q_idx]:
                        valid = False
                        break
                if valid:
                    W[i].add(lid)
                    wcount[i] += 1
        return W, wcount

    def try_greedy_heap(in_C, W, wcount, rng, max_time=10.0):
        in_C = in_C.copy()
        W = [set(s) for s in W]
        wcount = wcount.copy()
        
        # Build initial heap
        heap = []
        for i in range(q2):
            if not in_C[i]:
                # Cost: number of witness lines destroyed
                # We use a simple cost: number of points x for which lid is a witness
                # This is an approximation for "damage".
                # Actually, we want to prioritize points that destroy FEW witness lines.
                cost = 0
                for lid in point_lines[i]:
                    for x_idx in line_points_list[lid]:
                        if x_idx != i and lid in W[x_idx]:
                            cost += 1
                heapq.heappush(heap, (cost, i))
                
        start_time = time.time()
        
        while heap:
            if time.time() - start_time > max_time or time.time() > _DEADLINE - _MARGIN:
                break
                
            cost, p_idx = heapq.heappop(heap)
            
            if in_C[p_idx]:
                continue
                
            # Verify cost is still accurate
            current_cost = 0
            safe = True
            # We need to check if it's safe (no wcount becomes 0)
            # And calculate the "true" cost for the heap if we were to re-push it
            # For now, just check safety.
            for lid in point_lines[p_idx]:
                for x_idx in line_points_list[lid]:
                    if x_idx != p_idx:
                        if lid in W[x_idx]:
                            current_cost += 1
                            if wcount[x_idx] == 1:
                                safe = False
                                # We can break early if unsafe, but we might still want to know cost?
                                # No, if unsafe, we skip.
                                # But we might want to re-evaluate later?
                                # For simplicity, if unsafe, we discard p_idx for this step.
                                # It might become safe later if other points are added/removed?
                                # Adding points only REMOVES witness lines. So if it's unsafe now, it will remain unsafe or become more unsafe?
                                # Wait. Adding a point p to C destroys witness lines for others.
                                # This can only REDUCE wcount.
                                # So if wcount[x] == 1 and L is a witness, and we add p, wcount[x] becomes 0.
                                # If we add OTHER points first, wcount[x] might become 0 for OTHER reasons.
                                # So if it's unsafe now, it will definitely be unsafe later (wcount can only decrease).
                                # Therefore, if p is unsafe now, it will NEVER be safe to add it later.
                                # We can permanently discard p_idx.
                                break
                if not safe:
                    break
            
            if not safe:
                continue
            
            # If safe, we can add p_idx.
            # But is it the BEST choice?
            # The heap order is based on initial cost.
            # Since wcount only decreases, the "danger" of a point might increase?
            # No, the cost (number of witnesses destroyed) is static in terms of "which lines are witnesses".
            # But the "safety" depends on wcount.
            # If a point was safe, it remains safe?
            # If we add another point q, it might destroy a witness for x, reducing wcount[x].
            # If wcount[x] was 2, and q destroys one, it becomes 1.
            # Then p, which destroys the other one, becomes UNSAFE.
            # So a point that is safe now might become unsafe later.
            # This means we cannot just greedily pick the lowest cost point and add it, because it might block future better points?
            # Actually, the standard greedy approach for this problem (Maximizing C) is:
            # Pick a point that is safe to add. Among safe points, pick one that minimizes "loss" or "damage".
            # Damage = number of witness lines destroyed.
            # This is what the heap does.
            # The issue is that the cost changes as W changes.
            # So we must update the heap.
            
            # Since we are using a lazy heap, we push updated costs for neighbors.
            
            # Commit the addition
            in_C[p_idx] = True
            for lid in point_lines[p_idx]:
                for x_idx in line_points_list[lid]:
                    if x_idx != p_idx:
                        if lid in W[x_idx]:
                            W[x_idx].discard(lid)
                            wcount[x_idx] -= 1
                            # If wcount[x_idx] becomes 0, we have a problem.
                            # But we checked safety, so this shouldn't happen.
                            
                            # Update heap for x_idx?
                            # x_idx is in C? No, x_idx is a point in the plane.
                            # We only add points NOT in C.
                            # So x_idx is a candidate if not in_C[x_idx].
                            if not in_C[x_idx]:
                                # Recalculate cost for x_idx
                                new_cost = 0
                                # Optimization: we can just push it with a high cost and let lazy eval handle it?
                                # Or calculate it.
                                for lid2 in point_lines[x_idx]:
                                    for y_idx in line_points_list[lid2]:
                                        if y_idx != x_idx and lid2 in W[y_idx]:
                                            new_cost += 1
                                heapq.heappush(heap, (new_cost, x_idx))
            
        return in_C, W, wcount

    best_in_C = init_C_parabola()
    best_size = int(np.sum(best_in_C))
    
    rng = random.Random(42)
    
    # Run multiple trials
    # Each trial gets a time slice.
    num_trials = 20
    time_per_trial = 25.0 # 20 * 25 = 500s, leaving 100s margin
    
    for trial in range(num_trials):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        in_C = init_C_parabola()
        W, wcount = compute_W_and_wcount(in_C)
        
        in_C, W, wcount = try_greedy_heap(in_C, W, wcount, rng, max_time=time_per_trial)
        
        cur_size = int(np.sum(in_C))
        if cur_size > best_size:
            best_size = cur_size
            best_in_C = in_C.copy()
            
    # Construct N = complement of C
    in_N = ~best_in_C
    
    N = []
    for i in range(q2):
        if in_N[i]:
            x = i // q
            y = i % q
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)
