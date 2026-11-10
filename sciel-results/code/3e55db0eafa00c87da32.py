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
    
    # Convert line_points to lists for faster iteration
    line_points_list = [list(s) for s in line_points]
    
    # Precompute for each point, the set of points on the same line (excluding itself)
    # This is useful for checking witness lines
    # Actually, we can just use line_points_list and point_lines
    
    def calculate_cost(p_idx, W):
        """Calculate the number of witness lines that p_idx would invalidate if added to C.
        More precisely, count how many points x would lose at least one witness line.
        But for safety, we need to know if any point would lose ALL witness lines.
        So we track for each point x, how many witness lines it has (wcount).
        The 'damage' of adding p is the number of x such that p is on a witness line of x
        and wcount[x] == 1 (i.e., x would have 0 witness lines left).
        """
        damage = 0
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        if len(W[x_idx]) == 1:
                            damage += 1
        return damage
    
    def is_safe(p_idx, W):
        """Check if adding p_idx to C leaves all points with at least one witness."""
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        # If this is the only witness for x_idx, it's unsafe
                        if len(W[x_idx]) == 1:
                            return False
        return True
    
    def add_point_to_C(p_idx, in_C, W):
        """Add p_idx to C and update W."""
        in_C[p_idx] = True
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        W[x_idx].discard(lid)
    
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
    
    def compute_W(in_C):
        W = [set() for _ in range(q2)]
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
        return W
    
    def greedy_expand(in_C, W, rng, max_steps=10000):
        """Greedy expand C using a priority queue."""
        in_C = in_C.copy()
        W = [set(s) for s in W]
        
        # Build initial heap
        heap = []
        for i in range(q2):
            if not in_C[i]:
                c = calculate_cost(i, W)
                heapq.heappush(heap, (c, i))
        
        steps = 0
        while heap and steps < max_steps and time.time() < _DEADLINE - _MARGIN:
            steps += 1
            cost, p_idx = heapq.heappop(heap)
            
            if in_C[p_idx]:
                continue
                
            # Verify cost is still accurate
            current_cost = calculate_cost(p_idx, W)
            if current_cost != cost:
                heapq.heappush(heap, (current_cost, p_idx))
                continue
                
            # Check safety
            if is_safe(p_idx, W):
                add_point_to_C(p_idx, in_C, W)
                
                # Update heap for affected points
                for lid in point_lines[p_idx]:
                    for x_idx in line_points_list[lid]:
                        if x_idx != p_idx and not in_C[x_idx]:
                            new_c = calculate_cost(x_idx, W)
                            heapq.heappush(heap, (new_c, x_idx))
        
        return in_C, W
    
    def local_search_swap(in_C, W, rng, max_iters=5000):
        """Try to swap points in C to allow more additions."""
        in_C = in_C.copy()
        W = [set(s) for s in W]
        
        c_points = [i for i in range(q2) if in_C[i]]
        n_points = [i for i in range(q2) if not in_C[i]]
        
        for _ in range(max_iters):
            if time.time() > _DEADLINE - _MARGIN:
                break
            if not c_points or not n_points:
                break
            
            # Pick a random point from C to try removing
            idx = rng.randrange(len(c_points))
            p_idx = c_points[idx]
            
            # Try removing p_idx from C
            # This is always safe, but we want to see if it allows adding more points
            # Remove p_idx
            in_C[p_idx] = False
            c_points.remove(p_idx)
            n_points.append(p_idx)
            
            # Update W: for each line through p_idx, p_idx is no longer in C
            # So lines that previously had p_idx in them might now be valid witnesses
            # for points on those lines
            for lid in point_lines[p_idx]:
                for x_idx in line_points_list[lid]:
                    if x_idx != p_idx:
                        # Check if lid is now a valid witness for x_idx
                        # i.e., no other point in C is on lid (except x_idx if x_idx in C)
                        valid = True
                        for q_idx in line_points_list[lid]:
                            if q_idx != x_idx and in_C[q_idx]:
                                valid = False
                                break
                        if valid:
                            W[x_idx].add(lid)
            
            # Now try to greedily add points from n_points
            added = 0
            heap = []
            for i in n_points:
                if not in_C[i]:
                    c = calculate_cost(i, W)
                    heapq.heappush(heap, (c, i))
            
            while heap:
                cost, p_add = heapq.heappop(heap)
                if in_C[p_add]:
                    continue
                current_cost = calculate_cost(p_add, W)
                if current_cost != cost:
                    heapq.heappush(heap, (current_cost, p_add))
                    continue
                if is_safe(p_add, W):
                    add_point_to_C(p_add, in_C, W)
                    if p_add in n_points:
                        n_points.remove(p_add)
                    c_points.append(p_add)
                    added += 1
                    # Update heap
                    for lid in point_lines[p_add]:
                        for x_idx in line_points_list[lid]:
                            if x_idx != p_add and not in_C[x_idx]:
                                new_c = calculate_cost(x_idx, W)
                                heapq.heappush(heap, (new_c, x_idx))
                else:
                    break
            
            # If we didn't add anything, put p_idx back
            if added == 0:
                in_C[p_idx] = True
                c_points.append(p_idx)
                if p_idx in n_points:
                    n_points.remove(p_idx)
                # Revert W changes (this is complex, so just recompute)
                # For now, skip revert and just keep the state if it's valid
                # Actually, let's just recompute W from scratch if we revert
                # But that's expensive. Let's just accept the state if valid.
                # Check validity
                valid = True
                for i in range(q2):
                    if len(W[i]) == 0:
                        valid = False
                        break
                if not valid:
                    # Revert
                    in_C[p_idx] = True
                    if p_idx not in c_points:
                        c_points.append(p_idx)
                    if p_idx in n_points:
                        n_points.remove(p_idx)
                    # Recompute W
                    W = compute_W(in_C)
        
        return in_C, W
    
    best_in_C = init_C_parabola()
    best_size = int(np.sum(best_in_C))
    
    rng = random.Random(42)
    
    for trial in range(30):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        in_C = init_C_parabola()
        W = compute_W(in_C)
        
        in_C, W = greedy_expand(in_C, W, rng, max_steps=10000)
        
        # Try local search swaps
        in_C, W = local_search_swap(in_C, W, rng, max_iters=2000)
        
        # Final greedy expand
        in_C, W = greedy_expand(in_C, W, rng, max_steps=5000)
        
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
