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
    
    def init_parabola():
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

    def compute_witness(in_C):
        """
        Returns a list of sets, W[i] contains line IDs that are valid witnesses for point i.
        A line L is a witness for i if i in L and (L \ {i}) is disjoint from C.
        """
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

    def is_valid(in_C, W):
        for i in range(q2):
            if len(W[i]) == 0:
                return False
        return True

    def calculate_cost(p_idx, W):
        """Calculate the number of witness lines that p_idx would invalidate."""
        cost = 0
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    if lid in W[x_idx]:
                        cost += 1
        return cost

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

    def remove_point_from_C(p_idx, in_C, W):
        """Remove p_idx from C and update W."""
        in_C[p_idx] = False
        for lid in point_lines[p_idx]:
            for x_idx in line_points_list[lid]:
                if x_idx != p_idx:
                    # Check if lid becomes a valid witness for x_idx
                    lp = line_points_list[lid]
                    valid = True
                    for q_idx in lp:
                        if q_idx != x_idx and in_C[q_idx]:
                            valid = False
                            break
                    if valid:
                        W[x_idx].add(lid)

    def local_search(in_C, W, rng, max_iters=2000):
        """
        Try to increase |C| by removing a point from C and adding two or more points.
        Or try swaps that maintain validity but allow further expansion.
        """
        in_C = in_C.copy()
        W = [set(s) for s in W]
        
        if not is_valid(in_C, W):
            return in_C, W
        
        best_size = int(np.sum(in_C))
        
        for _ in range(max_iters):
            if time.time() > _DEADLINE - _MARGIN:
                break
                
            # Try to find a point in C that can be removed to allow more additions
            # or try a random swap
            if rng.random() < 0.3:
                # Try removing a random point from C and see if we can add more
                c_points = [i for i in range(q2) if in_C[i]]
                if not c_points:
                    break
                rem = rng.choice(c_points)
                
                # Remove rem
                remove_point_from_C(rem, in_C, W)
                
                if not is_valid(in_C, W):
                    # Invalid, restore
                    add_point_to_C(rem, in_C, W)
                    continue
                
                # Try to add as many points as possible
                added = 0
                heap = []
                for i in range(q2):
                    if not in_C[i]:
                        c = calculate_cost(i, W)
                        heapq.heappush(heap, (c, i))
                
                while heap:
                    if time.time() > _DEADLINE - _MARGIN:
                        break
                    cost, p_idx = heapq.heappop(heap)
                    if in_C[p_idx]:
                        continue
                    current_cost = calculate_cost(p_idx, W)
                    if current_cost != cost:
                        heapq.heappush(heap, (current_cost, p_idx))
                        continue
                    if is_safe(p_idx, W):
                        add_point_to_C(p_idx, in_C, W)
                        added += 1
                        for lid in point_lines[p_idx]:
                            for x_idx in line_points_list[lid]:
                                if x_idx != p_idx and not in_C[x_idx]:
                                    new_c = calculate_cost(x_idx, W)
                                    heapq.heappush(heap, (new_c, x_idx))
                
                cur_size = int(np.sum(in_C))
                if cur_size > best_size:
                    best_size = cur_size
                    # Keep this state
                else:
                    # Restore
                    # This is complex, so we just revert
                    # For simplicity, we'll restart if not improved
                    break
            else:
                # Try a random swap: remove one from C, add one not in C
                c_points = [i for i in range(q2) if in_C[i]]
                nc_points = [i for i in range(q2) if not in_C[i]]
                if not c_points or not nc_points:
                    break
                rem = rng.choice(c_points)
                add = rng.choice(nc_points)
                
                # Remove rem
                remove_point_from_C(rem, in_C, W)
                
                if not is_valid(in_C, W):
                    add_point_to_C(rem, in_C, W)
                    continue
                
                # Try to add
                if is_safe(add, W):
                    add_point_to_C(add, in_C, W)
                    cur_size = int(np.sum(in_C))
                    if cur_size > best_size:
                        best_size = cur_size
                    # Keep the swap
                else:
                    # Can't add, restore
                    add_point_to_C(rem, in_C, W)
        
        return in_C, W

    best_in_C = None
    best_size = -1
    
    rng = random.Random(42)
    
    for trial in range(100):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        rng.seed(42 + trial)
        
        # Initialize C with Parabola
        in_C = init_parabola()
        
        # Initialize W
        W = compute_witness(in_C)
        
        if not is_valid(in_C, W):
            continue
        
        # Build Heap
        heap = []
        for i in range(q2):
            if not in_C[i]:
                c = calculate_cost(i, W)
                heapq.heappush(heap, (c, i))
                
        # Greedy Expansion with Priority Queue
        while heap:
            if time.time() > _DEADLINE - _MARGIN:
                break
                
            cost, p_idx = heapq.heappop(heap)
            
            if in_C[p_idx]:
                continue
                
            # Verify cost is still accurate (Lazy Update)
            current_cost = calculate_cost(p_idx, W)
            if current_cost != cost:
                heapq.heappush(heap, (current_cost, p_idx))
                continue
                
            # Check safety
            if is_safe(p_idx, W):
                add_point_to_C(p_idx, in_C, W)
                
                # Push neighbors back into heap
                for lid in point_lines[p_idx]:
                    for x_idx in line_points_list[lid]:
                        if x_idx != p_idx and not in_C[x_idx]:
                            new_c = calculate_cost(x_idx, W)
                            heapq.heappush(heap, (new_c, x_idx))
        
        # Local search phase
        in_C, W = local_search(in_C, W, rng, max_iters=500)
        
        cur_size = int(np.sum(in_C))
        if cur_size > best_size:
            best_size = cur_size
            best_in_C = in_C.copy()
            
        if time.time() > _DEADLINE - _MARGIN:
            break
    
    # If no valid solution found, fall back to empty set (N = all points)
    if best_in_C is None:
        best_in_C = np.zeros(q2, dtype=bool)
    
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
