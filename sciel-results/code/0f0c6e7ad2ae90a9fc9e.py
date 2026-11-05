"""Strong algorithm for bounded-lcm set maximization on [1,512]."""
import time
from math import gcd, isqrt

N = 512
DEADLINE = 85.0

def _lcm(a, b):
    return a * b // gcd(a, b)

def _validate(s):
    s = set(s)
    if not s:
        return False
    for x in s:
        if x < 1 or x > N:
            return False
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if _lcm(lst[i], lst[j]) > N:
                return False
    return True

def _divisors_of(n, limit=N):
    res = set()
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            if i <= limit:
                res.add(i)
            j = n // i
            if j != i and j <= limit:
                res.add(j)
    return res

def construct():
    start = time.time()

    # Precompute LCM compatibility as bitmasks
    # compat_mask[v] has bit (u-1) set iff lcm(v,u) <= N
    compat_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        mask = 0
        for u in range(1, N + 1):
            if _lcm(v, u) <= N:
                mask |= (1 << (u - 1))
        compat_mask[v] = mask

    def is_compatible(v, S_mask):
        return (S_mask & ~compat_mask[v]) == 0

    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << (v - 1))
        return m

    def mask_to_set(m):
        s = set()
        r = m
        while r:
            lb = r & (-r)
            s.add(lb.bit_length())
            r &= r - 1
        return s

    def greedy_by_degree(seed_set):
        S = set(seed_set)
        S_mask = set_to_mask(S)
        remaining_mask = 0
        for v in range(1, N + 1):
            if v not in S and is_compatible(v, S_mask):
                remaining_mask |= (1 << (v - 1))
        
        while remaining_mask:
            best_v = None
            best_deg = -1
            r = remaining_mask
            while r:
                lb = r & (-r)
                v = lb.bit_length()
                deg = (compat_mask[v] & remaining_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                r &= r - 1
            
            if best_v is None:
                break
            S.add(best_v)
            S_mask |= (1 << (best_v - 1))
            remaining_mask &= compat_mask[best_v]
            remaining_mask &= ~(1 << (best_v - 1))
        
        return S

    def greedy_augment(seed_set):
        S = set(seed_set)
        S_mask = set_to_mask(S)
        changed = True
        while changed:
            changed = False
            for v in range(1, N + 1):
                if v in S:
                    continue
                if is_compatible(v, S_mask):
                    S.add(v)
                    S_mask |= (1 << (v - 1))
                    changed = True
        return S

    # Find best initial solution
    best_S = set()
    best_size = 0

    # Try divisor sets of highly composite numbers
    M_sorted = sorted(range(1, N + 1), key=lambda m: len(_divisors_of(m, N)), reverse=True)
    tried_M = set()
    for M in M_sorted[:40]:
        if time.time() - start > DEADLINE - 25:
            break
        if M in tried_M:
            continue
        tried_M.add(M)
        d = _divisors_of(M, N)
        if not d:
            continue
        S1 = greedy_augment(d)
        if len(S1) > best_size:
            best_size = len(S1)
            best_S = S1
        S2 = greedy_by_degree(d)
        if len(S2) > best_size:
            best_size = len(S2)
            best_S = S2

    # Try from empty
    if time.time() - start < DEADLINE - 30:
        S3 = greedy_by_degree(set())
        if len(S3) > best_size:
            best_size = len(S3)
            best_S = S3

    # Local search: try to improve by removing one and adding more
    def local_search(S, max_iters=200):
        S = set(S)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() - start < DEADLINE - 15:
            improved = False
            iter_count += 1
            S_mask = set_to_mask(S)
            for v in sorted(S):
                S_no_v_mask = S_mask & ~(1 << (v - 1))
                # Find candidates compatible with S\{v}
                cand_mask = 0
                for u in range(1, N + 1):
                    if u != v and is_compatible(u, S_no_v_mask):
                        cand_mask |= (1 << (u - 1))
                
                # Try single additions
                r = cand_mask
                while r:
                    lb = r & (-r)
                    u = lb.bit_length()
                    if u != v:
                        S.add(u)
                        S.remove(v)
                        improved = True
                        break
                    r &= r - 1
                if improved:
                    break
                
                # Try double additions: remove v, add two
                cand_list = []
                r = cand_mask
                while r:
                    lb = r & (-r)
                    u = lb.bit_length()
                    if u != v:
                        cand_list.append(u)
                    r &= r - 1
                
                if len(cand_list) >= 2:
                    found = False
                    for i in range(len(cand_list)):
                        if time.time() - start > DEADLINE - 15:
                            break
                        for j in range(i + 1, len(cand_list)):
                            u1, u2 = cand_list[i], cand_list[j]
                            if _lcm(u1, u2) <= N:
                                S.add(u1)
                                S.add(u2)
                                S.remove(v)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if improved:
                        break
        return S

    # Apply local search
    if time.time() - start < DEADLINE - 20:
        improved_S = local_search(best_S, max_iters=200)
        if len(improved_S) > best_size:
            best_size = len(improved_S)
            best_S = improved_S

    # Branch and bound for maximum independent set
    # Conflict graph: edge between u,v if lcm(u,v) > N
    # We want max independent set = max subset with no conflicts
    # Equivalent: max clique in complement graph (compatibility graph)
    
    def greedy_color_bound(remaining_mask):
        """Compute upper bound on clique size in remaining graph using greedy coloring.
        In the compatibility graph, we're looking for max clique.
        Upper bound: color the vertices of remaining subgraph such that adjacent
        vertices get different colors. Number of colors = upper bound on clique size.
        """
        if not remaining_mask:
            return 0
        
        # Greedy coloring: repeatedly pick a vertex, assign it the smallest
        # available color, remove it and its neighbors from consideration
        # Actually for upper bound on max clique, we use:
        # Greedily color the vertices so that no two adjacent (in compatibility)
        # vertices share a color. The number of colors used is an upper bound
        # on the clique number (since a clique needs all distinct colors).
        
        # Simpler approach: greedy coloring of the remaining subgraph
        # where "adjacent" means compatible (edge in compatibility graph)
        # Upper bound on max clique = min number of colors needed
        # We use a greedy algorithm to get an upper bound
        
        colors = {}
        uncolored = remaining_mask
        color_count = 0
        
        while uncolored:
            color_count += 1
            # Pick a vertex from uncolored
            lb = uncolored & (-uncolored)
            v = lb.bit_length()
            # Remove v and all its compatible neighbors from uncolored
            # (they can't share this color)
            neighbors = compat_mask[v] & uncolored
            uncolored &= ~(neighbors | (1 << (v - 1)))
        
        return color_count

    def branch_and_bound(initial_mask, current_size, deadline_margin=10):
        """Branch and bound to find max independent set (max clique in compat graph).
        initial_mask: bitmask of vertices still available
        current_size: size of the current solution found so far
        Returns the best set found.
        """
        best_set = set()
        best_sz = current_size
        nodes_explored = 0
        
        # Order vertices by degree in the remaining graph for better pruning
        def get_ordered_vertices(mask):
            """Return vertices in order of decreasing degree within mask."""
            verts = []
            r = mask
            while r:
                lb = r & (-r)
                v = lb.bit_length()
                deg = (compat_mask[v] & mask).bit_count()
                verts.append((deg, v))
                r &= r - 1
            verts.sort(key=lambda x: -x[0])
            return [v for _, v in verts]
        
        def bb_search(available_mask, selected_set, selected_mask):
            nonlocal best_set, best_sz, nodes_explored
            
            if time.time() - start > DEADLINE - deadline_margin:
                return
            
            if not available_mask:
                if len(selected_set) > best_sz:
                    best_sz = len(selected_set)
                    best_set = set(selected_set)
                return
            
            # Upper bound: current selected + greedy color bound on remaining
            ub = len(selected_set) + greedy_color_bound(available_mask)
            if ub <= best_sz:
                return
            
            nodes_explored += 1
            
            # Pick a vertex with high degree in the available subgraph
            # to branch on
            best_v = None
            best_deg = -1
            r = available_mask
            while r:
                lb = r & (-r)
                v = lb.bit_length()
                deg = (compat_mask[v] & available_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                r &= r - 1
            
            if best_v is None:
                return
            
            v = best_v
            bit = 1 << (v - 1)
            
            # Branch 1: include v
            new_selected = set(selected_set)
            new_selected.add(v)
            new_selected_mask = selected_mask | bit
            # Remaining vertices must be compatible with v
            new_available = available_mask & compat_mask[v] & ~bit
            bb_search(new_available, new_selected, new_selected_mask)
            
            # Branch 2: exclude v
            bb_search(available_mask & ~bit, selected_set, selected_mask)
        
        bb_search(initial_mask, set(), 0)
        return best_set

    # Run B&B if we have time
    if time.time() - start < DEADLINE - 20:
        # Start from the best solution found so far
        # Use B&B on the full graph with the current best as lower bound
        full_mask = 0
        for v in range(1, N + 1):
            full_mask |= (1 << (v - 1))
        
        bb_result = branch_and_bound(full_mask, best_size, deadline_margin=10)
        if len(bb_result) > best_size:
            best_size = len(bb_result)
            best_S = bb_result

    # Final local search polish
    if time.time() - start < DEADLINE - 5:
        improved_S = local_search(best_S, max_iters=100)
        if len(improved_S) > best_size:
            best_size = len(improved_S)
            best_S = improved_S

    # Validate
    if not _validate(best_S):
        best_S = set(range(1, 21))

    return list(best_S)
