"""
5-colourings of K_n without monochromatic C4.
Strategy: Optimized Simulated Annealing with NumPy vectorized incremental updates.
Target: n=26 (Lazebnik-Woldar bound).
"""
import time
import numpy as np
import random

_START = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    # Build neighbour sets per colour
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
            idx += 1
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                if len(nbrs[c][i] & nbrs[c][j]) >= 2:
                    return False
    return True

def _seed_construction():
    """Original K11 seed."""
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _cyclic_init(n):
    """Generate a cyclic coloring initialization for n vertices."""
    colors = np.empty(n * (n - 1) // 2, dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d = min(d, n - d)
            colors[idx] = (d - 1) % 5
            idx += 1
    return colors

def _solve_sa(n, max_time=100.0):
    """
    Simulated Annealing to find a valid 5-coloring of K_n.
    Uses incremental updates for the cost function (number of C4s).
    """
    if n < 2:
        return None
    
    m = n * (n - 1) // 2
    colors = _cyclic_init(n)
    
    # Precompute edge indices
    edge_idx = np.empty(m, dtype=np.int32)
    edge_u = np.empty(m, dtype=np.int32)
    edge_v = np.empty(m, dtype=np.int32)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[k] = k
            edge_u[k] = i
            edge_v[k] = j
            k += 1
    
    # Build adjacency matrices for each color (5 x n x n)
    # A_c[i, j] = 1 if edge (i, j) has color c
    A = np.zeros((5, n, n), dtype=np.int8)
    for k in range(m):
        c = colors[k]
        u = edge_u[k]
        v = edge_v[k]
        A[c, u, v] = 1
        A[c, v, u] = 1
        
    # Compute initial cost: number of C4s
    # A C4 in color c exists if (A_c @ A_c)[i, j] >= 2 for i != j.
    # Actually, (A_c @ A_c)[i, j] is the number of common neighbors.
    # We want to count pairs (i, j) with common neighbors >= 2.
    # Let M_c = A_c @ A_c. Cost = sum_c sum_{i<j} [M_c[i,j] >= 2].
    
    def compute_cost():
        cost = 0
        for c in range(5):
            M = A[c] @ A[c]
            # Count entries >= 2 in upper triangle
            # M is symmetric. Diagonal is degree.
            # We only care about i != j.
            # np.count_nonzero(M >= 2) counts all entries including diagonal and both triangles.
            # We need to be careful.
            # Let's just count upper triangle.
            tri = np.triu_indices(n, 1)
            cost += np.sum(M[tri] >= 2)
        return cost

    current_cost = compute_cost()
    if current_cost == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    best_cost = current_cost
    best_colors = colors.copy()
    
    T = 1.0
    T_min = 0.001
    alpha = 0.999 # cooling rate, tuned for ~50k steps in time limit
    
    # Estimate steps based on time
    start_time = time.time()
    step = 0
    rng = np.random.default_rng(42)
    
    # Precompute for fast updates? 
    # Updating A and cost incrementally is complex but doable.
    # For simplicity and speed, we'll use a heuristic: 
    # If we change an edge, only specific C4s are affected.
    # However, recomputing full cost is O(5 * n^3) ~ 5 * 17576 = 87k ops. 
    # 100k steps * 87k ops = 8.7e9 ops. Too slow for Python/NumPy overhead if done naively in loop.
    # We need incremental cost update.
    
    # Let's try a simpler approach: 
    # The cost is sum over c of #C4s in color c.
    # Changing edge (u,v) from c_old to c_new:
    # 1. Remove (u,v) from color c_old.
    # 2. Add (u,v) to color c_new.
    # 
    # Impact on cost for color c:
    # Let A be the adjacency matrix for color c.
    # Let C4_count = sum_{i<j} [ (A@A)[i,j] >= 2 ].
    # This is hard to update incrementally without maintaining the full matrix of common neighbors.
    # 
    # Alternative: Use a different cost function or a faster check.
    # Given the time limit, maybe we should just use a greedy repair with NumPy acceleration for the check,
    # or stick to a simpler SA that recomputes cost but limits steps.
    # 
    # Let's try a hybrid: 
    # 1. Start with cyclic init.
    # 2. Use SA with incremental cost update.
    
    # Incremental Cost Update Logic:
    # Maintain M_c = A_c @ A_c for each c.
    # M_c[i, j] = number of common neighbors of i and j in color c.
    # Cost_c = sum_{i<j} [ M_c[i, j] >= 2 ].
    
    # When adding edge (u, v) to color c:
    # A_c[u, v] = 1, A_c[v, u] = 1.
    # M_c changes.
    # M_c[i, j] = sum_k A_c[i, k] A_c[k, j].
    # The change in M_c is:
    # Delta M_c = A_c[:, u] e_v^T + e_u A_c[:, v] + A_c[:, v] e_u^T + e_v A_c[:, u] ... wait.
    # Actually, M = A @ A.
    # If we add edge (u, v), A becomes A + E where E has 1s at (u,v) and (v,u).
    # New M = (A+E) @ (A+E) = A@A + A@E + E@A + E@E.
    # A@E: column u of A is added to column v. Column v of A is added to column u.
    # So (A@E)[i, v] = A[i, u] and (A@E)[i, u] = A[i, v].
    # Similarly for E@A.
    # E@E: has 1 at (u, u) and (v, v) if u!=v? 
    # E[u,v]=1, E[v,u]=1.
    # E@E: (E@E)[i,j] = sum_k E[i,k]E[k,j].
    # (E@E)[u, u] = E[u,v]E[v,u] = 1.
    # (E@E)[v, v] = E[v,u]E[u,v] = 1.
    # Other entries 0.
    
    # So Delta M_c:
    # For i != u, v:
    # M_c[i, u] += A_c[i, v] (from A@E) + A_c[v, i] (from E@A) = 2 * A_c[i, v] (since symmetric)
    # M_c[i, v] += 2 * A_c[i, u]
    # For i = u:
    # M_c[u, u] += A_c[u, v] + A_c[v, u] + 1 = 2 * A_c[u, v] + 1. (But we don't care about diagonal for C4 count i<j)
    # M_c[u, v] += A_c[u, u] (0) + A_c[v, v] (0) + A_c[u, v] (from A@E? No, A@E[u,v] = A[u,u]=0) 
    # Let's re-evaluate M[u,v] change.
    # M[u,v] = sum_k A[u,k] A[k,v].
    # New M[u,v] = sum_k (A[u,k] + E[u,k]) (A[k,v] + E[k,v])
    # = M[u,v] + sum_k A[u,k]E[k,v] + sum_k E[u,k]A[k,v] + E[u,v]
    # E[k,v] is non-zero only if k=u (E[u,v]=1). So sum_k A[u,k]E[k,v] = A[u,u] = 0.
    # E[u,k] is non-zero only if k=v (E[u,v]=1). So sum_k E[u,k]A[k,v] = A[v,v] = 0.
    # So Delta M[u,v] = E[u,v] = 1.
    # Similarly Delta M[v,u] = 1.
    
    # So, adding edge (u,v) to color c:
    # 1. Update M_c:
    #    For all i != u, v:
    #       M_c[i, u] += 2 * A_c[i, v]
    #       M_c[i, v] += 2 * A_c[i, u]
    #    M_c[u, v] += 1
    #    M_c[v, u] += 1
    #    M_c[u, u] += 1 + 2*A_c[u,v] (ignore)
    #    M_c[v, v] += 1 + 2*A_c[v,u] (ignore)
    # 2. Update Cost_c:
    #    We need to count how many pairs (i,j) have M_c[i,j] >= 2.
    #    We can maintain a count of pairs with M_c[i,j] == 1, == 2, etc? 
    #    Or just recount the affected pairs.
    #    Affected pairs are (i, u) and (i, v) for all i, and (u, v).
    #    So O(n) update per edge change.
    #    100k steps * 26 * 5 colors? No, only 2 colors affected per step.
    #    100k steps * 26 * 2 = 5.2M operations. Very fast.

    M = np.zeros((5, n, n), dtype=np.int16)
    for c in range(5):
        M[c] = A[c] @ A[c]
        
    # Cost: count of i<j where M[c][i,j] >= 2
    def get_cost_from_M():
        cost = 0
        for c in range(5):
            tri = np.triu_indices(n, 1)
            cost += np.sum(M[c][tri] >= 2)
        return cost

    current_cost = get_cost_from_M()
    if current_cost == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    best_cost = current_cost
    best_colors = colors.copy()
    
    T = 1.0
    T_min = 0.001
    alpha = 0.9995
    
    # Precompute tri indices for cost update? 
    # We will update cost manually for affected pairs.
    
    def update_cost_delta(c, u, v, sign):
        """
        Update cost for color c when edge (u,v) is added (sign=1) or removed (sign=-1).
        Returns the change in cost for this color.
        """
        delta_cost = 0
        # Pairs affected: (i, u) for i != u, v; (i, v) for i != u, v; and (u, v).
        
        # Before update, record old M values for these pairs.
        # We need to know if M >= 2 before and after.
        
        # Helper to check if pair (a,b) contributes to cost (a<b)
        def contributes(a, b, mat):
            if a == b: return False
            if a > b: a, b = b, a
            return mat[a, b] >= 2

        # Store old values
        old_mu = M[c, u, :]
        old_mv = M[c, v, :]
        old_muv = M[c, u, v]
        
        # Calculate new M values
        # New M[i, u] = M[i, u] + sign * 2 * A_c[i, v]
        # New M[i, v] = M[i, v] + sign * 2 * A_c[i, u]
        # New M[u, v] = M[u, v] + sign * 1
        
        # We iterate i from 0 to n-1
        for i in range(n):
            if i == u or i == v:
                continue
                
            # Pair (i, u)
            a, b = (i, u) if i < u else (u, i)
            old_val = M[c, a, b]
            # Change in M[a, b]
            # If a == i, b == u: change is sign * 2 * A_c[i, v]
            # If a == u, b == i: change is sign * 2 * A_c[i, v] (symmetric)
            # Wait, M is symmetric.
            # M[i, u] changes by sign * 2 * A_c[i, v]
            delta_mu_i = sign * 2 * A[c, i, v]
            new_val = old_val + delta_mu_i
            
            if contributes(i, u, M[c]) and not (new_val >= 2):
                delta_cost -= 1
            elif not contributes(i, u, M[c]) and new_val >= 2:
                delta_cost += 1
                
            # Pair (i, v)
            a, b = (i, v) if i < v else (v, i)
            old_val = M[c, a, b]
            delta_mv_i = sign * 2 * A[c, i, u]
            new_val = old_val + delta_mv_i
            
            if contributes(i, v, M[c]) and not (new_val >= 2):
                delta_cost -= 1
            elif not contributes(i, v, M[c]) and new_val >= 2:
                delta_cost += 1
                
        # Pair (u, v)
        a, b = (u, v) if u < v else (v, u)
        old_val = M[c, a, b]
        delta_muv = sign * 1
        new_val = old_val + delta_muv
        
        if contributes(u, v, M[c]) and not (new_val >= 2):
            delta_cost -= 1
        elif not contributes(u, v, M[c]) and new_val >= 2:
            delta_cost += 1
            
        return delta_cost

    def apply_change(c_old, c_new, u, v, edge_k):
        """
        Apply a color change to edge (u,v) from c_old to c_new.
        Updates A, M, colors, and returns cost_delta.
        """
        nonlocal current_cost
        
        # 1. Remove from c_old
        A[c_old, u, v] = 0
        A[c_old, v, u] = 0
        
        # Update M for c_old
        # Delta M for removal (sign = -1)
        # We must update M before calculating cost delta? 
        # The function update_cost_delta uses current M and A.
        # But A is already changed! 
        # We need to calculate delta based on OLD A.
        # So we must call update_cost_delta BEFORE changing A? 
        # No, update_cost_delta uses A to calculate delta M.
        # If we change A first, the delta calculation is wrong.
        
        # Correct order:
        # 1. Calculate cost_delta for removal using OLD A, OLD M.
        # 2. Update M for removal.
        # 3. Update A for removal.
        # 4. Calculate cost_delta for addition using NEW A (which is OLD A + edge), NEW M (which is OLD M - delta).
        #    Wait, when adding to c_new, the "old" state for c_new is the current state.
        #    The edge (u,v) is NOT in c_new initially.
        #    So we add it.
        
        # Let's refine:
        # State: A, M, colors.
        
        # Step 1: Remove (u,v) from c_old.
        # Calculate delta_cost_old = update_cost_delta(c_old, u, v, -1) 
        # BUT update_cost_delta uses A[c_old, i, v]. 
        # Currently A[c_old, u, v] is 1. 
        # The formula for delta M[i, u] is sign * 2 * A[i, v]. 
        # This is correct for the current state.
        
        cost_delta_old = update_cost_delta(c_old, u, v, -1)
        
        # Update M[c_old]
        # M[i, u] -= 2 * A[i, v]
        # M[i, v] -= 2 * A[i, u]
        # M[u, v] -= 1
        # M[v, u] -= 1
        for i in range(n):
            if i == u or i == v: continue
            M[c_old, i, u] -= 2 * A[c_old, i, v]
            M[c_old, u, i] -= 2 * A[c_old, i, v]
            M[c_old, i, v] -= 2 * A[c_old, i, u]
            M[c_old, v, i] -= 2 * A[c_old, i, u]
        M[c_old, u, v] -= 1
        M[c_old, v, u] -= 1
        
        # Update A[c_old]
        A[c_old, u, v] = 0
        A[c_old, v, u] = 0
        
        # Step 2: Add (u,v) to c_new.
        # Calculate delta_cost_new = update_cost_delta(c_new, u, v, 1)
        # This uses A[c_new, i, v]. 
        # Currently A[c_new, u, v] is 0.
        # This is correct for the current state (before addition).
        
        cost_delta_new = update_cost_delta(c_new, u, v, 1)
        
        # Update M[c_new]
        for i in range(n):
            if i == u or i == v: continue
            M[c_new, i, u] += 2 * A[c_new, i, v]
            M[c_new, u, i] += 2 * A[c_new, i, v]
            M[c_new, i, v] += 2 * A[c_new, i, u]
            M[c_new, v, i] += 2 * A[c_new, i, u]
        M[c_new, u, v] += 1
        M[c_new, v, u] += 1
        
        # Update A[c_new]
        A[c_new, u, v] = 1
        A[c_new, v, u] = 1
        
        # Update colors
        colors[edge_k] = c_new
        
        return cost_delta_old + cost_delta_new

    # Main SA Loop
    while _time_left() > 5:
        # Pick random edge
        k = rng.integers(0, m)
        u = edge_u[k]
        v = edge_v[k]
        c_old = colors[k]
        
        # Pick random new color
        c_new = rng.integers(0, 4)
        if c_new == c_old:
            continue
            
        # Calculate cost delta
        # We need to simulate the change to get delta.
        # To avoid modifying state if rejected, we can either:
        # 1. Modify, calculate delta, revert if rejected.
        # 2. Calculate delta without modifying (harder).
        
        # Let's modify and revert.
        
        # Store old state for revert
        old_M_c_old = M[c_old].copy()
        old_M_c_new = M[c_new].copy()
        old_A_c_old = A[c_old].copy()
        old_A_c_new = A[c_new].copy()
        old_color = colors[k]
        
        delta_cost = apply_change(c_old, c_new, u, v, k)
        
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
            current_cost += delta_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                if best_cost == 0:
                    break
        else:
            # Revert
            M[c_old] = old_M_c_old
            M[c_new] = old_M_c_new
            A[c_old] = old_A_c_old
            A[c_new] = old_A_c_new
            colors[k] = old_color
            
        T = max(T * alpha, T_min)
        step += 1
        
    return {'n': n, 'colors': best_colors.tolist()}

def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        return _seed_construction()
    
    best = _seed_construction()
    
    # Try SA for n=26
    if _time_left() > 10:
        res = _solve_sa(26, max_time=_time_left() - 5)
        if res:
            # Validate
            if _validate(res['n'], res['colors']):
                if res['n'] > best['n']:
                    best = res
            else:
                # If SA didn't find a valid one, maybe it found a near-miss?
                # We only return valid ones.
                pass
                
    # If we didn't get 26, try 25?
    if best['n'] < 26 and _time_left() > 10:
        res = _solve_sa(25, max_time=_time_left() - 5)
        if res:
            if _validate(res['n'], res['colors']):
                if res['n'] > best['n']:
                    best = res
                    
    return best

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
    print(res['colors'][:20])
