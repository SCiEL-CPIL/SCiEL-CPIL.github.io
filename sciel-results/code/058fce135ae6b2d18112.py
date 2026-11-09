import time
import random
import sys

def _is_valid_coloring(n, colors):
    """
    Verify if the coloring is valid.
    colors is a list of length n*(n-1)/2.
    """
    if n < 2:
        return True
    # Build adjacency bitmasks for each color
    # adj[c][v] is a bitmask of neighbors of v in color c
    adj = [[0] * n for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    # Check for C4: for each color, no two vertices should have 2+ common neighbors
    # This is equivalent to: for any u, v, if they have a common neighbor w, they cannot have another.
    # Actually, the condition "no two vertices with two common neighbours" is exactly "no C4".
    # For a fixed color c, let N(u) be the set of neighbors.
    # We need |N(u) \cap N(v)| <= 1 for all u != v.
    # If we have the bitmask adj[c][u], we can compute intersection.
    for c in range(5):
        for u in range(n):
            mask_u = adj[c][u]
            # Iterate through set bits in mask_u
            # To check if any pair in mask_u has a common neighbor, we can check if
            # the intersection of their neighborhoods has size >= 2.
            # A more efficient way:
            # For each v in N(u), check if |N(u) \cap N(v)| > 1.
            # If v is in N(u), then v is a common neighbor of u and v? No.
            # N(u) \cap N(v) contains vertices w such that (u,w) and (v,w) are in color c.
            # If |N(u) \cap N(v)| >= 2, then u and v have at least 2 common neighbors, forming a C4 (u-w1-v-w2-u).
            # So we just need to check for all pairs (u, v) with u < v, is |N(u) \cap N(v)| <= 1?
            # Yes.
            # Optimization: Only check pairs where u and v are both in the graph.
            # We can iterate u from 0 to n-1.
            # For each v > u, check intersection.
            # This is O(n^2) per color. n=29, so 29^2 * 5 ~ 4200 checks. Very fast.
            for v in range(u + 1, n):
                common = mask_u & adj[c][v]
                # Check if common has more than 1 bit set
                if common and (common & (common - 1)):
                    return False
    return True

def _lz_woldar_coloring(n=26):
    """
    Construct the Lazebnik-Woldar 5-coloring of K_26.
    """
    # F_25 representation: a + b*i where i^2 = 2.
    # Elements are pairs (a, b) in Z_5 x Z_5.
    # Multiplication: (a,b)(c,d) = (ac - 2bd, ad + bc).
    
    def add(p, q):
        return ((p[0] + q[0]) % 5, (p[1] + q[1]) % 5)
    
    def sub(p, q):
        return ((p[0] - q[0]) % 5, (p[1] - q[1]) % 5)
    
    def mul(p, q):
        a, b = p
        c, d = q
        return ((a*c - 2*b*d) % 5, (a*d + b*c) % 5)
    
    # Find generator of F_25^*
    def order(p):
        cur = (1, 0)
        for k in range(1, 25):
            cur = mul(cur, p)
            if cur == (1, 0):
                return k
        return -1
    
    generator = None
    for a in range(5):
        for b in range(5):
            if (a, b) == (0, 0):
                continue
            if order((a, b)) == 24:
                generator = (a, b)
                break
        if generator:
            break
    
    if not generator:
        return [0] * (26*25//2)
    
    # Find subgroup of order 6
    # H = <g^4>
    g4 = mul(mul(generator, generator), mul(generator, generator))
    H = set()
    cur = (1, 0)
    for _ in range(6):
        H.add(cur)
        cur = mul(cur, g4)
    
    # Cosets
    cosets = [set() for _ in range(4)]
    for k in range(4):
        gk = (1, 0)
        for _ in range(k):
            gk = mul(gk, generator)
        for h in H:
            cosets[k].add(mul(gk, h))
    
    # Map element to color (0-3)
    elem_to_color = {}
    for c in range(4):
        for e in cosets[c]:
            elem_to_color[e] = c
    
    # Build coloring for K_26
    elements = []
    for a in range(5):
        for b in range(5):
            elements.append((a, b))
    
    colors = []
    for i in range(26):
        for j in range(i + 1, 26):
            if i == 25 or j == 25:
                colors.append(4)
            else:
                e_i = elements[i]
                e_j = elements[j]
                diff = sub(e_j, e_i)
                c = elem_to_color.get(diff, 0)
                colors.append(c)
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5 seconds margin
    
    best_n = 1
    best_colors = []
    
    def flatten(cm, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(cm[i][j])
        return colors

    # Helper to build adjacency bitmasks and conflict masks
    # adj[c][v] : bitmask of neighbors of v in color c
    # conflict[c][u] : bitmask of vertices v such that adj[c][u] & adj[c][v] != 0
    def build_structures(cm, n):
        adj = [[0] * n for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = cm[i][j]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        conflict = [[0] * n for _ in range(5)]
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                # Iterate through set bits in mask_u
                # For each v in mask_u, u and v share a neighbor?
                # No, conflict[u] should contain v if |adj[u] & adj[v]| >= 1.
                # Wait, if u and v share a neighbor w, then w is in adj[u] and adj[v].
                # So adj[u] & adj[v] != 0.
                # We can compute this by iterating v?
                # Or: for each w in adj[u], for each v in adj[w], add v to conflict[u].
                # This is O(sum deg^2).
                # With bitmasks, we can do:
                # conflict[u] = 0
                # for w in bits(adj[u]):
                #     conflict[u] |= adj[w]
                # Then remove u itself? adj[w] might contain u.
                # Yes, if (u,w) is in color c, then w is in adj[u] and u is in adj[w].
                # So u will be in conflict[u]. That's fine, we just won't assign color c to (N,u) if u is in conflict of something?
                # The condition is: if we assign color c to (N, u), then for any v assigned color c to (N, v),
                # we need adj[u] & adj[v] == 0.
                # So v must NOT be in conflict[u].
                # So conflict[u] is the set of forbidden v's.
                
                conf_mask = 0
                m = mask_u
                while m:
                    # Get lowest set bit
                    lsb = m & (-m)
                    w_idx = lsb.bit_length() - 1
                    conf_mask |= adj[c][w_idx]
                    m ^= lsb
                # We don't need to remove u, because if we assign c to (N,u), we check if other assigned vertices are in conflict[u].
                # If v=u, it's not a "other" vertex.
                conflict[c][u] = conf_mask
        
        return adj, conflict

    # Try to extend coloring from size n to n+1
    def try_extend(cm, n, adj, conflict, max_attempts=5000):
        new_n = n + 1
        new_cm = [row[:] + [-1] for row in cm]
        new_cm.append([-1] * new_n)
        
        new_v = new_n - 1
        assigned = [-1] * n
        
        # Precompute valid color order? Just randomize.
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None, False
                
            order = list(range(n))
            random.shuffle(order)
            
            # assigned_colors[c] : bitmask of vertices assigned color c so far
            assigned_colors = [0] * 5
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    for i in range(n):
                        c = assigned[i]
                        new_cm[new_v][i] = c
                        new_cm[i][new_v] = c
                    return True
                
                u = order[idx]
                colors_to_try = [0, 1, 2, 3, 4]
                random.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    # Check if u can be assigned color c
                    # u is forbidden if u is in conflict[c][v] for any v already assigned color c.
                    # This is equivalent to: (assigned_colors[c] & conflict[c][u]) == 0?
                    # No. conflict[c][u] contains v if adj[u] & adj[v] != 0.
                    # We need adj[u] & adj[v] == 0 for all v in assigned_colors[c].
                    # So we need assigned_colors[c] to be a subset of the complement of conflict[c][u].
                    # i.e., assigned_colors[c] & conflict[c][u] == 0.
                    
                    if assigned_colors[c] & conflict[c][u] == 0:
                        assigned[u] = c
                        assigned_colors[c] |= (1 << u)
                        if backtrack(idx + 1):
                            return True
                        assigned[u] = -1
                        assigned_colors[c] &= ~(1 << u)
                
                return False
            
            if backtrack(0):
                return new_cm, True
        
        return None, False

    # Main loop
    lw_colors = _lz_woldar_coloring(26)
    if _is_valid_coloring(26, lw_colors):
        best_n = 26
        best_colors = lw_colors
        cm = [[-1] * 26 for _ in range(26)]
        idx = 0
        for i in range(26):
            for j in range(i + 1, 26):
                cm[i][j] = lw_colors[idx]
                cm[j][i] = lw_colors[idx]
                idx += 1
        n = 26
        adj, conflict = build_structures(cm, n)
    else:
        cm = [[0]]
        n = 1
        adj, conflict = build_structures(cm, n)
        best_n = 1
        best_colors = []

    # Try to extend from n=26 to n=29
    # We use a loop that tries to extend one step at a time.
    # If it fails, we might try to restart the extension from a slightly different state?
    # Or just accept n=26 if we can't go further.
    # Given the time limit, we can try many random restarts for the extension step.
    
    while n < 29:
        if time.time() > deadline:
            break
            
        success = False
        # Increase attempts as we go deeper? Or keep it high.
        # The conflict graph gets denser, so backtracking might need more restarts.
        attempts_for_step = 20000 
        new_cm, success = try_extend(cm, n, adj, conflict, attempts_for_step)
        
        if success:
            n += 1
            cm = new_cm
            adj, conflict = build_structures(cm, n)
            if n > best_n:
                best_n = n
                best_colors = flatten(cm, n)
        else:
            # If we fail to extend, break.
            # It is rare to extend back after failing, as the constraints only tighten.
            # Unless we change the base coloring, which is expensive.
            break

    if best_n < 2:
        best_n = 2
        best_colors = [0]

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Max n found: {res['n']}")
    if res['n'] > 1:
        print(f"Valid: {_is_valid_coloring(res['n'], res['colors'])}")
