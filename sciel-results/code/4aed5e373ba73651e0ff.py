import random

def construct(k, g):
    if k != 3 or g != 16:
        # Fallback for other cases if needed, but task is specific
        # We'll just return a small graph if it doesn't match, or raise
        # Assuming k=3, g=16
        pass

    n_target = 936
    group_order = 234
    
    edges_base = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    
    # Precompute non-backtracking closed walks in K4 of length < 16
    # Walk is a sequence of vertices v0, v1, ..., vl = v0
    # v_{i+1} != v_{i-1}
    
    walks = []
    
    # DFS to generate walks
    # Start at 0
    # We need to generate walks for all start nodes?
    # Yes, but due to symmetry of K4, the structure of walks is similar.
    # However, voltages break symmetry.
    # We must check walks starting from all nodes.
    
    def get_walks(start):
        res = []
        # state: (curr, prev, length, coeffs)
        # coeffs: list of 6 ints
        
        def dfs(curr, prev, length, coeffs):
            if length >= 16:
                return
            for next_node in range(4):
                if next_node == prev:
                    continue
                
                # Update coeffs
                new_coeffs = coeffs[:]
                # Determine edge index and sign
                if curr < next_node:
                    u, v = curr, next_node
                else:
                    u, v = next_node, curr
                
                idx = -1
                if (u,v) == (0,1): idx = 0
                elif (u,v) == (0,2): idx = 1
                elif (u,v) == (0,3): idx = 2
                elif (u,v) == (1,2): idx = 3
                elif (u,v) == (1,3): idx = 4
                elif (u,v) == (2,3): idx = 5
                
                if curr < next_node:
                    new_coeffs[idx] += 1
                else:
                    new_coeffs[idx] -= 1
                
                new_len = length + 1
                
                if next_node == start:
                    if new_len < 16:
                        res.append((new_coeffs, new_len))
                else:
                    dfs(next_node, curr, new_len, new_coeffs)
        
        # Start steps
        for next_node in range(4):
            if next_node == start:
                continue
            new_coeffs = [0]*6
            if start < next_node:
                u, v = start, next_node
            else:
                u, v = next_node, start
            idx = -1
            if (u,v) == (0,1): idx = 0
            elif (u,v) == (0,2): idx = 1
            elif (u,v) == (0,3): idx = 2
            elif (u,v) == (1,2): idx = 3
            elif (u,v) == (1,3): idx = 4
            elif (u,v) == (2,3): idx = 5
            
            if start < next_node:
                new_coeffs[idx] += 1
            else:
                new_coeffs[idx] -= 1
                
            dfs(next_node, start, 2, new_coeffs)
            
        return res

    all_walks = []
    for s in range(4):
        all_walks.extend(get_walks(s))
        
    # Remove duplicates?
    # Different walks might have same coeffs?
    # If so, checking one is enough.
    # Convert to tuple for set
    unique_walks = set()
    for w in all_walks:
        unique_walks.add(tuple(w[0]))
    unique_walks = list(unique_walks)
    
    # Find valid voltages
    found_vs = None
    for _ in range(2000):
        vs = [random.randint(1, group_order-1) for _ in range(6)]
        valid = True
        for coeffs in unique_walks:
            vol_sum = 0
            for i in range(6):
                if coeffs[i] != 0:
                    vol_sum += coeffs[i] * vs[i]
            vol_sum %= group_order
            if vol_sum == 0:
                valid = False
                break
        
        if valid:
            # Check connectivity heuristic: gcd of vs and group_order?
            # Just check if any vs is coprime to 234
            import math
            if any(math.gcd(v, group_order) == 1 for v in vs):
                found_vs = vs
                break
                
    if found_vs is None:
        # Try more or use deterministic
        # For safety, use a known good pattern if available
        # Or just return the first valid one even if connectivity is weak?
        # No, must be connected.
        # If not found, raise error or fallback
        raise RuntimeError("No valid voltages found")
        
    vs = found_vs
    
    n = 4 * group_order
    edges = []
    
    for u in range(4):
        for v_pair in edges_base:
            if u not in v_pair:
                continue
            v = v_pair[0] if v_pair[1] == u else v_pair[1]
            
            if u < v:
                u_edge, v_edge = u, v
            else:
                u_edge, v_edge = v, u
                
            idx = -1
            if (u_edge, v_edge) == (0,1): idx = 0
            elif (u_edge, v_edge) == (0,2): idx = 1
            elif (u_edge, v_edge) == (0,3): idx = 2
            elif (u_edge, v_edge) == (1,2): idx = 3
            elif (u_edge, v_edge) == (1,3): idx = 4
            elif (u_edge, v_edge) == (2,3): idx = 5
            
            if u < v:
                vol = vs[idx]
            else:
                vol = -vs[idx]
                
            for i in range(group_order):
                src = u * group_order + i
                dst_fiber = (i + vol) % group_order
                dst = v * group_order + dst_fiber
                if src < dst:
                    edges.append((src, dst))
                else:
                    edges.append((dst, src))
                    
    return n, edges
