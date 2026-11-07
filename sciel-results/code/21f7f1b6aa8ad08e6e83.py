def is_valid(adj, deg):
    # Group vertices by degree
    deg_groups = {}
    for i in range(N):
        d = deg[i]
        if d not in deg_groups:
            deg_groups[d] = []
        deg_groups[d].append(i)
        
    for d, vertices in deg_groups.items():
        if len(vertices) < 2:
            continue
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                s = vertices[i]
                t = vertices[j]
                # Check path s-u-v-t
                mask_u = adj[s] & ~(1 << t)
                mask_v = adj[t] & ~(1 << s)
                
                u_mask = mask_u
                while u_mask:
                    u_bit = u_mask & -u_mask
                    u = u_bit.bit_length() - 1
                    u_mask ^= u_bit
                    
                    possible_v = mask_v & adj[u] & ~(1 << u)
                    if possible_v:
                        return False
    return True
