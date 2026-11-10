import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_edges = []
    best_count = 0

    def creates_q3(u, v, adj):
        # Check if adding edge (u,v) creates a Q3 containing (u,v)
        
        # Get neighbors of u and v excluding each other
        mask_u = adj[u] & ~(1 << v)
        mask_v = adj[v] & ~(1 << u)
        
        # Extract lists for iteration
        nu = []
        m = mask_u
        while m:
            lsb = m & -m
            nu.append(lsb.bit_length() - 1)
            m ^= lsb
            
        nv = []
        m = mask_v
        while m:
            lsb = m & -m
            nv.append(lsb.bit_length() - 1)
            m ^= lsb

        len_nu = len(nu)
        len_nv = len(nv)
        
        for i in range(len_nu):
            a = nu[i]
            mask_a = adj[a]
            for j in range(i + 1, len_nu):
                b = nu[j]
                mask_b = adj[b]
                
                for k in range(len_nv):
                    c = nv[k]
                    if c == a or c == b:
                        continue
                    mask_c = adj[c]
                    for l in range(k + 1, len_nv):
                        d = nv[l]
                        if d == a or d == b or d == c:
                            continue
                        
                        # Check Case 1: (a,c) and (b,d) are edges
                        if (mask_a >> c) & 1 and (mask_b >> d) & 1:
                            used = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            e_cand = mask_a & mask_b & ~used
                            f_cand = mask_c & adj[d] & ~used
                            
                            if e_cand and f_cand:
                                e_mask = e_cand
                                while e_mask:
                                    lsb = e_mask & -e_mask
                                    e = lsb.bit_length() - 1
                                    e_mask ^= lsb
                                    if adj[e] & f_cand:
                                        return True

                        # Check Case 2: (a,d) and (b,c) are edges
                        if (mask_a >> d) & 1 and (mask_b >> c) & 1:
                            used = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c) | (1 << d)
                            e_cand = mask_a & mask_b & ~used
                            f_cand = mask_c & adj[d] & ~used
                            
                            if e_cand and f_cand:
                                e_mask = e_cand
                                while e_mask:
                                    lsb = e_mask & -e_mask
                                    e = lsb.bit_length() - 1
                                    e_mask ^= lsb
                                    if adj[e] & f_cand:
                                        return True

        return False

    # Greedy with random restarts and kicks
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [0] * n
        current_edges = []

        # Greedy phase
        for u, v in all_edges:
            if not creates_q3(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.append((u, v))

        # Local search: try kicks
        # Reduced to 3 kicks to allow more restarts
        for _ in range(3):
            if time.time() > deadline:
                break
            if not current_edges:
                break

            # Remove a random edge
            idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[idx]
            adj[u_rem] &= ~(1 << u_rem) # BUG: should be v_rem
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.pop(idx)

            # Try to add edges in random order
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if not creates_q3(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.append((u, v))

        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges.copy()

    return [[u, v] for u, v in best_edges]
