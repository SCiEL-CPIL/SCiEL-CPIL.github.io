import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin before 300s limit

    q = 37
    n_core = 37
    n_new = 5
    n_total = n_core + n_new

    # Build Paley graph on 37 vertices
    residues = {(x * x) % q for x in range(1, q)}
    core_adj = [0] * n_core

    for i in range(n_core):
        for j in range(i + 1, n_core):
            if (j - i) % q in residues:
                core_adj[i] |= (1 << j)
                core_adj[j] |= (1 << i)

    # Precompute subsets of core
    core_k4s = []
    core_i4s = []
    core_k3s = []
    core_i3s = []
    
    # Helper to check if a set of vertices is a clique or independent
    def is_clique(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if not (core_adj[vertices[i]] & (1 << vertices[j])):
                    return False
        return True

    def is_independent(vertices):
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if (core_adj[vertices[i]] & (1 << vertices[j])):
                    return False
        return True

    # Precompute K4, I4, K3, I3
    # This is C(37,4) ~ 66k, C(37,3) ~ 7764. Fast enough.
    for combo in itertools.combinations(range(n_core), 4):
        if is_clique(combo):
            core_k4s.append(combo)
        elif is_independent(combo):
            core_i4s.append(combo)
            
    for combo in itertools.combinations(range(n_core), 3):
        if is_clique(combo):
            core_k3s.append(combo)
        elif is_independent(combo):
            core_i3s.append(combo)

    # Convert to bitmasks for faster intersection
    core_k4_masks = [0]*len(core_k4s)
    for i, c in enumerate(core_k4s):
        m = 0
        for v in c: m |= (1 << v)
        core_k4_masks[i] = m
        
    core_i4_masks = [0]*len(core_i4s)
    for i, c in enumerate(core_i4s):
        m = 0
        for v in c: m |= (1 << v)
        core_i4_masks[i] = m

    # For K3/I3, we need to check against new vertex connections.
    # We will store the lists of vertices for K3/I3 to check specific connections.
    # But to check "Does Core induced by Mask contain an Edge/Non-Edge", we need core_adj.
    
    def count_violations(new_conn, new_adj):
        """
        new_conn: list of 5 integers, bitmask of core vertices connected to new vertex i
        new_adj: list of 5 integers, bitmask of new vertices connected to new vertex i (only lower bits matter)
        """
        violations = 0
        
        # 1. Core K4 + New Vertex (K5)
        # If new vertex i is connected to all vertices in a Core K4, it forms a K5.
        for k4_mask in core_k4_masks:
            for i in range(n_new):
                if (new_conn[i] & k4_mask) == k4_mask:
                    violations += 1
                    
        # 2. Core I4 + New Vertex (I5)
        # If new vertex i is NOT connected to any vertex in a Core I4, it forms an I5.
        for i4_mask in core_i4_masks:
            for i in range(n_new):
                if (new_conn[i] & i4_mask) == 0:
                    violations += 1

        # 3. 2 Core + 3 New (K5)
        # 3 New form K3. 2 Core form Edge. All cross-edges present.
        # Intersection of connections of 3 new vertices: M = C[n1] & C[n2] & C[n3]
        # We need to know if Core induced by M contains an Edge.
        new_k3s = []
        new_i3s = []
        for combo in itertools.combinations(range(n_new), 3):
            n1, n2, n3 = combo
            # Check if K3 in new graph
            if (new_adj[n1] & (1 << n2)) and (new_adj[n1] & (1 << n3)) and (new_adj[n2] & (1 << n3)):
                new_k3s.append(combo)
            # Check if I3 in new graph
            elif not (new_adj[n1] & (1 << n2)) and not (new_adj[n1] & (1 << n3)) and not (new_adj[n2] & (1 << n3)):
                new_i3s.append(combo)

        for combo in new_k3s:
            n1, n2, n3 = combo
            M = new_conn[n1] & new_conn[n2] & new_conn[n3]
            # Check if Core induced by M has an edge
            # Iterate through bits in M
            m = M
            has_edge = False
            while m:
                lsb = m & -m
                c1 = lsb.bit_length() - 1
                # Check if any neighbor of c1 in Core is in M
                if core_adj[c1] & M:
                    has_edge = True
                    break
                m ^= lsb
            if has_edge:
                violations += 1

        # 4. 2 Core + 3 New (I5)
        # 3 New form I3. 2 Core form Non-Edge. All cross-edges absent.
        # Intersection of NON-connections: M = ~C[n1] & ~C[n2] & ~C[n3] & CoreMask
        core_mask = (1 << n_core) - 1
        for combo in new_i3s:
            n1, n2, n3 = combo
            M = (~new_conn[n1]) & (~new_conn[n2]) & (~new_conn[n3]) & core_mask
            # Check if Core induced by M has a Non-Edge
            # i.e. Is it NOT a Clique?
            # If M has < 2 bits, no non-edge possible (vacuously clique? No, we need a pair).
            # If M has >= 2 bits, check if any pair is not connected.
            m = M
            has_non_edge = False
            verts_in_m = []
            while m:
                lsb = m & -m
                verts_in_m.append(lsb.bit_length() - 1)
                m ^= lsb
            
            if len(verts_in_m) >= 2:
                for i in range(len(verts_in_m)):
                    for j in range(i+1, len(verts_in_m)):
                        if not (core_adj[verts_in_m[i]] & (1 << verts_in_m[j])):
                            has_non_edge = True
                            break
                    if has_non_edge: break
            if has_non_edge:
                violations += 1

        # 5. 3 Core + 2 New (K5)
        # 2 New form Edge. 3 Core form K3. All cross-edges present.
        new_k2s = []
        new_i2s = []
        for combo in itertools.combinations(range(n_new), 2):
            n1, n2 = combo
            if (new_adj[n1] & (1 << n2)):
                new_k2s.append(combo)
            else:
                new_i2s.append(combo)

        for combo in new_k2s:
            n1, n2 = combo
            M = new_conn[n1] & new_conn[n2]
            # Check if Core induced by M contains a K3
            # Iterate through core_k3s and check if k3 is subset of M
            for k3 in core_k3s:
                k3_mask = 0
                for v in k3: k3_mask |= (1 << v)
                if (M & k3_mask) == k3_mask:
                    violations += 1
                    break # Only count once per K3? No, each K3+Edge is a distinct K5.
                    # Wait, if M contains multiple K3s, each forms a distinct K5.
                    # So we should count all.
                    # But `break` stops counting others. Remove break.
                    # Actually, if M contains K3, it's a violation.
                    # We just need to know if there is ANY K3.
                    # The task is to count violations? Or just check validity?
                    # The current code counts. But for SA, we just need a score.
                    # Counting is better for landscape.
                    # Let's count all.
                    
        # Re-do 5 properly
        violations -= 0 # dummy
        # Let's just re-implement 5 and 6 cleanly to avoid double counting or errors.
        
        # Actually, the previous loops for 3 and 4 counted violations per subset.
        # For 5: For each K2 in new, for each K3 in core, if K3 subset of M, violation++.
        # For 6: For each I2 in new, for each I3 in core, if I3 subset of M_not, violation++.
        
        # Let's refine 5
        for combo in new_k2s:
            n1, n2 = combo
            M = new_conn[n1] & new_conn[n2]
            for k3 in core_k3s:
                k3_mask = 0
                for v in k3: k3_mask |= (1 << v)
                if (M & k3_mask) == k3_mask:
                    violations += 1

        # 6. 3 Core + 2 New (I5)
        for combo in new_i2s:
            n1, n2 = combo
            M_not = (~new_conn[n1]) & (~new_conn[n2]) & core_mask
            for i3 in core_i3s:
                i3_mask = 0
                for v in i3: i3_mask |= (1 << v)
                if (M_not & i3_mask) == i3_mask:
                    violations += 1

        # 7. 4 Core + 1 New (Already covered by 1 and 2)
        
        # 8. 5 New (K5 or I5)
        # Check if all 5 new form K5
        is_k5_new = True
        for i in range(n_new):
            for j in range(i+1, n_new):
                if not (new_adj[i] & (1 << j)):
                    is_k5_new = False
                    break
            if not is_k5_new: break
        if is_k5_new:
            violations += 1

        # Check if all 5 new form I5
        is_i5_new = True
        for i in range(n_new):
            for j in range(i+1, n_new):
                if (new_adj[i] & (1 << j)):
                    is_i5_new = False
                    break
            if not is_i5_new: break
        if is_i5_new:
            violations += 1

        return violations

    best_violations = float('inf')
    best_state = None

    # State: (new_conn, new_adj)
    # new_conn: list of 5 ints
    # new_adj: list of 5 ints

    for attempt in range(50):
        if time.time() > deadline:
            break

        # Random Init
        new_conn = [0] * n_new
        for i in range(n_new):
            for j in range(n_core):
                if random.random() < 0.5:
                    new_conn[i] |= (1 << j)
        
        new_adj = [0] * n_new
        for i in range(n_new):
            for j in range(i+1, n_new):
                if random.random() < 0.5:
                    new_adj[i] |= (1 << j)
                    new_adj[j] |= (1 << i)

        curr_v = count_violations(new_conn, new_adj)
        
        if curr_v < best_violations:
            best_violations = curr_v
            best_state = (new_conn[:], new_adj[:])
            
        if best_violations == 0:
            break

        # Local Search
        curr_conn = best_state[0][:]
        curr_adj = best_state[1][:]
        curr_v = best_violations
        temp = 1.0
        
        for step in range(10000):
            if time.time() > deadline:
                break
                
            # Choose move: flip new-core or new-new
            if random.random() < 0.5:
                # Flip new-core edge
                i = random.randint(0, n_new - 1)
                j = random.randint(0, n_core - 1)
                curr_conn[i] ^= (1 << j)
                new_v = count_violations(curr_conn, curr_adj)
                if new_v < curr_v or random.random() < pow(2.718, -(new_v - curr_v) / temp):
                    curr_v = new_v
                    if new_v < best_violations:
                        best_violations = new_v
                        best_state = (curr_conn[:], curr_adj[:])
                        if best_violations == 0:
                            break
                else:
                    curr_conn[i] ^= (1 << j)
            else:
                # Flip new-new edge
                i = random.randint(0, n_new - 1)
                j = random.randint(0, n_new - 1)
                if i == j: continue
                curr_adj[i] ^= (1 << j)
                curr_adj[j] ^= (1 << i)
                new_v = count_violations(curr_conn, curr_adj)
                if new_v < curr_v or random.random() < pow(2.718, -(new_v - curr_v) / temp):
                    curr_v = new_v
                    if new_v < best_violations:
                        best_violations = new_v
                        best_state = (curr_conn[:], curr_adj[:])
                        if best_violations == 0:
                            break
                else:
                    curr_adj[i] ^= (1 << j)
                    curr_adj[j] ^= (1 << i)
                    
            temp *= 0.9999

    if best_violations == 0 and best_state is not None:
        new_conn, new_adj = best_state
        edges = []
        # Core edges
        for i in range(n_core):
            for j in range(i + 1, n_core):
                if (j - i) % q in residues:
                    edges.append((i, j))
        # New-Core edges
        for i in range(n_new):
            v = n_core + i
            for j in range(n_core):
                if (new_conn[i] >> j) & 1:
                    edges.append((j, v))
        # New-New edges
        for i in range(n_new):
            for j in range(i+1, n_new):
                if (new_adj[i] >> j) & 1:
                    edges.append((n_core + i, n_core + j))
        return n_total, edges
    else:
        # Fallback to Paley 37
        edges = []
        for i in range(n_core):
            for j in range(i + 1, n_core):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return n_core, edges
