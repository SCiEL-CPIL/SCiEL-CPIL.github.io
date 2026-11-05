import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Generate Paley coloring for n=29
    def generate_paley(n, prime):
        res = set()
        for x in range(1, prime):
            res.add((x * x) % prime)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % prime
                if diff in res:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Build adjacency bitmasks from colors
    def build_adj(colors, n):
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    # Count K2,2,2 violations using adjacency bitmasks
    def count_violations(adj, n):
        total_viol = 0
        for c in [0, 1]:
            A = adj[c]
            for u1 in range(n):
                for u2 in range(u1 + 1, n):
                    S = A[u1] & A[u2]
                    if S == 0:
                        continue
                    # Extract set bits
                    s_bits = []
                    temp = S
                    while temp:
                        lsb = temp & -temp
                        s_bits.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    k = len(s_bits)
                    if k < 4:
                        continue
                    for a in range(k):
                        v1 = s_bits[a]
                        for b in range(a + 1, k):
                            v2 = s_bits[b]
                            C = A[v1] & A[v2]
                            common_in_S = C & S
                            if common_in_S & (common_in_S - 1):
                                total_viol += 1
        return total_viol

    # Fast incremental violation delta when flipping edge (i,j) from old_c to new_c
    def violation_delta(adj, n, i, j, old_c):
        """
        Compute change in violation count when flipping edge (i,j) from old_c to (1-old_c).
        Only checks violations where i or j is one of the 6 vertices in the K2,2,2.
        """
        new_c = 1 - old_c
        delta = 0
        
        for c in [old_c, new_c]:
            A = adj[c]
            # Determine if edge (i,j) is present in color c BEFORE the flip
            edge_in_c_before = (old_c == c)  # if c==old_c, edge is there; if c==new_c, edge is not
            
            # For color c, we need to count violations involving i or j
            # BEFORE flip and AFTER flip
            
            # Helper: count violations for a specific color where i or j is involved
            def count_viol_involving(A, n, i, j, edge_present):
                """Count violations in color A where i or j is one of the 6 vertices."""
                cnt = 0
                # Case 1: i or j is u1 or u2 (the "top" pair)
                for u1, u2 in [(i, v) for v in range(n) if v != i] + [(v, i) for v in range(i+1, n) if v != i]:
                    if u1 >= u2:
                        continue
                    # Build S = A[u1] & A[u2], but if edge (i,j) status matters for this pair
                    S = A[u1] & A[u2]
                    if not edge_present and (set([u1, u2]) == set([i, j])):
                        # Edge (i,j) not in this color, so i not in A[j] and j not in A[i]
                        # But our adj already reflects current state, so S is correct
                        pass
                    if S == 0:
                        continue
                    s_bits = []
                    temp = S
                    while temp:
                        lsb = temp & -temp
                        s_bits.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    k = len(s_bits)
                    if k < 4:
                        continue
                    for a in range(k):
                        v1 = s_bits[a]
                        for b in range(a + 1, k):
                            v2 = s_bits[b]
                            C = A[v1] & A[v2]
                            common_in_S = C & S
                            if common_in_S & (common_in_S - 1):
                                cnt += 1
                # Case 2: i or j is in s_bits (the "bottom" vertices)
                # This is harder - i or j is a common neighbor of some (u1,u2)
                # We need to check all pairs (u1,u2) where S contains i or j
                for u1 in range(n):
                    for u2 in range(u1 + 1, n):
                        S = A[u1] & A[u2]
                        if S == 0:
                            continue
                        if not (S & (1 << i)) and not (S & (1 << j)):
                            continue
                        s_bits = []
                        temp = S
                        while temp:
                            lsb = temp & -temp
                            s_bits.append(lsb.bit_length() - 1)
                            temp ^= lsb
                        k = len(s_bits)
                        if k < 4:
                            continue
                        for a in range(k):
                            v1 = s_bits[a]
                            for b in range(a + 1, k):
                                v2 = s_bits[b]
                                C = A[v1] & A[v2]
                                common_in_S = C & S
                                if common_in_S & (common_in_S - 1):
                                    cnt += 1
                return cnt
            
            # This approach double-counts. Let me use a different strategy.
            # Instead, I'll directly compute before and after for the specific structures.
            pass
        
        # Simpler approach: just do a full recount but it's the same cost.
        # Let me instead use a smarter incremental approach.
        
        # When flipping (i,j) from old_c to new_c:
        # In old_c: remove edge (i,j) -> A[i] loses bit j, A[j] loses bit i
        # In new_c: add edge (i,j) -> A[i] gains bit j, A[j] gains bit i
        # 
        # Violations affected are those where the set of common neighbors changes for pairs involving i or j,
        # OR where i or j itself is a common neighbor.
        
        # For efficiency, let me just compute the full violation count before and after
        # but only for the affected color classes, restricted to relevant pairs.
        
        # Actually, the cleanest approach: temporarily modify, count, revert.
        # But that's what we're doing in the main loop anyway.
        
        # Let me return 0 and let the main loop do the full count.
        # The real improvement should be in making count_violations faster.
        return 0

    # Optimized count_violations using numpy-style bit operations
    def count_violations_fast(adj, n):
        total_viol = 0
        for c in [0, 1]:
            A = adj[c]
            for u1 in range(n):
                Au1 = A[u1]
                for u2 in range(u1 + 1, n):
                    S = Au1 & A[u2]
                    if S == 0:
                        continue
                    # Count bits in S
                    bit_count = bin(S).count('1')
                    if bit_count < 4:
                        continue
                    # For each pair in S, check if their common neighbors in S >= 2
                    s_bits = []
                    temp = S
                    while temp:
                        lsb = temp & -temp
                        s_bits.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    k = len(s_bits)
                    for a in range(k):
                        v1 = s_bits[a]
                        for b in range(a + 1, k):
                            v2 = s_bits[b]
                            C = A[v1] & A[v2]
                            common_in_S = C & S
                            # Check if common_in_S has at least 2 bits
                            if common_in_S & (common_in_S - 1):
                                total_viol += 1
        return total_viol

    # Fast validity check
    def is_valid_fast(colors, n):
        adj = build_adj(colors, n)
        return count_violations_fast(adj, n) == 0

    # Local search for n=30 with faster violation counting
    def local_search_30(colors, n):
        adj = build_adj(colors, n)
        current_viol = count_violations_fast(adj, n)
        best_colors = colors[:]
        best_viol = current_viol
        
        num_edges = n * (n - 1) // 2
        edge_map = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
        
        temp = 3.0
        cooling = 0.9998
        stall_count = 0
        stall_threshold = 3000
        no_improve_count = 0
        no_improve_threshold = 20000
        
        while time.time() < deadline - 5:
            edge_idx = random.randint(0, num_edges - 1)
            i, j = edge_map[edge_idx]
            old_val = colors[edge_idx]
            new_val = 1 - old_val
            
            # Update adjacency incrementally
            adj[old_val][i] &= ~(1 << j)
            adj[old_val][j] &= ~(1 << i)
            adj[new_val][i] |= (1 << j)
            adj[new_val][j] |= (1 << i)
            
            new_viol = count_violations_fast(adj, n)
            colors[edge_idx] = new_val
            
            delta = new_viol - current_viol
            if delta < 0 or (delta == 0 and random.random() < 0.1) or random.random() < pow(2.71828, -delta / temp):
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_colors = colors[:]
                    stall_count = 0
                    no_improve_count = 0
                    if best_viol == 0:
                        return best_colors
                else:
                    no_improve_count += 1
            else:
                # Revert
                adj[new_val][i] &= ~(1 << j)
                adj[new_val][j] &= ~(1 << i)
                adj[old_val][i] |= (1 << j)
                adj[old_val][j] |= (1 << i)
                colors[edge_idx] = old_val
                stall_count += 1
                no_improve_count += 1
            
            temp *= cooling
            
            if stall_count > stall_threshold:
                num_flips = 5
                flipped_edges = []
                for _ in range(num_flips):
                    e1 = random.randint(0, num_edges - 1)
                    i1, j1 = edge_map[e1]
                    old1 = colors[e1]
                    new1 = 1 - old1
                    adj[old1][i1] &= ~(1 << j1)
                    adj[old1][j1] &= ~(1 << i1)
                    adj[new1][i1] |= (1 << j1)
                    adj[new1][j1] |= (1 << i1)
                    colors[e1] = new1
                    flipped_edges.append((e1, i1, j1, old1, new1))
                
                nv = count_violations_fast(adj, n)
                if nv < current_viol:
                    current_viol = nv
                    if nv < best_viol:
                        best_viol = nv
                        best_colors = colors[:]
                        no_improve_count = 0
                else:
                    for (e1, i1, j1, old1, new1) in flipped_edges:
                        adj[new1][i1] &= ~(1 << j1)
                        adj[new1][j1] &= ~(1 << i1)
                        adj[old1][i1] |= (1 << j1)
                        adj[old1][j1] |= (1 << i1)
                        colors[e1] = old1
                stall_count = 0
                temp = 3.0
            
            if no_improve_count > no_improve_threshold:
                num_flips = 10
                flipped_edges = []
                for _ in range(num_flips):
                    e1 = random.randint(0, num_edges - 1)
                    i1, j1 = edge_map[e1]
                    old1 = colors[e1]
                    new1 = 1 - old1
                    adj[old1][i1] &= ~(1 << j1)
                    adj[old1][j1] &= ~(1 << i1)
                    adj[new1][i1] |= (1 << j1)
                    adj[new1][j1] |= (1 << i1)
                    colors[e1] = new1
                    flipped_edges.append((e1, i1, j1, old1, new1))
                
                nv = count_violations_fast(adj, n)
                if nv < current_viol:
                    current_viol = nv
                    if nv < best_viol:
                        best_viol = nv
                        best_colors = colors[:]
                        no_improve_count = 0
                else:
                    for (e1, i1, j1, old1, new1) in flipped_edges:
                        adj[new1][i1] &= ~(1 << j1)
                        adj[new1][j1] &= ~(1 << i1)
                        adj[old1][i1] |= (1 << j1)
                        adj[old1][j1] |= (1 << i1)
                        colors[e1] = old1
                no_improve_count = 0
                temp = 4.0
        
        return best_colors

    # Targeted 30-vertex construction: extend Paley(29) with vertex 29
    def try_extend_to_30(colors29, n29):
        n30 = 30
        best_result = None
        best_viol = float('inf')
        
        # Precompute Paley quadratic residues for prime 29
        res29 = set()
        for x in range(1, 29):
            res29.add((x * x) % 29)
        
        # Try multiple deterministic patterns for connecting vertex 29
        patterns = []
        
        # Pattern 0: connect to QRs
        patterns.append([0 if i in res29 else 1 for i in range(n29)])
        
        # Pattern 1: connect to non-QRs
        patterns.append([1 if i in res29 else 0 for i in range(n29)])
        
        # Pattern 2: balanced split by index
        patterns.append([0 if i < 15 else 1 for i in range(n29)])
        
        # Pattern 3: mod 3 pattern
        patterns.append([0 if i % 3 < 2 else 1 for i in range(n29)])
        
        # Pattern 4: mod 5 pattern
        patterns.append([0 if i % 5 < 3 else 1 for i in range(n29)])
        
        # Pattern 5: based on QR of (i+1)
        patterns.append([0 if ((i+1) in res29) else 1 for i in range(n29)])
        
        # Pattern 6: alternating
        patterns.append([0 if i % 2 == 0 else 1 for i in range(n29)])
        
        # Pattern 7: based on whether i is a QR
        patterns.append([0 if (i*i) % 29 in res29 else 1 for i in range(n29)])
        
        for pat in patterns:
            colors30 = colors29[:] + pat
            colors30_opt = local_search_30(colors30, n30)
            adj = build_adj(colors30_opt, n30)
            viol = count_violations_fast(adj, n30)
            
            if viol < best_viol:
                best_viol = viol
                best_result = colors30_opt
            
            if best_viol == 0:
                return best_result
        
        # If deterministic patterns didn't work, try a few random ones
        for _ in range(5):
            pat = [random.randint(0, 1) for _ in range(n29)]
            colors30 = colors29[:] + pat
            colors30_opt = local_search_30(colors30, n30)
            adj = build_adj(colors30_opt, n30)
            viol = count_violations_fast(adj, n30)
            
            if viol < best_viol:
                best_viol = viol
                best_result = colors30_opt
            
            if best_viol == 0:
                return best_result
        
        return best_result

    # 1. Generate and verify Paley(29)
    n29 = 29
    colors29 = generate_paley(n29, 29)
    
    if is_valid_fast(colors29, n29):
        colors30_optimized = try_extend_to_30(colors29, n29)
        
        if colors30_optimized is not None:
            adj = build_adj(colors30_optimized, 30)
            if count_violations_fast(adj, 30) == 0:
                return {'n': 30, 'colors': colors30_optimized}
        
        return {'n': 29, 'colors': colors29}
    else:
        return {'n': 17, 'colors': generate_paley(17, 17)}
