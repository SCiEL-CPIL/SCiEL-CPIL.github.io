"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill
    
    N = 512
    
    # Precompute LCM check: build conflict graph using bitsets
    # adj[i] (0-indexed) is a Python int bitset where bit j is 1 if lcm(i+1, j+1) > N
    adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = math.gcd(a, b)
            if a * b // g > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Greedy independent set for initial lower bound
    def greedy_is():
        avail = (1 << N) - 1
        result = set()
        while avail:
            # Pick node with minimum degree in available subgraph
            best_v = -1
            best_deg = N + 1
            for v in range(N):
                if (avail >> v) & 1:
                    d = bin(adj[v] & avail).count('1')
                    if d < best_deg:
                        best_deg = d
                        best_v = v
            if best_v == -1:
                break
            result.add(best_v + 1)
            avail &= ~(1 << best_v)
            avail &= ~adj[best_v]
        return result
    
    greedy_result = greedy_is()
    best_size = len(greedy_result)
    best_set = set(greedy_result)
    
    # Also try a greedy starting from high-composite numbers
    def greedy_is_v2():
        avail = (1 << N) - 1
        result = set()
        # Try multiple starting orders
        for start_order in ['desc_degree', 'asc_degree', 'high_composite']:
            test_avail = (1 << N) - 1
            test_result = set()
            while test_avail:
                best_v = -1
                if start_order == 'desc_degree':
                    best_deg = -1
                    for v in range(N):
                        if (test_avail >> v) & 1:
                            d = bin(adj[v] & test_avail).count('1')
                            if d > best_deg:
                                best_deg = d
                                best_v = v
                elif start_order == 'asc_degree':
                    best_deg = N + 1
                    for v in range(N):
                        if (test_avail >> v) & 1:
                            d = bin(adj[v] & test_avail).count('1')
                            if d < best_deg:
                                best_deg = d
                                best_v = v
                else:  # high_composite: prefer numbers with many divisors
                    best_score = -1
                    for v in range(N):
                        if (test_avail >> v) & 1:
                            num = v + 1
                            # Count divisors as proxy for "high composite"
                            cnt = 0
                            for d in range(1, int(math.isqrt(num)) + 1):
                                if num % d == 0:
                                    cnt += 1
                                    if d * d != num:
                                        cnt += 1
                            d_in_avail = bin(adj[v] & test_avail).count('1')
                            score = cnt - d_in_avail
                            if score > best_score:
                                best_score = score
                                best_v = v
                if best_v == -1:
                    break
                test_result.add(best_v + 1)
                test_avail &= ~(1 << best_v)
                test_avail &= ~adj[best_v]
            if len(test_result) > len(result):
                result = set(test_result)
        return result
    
    greedy_result2 = greedy_is_v2()
    if len(greedy_result2) > best_size:
        best_size = len(greedy_result2)
        best_set = set(greedy_result2)
    
    # Check time
    if time.time() > deadline - 15:
        return list(best_set)
    
    # Branch and bound with greedy coloring upper bound
    # The coloring bound: color the conflict graph on available nodes greedily.
    # The number of colors is an upper bound on the independent set size.
    
    def color_bound(avail_mask):
        """Greedy coloring of the conflict graph induced by avail_mask.
        Returns (num_colors, color_assignment) where color_assignment[v] = color index."""
        if not avail_mask:
            return 0, {}
        
        # Get list of available nodes
        nodes = []
        m = avail_mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            nodes.append(v)
            m ^= lb
        
        # Sort by degree in avail (descending) for better coloring
        nodes.sort(key=lambda v: bin(adj[v] & avail_mask).count('1'), reverse=True)
        
        color = {}
        for v in nodes:
            # Find smallest color not used by neighbors of v in avail
            used = set()
            nb_mask = adj[v] & avail_mask
            nm = nb_mask
            while nm:
                lb = nm & (-nm)
                u = lb.bit_length() - 1
                if u in color:
                    used.add(color[u])
                nm ^= lb
            c = 0
            while c in used:
                c += 1
            color[v] = c
        
        if not color:
            return 0, color
        return max(color.values()) + 1, color
    
    # B&B solver
    # To avoid expensive set copies, track the selected set via a bitset too
    best_mask = 0
    for v in best_set:
        best_mask |= (1 << (v - 1))
    
    def solve(avail, cur_size, cur_mask):
        nonlocal best_size, best_set, best_mask
        
        if time.time() > deadline - 3:
            return
        
        if not avail:
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
                best_set = set()
                m = cur_mask
                while m:
                    lb = m & (-m)
                    best_set.add(lb.bit_length())
                    m ^= lb
            return
        
        # Upper bound via coloring
        num_colors, _ = color_bound(avail)
        if cur_size + num_colors <= best_size:
            return
        
        # Also use simple popcount bound as a quick check
        remaining_count = bin(avail).count('1')
        if cur_size + remaining_count <= best_size:
            return
        
        # Pick branching node: maximum degree in avail
        best_v = -1
        best_deg = -1
        m = avail
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            d = bin(adj[v] & avail).count('1')
            if d > best_deg:
                best_deg = d
                best_v = v
            m ^= lb
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_avail = avail & ~(1 << best_v) & ~adj[best_v]
        solve(new_avail, cur_size + 1, cur_mask | (1 << best_v))
        
        # Branch 2: exclude best_v
        new_avail = avail & ~(1 << best_v)
        solve(new_avail, cur_size, cur_mask)
    
    all_nodes = (1 << N) - 1
    solve(all_nodes, 0, best_mask)
    
    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lb = m & (-m)
        result.append(lb.bit_length())
        m ^= lb
    
    if not result:
        result = list(best_set)
    
    return result

# EVOLVE-BLOCK-END
