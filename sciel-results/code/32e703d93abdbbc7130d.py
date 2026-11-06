import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency bitmasks for fast intersection
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        remaining_mask = [0] * n
        for u, v in edges:
            remaining_mask[u] |= (1 << v)
            remaining_mask[v] |= (1 << u)
        
        partition = []
        edge_count = len(edges)
        processed = 0
        
        while processed < edge_count and time.time() < deadline:
            # Find most constrained edge: minimize |common neighbors in remaining|
            best_u, best_v = -1, -1
            min_common = n + 1
            
            # Sample a subset of edges if too many to check
            edge_list = []
            for u in range(n):
                mask = remaining_mask[u]
                while mask:
                    lsb = mask & (-mask)
                    v = lsb.bit_length() - 1
                    if v > u:
                        edge_list.append((u, v))
                    mask ^= lsb
            
            if not edge_list:
                break
            
            # Check all edges but break early if time is tight
            check_all = len(edge_list) <= 2000 or time.time() < deadline - 5
            for u, v in edge_list:
                if not check_all and time.time() > deadline - 2:
                    break
                # Common neighbors in remaining graph
                common = remaining_mask[u] & remaining_mask[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                cnt = common.bit_count()
                if cnt < min_common:
                    min_common = cnt
                    best_u, best_v = u, v
            
            if best_u == -1:
                break
            
            u, v = best_u, best_v
            
            # Greedy clique growth from {u, v}
            clique = [u, v]
            # Candidates: vertices connected to all clique members in remaining
            cand_mask = remaining_mask[u] & remaining_mask[v]
            cand_mask &= ~(1 << u)
            cand_mask &= ~(1 << v)
            
            while cand_mask and time.time() < deadline:
                # Pick candidate with max degree within candidates
                best_cand = -1
                best_deg = -1
                temp = cand_mask
                while temp:
                    lsb = temp & (-temp)
                    w = lsb.bit_length() - 1
                    # Count connections to other candidates
                    deg = (cand_mask & remaining_mask[w]).bit_count()
                    deg -= 1  # don't count self
                    if deg > best_deg:
                        best_deg = deg
                        best_cand = w
                    temp ^= lsb
                
                if best_cand == -1:
                    break
                
                # Verify best_cand connects to all clique members
                valid = True
                for c in clique:
                    if not (remaining_mask[best_cand] >> c) & 1:
                        valid = False
                        break
                if not valid:
                    break
                
                clique.append(best_cand)
                cand_mask &= ~(1 << best_cand)
                
                # Update candidates: must connect to new clique member
                cand_mask &= remaining_mask[best_cand]
                # Remove clique members from candidates
                for c in clique:
                    cand_mask &= ~(1 << c)
            
            clique_list = sorted(clique)
            
            # Verify validity
            valid = True
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if not (remaining_mask[a] >> b) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid or len(clique_list) < 2:
                clique_list = [u, v]
                clique = [u, v]
            
            partition.append(clique_list)
            
            # Remove all edges within this clique
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    remaining_mask[a] &= ~(1 << b)
                    remaining_mask[b] &= ~(1 << a)
                    processed += 1
        
        # Fallback for remaining edges
        for u in range(n):
            mask = remaining_mask[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if v > u:
                    partition.append([u, v])
                mask ^= lsb
        
        return partition

    result1 = solve_instance(32, 0.6, 601, 60)
    result2 = solve_instance(64, 0.55, 602, 65)
    
    return [result1, result2]
