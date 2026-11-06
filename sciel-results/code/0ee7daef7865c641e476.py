import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 115
    
    # Fallback: n=17 quadratic residue coloring
    def fallback():
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(0 if (j - i) % 17 in squares else 1)
        return {'n': n, 'colors': colors}
    
    best_result = fallback()
    best_n = 17

    def is_valid_coloring(n, colors):
        """
        Check if the coloring of K_n has no monochromatic K2,2,2.
        colors: list of 0/1 of length n*(n-1)/2
        """
        edge_to_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = idx
                idx += 1
        
        partitions = [
            [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
        ]
        
        for vertices in combinations(range(n), 6):
            v = vertices
            for part in partitions:
                p0 = [v[part[0][0]], v[part[0][1]]]
                p1 = [v[part[1][0]], v[part[1][1]]]
                p2 = [v[part[2][0]], v[part[2][1]]]
                
                c0 = 0
                c1 = 0
                # Edges between p0, p1
                for u in p0:
                    for w in p1:
                        idx_e = edge_to_idx[(min(u,w), max(u,w))]
                        if colors[idx_e] == 0: c0 += 1
                        else: c1 += 1
                # Edges between p0, p2
                for u in p0:
                    for w in p2:
                        idx_e = edge_to_idx[(min(u,w), max(u,w))]
                        if colors[idx_e] == 0: c0 += 1
                        else: c1 += 1
                # Edges between p1, p2
                for u in p1:
                    for w in p2:
                        idx_e = edge_to_idx[(min(u,w), max(u,w))]
                        if colors[idx_e] == 0: c0 += 1
                        else: c1 += 1
                
                if c0 == 12 or c1 == 12:
                    return False
        return True

    def paley_coloring(n, p):
        """
        Generate a coloring based on Paley graph over GF(p).
        If n > p, this is not directly applicable as a single Paley graph.
        We will use this for n=p.
        """
        if n != p:
            return None
        
        squares = np.zeros(p, dtype=int)
        for x in range(1, p):
            squares[x*x % p] = 1
            
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                # In Paley graph, edge exists if diff is a quadratic residue.
                # We map residue to 0, non-residue to 1.
                colors.append(1 if squares[diff] == 1 else 0)
        return colors

    # Primes suitable for Paley: 5, 13, 17, 29
    # Paley graph P_q exists for q = p^k where p is prime and q ≡ 1 (mod 4).
    # 5 ≡ 1, 13 ≡ 1, 17 ≡ 1, 29 ≡ 1.
    # So Paley graphs exist for 5, 13, 17, 29.
    
    paley_primes = [5, 13, 17, 29]
    
    # Strategy:
    # 1. Try Paley colorings for primes in descending order (29, 17, 13, 5).
    #    If a Paley coloring is valid, return it.
    # 2. If not, fall back to local search.
    
    for p in sorted(paley_primes, reverse=True):
        if time.time() > deadline - 30:
            break
            
        if p <= best_n:
            continue
            
        colors = paley_coloring(p, p)
        if colors is None:
            continue
            
        # Quick check: Is it valid?
        # For n=29, is_valid_coloring takes ~5s.
        # We can afford this check.
        
        # Optimization: Only do full check if we are close to deadline or if we want to be sure.
        # Let's assume Paley is likely valid for small n.
        # For n=29, it's the frontier.
        
        # To save time, let's run a few steps of local search to see if violations are 0.
        # If local search starts with 0 violations, we are done.
        
        # Let's integrate the local search here.
        
        def local_search(n, initial_colors, time_limit):
            """
            Optimized local search using NumPy for vectorized updates where possible.
            """
            colors = np.array(initial_colors, dtype=np.int8)
            
            # Precompute edge indices
            edge_to_idx = {}
            edges = []
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    edge_to_idx[(i, j)] = idx
                    edges.append((i, j))
                    idx += 1
            num_edges = idx
            
            # Precompute instances (K2,2,2 structures)
            # This is the expensive part.
            # C(29, 6) * 15 = 7.5M instances.
            # Storing them is heavy.
            # Alternative: On-the-fly violation detection?
            # No, we need incremental updates.
            
            # Let's try to build the instance structure only if we have time.
            # If n=29, this takes ~5-10s to build in Python.
            
            if time.time() > time_limit - 15:
                return None, None
            
            instances = []
            edge_to_instances = [[] for _ in range(num_edges)]
            
            partitions = [
                [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
                [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
                [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
                [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
                [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
            ]
            
            gen_start = time.time()
            for vertices in combinations(range(n), 6):
                if time.time() > time_limit - 10:
                    return None, None
                v = vertices
                for part in partitions:
                    p0 = [v[part[0][0]], v[part[0][1]]]
                    p1 = [v[part[1][0]], v[part[1][1]]]
                    p2 = [v[part[2][0]], v[part[2][1]]]
                    
                    edge_indices = []
                    for u in p0:
                        for w in p1:
                            edge_indices.append(edge_to_idx[(min(u,w), max(u,w))])
                    for u in p0:
                        for w in p2:
                            edge_indices.append(edge_to_idx[(min(u,w), max(u,w))])
                    for u in p1:
                        for w in p2:
                            edge_indices.append(edge_to_idx[(min(u,w), max(u,w))])
                    
                    inst_id = len(instances)
                    instances.append(edge_indices)
                    for e in edge_indices:
                        edge_to_instances[e].append(inst_id)
            
            num_instances = len(instances)
            
            # Initialize counts
            # count_0[i] = number of edges in instance i colored 0
            # count_1[i] = number of edges in instance i colored 1
            # Note: count_0 + count_1 = 12 for all i
            
            count_0 = np.zeros(num_instances, dtype=np.int8)
            
            # Vectorized initialization is hard because instances are irregular lists.
            # We'll do it in a loop, but it's 7.5M * 12 ops = 90M ops. Might be slow in Python.
            # Let's try to optimize this.
            
            # Alternative: Just use the is_valid_coloring check?
            # If we can't build the structure fast, we can't do efficient local search.
            
            # Let's check time.
            if time.time() - gen_start > 10:
                return None, None
                
            # Initialize counts
            for i in range(num_instances):
                # Sum of 1 - colors[edge] for edge in instances[i]
                # colors is 0/1.
                # count_0 = sum(1 - colors[e] for e in instances[i])
                # Using numpy: 1 - colors[np.array(instances[i])]
                # This creates a temp array. 7.5M times. Might be slow.
                
                # Let's just do a simple loop for now, or use a precomputed sum?
                # Actually, we can precompute the edge colors as a numpy array.
                # count_0[i] = np.sum(1 - colors[np.array(instances[i])])
                
                # To speed up, let's convert instances to a list of numpy arrays?
                # No, that's memory heavy.
                
                # Let's just do the loop. If it's too slow, we fallback.
                s = 0
                for e in instances[i]:
                    if colors[e] == 0:
                        s += 1
                count_0[i] = s
            
            # Identify violations
            # Violation if count_0 == 12 or count_0 == 0 (since count_1 = 12 - count_0)
            is_viol = (count_0 == 12) | (count_0 == 0)
            violating_indices = np.where(is_viol)[0]
            
            if len(violating_indices) == 0:
                return colors.tolist(), n
            
            # Local Search
            # We need to update counts when flipping an edge.
            # Flip edge e:
            #   colors[e] ^= 1
            #   For each instance i containing e:
            #       if old_color was 0, count_0[i] -= 1
            #       if old_color was 1, count_0[i] += 1
            #   Update is_viol for those instances.
            
            # To make this fast, we need to quickly find instances containing e.
            # edge_to_instances[e] gives this list.
            
            # We also need to maintain the set of violating instances efficiently.
            # A Python set is good.
            
            viol_set = set(violating_indices.tolist())
            
            max_steps = 50000
            step = 0
            while len(viol_set) > 0 and step < max_steps:
                if time.time() > time_limit - 2:
                    break
                
                # Pick a random violating instance
                if not viol_set:
                    break
                inst_id = random.choice(list(viol_set))
                
                # Pick a random edge in this instance
                e_idx = random.choice(instances[inst_id])
                
                old_color = colors[e_idx]
                new_color = 1 - old_color
                colors[e_idx] = new_color
                
                # Update counts for all instances involving e_idx
                for inst_id_2 in edge_to_instances[e_idx]:
                    if old_color == 0:
                        count_0[inst_id_2] -= 1
                    else:
                        count_0[inst_id_2] += 1
                    
                    # Check violation status
                    c0 = count_0[inst_id_2]
                    is_viol_now = (c0 == 12) or (c0 == 0)
                    
                    if is_viol_now:
                        viol_set.add(inst_id_2)
                    else:
                        viol_set.discard(inst_id_2)
                
                step += 1
                
                if len(viol_set) == 0:
                    return colors.tolist(), n
            
            return None, None

        # Run local search on the Paley coloring
        # Give it a reasonable time budget
        search_time = min(30, deadline - time.time() - 5)
        if search_time > 5:
            res_colors, res_n = local_search(p, colors, time.time() + search_time)
            if res_colors is not None and res_n > best_n:
                # Verify
                if is_valid_coloring(res_n, res_colors):
                    best_result = {'n': res_n, 'colors': res_colors}
                    best_n = res_n
                    if best_n == 29:
                        return best_result

    # If we haven't found a better solution, try random local search for n=28, 27
    for n in [28, 27]:
        if time.time() > deadline - 15:
            break
        if n <= best_n:
            continue
            
        # Random start
        num_edges = n * (n - 1) // 2
        random_colors = [random.randint(0, 1) for _ in range(num_edges)]
        
        search_time = min(20, deadline - time.time() - 5)
        if search_time > 5:
            res_colors, res_n = local_search(n, random_colors, time.time() + search_time)
            if res_colors is not None and res_n > best_n:
                if is_valid_coloring(res_n, res_colors):
                    best_result = {'n': res_n, 'colors': res_colors}
                    best_n = res_n
                    if best_n == 29:
                        return best_result

    return best_result
