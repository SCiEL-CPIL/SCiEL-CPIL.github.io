import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 115
    
    def is_valid_coloring(n, colors):
        """Check if the coloring of K_n has no monochromatic K2,2,2."""
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
        
        colors_arr = np.array(colors, dtype=np.int8)
        
        for vertices in combinations(range(n), 6):
            v = vertices
            for part in partitions:
                p0 = [v[part[0][0]], v[part[0][1]]]
                p1 = [v[part[1][0]], v[part[1][1]]]
                p2 = [v[part[2][0]], v[part[2][1]]]
                
                c0 = 0
                c1 = 0
                for u in p0:
                    for w in p1:
                        idx_e = edge_to_idx[(min(u,w), max(u,w))]
                        if colors_arr[idx_e] == 0: c0 += 1
                        else: c1 += 1
                for u in p0:
                    for w in p2:
                        idx_e = edge_to_idx[(min(u,w), max(u,w))]
                        if colors_arr[idx_e] == 0: c0 += 1
                        else: c1 += 1
                for u in p1:
                    for w in p2:
                        idx_e = edge_to_idx[(min(u,w), max(u,w))]
                        if colors_arr[idx_e] == 0: c0 += 1
                        else: c1 += 1
                
                if c0 == 12 or c1 == 12:
                    return False
        return True

    def paley_coloring(n, p):
        """Generate a coloring based on Paley graph over GF(p)."""
        if n != p:
            return None
        
        squares = np.zeros(p, dtype=int)
        for x in range(1, p):
            squares[x*x % p] = 1
            
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                colors.append(1 if squares[diff] == 1 else 0)
        return colors

    def build_instances(n):
        """
        Build all K2,2,2 instances and their edge incidence.
        Returns:
            instances: list of lists of edge indices (one per instance)
            edge_to_instances: list of lists of instance indices for each edge
        """
        edge_to_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = idx
                idx += 1
        num_edges = idx
        
        partitions = [
            [(0,1),(2,3),(4,5)], [(0,1),(2,4),(3,5)], [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)], [(0,2),(1,4),(3,5)], [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)], [(0,3),(1,4),(2,5)], [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)], [(0,4),(1,3),(2,5)], [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)], [(0,5),(1,3),(2,4)], [(0,5),(1,4),(2,3)]
        ]
        
        instances = []
        edge_to_instances = [[] for _ in range(num_edges)]
        
        for vertices in combinations(range(n), 6):
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
        
        return instances, edge_to_instances

    def vectorized_local_search(n, initial_colors, instances, edge_to_instances, time_limit, seed=42):
        """
        Local search with scored edge-flip heuristic.
        For each candidate flip, compute the net change in violation count
        before committing.  Prefer flips that reduce violations.
        """
        rng = random.Random(seed)
        colors = np.array(initial_colors, dtype=np.int8)
        num_edges = n * (n - 1) // 2
        num_instances = len(instances)

        # Precompute instance edge arrays
        inst_arrays = [np.array(inst, dtype=np.int32) for inst in instances]

        # Initialize count_0: number of edges colored 0 in each instance
        count_0 = np.zeros(num_instances, dtype=np.int8)
        for i in range(num_instances):
            count_0[i] = np.sum(colors[inst_arrays[i]] == 0)

        # Identify violations
        is_viol = (count_0 == 12) | (count_0 == 0)
        viol_set = set(np.where(is_viol)[0].tolist())

        if len(viol_set) == 0:
            return colors.tolist(), n

        best_colors = colors.copy()
        best_viol = len(viol_set)

        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > time_limit - 2:
                break

            if restart > 0:
                # Perturb: flip some random edges
                num_flips = max(1, num_edges // 20)
                for _ in range(num_flips):
                    e_idx = rng.randint(0, num_edges - 1)
                    old_color = colors[e_idx]
                    colors[e_idx] = 1 - old_color
                    delta = -1 if old_color == 0 else 1
                    for inst_id_2 in edge_to_instances[e_idx]:
                        count_0[inst_id_2] += delta
                is_viol = (count_0 == 12) | (count_0 == 0)
                viol_set = set(np.where(is_viol)[0].tolist())

            if len(viol_set) == 0:
                return colors.tolist(), n

            if len(viol_set) < best_viol:
                best_viol = len(viol_set)
                best_colors = colors.copy()

            # Scored greedy: pick the flip that most reduces violations
            max_iter = 8000
            for _ in range(max_iter):
                if time.time() > time_limit - 2:
                    break
                if not viol_set:
                    break

                # Sample candidate edges from violating instances
                candidate_edges = set()
                for _ in range(20):
                    inst_id = rng.choice(list(viol_set))
                    e_idx = inst_arrays[inst_id][rng.randint(0, 11)]
                    candidate_edges.add(e_idx)

                best_score = 0
                best_edge = None
                for e_idx in candidate_edges:
                    old_color = colors[e_idx]
                    delta = -1 if old_color == 0 else 1
                    # Score: number of violations fixed minus new violations created
                    score = 0
                    for inst_id_2 in edge_to_instances[e_idx]:
                        c0 = count_0[inst_id_2]
                        is_viol_before = (c0 == 12) or (c0 == 0)
                        c0_after = c0 + delta
                        is_viol_after = (c0_after == 12) or (c0_after == 0)
                        if is_viol_before and not is_viol_after:
                            score += 1
                        elif not is_viol_before and is_viol_after:
                            score -= 1
                    if score > best_score:
                        best_score = score
                        best_edge = e_idx

                if best_edge is None:
                    # No improving move; accept a random move to escape
                    e_idx = rng.choice(list(candidate_edges))
                else:
                    e_idx = best_edge

                old_color = colors[e_idx]
                colors[e_idx] = 1 - old_color
                delta = -1 if old_color == 0 else 1
                affected = edge_to_instances[e_idx]

                new_viol_set = set()
                for inst_id_2 in affected:
                    count_0[inst_id_2] += delta
                    c0 = count_0[inst_id_2]
                    if c0 == 12 or c0 == 0:
                        new_viol_set.add(inst_id_2)

                viol_set -= set(affected)
                viol_set |= new_viol_set

                if len(viol_set) < best_viol:
                    best_viol = len(viol_set)
                    best_colors = colors.copy()

                if len(viol_set) == 0:
                    return colors.tolist(), n

        if best_viol < len(viol_set):
            return best_colors.tolist(), n
        return None, None

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

    # Strategy:
    # 1. Try Paley colorings for primes in descending order (29, 17, 13, 5).
    # 2. For each, run vectorized local search to fix violations.
    # 3. If not, try random starts with local search.
    
    paley_primes = [29, 17, 13, 5]
    
    for p in paley_primes:
        if time.time() > deadline - 20:
            break
            
        if p <= best_n:
            continue
            
        colors = paley_coloring(p, p)
        if colors is None:
            continue
        
        # Quick validity check
        if is_valid_coloring(p, colors):
            best_result = {'n': p, 'colors': colors}
            best_n = p
            if best_n >= 29:
                return best_result
            continue
        
        # Build instances and run local search
        search_time = min(25, deadline - time.time() - 5)
        if search_time < 5:
            continue
        
        try:
            instances, edge_to_instances = build_instances(p)
        except Exception:
            continue
        
        if time.time() > deadline - 10:
            continue
        
        res_colors, res_n = vectorized_local_search(
            p, colors, instances, edge_to_instances,
            time.time() + search_time, seed=random.randint(0, 1000000)
        )
        
        if res_colors is not None and res_n > best_n:
            if is_valid_coloring(res_n, res_colors):
                best_result = {'n': res_n, 'colors': res_colors}
                best_n = res_n
                if best_n >= 29:
                    return best_result

    # Try multi-start random search for n=29 (and 28) if not yet found
    for n in [29, 28]:
        if time.time() > deadline - 15:
            break
        if n <= best_n:
            continue

        num_edges = n * (n - 1) // 2
        max_starts = 5
        for start in range(max_starts):
            if time.time() > deadline - 10:
                break

            random_colors = [random.randint(0, 1) for _ in range(num_edges)]

            search_time = min(12, deadline - time.time() - 5)
            if search_time < 4:
                continue

            try:
                instances, edge_to_instances = build_instances(n)
            except Exception:
                continue

            if time.time() > deadline - 8:
                continue

            res_colors, res_n = vectorized_local_search(
                n, random_colors, instances, edge_to_instances,
                time.time() + search_time, seed=random.randint(0, 1000000)
            )

            if res_colors is not None and res_n > best_n:
                if is_valid_coloring(res_n, res_colors):
                    best_result = {'n': res_n, 'colors': res_colors}
                    best_n = res_n
                    if best_n >= 29:
                        return best_result
                    break  # Found a valid coloring for this n; move to next n

    return best_result
