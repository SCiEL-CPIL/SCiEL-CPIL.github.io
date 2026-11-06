import time
import numpy as np
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s safety margin
    
    def get_paley_colors(n, p):
        """Generate a coloring based on Paley graph over GF(p) for the first n vertices."""
        if n > p:
            return None
        squares = np.zeros(p, dtype=np.int8)
        for x in range(1, p):
            squares[x * x % p] = 1
        colors = np.zeros(n * (n - 1) // 2, dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                colors[idx] = squares[diff]
                idx += 1
        return colors

    def build_instances(n):
        """
        Build all K2,2,2 instances and their edge incidence.
        Returns:
            inst_edges: np.array of shape (num_instances, 12) with edge indices
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
        
        inst_edge_list = []
        
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
                
                inst_edge_list.append(edge_indices)
        
        inst_edges = np.array(inst_edge_list, dtype=np.int32)
        num_instances = len(inst_edge_list)
        
        # Build reverse mapping
        edge_to_instances = [[] for _ in range(num_edges)]
        for i, edges in enumerate(inst_edge_list):
            for e in edges:
                edge_to_instances[e].append(i)
        
        return inst_edges, edge_to_instances, num_edges

    def count_violations(colors, inst_edges):
        """Count number of monochromatic K2,2,2 instances."""
        # colors is int8 array of 0s and 1s
        # inst_edges is int32 array of shape (N, 12)
        # For each instance, check if all edges are 0 or all are 1.
        # This can be done by summing colors for each instance.
        # sum == 0 -> all 1s (if colors are 0/1, wait, if colors are 0/1, sum==0 means all 0, sum==12 means all 1).
        # Let's say color 0 is red, 1 is blue.
        # sum == 0 -> all red.
        # sum == 12 -> all blue.
        sums = np.sum(colors[inst_edges], axis=1)
        viol_mask = (sums == 0) | (sums == 12)
        return np.sum(viol_mask)

    def local_search(n, initial_colors, inst_edges, edge_to_instances, time_limit, seed=42):
        """
        Local search to eliminate monochromatic K2,2,2.
        """
        rng = random.Random(seed)
        colors = initial_colors.copy()
        num_edges = len(colors)
        num_instances = inst_edges.shape[0]
        
        # Initialize sums
        sums = np.sum(colors[inst_edges], axis=1)
        viol_count = np.sum((sums == 0) | (sums == 12))
        
        if viol_count == 0:
            return colors, n
        
        best_colors = colors.copy()
        best_viol = viol_count
        T = 2.0
        T_min = 0.05
        cooling = 0.9995
        
        # Precompute affected instances for each edge for faster access?
        # edge_to_instances is a list of lists.
        
        while viol_count > 0 and time.time() < time_limit:
            # Pick a random edge to flip
            # Better: Pick an edge involved in a violation
            # Find violating instances
            viol_mask = (sums == 0) | (sums == 12)
            if not np.any(viol_mask):
                break
            
            viol_indices = np.where(viol_mask)[0]
            # Pick a random violating instance
            inst_id = viol_indices[rng.randint(0, len(viol_indices) - 1)]
            # Pick a random edge in this instance
            e_idx = inst_edges[inst_id, rng.randint(0, 11)]
            
            old_color = colors[e_idx]
            new_color = 1 - old_color
            
            # Calculate delta in sums for affected instances
            # If old_color is 0, new is 1 -> sum increases by 1
            # If old_color is 1, new is 0 -> sum decreases by 1
            delta = 1 if old_color == 0 else -1
            
            affected = edge_to_instances[e_idx]
            
            # Update sums
            old_sums = sums[affected].copy()
            sums[affected] += delta
            
            # Calculate change in violation count
            old_viol = np.sum((old_sums == 0) | (old_sums == 12))
            new_viol = np.sum((sums[affected] == 0) | (sums[affected] == 12))
            delta_viol = new_viol - old_viol
            
            # Simulated Annealing acceptance
            if delta_viol < 0:
                accept = True
            elif delta_viol == 0:
                accept = True
            else:
                # Accept with probability exp(-delta_viol / T)
                # delta_viol is integer, T is float
                if T > 0:
                    prob = np.exp(-delta_viol / T)
                    accept = rng.random() < prob
                else:
                    accept = False
            
            if accept:
                colors[e_idx] = new_color
                viol_count += delta_viol
                if viol_count < best_viol:
                    best_viol = viol_count
                    best_colors = colors.copy()
                    if best_viol == 0:
                        return best_colors, n
            else:
                # Revert sums
                sums[affected] -= delta
            
            T = max(T_min, T * cooling)
            
        return best_colors, n

    # Try n=30 first
    # We need an initial coloring. Paley for p=31 (prime > 30)
    # Or random
    best_n = 0
    best_colors = []
    
    # Strategy: Try to solve n=30. If fails, try n=29.
    target_ns = [30, 29]
    
    for n in target_ns:
        if time.time() > deadline - 10: # Need time for setup and search
            break
            
        print(f"Trying n={n}")
        inst_edges, edge_to_instances, num_edges = build_instances(n)
        
        # Try multiple initial colorings
        # 1. Paley-like (if n <= 31, use p=31)
        # 2. Random
        initializations = []
        
        # Paley p=31
        p = 31
        squares = np.zeros(p, dtype=np.int8)
        for x in range(1, p):
            squares[x * x % p] = 1
        paley_colors = np.zeros(num_edges, dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                paley_colors[idx] = squares[diff]
                idx += 1
        initializations.append(paley_colors)
        
        # Random
        for _ in range(5):
            initializations.append(np.random.randint(0, 2, size=num_edges).astype(np.int8))
            
        # Run local search for each initialization
        for init_idx, init_colors in enumerate(initializations):
            if time.time() > deadline - 5:
                break
                
            # Allocate time per initialization
            time_per_init = max(1.0, (deadline - time.time()) / (len(initializations) - init_idx))
            time_limit = time.time() + time_per_init
            
            res_colors, res_n = local_search(n, init_colors, inst_edges, edge_to_instances, time_limit, seed=init_idx)
            
            if res_n == n:
                # Verify
                v = count_violations(res_colors, inst_edges)
                if v == 0:
                    if n > best_n:
                        best_n = n
                        best_colors = res_colors.tolist()
                    # If we found n=30, we can stop early as it's the max possible for improvement over 29
                    if n == 30:
                        return {'n': n, 'colors': best_colors}
                        
        # If we didn't find a valid coloring for this n, continue to next n
        
    # Fallback to n=17 if nothing else worked (should not happen)
    if best_n == 0:
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        best_colors = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
        best_n = 17
        
    return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END
