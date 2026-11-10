import random
import time
import math

def construct():
        d = 7
        n = 1 << d
        m = d * (1 << (d - 1)) # 7 * 64 = 448 edges
        
        # Build edges and conflicts
        edges = []
        edge_index = {}
        for u in range(n):
            for i in range(d):
                v = u ^ (1 << i)
                if u < v:
                    idx = len(edges)
                    edges.append((u, v))
                    edge_index[(u, v)] = idx
                    edge_index[(v, u)] = idx
        
        # conflicts[i] = list of edge indices j such that edge i and j are in a common face
        conflicts = [[] for _ in range(m)]
        for idx, (u, v) in enumerate(edges):
            diff = u ^ v
            i = diff.bit_length() - 1
            for j in range(d):
                if j == i: continue
                c = u ^ (1 << j)
                e = v ^ (1 << j)
                # The other 3 edges in this face
                e1 = edge_index.get((min(u, c), max(u, c)))
                e2 = edge_index.get((min(v, e), max(v, e)))
                e3 = edge_index.get((min(c, e), max(c, e)))
                for k in (e1, e2, e3):
                    if k is not None and k != idx:
                        conflicts[idx].append(k)
        
        # Deduplicate conflicts? Not strictly necessary for logic, but good for speed.
        # Actually, an edge can conflict with another via multiple faces? No, two edges in Q_n are in at most one common face unless they are disjoint? 
        # If they share a vertex, they are in 0 or 1 face? 
        # If they are disjoint, they can be in multiple faces? 
        # In Q_n, two edges are in a common face iff they are disjoint and their endpoints form a C4, OR they share a vertex?
        # Wait, a face is a C4. The edges of a face are disjoint pairs.
        # So "conflict" means "cannot be in the same C4-free set together".
        # Two edges conflict if they are part of the same face.
        # In Q_n, two edges are in a common face iff they are disjoint and the distance between their midpoints is 1? 
        # Actually, just checking the 3 other edges of the face formed by extending one edge in direction j is sufficient.
        
        # Simulated Annealing
        def run_sa(seed, time_limit):
            rng = random.Random(seed)
            start_time = time.time()
            
            # Init: Random Greedy
            chosen = [False] * m
            order = list(range(m))
            rng.shuffle(order)
            for idx in order:
                ok = True
                for k in conflicts[idx]:
                    if chosen[k]:
                        ok = False
                        break
                if ok:
                    chosen[idx] = True
            
            current_size = sum(chosen)
            best_size = current_size
            best_chosen = chosen[:]
            
            T = 10.0
            T_min = 0.01
            cooling = 0.995
            
            while time.time() - start_time < time_limit:
                # Pick a move
                # 70% chance: Try to add a random unselected edge (removing conflicts)
                # 30% chance: Try to remove a random selected edge
                if rng.random() < 0.7:
                    # Pick random unselected
                    while True:
                        idx = rng.randint(0, m - 1)
                        if not chosen[idx]:
                            break
                    
                    conf = [k for k in conflicts[idx] if chosen[k]]
                    # Remove conf, add idx
                    for k in conf:
                        chosen[k] = False
                    chosen[idx] = True
                    
                    delta = 1 - len(conf)
                    new_size = current_size + delta
                    current_size = new_size
                    
                    if new_size > best_size:
                        best_size = new_size
                        best_chosen = chosen[:]
                    elif delta < 0:
                        # Accept with prob exp(delta/T)
                        if rng.random() < math.exp(delta / T):
                            pass # Already applied
                        else:
                            # Revert
                            chosen[idx] = False
                            for k in conf:
                                chosen[k] = True
                            current_size -= delta
                else:
                    # Pick random selected
                    while True:
                        idx = rng.randint(0, m - 1)
                        if chosen[idx]:
                            break
                    
                    chosen[idx] = False
                    delta = -1
                    new_size = current_size + delta
                    current_size = new_size
                    
                    if delta < 0:
                        if rng.random() < math.exp(delta / T):
                            pass
                        else:
                            chosen[idx] = True
                            current_size -= delta
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            return best_chosen, best_size

        deadline = time.time() + 110
        best_global = None
        best_global_size = 0
        
        # Run multiple SA instances
        seed = 42
        while time.time() < deadline - 5:
            time_per_sa = 5.0
            if time.time() + time_per_sa > deadline - 5:
                time_per_sa = deadline - 5 - time.time()
            
            res_chosen, res_size = run_sa(seed, time_per_sa)
            if res_size > best_global_size:
                best_global_size = res_size
                best_global = res_chosen
            
            seed += 1
        
        # Final polish: Greedy add
        if best_global:
            chosen = best_global[:]
            improved = True
            while improved:
                improved = False
                for i in range(m):
                    if not chosen[i]:
                        ok = True
                        for k in conflicts[i]:
                            if chosen[k]:
                                ok = False
                                break
                        if ok:
                            chosen[i] = True
                            improved = True
            
            # Convert to output
            out = []
            for i in range(m):
                if chosen[i]:
                    out.append(list(edges[i]))
            return out
        
        # Fallback
        return []
