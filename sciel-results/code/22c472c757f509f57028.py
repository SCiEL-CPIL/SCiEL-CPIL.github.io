import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin 5s

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency masks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Bron-Kerbosch to find maximal cliques
        max_cliques = []
        
        def bk(R, P, X):
            if P == 0 and X == 0:
                # Extract vertices from R
                clique = [i for i in range(n) if (R >> i) & 1]
                max_cliques.append(clique)
                return
            
            # Pivot selection
            # Choose u in P|X with max |P & N(u)|
            candidates = P | X
            pivot = -1
            max_deg = -1
            temp = candidates
            while temp:
                lsb = temp & -temp
                u = lsb.bit_length() - 1
                deg = (P & adj[u]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = u
                temp ^= lsb
                
            if pivot == -1:
                return
                
            cand = P & ~adj[pivot]
            
            while cand:
                lsb = cand & -cand
                v = lsb.bit_length() - 1
                cand ^= lsb
                
                bk(R | lsb, P & adj[v], X & adj[v])
                
                P &= ~lsb
                X |= lsb
                
        bk(0, (1 << n) - 1, 0)
        
        # Greedy Edge Cover
        uncovered = set(edges)
        solution = []
        
        # Precompute edges for each clique as a set for fast intersection
        # Also store the clique vertices
        clique_data = []
        for clique in max_cliques:
            c_edges = set()
            for i in range(len(clique)):
                u = clique[i]
                for j in range(i+1, len(clique)):
                    v = clique[j]
                    c_edges.add((u, v))
            clique_data.append((clique, c_edges))
            
        # Sort by size descending? Might help tie-breaking or just let greedy handle it.
        # Greedy picks max intersection.
        
        while uncovered:
            if time.time() > deadline:
                # Emergency fallback: return trivial partition for remaining?
                # We must return a VALID partition.
                # If we stop here, we have `solution` covering some edges.
                # We need to cover the rest.
                # The safest valid fallback is to add each remaining edge as a clique of size 2.
                for e in uncovered:
                    solution.append(list(e))
                break
                
            best_idx = -1
            best_count = -1
            for i, (clique, c_edges) in enumerate(clique_data):
                if not c_edges:
                    continue
                # Intersection
                count = len(c_edges & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
                    if best_count == len(uncovered):
                        break
                        
            if best_idx == -1:
                # Should not happen, but if it does, fallback
                for e in uncovered:
                    solution.append(list(e))
                break
                
            solution.append(clique_data[best_idx][0])
            uncovered -= clique_data[best_idx][1]
            # Mark as used to avoid re-checking
            clique_data[best_idx] = (clique_data[best_idx][0], set())
            
        return solution

    res1 = solve(32, 0.6, 601)
    res2 = solve(64, 0.55, 602)
    
    return [res1, res2]
