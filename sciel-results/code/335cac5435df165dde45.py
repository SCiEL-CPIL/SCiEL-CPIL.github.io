import time
from random import Random

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin before the 150s kill
    
    def instance(n, p, seed):
        rng = Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
    
    def popcount(x):
        return bin(x).count('1')
    
    def solve(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        cliques = []
        
        while True:
            if time.time() > deadline:
                break
                
            # Find the edge (u, v) with the maximum common neighbors in the uncovered graph
            best_u = -1
            best_v = -1
            best_count = -1
            
            for u in range(n):
                if adj[u] == 0:
                    continue
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        count = popcount(common)
                        if count > best_count:
                            best_count = count
                            best_u = u
                            best_v = v
                            
            if best_u == -1:
                break
                
            # Build clique starting with best_u, best_v
            clique = [best_u, best_v]
            # Candidates are common neighbors
            candidates_mask = adj[best_u] & adj[best_v]
            
            # Greedy add vertices to make a larger clique
            # We'll iteratively pick the best candidate that is connected to all current clique members
            while True:
                if time.time() > deadline:
                    break
                # Find best candidate: one connected to all in clique, with max connections to other candidates
                best_w = -1
                best_score = -1
                
                # Iterate through candidates_mask bits
                m = candidates_mask
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    
                    if w == best_u or w == best_v:
                        continue
                    
                    # Check if w is connected to all in clique
                    ok = True
                    for c in clique:
                        if not ((adj[w] >> c) & 1):
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    # Score: number of connections to other remaining candidates
                    # Actually, for clique maximization, we just want to add it
                    # To get a larger clique, prefer vertices with higher degree in the candidate set
                    # But since all candidates are connected to all clique members, 
                    # the key is which candidates are mutually connected
                    # Simpler: just add the first valid one, or the one with max degree
                    score = popcount(adj[w] & candidates_mask)
                    if score > best_score:
                        best_score = score
                        best_w = w
                
                if best_w == -1:
                    break
                
                clique.append(best_w)
                # Update candidates_mask: remove best_w and its non-neighbors from candidates
                candidates_mask &= ~(1 << best_w)
                candidates_mask &= adj[best_w]
                
                if not candidates_mask:
                    break
            
            cliques.append(clique)
            
            # Remove edges in this clique from adj
            for i in range(len(clique)):
                ci = clique[i]
                for j in range(i + 1, len(clique)):
                    cj = clique[j]
                    adj[ci] &= ~(1 << cj)
                    adj[cj] &= ~(1 << ci)
                    
        return cliques
    
    c1 = solve(32, 0.6, 601)
    c2 = solve(64, 0.55, 602)
    
    return [c1, c2]
