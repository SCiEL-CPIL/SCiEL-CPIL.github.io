"""
Schütte's problem k=4: tournament where every 4-set has a common dominator.
Strategy: Probabilistic dominator-based construction + targeted repair.
"""
import numpy as np
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 115.0
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    def orient_to_adj(n, orient):
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if orient[idx] == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
                idx += 1
        return adj
    
    def adj_to_orient(n, adj):
        orient = []
        for i in range(n):
            for j in range(i+1, n):
                orient.append(0 if adj[i][j] == 1 else 1)
        return orient
    
    def get_out_masks(n, adj):
        out_mask = [0] * n
        for v in range(n):
            mask = 0
            for w in range(n):
                if adj[v][w]:
                    mask |= (1 << w)
            out_mask[v] = mask
        return out_mask
    
    def find_violations(n, adj, out_mask):
        violations = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            found = False
            for v in range(n):
                if (out_mask[v] & quad_mask) == quad_mask:
                    found = True
                    break
            if not found:
                violations.append(quad)
        return violations
    
    def try_repair(n, adj, out_mask, max_iters=20000):
        """Targeted repair: for each violation, find the vertex that's closest to dominating it, flip edges."""
        for it in range(max_iters):
            if time_left() < 5:
                break
            violations = find_violations(n, adj, out_mask)
            if not violations:
                return adj, out_mask, True
            
            # Pick a random violation
            quad = random.choice(violations)
            a, b, c, d = quad
            quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            
            # Find vertex that dominates the most of the quad
            best_v = -1
            best_count = -1
            for v in range(n):
                count = bin(out_mask[v] & quad_mask).count('1')
                if count > best_count:
                    best_count = count
                    best_v = v
            
            # Flip edges from best_v to undominated vertices in quad
            if best_count >= 2:  # At least 2 of 4 already dominated
                for w in quad:
                    if not adj[best_v][w]:
                        adj[best_v][w] = 1
                        adj[w][best_v] = 0
                        out_mask[best_v] |= (1 << w)
                        out_mask[w] &= ~(1 << best_v)
            
            # Occasionally do random perturbation to escape local minima
            if it % 50 == 0 and best_count < 3:
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i != j:
                    adj[i][j] = 1 - adj[i][j]
                    adj[j][i] = 1 - adj[j][i]
                    out_mask[i] = (out_mask[i] & ~(1 << j)) | ((1 - adj[i][j]) << j) if adj[i][j] else (out_mask[i] & ~(1 << j))
                    out_mask[j] = (out_mask[j] & ~(1 << i)) | ((1 - adj[j][i]) << i) if adj[j][i] else (out_mask[j] & ~(1 << i))
                    # Recompute masks properly
                    out_mask[i] = 0
                    for w in range(n):
                        if adj[i][w]:
                            out_mask[i] |= (1 << w)
                    out_mask[j] = 0
                    for w in range(n):
                        if adj[j][w]:
                            out_mask[j] |= (1 << w)
        
        violations = find_violations(n, adj, out_mask)
        return adj, out_mask, len(violations) == 0
    
    def dominator_based_construction(n, num_dominators, seed=42):
        """
        Construct tournament with explicit dominator structure.
        First num_dominators vertices are 'dominators', rest are 'clients'.
        Each dominator dominates a random subset of clients.
        Edges among dominators and within clients are random.
        """
        random.seed(seed)
        adj = [[0]*n for _ in range(n)]
        
        D = list(range(num_dominators))
        C = list(range(num_dominators, n))
        
        # Orient edges among dominators randomly
        for i in range(len(D)):
            for j in range(i+1, len(D)):
                if random.random() < 0.5:
                    adj[D[i]][D[j]] = 1
                else:
                    adj[D[j]][D[i]] = 1
        
        # Orient edges among clients randomly
        for i in range(len(C)):
            for j in range(i+1, len(C)):
                if random.random() < 0.5:
                    adj[C[i]][C[j]] = 1
                else:
                    adj[C[j]][C[i]] = 1
        
        # Orient edges between dominators and clients:
        # Each dominator dominates ~half the clients on average
        for d in D:
            for c in C:
                if random.random() < 0.5:
                    adj[d][c] = 1
                else:
                    adj[c][d] = 1
        
        return adj
    
    # Fallback: Paley(83)
    p = 83
    squares = set(x * x % p for x in range(1, p))
    adj_paley = [[0]*p for _ in range(p)]
    for i in range(p):
        for j in range(i+1, p):
            if (j - i) % p in squares:
                adj_paley[i][j] = 1
            else:
                adj_paley[j][i] = 1
    best_n = p
    best_orient = adj_to_orient(p, adj_paley)
    
    # Try various n with dominator-based construction + repair
    for n in range(67, 83):
        if time_left() < 15:
            break
        
        # Try different numbers of dominators
        for num_dom in [5, 7, 9, 11, 13]:
            if time_left() < 10:
                break
            
            # Try multiple seeds
            for seed in range(20):
                if time_left() < 8:
                    break
                
                adj = dominator_based_construction(n, num_dom, seed=seed)
                out_mask = get_out_masks(n, adj)
                adj, out_mask, success = try_repair(n, adj, out_mask, max_iters=10000)
                
                if success:
                    # Verify
                    final_violations = find_violations(n, adj, out_mask)
                    if not final_violations:
                        best_n = n
                        best_orient = adj_to_orient(n, adj)
                        break
            
            if best_n < 83:
                break
        
        if best_n < 83:
            break
    
    return {'n': best_n, 'orientation': best_orient}
