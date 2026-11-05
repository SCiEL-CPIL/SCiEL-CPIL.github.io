import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n = 18
    FULL = (1 << n) - 1
    
    def popcount(x):
        return bin(x).count('1')
    
    def get_bitmask(edges):
        bm = [0] * n
        for u, v in edges:
            bm[u] |= (1 << v)
            bm[v] |= (1 << u)
        return bm
    
    def is_k4_free(bm):
        # Check all 4-cliques
        for a in range(n):
            for b in range(a+1, n):
                if not (bm[a] & (1 << b)):
                    continue
                common_ab = bm[a] & bm[b]
                for c in range(b+1, n):
                    if not (common_ab & (1 << c)):
                        continue
                    common_abc = common_ab & bm[c]
                    for d in range(c+1, n):
                        if common_abc & (1 << d):
                            return False
        return True
    
    def max_triangle_free(bm):
        """Find largest triangle-free induced subgraph using B&B."""
        best = [0]
        best_set = [0]
        
        def bb(current, candidates, depth):
            if time.time() > deadline:
                return
            # Upper bound
            if depth + popcount(candidates) <= best[0]:
                return
            if candidates == 0:
                if depth > best[0]:
                    best[0] = depth
                    best_set[0] = current
                return
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with fewest conflicts
            v = None
            best_conf = -1
            for i in range(n):
                if candidates & (1 << i):
                    # Count how many candidates would be excluded if we add i
                    # This is complex; just pick first for now
                    v = i
                    break
            
            if v is None:
                return
            
            # Option 1: exclude v
            bb(current, candidates & ~(1 << v), depth)
            
            # Option 2: include v (if compatible)
            # Check if v is compatible with current
            compatible = True
            cur_bits = current
            while cur_bits:
                lsb = cur_bits & (-cur_bits)
                u = lsb.bit_length() - 1
                if bm[v] & bm[u] & current:
                    compatible = False
                    break
                cur_bits ^= lsb
            
            if compatible:
                # Remove from candidates any vertex w that would form a triangle with v and some u in current
                # w is incompatible if there exists u in current such that bm[v]&bm[u]&bm[w] != 0
                # Actually, w is incompatible with (current ∪ {v}) if there exist u1, u2 in (current ∪ {v}) such that u1-u2-v or similar form triangle
                # Simpler: for each w in candidates, check if w is compatible with (current | (1<<v))
                new_current = current | (1 << v)
                new_candidates = 0
                cand_bits = candidates & ~(1 << v)
                while cand_bits:
                    lsb = cand_bits & (-cand_bits)
                    w = lsb.bit_length() - 1
                    # Check if w is compatible with new_current
                    ok = True
                    nc_bits = new_current
                    while nc_bits:
                        lsb2 = nc_bits & (-nc_bits)
                        u2 = lsb2.bit_length() - 1
                        if bm[w] & bm[u2] & new_current:
                            ok = False
                            break
                        nc_bits ^= lsb2
                    if ok:
                        new_candidates |= lsb
                    cand_bits ^= lsb
                
                bb(new_current, new_candidates, depth + 1)
        
        bb(0, FULL, 0)
        return best[0]
    
    # Initial graph: random K4-free
    def random_k4_free():
        while True:
            bm = [0] * n
            edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.5:
                        bm[u] |= (1 << v)
                        bm[v] |= (1 << u)
                        edges.append([u, v])
            if is_k4_free(bm):
                return bm, edges
    
    bm, edges = random_k4_free()
    best_score = max_triangle_free(bm)
    best_bm = bm[:]
    best_edges = edges[:]
    
    # Local search
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        
        # Try flipping edge (u,v)
        has_edge = bm[u] & (1 << v)
        
        if has_edge:
            # Remove edge
            new_bm = bm[:]
            new_bm[u] &= ~(1 << v)
            new_bm[v] &= ~(1 << u)
            if is_k4_free(new_bm):
                s = max_triangle_free(new_bm)
                if s < best_score:
                    best_score = s
                    best_bm = new_bm
                    best_edges = [[a, b] for a in range(n) for b in range(a+1, n) if new_bm[a] & (1 << b)]
                    bm = new_bm
        else:
            # Add edge
            new_bm = bm[:]
            new_bm[u] |= (1 << v)
            new_bm[v] |= (1 << u)
            if is_k4_free(new_bm):
                s = max_triangle_free(new_bm)
                if s < best_score:
                    best_score = s
                    best_bm = new_bm
                    best_edges = [[a, b] for a in range(n) for b in range(a+1, n) if new_bm[a] & (1 << b)]
                    bm = new_bm
    
    return best_edges

BET: {mechanism: "Local search with bitmask B&B for exact triangle-free subgraph size, starting from random K4-free graphs and flipping edges to minimize the largest triangle-free induced subgraph", q: 0.6, delta: [0.1, 0.5]}
