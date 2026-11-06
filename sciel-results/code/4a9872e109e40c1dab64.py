"""K4-free graph on 18 vertices minimizing max triangle-free induced subgraph."""
import time
import random

# EVOLVE-BLOCK-START

def _has_k4(adj):
    """Check if adjacency (list of sets) contains a K4."""
    n = len(adj)
    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            common = adj[a] & adj[b]
            for c in common:
                if c <= b:
                    continue
                if c in adj[a] and c in adj[b] and (adj[a] & adj[b] & adj[c]):
                    # Found K4: a, b, c, and some vertex in the intersection
                    # Actually need a 4th vertex. Let me redo.
                    pass
    # Better: iterate over all 4-tuples? Too slow.
    # Use: for each edge (a,b), for each c in N(a)∩N(b) with c>b,
    # check if N(a)∩N(b)∩N(c) is non-empty (giving a 4th vertex d)
    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            common_ab = adj[a] & adj[b]
            for c in common_ab:
                if c <= b:
                    continue
                common_abc = common_ab & adj[c]
                if common_abc:
                    return True
    return False

def _max_triangle_free(adj, deadline=None):
    """Exact largest triangle-free induced subgraph size."""
    n = len(adj)
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    full_mask = (1 << n) - 1
    
    def is_independent(mask):
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if adj_mask[v] & (m ^ lsb):
                return False
            m ^= lsb
        return True
    
    def can_add(v, current_mask):
        """Can vertex v be added to current_mask keeping it triangle-free?"""
        neigh_in_set = current_mask & adj_mask[v]
        return is_independent(neigh_in_set)
    
    best = [0]
    
    # Greedy lower bound
    for start in range(n):
        current_mask = 1 << start
        candidates = full_mask ^ (1 << start)
        improved = True
        while improved:
            improved = False
            best_v = -1
            best_count = -1
            c = candidates
            while c:
                lsb = c & (-c)
                w = lsb.bit_length() - 1
                if can_add(w, current_mask):
                    new_mask = current_mask | lsb
                    new_cands = candidates & ~lsb
                    count = 0
                    fc = new_cands
                    while fc:
                        lsb2 = fc & (-fc)
                        u = lsb2.bit_length() - 1
                        if can_add(u, new_mask):
                            count += 1
                        fc ^= lsb2
                    if count > best_count:
                        best_count = count
                        best_v = w
                c ^= lsb
            if best_v >= 0:
                current_mask |= (1 << best_v)
                candidates &= ~(1 << best_v)
                improved = True
        sz = bin(current_mask).count('1')
        if sz > best[0]:
            best[0] = sz
    
    # Branch and bound
    def bb(current_mask, candidates, size):
        if deadline is not None and time.time() > deadline:
            return
        if not candidates:
            if size > best[0]:
                best[0] = size
            return
        upper = size + bin(candidates).count('1')
        if upper <= best[0]:
            return
        lsb = candidates & (-candidates)
        v = lsb.bit_length() - 1
        # Include v
        if can_add(v, current_mask):
            bb(current_mask | lsb, candidates & ~lsb, size + 1)
        # Exclude v
        bb(current_mask, candidates & ~lsb, size)
    
    bb(0, full_mask, 0)
    return best[0]

def _edges_to_adj(edges, n=18):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _adj_to_edges(adj, n=18):
    edges = []
    for i in range(n):
        for j in sorted(adj[i]):
            if i < j:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Baseline: K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    baseline_edges = []
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    baseline_edges.append([a, b])
    
    best_adj = _edges_to_adj(baseline_edges)
    best_score = _max_triangle_free(best_adj, deadline=deadline)
    best_edges = baseline_edges
    
    random.seed(12345)
    
    for _ in range(5000):
        if time.time() > deadline - 15:
            break
        
        if random.random() < 0.4:
            # Random graph, then remove edges to make K4-free
            adj = [set() for _ in range(18)]
            for i in range(18):
                for j in range(i + 1, 18):
                    if random.random() < 0.35:
                        adj[i].add(j)
                        adj[j].add(i)
            # Remove edges to eliminate K4s
            for _inner in range(50):
                if not _has_k4(adj):
                    break
                # Find a K4 and remove an edge
                found = False
                for a in range(18):
                    if found:
                        break
                    for b in adj[a]:
                        if b <= a or found:
                            continue
                        common = adj[a] & adj[b]
                        for c in common:
                            if c <= b or found:
                                continue
                            common_abc = common & adj[c]
                            if common_abc:
                                # Remove edge (a,b)
                                adj[a].discard(b)
                                adj[b].discard(a)
                                found = True
                                break
                if not found:
                    break
        else:
            # Perturb best
            adj = [set(s) for s in best_adj]
            num_ops = random.randint(1, 6)
            for _ in range(num_ops):
                u = random.randint(0, 17)
                v = random.randint(0, 17)
                if u == v:
                    continue
                if random.random() < 0.5:
                    if v not in adj[u]:
                        adj[u].add(v)
                        adj[v].add(u)
                        if _has_k4(adj):
                            adj[u].discard(v)
                            adj[v].discard(u)
                else:
                    if v in adj[u]:
                        adj[u].discard(v)
                        adj[v].discard(u)
        
        edges = _adj_to_edges(adj)
        adj_check = _edges_to_adj(edges)
        if _has_k4(adj_check):
            continue
        
        score = _max_triangle_free(adj_check, deadline=deadline)
        if score < best_score:
            best_score = score
            best_adj = adj_check
            best_edges = edges
    
    return best_edges

# EVOLVE-BLOCK-END
