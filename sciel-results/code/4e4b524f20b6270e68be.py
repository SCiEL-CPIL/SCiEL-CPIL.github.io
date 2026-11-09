import time
import random

def _has_k222(triple, edge_set, N):
    a, b, c = triple
    ts = edge_set
    
    key_bc = (b, c) if b < c else (c, b)
    key_ac = (a, c) if a < c else (c, a)
    key_ab = (a, b) if a < b else (b, a)
    
    if key_bc not in N or key_ac not in N or key_ab not in N:
        return False
        
    a_primes = N[key_bc]
    b_primes = N[key_ac]
    c_primes = N[key_ab]
    
    if not a_primes or not b_primes or not c_primes:
        return False

    for a_prime in a_primes:
        if a_prime in (a, b, c): continue
        for b_prime in b_primes:
            if b_prime in (a, b, c, a_prime): continue
            
            t4 = tuple(sorted((a_prime, b_prime, c)))
            if t4 not in ts: continue
            
            for c_prime in c_primes:
                if c_prime in (a, b, c, a_prime, b_prime): continue
                
                t5 = tuple(sorted((a_prime, b, c_prime)))
                if t5 not in ts: continue
                
                t6 = tuple(sorted((a, b_prime, c_prime)))
                if t6 not in ts: continue
                
                t7 = tuple(sorted((a_prime, b_prime, c_prime)))
                if t7 not in ts: continue
                
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    all_triples = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                all_triples.append((a, b, c))
                
    random.seed(42)
    
    best_edges = set()
    
    def update_N_incremental(N, triple, add):
        a, b, c = triple
        pairs = [(a, b), (a, c), (b, c)]
        for p in pairs:
            key = (p[0], p[1]) if p[0] < p[1] else (p[1], p[0])
            if p[0] == a and p[1] == b: other = c
            elif p[0] == a and p[1] == c: other = b
            else: other = a
            
            if add:
                if key not in N: N[key] = set()
                N[key].add(other)
            else:
                if key in N:
                    N[key].discard(other)
                    if not N[key]: del N[key]

    def try_add(edge_set, N, triple):
        if triple in edge_set: return False
        if _has_k222(triple, edge_set, N): return False
        edge_set.add(triple)
        update_N_incremental(N, triple, True)
        return True

    def remove_edge(edge_set, N, triple):
        if triple not in edge_set: return
        edge_set.discard(triple)
        update_N_incremental(N, triple, False)

    current_edges = set()
    current_N = {}
    
    # Greedy Init
    shuffled = all_triples[:]
    random.shuffle(shuffled)
    for t in shuffled:
        if time.time() > deadline: break
        try_add(current_edges, current_N, t)
        
    if len(current_edges) > len(best_edges):
        best_edges = set(current_edges)
        
    # Local Search
    while time.time() < deadline:
        kick_size = random.choice([1, 2, 3])
        if len(current_edges) < kick_size:
            kick_size = len(current_edges)
            if kick_size == 0: break
            
        to_remove = random.sample(list(current_edges), kick_size)
        backup_edges = set(current_edges)
        backup_N = dict(current_N)
        
        for t in to_remove:
            remove_edge(current_edges, current_N, t)
            
        candidates = [t for t in all_triples if t not in current_edges]
        random.shuffle(candidates)
        
        for t in candidates:
            if time.time() > deadline: break
            try_add(current_edges, current_N, t)
            
        if len(current_edges) >= len(backup_edges):
            if len(current_edges) > len(best_edges):
                best_edges = set(current_edges)
        else:
            current_edges = backup_edges
            current_N = backup_N

    return [list(t) for t in best_edges]
