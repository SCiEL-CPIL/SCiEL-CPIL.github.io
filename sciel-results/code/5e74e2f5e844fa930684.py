import random
import time

def construct():
    N = 180
    vertices = list(range(1, N + 1))
    
    # Precompute forbidden triples
    # 1/a = 1/b + 1/c => c = ab / (b-a)
    # b > a, c > a
    triples = []
    vertex_to_triples = {i: [] for i in vertices}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c <= N and c != a and c != b:
                    if len({a, b, c}) == 3:
                        triples.append((a, b, c))
                        idx = len(triples) - 1
                        vertex_to_triples[a].append(idx)
                        vertex_to_triples[b].append(idx)
                        vertex_to_triples[c].append(idx)
    
    start_time = time.time()
    best_set = set()
    best_size = 0
    
    def repair_and_improve(initial_set, vertex_order):
        S = set(initial_set)
        
        # 1. Repair: Remove vertices to eliminate all violations
        while True:
            # Find violations
            viol = []
            for idx, t in enumerate(triples):
                if t[0] in S and t[1] in S and t[2] in S:
                    viol.append(idx)
            
            if not viol:
                break
            
            # Pick a random violated triple
            v_idx = random.choice(viol)
            a, b, c = triples[v_idx]
            
            # Count how many violated triples each of a, b, c is in
            counts = {a: 0, b: 0, c: 0}
            for v_idx2 in viol:
                t2 = triples[v_idx2]
                if a in t2: counts[a] += 1
                if b in t2: counts[b] += 1
                if c in t2: counts[c] += 1
            
            # Remove the one with the highest count (random tie-break)
            max_count = max(counts.values())
            candidates = [v for v in [a, b, c] if counts[v] == max_count]
            v_to_remove = random.choice(candidates)
            S.remove(v_to_remove)
        
        # 2. Improve: Greedily add vertices in the given order
        improved = True
        while improved:
            improved = False
            for v in vertex_order:
                if v in S:
                    continue
                safe = True
                for t_idx in vertex_to_triples[v]:
                    t = triples[t_idx]
                    others = [x for x in t if x != v]
                    if others[0] in S and others[1] in S:
                        safe = False
                        break
                if safe:
                    S.add(v)
                    improved = True
        
        return S

    for _ in range(20000):
        if time.time() - start_time > 50:
            break
        
        p = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
        init_set = set(v for v in vertices if random.random() < p)
        
        # Shuffle vertex order for the greedy improvement phase
        shuffled_vertices = vertices.copy()
        random.shuffle(shuffled_vertices)
        
        current_set = repair_and_improve(init_set, shuffled_vertices)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    return sorted(best_set)
