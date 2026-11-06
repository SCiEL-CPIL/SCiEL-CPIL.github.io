import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 16
    triples = []
    triple_indices = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triple_indices[(a, b, c)] = len(triples)
                triples.append((a, b, c))
    num_triples = len(triples)
    
    constraints = []
    triple_to_constraints = [[] for _ in range(num_triples)]
    
    def get_idx(a, b, c):
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        return triple_indices[(a, b, c)]

    for v1 in range(n):
        for v2 in range(v1 + 1, n):
            for v3 in range(v2 + 1, n):
                for v4 in range(v3 + 1, n):
                    for v5 in range(v4 + 1, n):
                        for v6 in range(v5 + 1, n):
                            verts = [v1, v2, v3, v4, v5, v6]
                            for j in range(1, 6):
                                pair1 = (v1, verts[j])
                                remaining = [v for i, v in enumerate(verts) if i != 0 and i != j]
                                pairings = [
                                    [(remaining[0], remaining[1]), (remaining[2], remaining[3])],
                                    [(remaining[0], remaining[2]), (remaining[1], remaining[3])],
                                    [(remaining[0], remaining[3]), (remaining[1], remaining[2])]
                                ]
                                for p in pairings:
                                    pair2 = p[0]
                                    pair3 = p[1]
                                    t_indices = []
                                    for a in pair1:
                                        for b in pair2:
                                            for c in pair3:
                                                t_idx = get_idx(a, b, c)
                                                t_indices.append(t_idx)
                                                triple_to_constraints[t_idx].append(len(constraints))
                                    constraints.append(t_indices)
                                    
    num_constraints = len(constraints)
    
    offsets = [0]
    for t in range(num_triples):
        offsets.append(offsets[-1] + len(triple_to_constraints[t]))
    data = []
    for t in range(num_triples):
        data.extend(triple_to_constraints[t])
    data = np.array(data, dtype=np.int32)
    offsets = np.array(offsets, dtype=np.int32)
    
    best_result = []
    best_score = 0
    
    # Multiple restarts
    while time.time() < deadline:
        in_set = np.zeros(num_triples, dtype=bool)
        counts = np.zeros(num_constraints, dtype=np.int32)
        
        # Greedy
        order = np.random.permutation(num_triples)
        for t in order:
            start = offsets[t]
            end = offsets[t+1]
            c_indices = data[start:end]
            if np.all(counts[c_indices] < 7):
                in_set[t] = True
                counts[c_indices] += 1
                
        current_size = np.sum(in_set)
        
        # Local Search
        # Time limit per restart: e.g., 10 seconds
        restart_deadline = time.time() + 10.0
        while time.time() < restart_deadline:
            t = np.random.randint(num_triples)
            if in_set[t]:
                continue
                
            start = offsets[t]
            end = offsets[t+1]
            c_indices = data[start:end]
            violated = counts[c_indices] == 7
            num_violated = np.sum(violated)
            
            if num_violated == 0:
                in_set[t] = True
                counts[c_indices] += 1
                current_size += 1
            elif num_violated == 1:
                c_idx = c_indices[np.where(violated)[0][0]]
                c_triples = constraints[c_idx]
                candidates = [x for x in c_triples if in_set[x] and x != t]
                if candidates:
                    t_prime = candidates[np.random.randint(len(candidates))]
                    in_set[t_prime] = False
                    start_p = offsets[t_prime]
                    end_p = offsets[t_prime+1]
                    c_indices_p = data[start_p:end_p]
                    counts[c_indices_p] -= 1
                    
                    in_set[t] = True
                    counts[c_indices] += 1
                    # current_size unchanged
                    
        if current_size > best_score:
            best_score = current_size
            best_result = [list(triples[i]) for i in range(num_triples) if in_set[i]]
            
    return best_result
