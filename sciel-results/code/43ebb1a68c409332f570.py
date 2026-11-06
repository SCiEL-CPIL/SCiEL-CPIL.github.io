import numpy as np
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    n = 16
    # Generate all triples
    triples = []
    triple_indices = {}
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triple_indices[(a, b, c)] = len(triples)
                triples.append((a, b, c))
    num_triples = len(triples)
    
    # Generate constraints
    # We need to map each constraint to its 8 triples.
    # And each triple to the list of constraints it belongs to.
    constraint_to_triples = []
    triple_to_constraints = [[] for _ in range(num_triples)]
    
    # Helper to get triple index
    def get_idx(a, b, c):
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        return triple_indices[(a, b, c)]

    # Iterate over all sets of 6 points
    for v1 in range(n):
        for v2 in range(v1 + 1, n):
            for v3 in range(v2 + 1, n):
                for v4 in range(v3 + 1, n):
                    for v5 in range(v4 + 1, n):
                        for v6 in range(v5 + 1, n):
                            verts = [v1, v2, v3, v4, v5, v6]
                            
                            # Generate 15 pairings
                            # Pair v1 with v2..v6
                            for j in range(1, 6):
                                pair1 = (verts[0], verts[j])
                                remaining = [v for i, v in enumerate(verts) if i != 0 and i != j]
                                
                                # 3 pairings of remaining 4
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
                                                triple_to_constraints[t_idx].append(len(constraint_to_triples))
                                    
                                    constraint_to_triples.append(t_indices)
                                    
    num_constraints = len(constraint_to_triples)
    
    best_size = -1
    best_present = None

    max_restarts = 1000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        random.seed(restart)
        
        present = np.ones(num_triples, dtype=bool)
        counts = np.full(num_constraints, 8, dtype=np.int32)
        
        # Initial state: all constraints are violated
        is_violated = np.ones(num_constraints, dtype=bool)
        violated_list = list(range(num_constraints))
        
        # Initialize viol_deg
        viol_deg = np.zeros(num_triples, dtype=np.int32)
        for c in violated_list:
            for t in constraint_to_triples[c]:
                viol_deg[t] += 1

        removed_count = 0
        
        while violated_list:
            if time.time() > deadline:
                break
                
            idx = random.randint(0, len(violated_list) - 1)
            c = violated_list[idx]
            
            if not is_violated[c]:
                violated_list[idx] = violated_list[-1]
                violated_list.pop()
                continue
                
            best_t = -1
            best_score = -1
            for t in constraint_to_triples[c]:
                if present[t]:
                    score = viol_deg[t]
                    score += random.random() * 0.1
                    if score > best_score:
                        best_score = score
                        best_t = t
                        
            present[best_t] = False
            removed_count += 1
            
            for c2 in triple_to_constraints[best_t]:
                counts[c2] -= 1
                if counts[c2] == 7:
                    if is_violated[c2]:
                        is_violated[c2] = False
                        for t2 in constraint_to_triples[c2]:
                            viol_deg[t2] -= 1
                        
        # Local Search Phase
        local_search_steps = 500
        for step in range(local_search_steps):
            if time.time() > deadline:
                break
            
            t = random.randint(0, num_triples - 1)
            if present[t]:
                continue
                
            violated_constraints = []
            for c in triple_to_constraints[t]:
                if counts[c] == 7:
                    violated_constraints.append(c)
                    
            if not violated_constraints:
                present[t] = True
                removed_count -= 1
                for c in triple_to_constraints[t]:
                    counts[c] += 1
                    if counts[c] == 8:
                        is_violated[c] = True
                        for t2 in constraint_to_triples[c]:
                            viol_deg[t2] += 1
                continue
            
            R_list = []
            current_violated = list(violated_constraints)
            
            while current_violated:
                if time.time() > deadline:
                    break
                
                freq = defaultdict(int)
                for c in current_violated:
                    for u in constraint_to_triples[c]:
                        if present[u] and u != t:
                            freq[u] += 1
                            
                if not freq:
                    break
                    
                best_u = max(freq, key=freq.get)
                R_list.append(best_u)
                
                new_current_violated = []
                for c in current_violated:
                    if best_u in constraint_to_triples[c]:
                        pass
                    else:
                        new_current_violated.append(c)
                current_violated = new_current_violated
                
            if len(R_list) == 0 or current_violated:
                continue
                
            delta_size = 1 - len(R_list)
            
            accept = False
            if delta_size > 0:
                accept = True
            elif delta_size == 0:
                accept = random.random() < 0.5
            else:
                accept = False
                
            if accept:
                present[t] = True
                removed_count -= 1
                for c in triple_to_constraints[t]:
                    counts[c] += 1
                    if counts[c] == 8:
                        is_violated[c] = True
                        for t2 in constraint_to_triples[c]:
                            viol_deg[t2] += 1
                            
                for u in R_list:
                    present[u] = False
                    removed_count += 1
                    for c in triple_to_constraints[u]:
                        counts[c] -= 1
                        if counts[c] == 7:
                            if is_violated[c]:
                                is_violated[c] = False
                                for t2 in constraint_to_triples[c]:
                                    viol_deg[t2] -= 1
                                
        current_size = num_triples - removed_count
        if current_size > best_size:
            best_size = current_size
            best_present = present.copy()
            
    if best_present is None:
        return []
        
    result = []
    for i in range(num_triples):
        if best_present[i]:
            result.append(list(triples[i]))
            
    return result
