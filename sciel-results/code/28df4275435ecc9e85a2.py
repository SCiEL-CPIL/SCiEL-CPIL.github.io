import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 16
    A = list(range(3))
    B = list(range(3, 16))
    
    base_edges = []
    base_set = set()
    for a in A:
        for b in B:
            base_edges.append([a, b])
            base_set.add((a, b))
            base_set.add((b, a))
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            base_edges.append([A[i], A[j]])
            base_set.add((A[i], A[j]])
            base_set.add((A[j], A[i]))
            
    candidate_edges = []
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidate_edges.append([B[i], B[j]])
            
    def is_q3_free(edge_set):
        for combo in combinations(range(n), 8):
            cnt = 0
            for i in range(8):
                for j in range(i+1, 8):
                    if (combo[i], combo[j]) in edge_set:
                        cnt += 1
            if cnt == 12:
                # Check degrees
                degs = [0]*8
                for i in range(8):
                    for j in range(i+1, 8):
                        if (combo[i], combo[j]) in edge_set:
                            degs[i] += 1
                            degs[j] += 1
                if all(d == 3 for d in degs):
                    return False
        return True

    best_edges = base_edges
    best_len = len(base_edges)
    
    while time.time() < deadline:
        random.shuffle(candidate_edges)
        current_edges = list(base_edges)
        current_set = set(base_set)
        
        for e in candidate_edges:
            if time.time() > deadline:
                break
            u, v = e
            if (u, v) in current_set:
                continue
            
            current_set.add((u, v))
            current_set.add((v, u))
            
            if is_q3_free(current_set):
                current_edges.append([u, v])
            else:
                current_set.remove((u, v))
                current_set.remove((v, u))
                
        if len(current_edges) > best_len:
            best_len = len(current_edges)
            best_edges = current_edges
            
    return best_edges
