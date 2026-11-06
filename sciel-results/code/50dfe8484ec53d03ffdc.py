import time
import random
from collections import defaultdict

D = 7
N = 2 ** D
NUM_EDGES = D * (1 << (D - 1))  # 448

edges = []
edge_to_idx = {}
idx_to_edge = {}

for u in range(N):
    for i in range(D):
        v = u ^ (1 << i)
        if u < v:
            idx = len(edges)
            edges.append((u, v))
            edge_to_idx[(u, v)] = idx
            idx_to_edge[idx] = (u, v)

square_set = set()
squares = []

for i in range(D):
    for j in range(i + 1, D):
        bi = 1 << i
        bj = 1 << j
        for u in range(N):
            e1_u, e1_v = u, u ^ bi
            e2_u, e2_v = u ^ bj, u ^ bi ^ bj
            e3_u, e3_v = u, u ^ bj
            e4_u, e4_v = u ^ bi, u ^ bi ^ bj
            
            def norm(u, v):
                return (u, v) if u < v else (v, u)
            
            e1 = edge_to_idx[norm(e1_u, e1_v)]
            e2 = edge_to_idx[norm(e2_u, e2_v)]
            e3 = edge_to_idx[norm(e3_u, e3_v)]
            e4 = edge_to_idx[norm(e4_u, e4_v)]
            
            sq = frozenset([e1, e2, e3, e4])
            if sq not in square_set:
                square_set.add(sq)
                squares.append(list(sq))

edge_to_squares = defaultdict(list)
for idx, sq in enumerate(squares):
    for e in sq:
        edge_to_squares[e].append(idx)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_solution = set()
    best_size = 0
    
    indices = list(range(NUM_EDGES))
    
    # Greedy Phase
    while time.time() < deadline - 5.0:
        random.shuffle(indices)
        current = set()
        for idx in indices:
            can_add = True
            for sq_idx in edge_to_squares[idx]:
                sq = squares[sq_idx]
                count = 0
                for e in sq:
                    if e != idx and e in current:
                        count += 1
                if count == 3:
                    can_add = False
                    break
            if can_add:
                current.add(idx)
        
        if len(current) > best_size:
            best_size = len(current)
            best_solution = current
            
        if best_size >= 304:
            break

    # Local Search Phase
    current = set(best_solution)
    
    while time.time() < deadline:
        e_add = random.randint(0, NUM_EDGES - 1)
        if e_add in current:
            continue
            
        blocking_sq = -1
        for sq_idx in edge_to_squares[e_add]:
            sq = squares[sq_idx]
            count = 0
            for e in sq:
                if e != e_add and e in current:
                    count += 1
            if count == 3:
                blocking_sq = sq_idx
                break
        
        if blocking_sq == -1:
            current.add(e_add)
            continue
            
        sq = squares[blocking_sq]
        others = [e for e in sq if e != e_add]
        e_remove = random.choice(others)
        
        current.remove(e_remove)
        
        can_add = True
        for sq_idx in edge_to_squares[e_add]:
            sq = squares[sq_idx]
            count = 0
            for e in sq:
                if e != e_add and e in current:
                    count += 1
            if count == 3:
                can_add = False
                break
        
        if can_add:
            current.add(e_add)
        else:
            current.add(e_remove)

    return [[u, v] for u, v in (idx_to_edge[i] for i in current)]
