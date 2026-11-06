"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

N = 16
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_INDEX = {(i, j): k for k, (i, j) in enumerate(ALL_EDGES)}
M = len(ALL_EDGES)

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Precompute Q3 edge-sets (as frozensets of edge-indices)
    # Q3 canonical edges
    q3_canonical = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_canonical.append((i, j))
    
    q3_edge_sets = set()
    vertices = list(range(N))
    
    # Limit enumeration to stay within time budget
    count = 0
    for A_img in combinations(vertices, 4):
        if time.time() > deadline - 60:
            break
        A_set = set(A_img)
        remaining = [v for v in vertices if v not in A_set]
        for B_img in combinations(remaining, 4):
            count += 1
            if count > 50000:
                break
            for permA in permutations(A_img):
                for permB in permutations(B_img):
                    mapping = {0: permA[0], 3: permA[1], 5: permA[2], 6: permA[3],
                               1: permB[0], 2: permB[1], 4: permB[2], 7: permB[3]}
                    edge_set = set()
                    for (i, j) in q3_canonical:
                        u = mapping[i]
                        v = mapping[j]
                        if u > v:
                            u, v = v, u
                        edge_set.add(EDGE_INDEX[(u, v)])
                    if len(edge_set) == 12:
                        q3_edge_sets.add(frozenset(edge_set))
        if count > 50000:
            break
    
    q3_list = list(q3_edge_sets)
    
    # Start with K3,13
    current = 0
    for a in range(3):
        for b in range(3, 16):
            idx = EDGE_INDEX[(a, b)]
            current |= (1 << idx)
    
    best = current
    best_edges = 39
    
    # Local search
    random.seed(42)
    for trial in range(100000):
        if time.time() > deadline:
            break
        missing = [idx for idx in range(M) if not (current >> idx) & 1]
        if not missing:
            break
        idx = random.choice(missing)
        new = current | (1 << idx)
        
        # Check if new contains any Q3
        has_q3 = False
        for q3 in q3_list:
            all_present = True
            for e in q3:
                if not (new >> e) & 1:
                    all_present = False
                    break
            if all_present:
                has_q3 = True
                break
        
        if not has_q3:
            current = new
            edges = bin(current).count('1')
            if edges > best_edges:
                best = new
                best_edges = edges
    
    edges_list = []
    for (i, j), idx in EDGE_INDEX.items():
        if (best >> idx) & 1:
            edges_list.append([i, j])
    return edges_list

# EVOLVE-BLOCK-END
