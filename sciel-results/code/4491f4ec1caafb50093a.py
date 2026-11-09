import time
import itertools
from collections import Counter

N = 16
ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]
EDGE_SET = set(ALL_EDGES)

# Precompute all 8-vertex subsets
SUBSETS_8 = list(itertools.combinations(range(N), 8))

# Precompute partitions
PARTITIONS_8 = []
indices_8 = range(8)
for a_idx in itertools.combinations(indices_8, 4):
    b_idx = tuple(i for i in indices_8 if i not in a_idx)
    PARTITIONS_8.append((a_idx, b_idx))

def find_q3_edges(edge_set):
    """
    Finds edges involved in Q3 configurations.
    Returns a Counter.
    """
    edge_counts = Counter()
    
    for verts in SUBSETS_8:
        for a_idx, b_idx in PARTITIONS_8:
            A = [verts[i] for i in a_idx]
            B = [verts[i] for i in b_idx]
            
            deg_A = [0]*4
            deg_B = [0]*4
            
            # Check edges and degrees
            for i in range(4):
                u = A[i]
                for j in range(4):
                    v = B[j]
                    if (u, v) in edge_set or (v, u) in edge_set:
                        deg_A[i] += 1
                        deg_B[j] += 1
            
            if min(deg_A) >= 3 and min(deg_B) >= 3:
                # This configuration supports a Q3
                # Mark all present edges in this bipartite graph
                for i in range(4):
                    u = A[i]
                    for j in range(4):
                        v = B[j]
                        if (u, v) in edge_set or (v, u) in edge_set:
                            e = (min(u, v), max(u, v))
                            edge_counts[e] += 1
                            
    return edge_counts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    current_edges = set(ALL_EDGES)
    
    # Optional: Add some randomness to tie-breaking to escape local minima?
    # Not strictly necessary for "remove until none left".
    
    while time.time() < deadline:
        counts = find_q3_edges(current_edges)
        
        if not counts:
            break
            
        max_count = max(counts.values())
        candidates = [e for e, c in counts.items() if c == max_count]
        
        # Pick one
        e_to_remove = candidates[0]
        
        current_edges.remove(e_to_remove)
        
        if time.time() > deadline - 10.0:
            break
            
    result = [[u, v] for u, v in current_edges]
    return result
