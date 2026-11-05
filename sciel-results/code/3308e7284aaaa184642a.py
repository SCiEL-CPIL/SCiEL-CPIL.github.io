import time
import random
from itertools import combinations

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial graph
    G = set()
    for a in range(3):
        for b in range(3, 16):
            G.add((a,b))
    # Add intra edges
    for a in range(3, 16):
        for b in range(a+1, 16):
            if random.random() < 0.7:
                G.add((a,b))
                
    best_G = G.copy()
    best_count = len(G)
    
    # Precompute 8-subsets
    subsets_8 = list(combinations(range(16), 8))
    
    def find_q3(G):
        # Randomly sample subsets
        random.shuffle(subsets_8)
        for S in subsets_8[:100]: # Check first 100
            verts = list(S)
            # Check if G[S] contains Q3
            # Use the matching method
            for A in combinations(verts, 4):
                A_set = set(A)
                B = [v for v in verts if v not in A_set]
                missing = set()
                for a in A:
                    for b in B:
                        e = (min(a,b), max(a,b))
                        if e not in G:
                            missing.add(e)
                # Check if missing has perfect matching
                a_to_idx = {v:i for i,v in enumerate(A)}
                b_to_idx = {v:i for i,v in enumerate(B)}
                miss_idx = set()
                for (u,v) in missing:
                    if u in a_to_idx and v in b_to_idx:
                        miss_idx.add((a_to_idx[u], b_to_idx[v]))
                    elif v in a_to_idx and u in b_to_idx:
                        miss_idx.add((a_to_idx[v], b_to_idx[u]))
                # Simple matching check
                if has_pm(miss_idx):
                    return S
        return None

    def has_pm(edges):
        if len(edges) < 4:
            return False
        # Backtracking
        def dfs(used_r, i):
            if i == 4:
                return True
            for j in range(4):
                if not used_r[j] and (i,j) in edges:
                    used_r[j] = True
                    if dfs(used_r, i+1):
                        return True
                    used_r[j] = False
            return False
        return dfs([False]*4, 0)

    # Local search
    for _ in range(1000):
        if time.time() > deadline:
            break
        # Try to add a random edge
        u = random.randint(0, 15)
        v = random.randint(0, 15)
        if u == v or (u,v) in G:
            continue
        G.add((u,v))
        q3 = find_q3(G)
        if q3:
            # Remove a random edge from the Q3
            # Re-find edges in q3
            verts = list(q3)
            edges_in_q3 = []
            for a in range(len(verts)):
                for b in range(a+1, len(verts)):
                    e = (min(verts[a], verts[b]), max(verts[a], verts[b]))
                    if e in G:
                        edges_in_q3.append(e)
            if edges_in_q3:
                rem_edge = random.choice(edges_in_q3)
                G.remove(rem_edge)
        else:
            if len(G) > best_count:
                best_count = len(G)
                best_G = G.copy()
                
    return [[u, v] for u, v in best_G]
