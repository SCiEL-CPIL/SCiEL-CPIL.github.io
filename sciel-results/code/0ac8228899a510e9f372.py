import time
import random
import itertools

Q3_ADJ = {
    0: [1, 2, 4],
    1: [0, 3, 5],
    2: [0, 3, 6],
    3: [1, 2, 7],
    4: [0, 5, 6],
    5: [1, 4, 7],
    6: [2, 4, 7],
    7: [3, 5, 6]
}
Q3_EDGES = set()
for u in Q3_ADJ:
    for v in Q3_ADJ[u]:
        if u < v:
            Q3_EDGES.add((u, v))

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 16
    edges = set()
    possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Start with a decent seed
    # K_3,13 has 39 edges.
    # Let's start with a random graph with 50 edges.
    random.shuffle(possible_edges)
    for e in possible_edges[:50]:
        edges.add(e)
        
    def has_edge(u, v):
        if u > v: u, v = v, u
        return (u, v) in edges

    def find_q3(must_contain=None):
        # Returns list of edges of a Q3, or None
        if must_contain:
            # Generate subsets of size 8 containing must_contain
            others = [v for v in range(n) if v not in must_contain]
            need = 8 - len(must_contain)
            for combo in itertools.combinations(others, need):
                subset = set(must_contain) | set(combo)
                # Check this subset
                mapping = {}
                q3_verts = list(range(8))
                
                def backtrack(idx):
                    if idx == 8:
                        return True
                    v = q3_verts[idx]
                    for s in subset:
                        if s in mapping.values(): continue
                        ok = True
                        for u in q3_verts[:idx]:
                            if (u, v) in Q3_EDGES:
                                if not has_edge(mapping[u], s):
                                    ok = False
                                    break
                        if ok:
                            mapping[v] = s
                            if backtrack(idx + 1):
                                return True
                            del mapping[v]
                    return False
                
                if backtrack(0):
                    q3_edges = []
                    for u in range(8):
                        for v in Q3_ADJ[u]:
                            if u < v:
                                q3_edges.append((mapping[u], mapping[v]))
                    return q3_edges
        else:
            for subset in itertools.combinations(range(n), 8):
                # Same logic
                mapping = {}
                q3_verts = list(range(8))
                
                def backtrack(idx):
                    if idx == 8:
                        return True
                    v = q3_verts[idx]
                    for s in subset:
                        if s in mapping.values(): continue
                        ok = True
                        for u in q3_verts[:idx]:
                            if (u, v) in Q3_EDGES:
                                if not has_edge(mapping[u], s):
                                    ok = False
                                    break
                        if ok:
                            mapping[v] = s
                            if backtrack(idx + 1):
                                return True
                            del mapping[v]
                    return False
                
                if backtrack(0):
                    q3_edges = []
                    for u in range(8):
                        for v in Q3_ADJ[u]:
                            if u < v:
                                q3_edges.append((mapping[u], mapping[v]))
                    return q3_edges
        return None

    best_edges = len(edges)
    
    while time.time() < deadline:
        # Check if current graph is Q3-free
        q3 = find_q3()
        if q3:
            # Remove random edge from Q3
            e = random.choice(q3)
            edges.discard(e)
        else:
            # Try to add edges
            non_edges = [e for e in possible_edges if e not in edges]
            random.shuffle(non_edges)
            for e in non_edges:
                if time.time() > deadline - 2: break
                u, v = e
                edges.add(e)
                # Check if this specific addition created a Q3
                # Only check subsets containing u and v
                q3 = find_q3(must_contain=[u, v])
                if q3:
                    edges.discard(e)
                else:
                    best_edges = max(best_edges, len(edges))
                    # Keep adding
                    continue
            if not non_edges:
                break

    return [[u, v] for u, v in edges]
