import random
from itertools import combinations

def construct():
    n = 20
    # Start with a high-density random graph
    # We want to maximize edges, so we start with a high probability and then remove edges.
    # A complete graph has n*(n-1)/2 = 190 edges.
    # ex(n, K3,3) is known to be related to the Zarankiewicz problem.
    # For n=20, a good upper bound is around 80-100 edges?
    # Let's start with p=0.8 and see.
    
    # Actually, let's start with a random graph with p=0.9 and use local search to remove edges.
    # Or better, start with a known construction if available, but we don't have one readily.
    # Let's use a simple random approach with local search.
    
    # To ensure determinism and reproducibility, we set a seed.
    random.seed(42)
    
    # Initialize adjacency matrix
    adj = [[0] * n for _ in range(n)]
    
    # Start with a high probability of edges
    p = 0.85
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i][j] = 1
                adj[j][i] = 1
                
    def has_k33():
        # Check for K3,3 subgraph
        # A K3,3 exists if there are two sets of 3 vertices {a,b,c} and {d,e,f}
        # such that all edges between them are present.
        # This is O(n^6) in worst case, which is 20^6 = 64,000,000. Might be slow in Python.
        # We can optimize by checking common neighbors.
        
        # For each pair of vertices (u, v), find their common neighbors.
        # If there are 3 vertices u,v,w that share 3 common neighbors, then we have a K3,3?
        # No, that's not quite right.
        # A K3,3 is formed by two disjoint sets of 3 vertices, A={a,b,c} and B={d,e,f},
        # such that for all a in A, b in B, edge (a,b) exists.
        
        # A faster way:
        # Iterate over all triples (a,b,c). Let N(a,b,c) be the set of vertices adjacent to all of a,b,c.
        # If |N(a,b,c)| >= 3, and there exists a triple (d,e,f) in N(a,b,c) such that
        # d,e,f are not in {a,b,c} and are mutually non-adjacent? No, K3,3 doesn't require non-edges within the sets.
        # Wait, K3,3 is bipartite. The vertices in A are not connected to each other in the subgraph,
        # and vertices in B are not connected to each other in the subgraph.
        # So we need A and B to be independent sets in the original graph? No, just that the induced subgraph on A∪B is K3,3.
        # This means no edges within A, no edges within B, and all edges between A and B.
        
        # Let's iterate over all pairs (u, v).
        # Common neighbors C = {w : adj[u][w]=1 and adj[v][w]=1}
        # If |C| >= 2, pick two common neighbors w1, w2.
        # Then we need a third vertex x such that x is adjacent to u, v, w1, w2.
        # And we need to form a K3,3.
        
        # Actually, a K3,3 is determined by two triples {a,b,c} and {d,e,f}.
        # The condition is that all 9 edges between them are present.
        # We can iterate over all triples (a,b,c) and (d,e,f).
        # Number of triples is C(20,3) = 1140.
        # Pairs of triples is 1140^2 = 1,299,600.
        # For each pair, check 9 edges. Total ~11 million checks. This is feasible.
        
        triples = list(combinations(range(n), 3))
        for i in range(len(triples)):
            a, b, c = triples[i]
            for j in range(i+1, len(triples)):
                d, e, f = triples[j]
                # Check if the two triples are disjoint
                if len(set((a,b,c)).intersection(set((d,e,f)))) > 0:
                    continue
                # Check all 9 edges
                # Edges: (a,d), (a,e), (a,f), (b,d), (b,e), (b,f), (c,d), (c,e), (c,f)
                if (adj[a][d] and adj[a][e] and adj[a][f] and
                    adj[b][d] and adj[b][e] and adj[b][f] and
                    adj[c][d] and adj[c][e] and adj[c][f]):
                    return True
        return False

    def count_edges():
        return sum(sum(row) for row in adj) // 2

    # Local search: while there is a K3,3, remove a random edge from it
    # To make it efficient, we can find a K3,3 and remove one of its edges.
    
    # First, let's just check if it's free.
    # If not, we need to remove edges.
    
    # A smarter local search:
    # 1. Find a K3,3.
    # 2. Remove one of the edges in that K3,3.
    # 3. Repeat.
    
    # To find a K3,3 efficiently, we can use the triple method but stop at the first one found.
    
    def find_k33():
        triples = list(combinations(range(n), 3))
        for i in range(len(triples)):
            a, b, c = triples[i]
            for j in range(i+1, len(triples)):
                d, e, f = triples[j]
                if len(set((a,b,c)).intersection(set((d,e,f)))) > 0:
                    continue
                if (adj[a][d] and adj[a][e] and adj[a][f] and
                    adj[b][d] and adj[b][e] and adj[b][f] and
                    adj[c][d] and adj[c][e] and adj[c][f]):
                    return [(a,d), (a,e), (a,f), (b,d), (b,e), (b,f), (c,d), (c,e), (c,f)]
        return None

    # Remove edges until no K3,3
    # This might be slow if done naively.
    # Let's limit the number of iterations.
    
    for _ in range(1000):
        k33_edges = find_k33()
        if k33_edges is None:
            break
        # Remove a random edge from the K3,3
        edge = random.choice(k33_edges)
        u, v = edge
        adj[u][v] = 0
        adj[v][u] = 0

    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges

# Note: The above approach might be too slow if it gets stuck in a local minimum or if the initial graph is too dense.
# Let's refine the local search to be more robust.
# Instead of random removal, we can try to remove edges that are part of many K3,3s.
# But for now, let's stick with a simpler version that is more likely to finish in time.
# We can also try to start with a sparser graph to reduce the number of removals.

def construct():
    n = 20
    random.seed(42)
    
    # Start with a moderate density
    p = 0.7
    adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i][j] = 1
                adj[j][i] = 1
                
    def find_k33():
        triples = list(combinations(range(n), 3))
        for i in range(len(triples)):
            a, b, c = triples[i]
            for j in range(i+1, len(triples)):
                d, e, f = triples[j]
                if len(set((a,b,c)).intersection(set((d,e,f)))) > 0:
                    continue
                if (adj[a][d] and adj[a][e] and adj[a][f] and
                    adj[b][d] and adj[b][e] and adj[b][f] and
                    adj[c][d] and adj[c][e] and adj[c][f]):
                    return [(a,d), (a,e), (a,f), (b,d), (b,e), (b,f), (c,d), (c,e), (c,f)]
        return None

    # Remove edges until no K3,3
    for _ in range(500):
        k33_edges = find_k33()
        if k33_edges is None:
            break
        edge = random.choice(k33_edges)
        u, v = edge
        adj[u][v] = 0
        adj[v][u] = 0

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges
