"""Improved program: structured greedy + local search for K3,3-free graph on 20 vertices."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    n = 20
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def is_k33_free(adj):
        """Check if graph is K3,3-free using common neighbor counting."""
        # For each pair of vertices, count common neighbors
        # K3,3 exists iff there exist two disjoint triples with >= 3 common neighbors
        # More precisely: K3,3 exists iff there exist 3 vertices with >= 3 common neighbors
        # that are disjoint from those 3 vertices
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = 0
                    for l in range(n):
                        if l != i and l != j and l != k and adj[i][l] and adj[j][l] and adj[k][l]:
                            common += 1
                            if common >= 3:
                                return False
        return True
    
    def edge_count(adj):
        return sum(sum(row) for row in adj) // 2
    
    def get_edges(adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K3,3-free."""
        # Adding edge (u,v) creates K3,3 iff there exist 2 vertices a,b in N(u)∩N(v)
        # and 2 vertices c,d in N(u)∩N(v) such that... 
        # Actually, adding (u,v) creates K3,3 iff there exist vertices a,b,c such that:
        # - a,b,c are common neighbors of u and v (i.e., connected to both u and v)
        # Wait, that's not right either.
        
        # K3,3 with edge (u,v): we need two sets of 3 with all cross edges.
        # If (u,v) is in the new K3,3, then u and v must be on opposite sides.
        # So we need 2 more vertices on u's side and 2 more on v's side.
        # u's side: {u, a, b} where a,b are neighbors of v (and not u)
        # v's side: {v, c, d} where c,d are neighbors of u (and not v)
        # Cross edges needed: a-c, a-d, b-c, b-d (in addition to u-c, u-d, v-a, v-b which exist)
        
        # So: add (u,v) is safe iff there do NOT exist a,b in N(v)\{u} and c,d in N(u)\{v}
        # such that all four cross edges a-c, a-d, b-c, b-d exist.
        
        Nv = [x for x in range(n) if x != u and adj[v][x]]
        Nu = [x for x in range(n) if x != v and adj[u][x]]
        
        for a, b in combinations(Nv, 2):
            for c, d in combinations(Nu, 2):
                if adj[a][c] and adj[a][d] and adj[b][c] and adj[b][d]:
                    return False
        return True
    
    def greedy_build(adj):
        """Try to add all possible edges that keep graph K3,3-free."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i][j] and can_add_edge(adj, i, j):
                        adj[i][j] = adj[j][i] = True
                        improved = True
                        break
                if improved:
                    break
        return adj
    
    def local_search_swap(adj):
        """Try edge swaps to improve: remove an edge, then try to add more."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each edge and then greedily adding
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        adj[i][j] = adj[j][i] = False
                        greedy_build(adj)
                        if edge_count(adj) > edge_count_before:
                            # Keep the change
                            improved = True
                            break
                        else:
                            adj[i][j] = adj[j][i] = True
                if improved:
                    break
        return adj
    
    best_edges = None
    best_score = -1
    
    # Strategy 1: Start with empty graph, greedily add edges
    adj = [[False] * n for _ in range(n)]
    greedy_build(adj)
    if is_k33_free(adj):
        score = edge_count(adj)
        if score > best_score:
            best_score = score
            best_edges = get_edges(adj)
    
    # Strategy 2: Start with random graphs and improve
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        random.seed(attempt)
        adj = [[False] * n for _ in range(n)]
        p = 0.3 + 0.4 * (attempt % 5) / 5  # Vary probability
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = adj[j][i] = True
        
        # If not K3,3-free, remove edges to fix
        while not is_k33_free(adj) and time.time() < deadline:
            # Find a violating triple
            removed = False
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        common = []
                        for l in range(n):
                            if l not in (i, j, k) and adj[i][l] and adj[j][l] and adj[k][l]:
                                common.append(l)
                        if len(common) >= 3:
                            # Remove an edge to break this K3,3
                            # Remove edge between i and a common neighbor
                            for l in common:
                                adj[i][l] = adj[l][i] = False
                                removed = True
                                break
                            if removed:
                                break
                    if removed:
                        break
                if removed:
                    break
            if not removed:
                break
        
        if is_k33_free(adj):
            greedy_build(adj)
            if is_k33_free(adj):
                score = edge_count(adj)
                if score > best_score:
                    best_score = score
                    best_edges = get_edges(adj)
    
    # Strategy 3: Start with known good constructions
    # Complete bipartite K_{2,18} has 36 edges
    adj = [[False] * n for _ in range(n)]
    for i in range(2):
        for j in range(2, 20):
            adj[i][j] = adj[j][i] = True
    greedy_build(adj)
    if is_k33_free(adj):
        score = edge_count(adj)
        if score > best_score:
            best_score = score
            best_edges = get_edges(adj)
    
    # Strategy 4: Random restarts with local search
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        random.seed(1000 + attempt)
        adj = [[False] * n for _ in range(n)]
        # Start with a random sparse graph
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.2:
                    adj[i][j] = adj[j][i] = True
        
        # Make it K3,3-free
        while not is_k33_free(adj) and time.time() < deadline:
            removed = False
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        common = []
                        for l in range(n):
                            if l not in (i, j, k) and adj[i][l] and adj[j][l] and adj[k][l]:
                                common.append(l)
                        if len(common) >= 3:
                            for l in common:
                                adj[i][l] = adj[l][i] = False
                                removed = True
                                break
                            if removed:
                                break
                    if removed:
                        break
                if removed:
                    break
            if not removed:
                break
        
        if is_k33_free(adj):
            greedy_build(adj)
            if is_k33_free(adj):
                score = edge_count(adj)
                if score > best_score:
                    best_score = score
                    best_edges = get_edges(adj)
    
    if best_edges is not None:
        return best_edges
    
    # Fallback: K2,18
    return [[a, b] for a in range(2) for b in range(2, 20)]
# EVOLVE-BLOCK-END
