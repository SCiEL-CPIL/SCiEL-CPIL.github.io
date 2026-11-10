import random
import itertools
import time
import math

def construct():
    n = 20
    
    # Precompute all triples
    triples = list(itertools.combinations(range(n), 3))
    
    def count_bad_triples(adj):
        """Returns the number of triples with >= 3 common neighbors."""
        bad = 0
        for A in triples:
            common = adj[A[0]] & adj[A[1]] & adj[A[2]]
            if len(common) >= 3:
                bad += 1
        return bad

    def is_valid(adj):
        return count_bad_triples(adj) == 0

    def try_add_edge(adj, u, v):
        """Tries to add edge (u,v). Returns True if added and valid."""
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        if count_bad_triples(adj) > 0:
            adj[u].remove(v)
            adj[v].remove(u)
            return False
        return True

    def try_remove_edge(adj, u, v):
        """Removes edge (u,v). Always valid if it was valid before."""
        if v not in adj[u]:
            return False
        adj[u].remove(v)
        adj[v].remove(u)
        return True

    def count_edges(adj):
        return sum(len(adj[i]) for i in range(n)) // 2

    def fast_local_search(adj, steps=2000):
        """
        Optimized local search using add and swap moves.
        """
        current_adj = [set(s) for s in adj]
        current_edges = count_edges(current_adj)
        best_adj = [set(s) for s in current_adj]
        best_edges = current_edges
        
        # Precompute list of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for step in range(steps):
            # Randomly select a move type
            if random.random() < 0.5:
                # Try to add a random missing edge
                u, v = random.choice(all_edges)
                if v not in current_adj[u]:
                    if try_add_edge(current_adj, u, v):
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = [set(s) for s in current_adj]
            else:
                # Try a swap: Add (u,v) and remove (x,y)
                # Pick random missing edge (u,v)
                u, v = random.choice(all_edges)
                if v in current_adj[u]:
                    continue
                
                # Pick random existing edge (x,y)
                x = random.randint(0, n-1)
                if not current_adj[x]:
                    continue
                y = random.choice(list(current_adj[x]))
                
                # Attempt swap
                # Temporarily add u-v
                current_adj[u].add(v)
                current_adj[v].add(u)
                # Temporarily remove x-y
                current_adj[x].remove(y)
                current_adj[y].remove(x)
                
                if count_bad_triples(current_adj) == 0:
                    # Accept swap (state changes, edge count same)
                    pass
                else:
                    # Revert
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
                    current_adj[x].add(y)
                    current_adj[y].add(x)
                    
        return best_adj, best_edges

    best_adj = None
    best_edges = 0
    
    # Try multiple random starts
    for trial in range(50):
        random.seed(trial)
        adj = [set() for _ in range(n)]
        
        # Greedy initialization: Add edges in random order
        # This ensures we start with a valid graph
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if try_add_edge(adj, u, v):
                pass # Added
        
        # Local search to improve
        res_adj, res_edges = fast_local_search(adj, steps=2000)
        
        if res_edges > best_edges:
            best_edges = res_edges
            best_adj = res_adj

    if best_adj is None:
        # Fallback
        return [[a, b] for a in range(2) for b in range(2, 20)]

    # Convert to list
    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
    return edges
