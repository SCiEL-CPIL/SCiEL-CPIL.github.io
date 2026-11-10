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

    # Simulated Annealing
    def simulated_annealing(initial_adj, start_temp=1.0, end_temp=0.01, cooling=0.99, steps=5000):
        current_adj = [set(s) for s in initial_adj]
        current_edges = count_edges(current_adj)
        best_adj = [set(s) for s in current_adj]
        best_edges = current_edges
        
        temp = start_temp
        for step in range(steps):
            temp *= cooling
            if temp < end_temp:
                break
            
            # Generate a move: Add or Remove a random edge
            move_type = random.random()
            if move_type < 0.7:
                # Try to add an edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if v in current_adj[u]:
                    continue
                
                if try_add_edge(current_adj, u, v):
                    new_edges = current_edges + 1
                    # Always accept improvement
                    current_edges = new_edges
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [set(s) for s in current_adj]
            else:
                # Try to remove an edge
                existing_edges = []
                for i in range(n):
                    for j in current_adj[i]:
                        if i < j:
                            existing_edges.append((i, j))
                if not existing_edges:
                    continue
                u, v = random.choice(existing_edges)
                
                # Remove it
                try_remove_edge(current_adj, u, v)
                new_edges = current_edges - 1
                
                # Accept if improvement (impossible) or with prob exp(delta/temp)
                delta = new_edges - current_edges # -1
                if random.random() < math.exp(delta / temp):
                    current_edges = new_edges
                else:
                    # Reject move: add it back
                    try_add_edge(current_adj, u, v)
        
        return best_adj, best_edges

    # Initialize with a random valid graph
    # Start with a sparse graph
    best_adj = None
    best_edges = 0
    
    # Try multiple random starts
    for trial in range(20):
        random.seed(trial)
        adj = [set() for _ in range(n)]
        # Generate a random graph with p=0.3
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.3:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Repair if invalid
        # Simple repair: remove edges from bad triples
        valid = False
        for _ in range(100):
            if is_valid(adj):
                valid = True
                break
            # Find a bad triple
            bad_triple = None
            for A in triples:
                common = adj[A[0]] & adj[A[1]] & adj[A[2]]
                if len(common) >= 3:
                    bad_triple = (A, common)
                    break
            if bad_triple:
                A, common = bad_triple
                v = random.choice(list(common))
                a = random.choice(A)
                adj[v].remove(a)
                adj[a].remove(v)
            else:
                # Should not happen if is_valid is false
                break
        
        if not valid:
            continue
            
        # Run SA
        res_adj, res_edges = simulated_annealing(adj, start_temp=1.0, end_temp=0.001, cooling=0.995, steps=10000)
        
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
