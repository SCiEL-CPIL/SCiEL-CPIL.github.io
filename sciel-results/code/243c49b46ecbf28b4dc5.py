import time
import random

def construct():
    n = 15
    deadline = time.time() + 85
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def build_deg_adj(edge_set):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        return deg, adj
    
    def count_violations(edge_set):
        """Count number of violating paths s-u-v-t where deg(s)==deg(t)."""
        deg, adj = build_deg_adj(edge_set)
        violations = 0
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        violations += 1
        return violations
    
    def is_valid(edge_set):
        return count_violations(edge_set) == 0
    
    def local_search(edge_set, max_iters=500):
        """Local search: try to add edges or swap edges to improve."""
        current = set(edge_set)
        best_count = len(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try adding a random missing edge
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in current]
            if missing:
                random.shuffle(missing)
                added = False
                for e in missing[:20]:  # Try up to 20 random missing edges
                    trial = current | {e}
                    if is_valid(trial):
                        current = trial
                        best_count = len(current)
                        added = True
                        break
                if added:
                    continue
            
            # Try removing a random edge to allow future additions
            if current and random.random() < 0.3:
                edge_to_remove = random.choice(list(current))
                trial = current - {edge_to_remove}
                if is_valid(trial):
                    current = trial
                    best_count = min(best_count, len(current))
            
            # Try edge swap: remove one, add another
            if current and random.random() < 0.4:
                missing = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in current]
                if missing:
                    e_remove = random.choice(list(current))
                    e_add = random.choice(missing[:50])
                    trial = (current - {e_remove}) | {e_add}
                    if is_valid(trial) and len(trial) > len(current):
                        current = trial
                        best_count = len(current)
        
        return current
    
    # Generate diverse initial solutions
    def random_graph_with_density(density):
        edges = set()
        for e in all_edges:
            if random.random() < density:
                edges.add(e)
        return edges
    
    def structured_graph():
        """Create a graph with degree variation."""
        edges = set()
        # Create clusters with different internal densities
        for i in range(n):
            for j in range(i + 1, n):
                # Higher probability for edges between "dissimilar" vertices
                prob = 0.5 + 0.3 * abs(i - j) / n
                if random.random() < prob:
                    edges.add((i, j))
        return edges
    
    best_edges = set()
    best_len = 0
    
    # Try multiple random restarts with different strategies
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart < 10:
            # Random with varying density
            density = 0.3 + 0.6 * random.random()
            init = random_graph_with_density(density)
        elif restart < 20:
            # Structured
            init = structured_graph()
        elif restart < 30:
            # Start sparse, grow
            init = random_graph_with_density(0.2)
        else:
            # Start dense, prune
            init = random_graph_with_density(0.8)
        
        # First make it valid by removing violating edges
        while not is_valid(init) and time.time() < deadline:
            # Find a violating edge and remove it
            deg, adj = build_deg_adj(init)
            found = False
            for u, v in list(init):
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            init.discard((u, v))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        
        if not is_valid(init):
            continue
        
        # Local search to maximize edges
        improved = local_search(init, max_iters=100)
        
        if len(improved) > best_len:
            best_len = len(improved)
            best_edges = improved
    
    # Final polish: try to add any remaining edges
    for e in all_edges:
        if e not in best_edges:
            trial = best_edges | {e}
            if is_valid(trial):
                best_edges = trial
    
    # Convert to proper format
    result = []
    for u, v in best_edges:
        if u < v:
            result.append([int(u), int(v)])
        else:
            result.append([int(v), int(u)])
    
    return result
