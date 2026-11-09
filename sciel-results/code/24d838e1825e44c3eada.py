"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    n = 15
    deadline = time.time() + 85.0
    random.seed(42)
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    best_adj = None
    best_count = 0
    
    def make_complete():
        adj = [set() for _ in range(n)]
        for u, v in all_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def count_violations(adj):
        """Count violations: for each edge (u,v), check if exists s in N(u)\{v}, t in N(v)\{u}, s!=t with deg(s)==deg(t)."""
        deg = [len(adj[i]) for i in range(n)]
        violations = 0
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Build degree->count map for N(u)\{v}
                    deg_count_u = defaultdict(int)
                    for s in adj[u]:
                        if s != v:
                            deg_count_u[deg[s]] += 1
                    # Check N(v)\{u}
                    for t in adj[v]:
                        if t != u:
                            d = deg[t]
                            if d in deg_count_u:
                                # Need s != t
                                if deg_count_u[d] > 1 or (deg_count_u[d] == 1 and t not in adj[u]):
                                    violations += 1
                                    break
        return violations
    
    def edge_count(adj):
        return sum(len(adj[i]) for i in range(n)) // 2
    
    def try_remove_worst(adj, max_iter=5000):
        """Repeatedly remove the edge that contributes most violations."""
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            viol = count_violations(adj)
            if viol == 0:
                break
            # Find edge with most violations
            deg = [len(adj[i]) for i in range(n)]
            worst_edge = None
            worst_viol = 0
            for u, v in all_edges:
                if v in adj[u]:
                    # Count violations for this edge
                    deg_count_u = defaultdict(int)
                    for s in adj[u]:
                        if s != v:
                            deg_count_u[deg[s]] += 1
                    ev = 0
                    for t in adj[v]:
                        if t != u:
                            d = deg[t]
                            if d in deg_count_u:
                                if deg_count_u[d] > 1 or (deg_count_u[d] == 1 and t not in adj[u]):
                                    ev += 1
                                    break
                    if ev > worst_viol:
                        worst_viol = ev
                        worst_edge = (u, v)
            if worst_edge is None:
                break
            u, v = worst_edge
            adj[u].discard(v)
            adj[v].discard(u)
        return adj
    
    def local_search_add(adj, max_time=5.0):
        """Try to add edges greedily and via random perturbation."""
        t_start = time.time()
        for _ in range(20):
            if time.time() - t_start > max_time:
                break
            improved = True
            while improved:
                if time.time() - t_start > max_time:
                    return adj
                improved = False
                random.shuffle(all_edges)
                for u, v in all_edges:
                    if time.time() - t_start > max_time:
                        return adj
                    if v not in adj[u]:
                        adj[u].add(v)
                        adj[v].add(u)
                        if count_violations(adj) == 0:
                            improved = True
                        else:
                            adj[u].discard(v)
                            adj[v].discard(u)
        return adj
    
    def local_search_remove_then_add(adj, max_time=8.0):
        """Remove some random edges then try to add back different ones."""
        t_start = time.time()
        current = [set(s) for s in adj]
        ec = edge_count(current)
        for _ in range(30):
            if time.time() - t_start > max_time:
                break
            # Remove 2-4 random edges
            num_remove = random.randint(2, 4)
            present = [(u,v) for u,v in all_edges if v in current[u]]
            if not present:
                break
            to_remove = random.sample(present, min(num_remove, len(present)))
            for u, v in to_remove:
                current[u].discard(v)
                current[v].discard(u)
            # Try to add edges
            random.shuffle(all_edges)
            for u, v in all_edges:
                if time.time() - t_start > max_time:
                    break
                if v not in current[u]:
                    current[u].add(v)
                    current[v].add(u)
                    if count_violations(current) == 0:
                        pass
                    else:
                        current[u].discard(v)
                        current[v].discard(u)
            if edge_count(current) > ec:
                ec = edge_count(current)
                adj = [set(s) for s in current]
        return adj
    
    # Strategy 1: Start complete, remove worst edges
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        adj = make_complete()
        adj = try_remove_worst(adj, max_iter=3000)
        adj = local_search_add(adj, max_time=4.0)
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = [set(s) for s in adj]
    
    # Strategy 2: Start from best, do remove-then-add local search
    if best_adj is not None:
        if time.time() < deadline - 15.0:
            adj = local_search_remove_then_add(best_adj, max_time=10.0)
            ec = edge_count(adj)
            if ec > best_count:
                best_count = ec
                best_adj = [set(s) for s in adj]
    
    # Strategy 3: Random initializations with different densities
    for restart in range(30):
        if time.time() > deadline - 8.0:
            break
        density = random.uniform(0.3, 0.7)
        adj = [set() for _ in range(n)]
        for u, v in all_edges:
            if random.random() < density:
                adj[u].add(v)
                adj[v].add(u)
        # Remove violations
        for _ in range(100):
            if time.time() > deadline - 8.0:
                break
            viol = count_violations(adj)
            if viol == 0:
                break
            # Remove random edge involved in violation
            deg = [len(adj[i]) for i in range(n)]
            violated_edges = []
            for u, v in all_edges:
                if v in adj[u]:
                    deg_count_u = defaultdict(int)
                    for s in adj[u]:
                        if s != v:
                            deg_count_u[deg[s]] += 1
                    for t in adj[v]:
                        if t != u:
                            d = deg[t]
                            if d in deg_count_u:
                                if deg_count_u[d] > 1 or (deg_count_u[d] == 1 and t not in adj[u]):
                                    violated_edges.append((u, v))
                                    break
            if not violated_edges:
                break
            u, v = random.choice(violated_edges)
            adj[u].discard(v)
            adj[v].discard(u)
        # Add edges
        adj = local_search_add(adj, max_time=3.0)
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = [set(s) for s in adj]
    
    # Final polish: try adding every possible edge
    if best_adj is not None:
        adj = [set(s) for s in best_adj]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if time.time() > deadline - 2.0:
                break
            if v not in adj[u]:
                adj[u].add(v)
                adj[v].add(u)
                if count_violations(adj) == 0:
                    pass
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
        ec = edge_count(adj)
        if ec > best_count:
            best_count = ec
            best_adj = adj
    
    if best_adj is None:
        # Fallback: star
        best_adj = [set() for _ in range(n)]
        for v in range(1, n):
            best_adj[0].add(v)
            best_adj[v].add(0)
    
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])
    
    return edges

# EVOLVE-BLOCK-END
