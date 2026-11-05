"""
K2,3-free graph on 20 vertices: maximize edges.

Approach:
- Start with a dense base graph (complement of a matching) or a known construction.
- Use local search / greedy edge addition to add as many edges as possible
  without creating a K2,3.
- Also try random restarts and simulated-annealing-like moves.
- Keep an explicit deadline guard to return best found before 150s.
"""

import time
import random
from collections import defaultdict

# Deadline: 145 seconds to be safe
DEADLINE = 145.0


def count_common_neighbors(adj, u, v):
    """Count common neighbors of u and v given adjacency as a list of sets."""
    return len(adj[u] & adj[v])


def is_k23_free(edges, n):
    """Check if graph with given edges on n vertices is K2,3-free."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for u in range(n):
        for v in range(u + 1, n):
            if len(adj[u] & adj[v]) >= 3:
                return False
    return True


def build_adj(edges, n):
    """Build adjacency list as list of sets."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def can_add_edge(adj, u, v):
    """Check if adding edge (u,v) keeps graph K2,3-free."""
    if v in adj[u]:
        return False
    # After adding (u,v), for any w not in {u,v}, the common neighbors of (u,v)
    # are the old common neighbors of u and v (excluding each other since they weren't adjacent)
    # plus u and v are now neighbors, but we need to check all pairs.
    # Actually, adding edge (u,v) can only create K2,3 involving the pair (u,v).
    # The common neighbors of u and v after addition = (adj[u] ∩ adj[v]) 
    # because u is not in adj[u] and v is not in adj[v].
    # Wait: after adding (u,v), u's neighbors include v, v's neighbors include u.
    # Common neighbors of u and v = (adj[u] ∪ {v}) ∩ (adj[v] ∪ {u}) 
    # = (adj[u] ∩ adj[v]) ∪ (adj[u] ∩ {u}) ∪ ({v} ∩ adj[v]) ∪ ({v} ∩ {u})
    # adj[u] ∩ {u} = empty, {v} ∩ adj[v] = empty (v not in its own adj), {v}∩{u}=empty.
    # So common neighbors = adj[u] ∩ adj[v].
    # So we need len(adj[u] ∩ adj[v]) <= 2.
    return len(adj[u] & adj[v]) <= 2


def greedy_construct(n=20, seed=42, time_limit=10.0):
    """Greedy edge addition: try to add edges in a random order until no more can be added."""
    random.seed(seed)
    start = time.time()
    adj = [set() for _ in range(n)]
    edges = []
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Try multiple random orderings
    best_count = 0
    best_edges = []
    
    while time.time() - start < time_limit:
        adj = [set() for _ in range(n)]
        edges = []
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    return best_edges


def local_search(edges, n=20, time_limit=5.0, seed=42):
    """Try to improve edge count by swapping edges."""
    random.seed(seed)
    start = time.time()
    adj = build_adj(edges, n)
    current_edges = edges[:]
    
    # Build set of existing edges for O(1) lookup
    edge_set = set()
    for u, v in current_edges:
        edge_set.add((min(u, v), max(u, v)))
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    missing = [(u, v) for u, v in all_pairs if (u, v) not in edge_set]
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        random.shuffle(missing)
        random.shuffle(current_edges)
        
        for u, v in missing:
            if can_add_edge(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                current_edges.append([u, v])
                edge_set.add((u, v))
                improved = True
        
        # Try swaps: remove an edge, add a missing edge
        if not improved:
            for i, (eu, ev) in enumerate(current_edges[:]):
                for u, v in missing[:20]:  # limit search
                    if (eu, ev) in edge_set and (u, v) not in edge_set:
                        # Remove (eu, ev)
                        adj[eu].remove(ev)
                        adj[ev].remove(eu)
                        edge_set.remove((min(eu, ev), max(eu, ev)))
                        
                        # Try adding (u, v)
                        if can_add_edge(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            edge_set.add((u, v))
                            current_edges.remove([eu, ev])
                            current_edges.append([u, v])
                            improved = True
                            break
                        else:
                            # Restore
                            adj[eu].add(ev)
                            adj[ev].add(eu)
                            edge_set.add((min(eu, ev), max(eu, ev)))
                    
                    if time.time() - start > time_limit - 1:
                        break
                if improved or time.time() - start > time_limit - 1:
                    break
    
    return current_edges


def dense_base(n=20):
    """Create a dense base graph: complement of a perfect matching (for even n).
    This gives n/2 * (n-2) edges... wait.
    Complement of perfect matching on 20 vertices: each vertex connected to all except its partner and itself.
    So degree 18, edges = 20*18/2 = 180. But this is very dense and likely has K2,3.
    
    Better: use a random sparse base and let greedy fill in.
    """
    return []


def construct():
    n = 20
    start_time = time.time()
    
    best_edges = []
    best_count = 0
    
    # Phase 1: Greedy with multiple seeds
    time_budget_per_phase = 20
    phase1_end = start_time + time_budget_per_phase
    
    for seed in range(100):
        if time.time() > phase1_end:
            break
        edges = greedy_construct(n, seed=seed, time_limit=0.5)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Phase 2: Local search on best found
    phase2_end = min(start_time + 60, start_time + time_budget_per_phase * 3)
    
    if best_count > 0 and time.time() < phase2_end:
        # Try local search on top few solutions
        candidates = [best_edges]
        # Also generate a few more greedy solutions for diversity
        for seed in range(10):
            if time.time() > phase2_end - 10:
                break
            e = greedy_construct(n, seed=1000 + seed, time_limit=0.3)
            if len(e) > best_count:
                best_count = len(e)
                best_edges = e[:]
            candidates.append(e)
        
        # Local search on each candidate
        for i, cand in enumerate(candidates[:5]):
            if time.time() > phase2_end - 5:
                break
            improved = local_search(cand, n=n, time_limit=8.0, seed=2000 + i)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = improved[:]
    
    # Phase 3: More aggressive search with random restarts
    phase3_end = start_time + DEADLINE - 5
    
    if time.time() < phase3_end:
        for iteration in range(1000):
            if time.time() > phase3_end:
                break
            
            # Random restart: start from random subset of best_edges or fresh greedy
            if random.random() < 0.5:
                # Start from a random subset of best edges
                k = random.randint(max(0, best_count - 10), best_count)
                seed_edges = random.sample(best_edges, k)
                adj = build_adj(seed_edges, n)
                edge_set = set((min(u, v), max(u, v)) for u, v in seed_edges)
                
                # Try to add missing edges
                all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
                missing = [(u, v) for u, v in all_pairs if (u, v) not in edge_set]
                random.shuffle(missing)
                
                for u, v in missing:
                    if time.time() > phase3_end:
                        break
                    if can_add_edge(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        seed_edges.append([u, v])
                        edge_set.add((u, v))
                
                if len(seed_edges) > best_count:
                    best_count = len(seed_edges)
                    best_edges = seed_edges[:]
            else:
                # Fresh greedy with random seed
                edges = greedy_construct(n, seed=random.randint(0, 10**6), time_limit=0.3)
                if len(edges) > best_count:
                    best_count = len(edges)
                    best_edges = edges[:]
    
    # Final validation
    if not is_k23_free(best_edges, n):
        # Fallback to seed
        best_edges = []
        for k in range(5):
            best_edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
    
    # Safety: ensure we return before deadline
    if time.time() > start_time + DEADLINE - 2:
        pass
    
    return best_edges


if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} edges")
