import random
import time

def construct():
    n = 20
    adj = [set() for _ in range(n)]
    common = [[0] * n for _ in range(n)]
    
    def can_add(u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        if common[u][v] > 2:
            return False
        for y in adj[v]:
            if common[u][y] > 1:
                return False
        for y in adj[u]:
            if common[v][y] > 1:
                return False
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        for y in adj[v]:
            common[u][y] += 1
            common[y][u] += 1
        for y in adj[u]:
            common[v][y] += 1
            common[y][v] += 1

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        for y in adj[v]:
            common[u][y] -= 1
            common[y][u] -= 1
        for y in adj[u]:
            common[v][y] -= 1
            common[y][v] -= 1

    def get_edge_count():
        return sum(len(s) for s in adj) // 2

    def get_edges_list():
        return [[u, v] for u in range(n) for v in adj[u] if u < v]

    # 1. Build Seed: Two Petersen Graphs + 3-regular Bipartite
    def build_petersen(offset):
        edges = []
        # Outer cycle
        for i in range(5):
            edges.append([offset + i, offset + (i + 1) % 5])
        # Inner cycle (star-like, but 5-cycle in Petersen representation)
        # Standard Petersen: 5,7,9,6,8 is a cycle?
        # Let's use the standard 5+5+5 structure.
        # Vertices 0-4 outer, 5-9 inner.
        # Inner edges: (5,7), (7,9), (9,6), (6,8), (8,5)
        inner_edges = [(5,7), (7,9), (9,6), (6,8), (8,5)]
        for u, v in inner_edges:
            edges.append([offset + u, offset + v])
        # Spokes
        for i in range(5):
            edges.append([offset + i, offset + 5 + i])
        return edges

    # Add Petersen 1 (0-9)
    for u, v in build_petersen(0):
        add_edge(u, v)
        
    # Add Petersen 2 (10-19)
    for u, v in build_petersen(10):
        add_edge(u, v)

    # Add 3-regular bipartite between 0-9 and 10-19
    # Generate 3 disjoint perfect matchings
    left = list(range(10))
    right = list(range(10, 20))
    for _ in range(3):
        random.shuffle(right)
        for i in range(10):
            u = left[i]
            v = right[i]
            if can_add(u, v):
                add_edge(u, v)
            else:
                # If invalid, try to find a valid partner?
                # For now, just skip or try another random right vertex
                # Since it's a seed, we want to get as many as possible
                # A 3-regular bipartite graph might have C4s which are bad?
                # No, C4 in bipartite means 2 common neighbors.
                # We need < 3 common neighbors.
                # So C4 is allowed (2 common neighbors).
                # But if we have multiple C4s sharing edges, we might get 3?
                # Let's just add if valid.
                pass

    best_count = get_edge_count()
    best_edges = get_edges_list()

    start_time = time.time()
    time_limit = 110.0

    # Local Search
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try to add a random missing edge
        for _ in range(50):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and v not in adj[u]:
                if can_add(u, v):
                    add_edge(u, v)
                    improved = True
                    break
        
        if improved:
            if get_edge_count() > best_count:
                best_count = get_edge_count()
                best_edges = get_edges_list()
            continue

        # 2. Try to swap: remove one, add one
        # Pick a random edge to remove
        edges_list = get_edges_list()
        if edges_list:
            u_rem, v_rem = random.choice(edges_list)
            remove_edge(u_rem, v_rem)
            
            # Try to add a random missing edge
            for _ in range(50):
                u_add = random.randint(0, n - 1)
                v_add = random.randint(0, n - 1)
                if u_add != v_add and v_add not in adj[u_add]:
                    if can_add(u_add, v_add):
                        add_edge(u_add, v_add)
                        if get_edge_count() > best_count:
                            best_count = get_edge_count()
                            best_edges = get_edges_list()
                        improved = True
                        break
            if not improved:
                # Revert removal
                add_edge(u_rem, v_rem)

        # 3. Try double swap: remove two, add two
        # This is more expensive, do less frequently
        if random.random() < 0.1:
            edges_list = get_edges_list()
            if len(edges_list) >= 2:
                e1 = random.choice(edges_list)
                e2 = random.choice([e for e in edges_list if e != e1])
                u1, v1 = e1
                u2, v2 = e2
                remove_edge(u1, v1)
                remove_edge(u2, v2)
                
                added = 0
                for _ in range(100):
                    u_add = random.randint(0, n - 1)
                    v_add = random.randint(0, n - 1)
                    if u_add != v_add and v_add not in adj[u_add]:
                        if can_add(u_add, v_add):
                            add_edge(u_add, v_add)
                            added += 1
                            if added == 2:
                                break
                            
                if added == 2:
                    if get_edge_count() > best_count:
                        best_count = get_edge_count()
                        best_edges = get_edges_list()
                else:
                    # Revert
                    add_edge(u1, v1)
                    add_edge(u2, v2)

    # Final Verification
    adj_check = [set() for _ in range(n)]
    for u, v in best_edges:
        adj_check[u].add(v)
        adj_check[v].add(u)
    
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            if len(adj_check[u] & adj_check[v]) > 2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to seed
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return best_edges
