"""Strong edge colouring of bounded-degree graphs via greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_MARGIN = 10  # seconds to finish before 150s wall-clock

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    def get_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Generate a random bounded-degree graph
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges), degree

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def conflicts(e1, e2, adj):
        """Check if two edges conflict (share vertex or are joined by an edge)."""
        u, v = e1
        ou, ov = e2
        if u == ou or u == ov or v == ou or v == ov:
            return True
        if ou in adj[u] or ov in adj[u] or ou in adj[v] or ov in adj[v]:
            return True
        return False

    def greedy_coloring(edges, adj):
        """Greedy coloring: assign smallest available color to each edge in order."""
        color = {}
        for e in edges:
            taken = set()
            for other_e, c in color.items():
                if conflicts(e, other_e, adj):
                    taken.add(c)
            c = 0
            while c in taken:
                c += 1
            color[e] = c
        return color

    def improve_coloring(edges, adj, initial_color, max_iter=5000):
        """Local search: try to reduce number of colors by moving edges to lower colors."""
        color = dict(initial_color)
        rng = random.Random(42)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            shuffled_edges = list(edges)
            rng.shuffle(shuffled_edges)
            
            for e in shuffled_edges:
                current_c = color[e]
                if current_c == 0:
                    continue
                for new_c in range(current_c):
                    can_move = True
                    for other_e, c in color.items():
                        if c != new_c or other_e == e:
                            continue
                        if conflicts(e, other_e, adj):
                            can_move = False
                            break
                    if can_move:
                        color[e] = new_c
                        improved = True
                        break
            
            if not improved:
                break
        
        return color

    def swap_improve(edges, adj, initial_color, max_iter=3000):
        """Try swapping colors of pairs of edges to enable further reductions."""
        color = dict(initial_color)
        rng = random.Random(123)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            shuffled = list(edges)
            rng.shuffle(shuffled)
            
            for i in range(len(shuffled) - 1):
                e1 = shuffled[i]
                for j in range(i + 1, len(shuffled)):
                    e2 = shuffled[j]
                    c1, c2 = color[e1], color[e2]
                    if c1 == c2:
                        continue
                    
                    # Try swapping colors of e1 and e2
                    color[e1], color[e2] = c2, c1
                    
                    # Check if this swap is valid
                    valid = True
                    for other_e, c in color.items():
                        if other_e == e1 or other_e == e2:
                            continue
                        if c == c2 and conflicts(e1, other_e, adj):
                            valid = False
                            break
                        if c == c1 and conflicts(e2, other_e, adj):
                            valid = False
                            break
                    
                    if not valid:
                        color[e1], color[e2] = c1, c2
                        continue
                    
                    # If swap is valid, try to further reduce
                    new_count = max(color.values()) + 1
                    old_count = max([c1, c2, 0]) + 1
                    if new_count < old_count:
                        improved = True
                    else:
                        # Revert if no improvement
                        color[e1], color[e2] = c1, c2
            
            if not improved:
                break
        
        return color

    def optimize_instance(n, d, seed, time_budget):
        """Optimize coloring for a single instance within time budget."""
        edges, degree = get_instance(n, d, seed)
        adj = build_adjacency(edges)
        
        # Start with greedy
        color = greedy_coloring(edges, adj)
        best_color = dict(color)
        best_count = max(color.values()) + 1 if color else 0
        
        elapsed_start = time.time()
        
        # Try multiple edge orderings
        orderings = []
        # Natural order
        orderings.append(list(edges))
        # Degree-based (high degree first)
        orderings.append(sorted(edges, key=lambda e: degree[e[0]] + degree[e[1]], reverse=True))
        # Reverse natural
        orderings.append(list(reversed(edges)))
        # Random shuffles
        for s in range(5):
            rng = random.Random(seed + 100 + s)
            shuffled = list(edges)
            rng.shuffle(shuffled)
            orderings.append(shuffled)
        
        for ordering in orderings:
            if time.time() > deadline:
                break
            if time.time() - elapsed_start > time_budget:
                break
            
            color = greedy_coloring(ordering, adj)
            current_count = max(color.values()) + 1 if color else 0
            
            if current_count < best_count:
                best_color = dict(color)
                best_count = current_count
            
            # Local search
            color = improve_coloring(edges, adj, color, max_iter=max(50, int(time_budget * 30)))
            current_count = max(color.values()) + 1 if color else 0
            
            if current_count < best_count:
                best_color = dict(color)
                best_count = current_count
            
            # Swap-based improvement
            color = swap_improve(edges, adj, color, max_iter=max(30, int(time_budget * 20)))
            current_count = max(color.values()) + 1 if color else 0
            
            if current_count < best_count:
                best_color = dict(color)
                best_count = current_count
        
        return [best_color[e] for e in edges], best_count

    results = []
    total_colors = 0
    
    # Process each instance
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    remaining_time = deadline - time.time()
    
    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline:
            # Not enough time, use simple greedy for remaining
            edges, _ = get_instance(n, d, seed)
            adj = build_adjacency(edges)
            color = greedy_coloring(edges, adj)
            results.append([color[e] for e in edges])
            continue
        
        time_budget = remaining_time / (len(instances) - idx)
        coloring, count = optimize_instance(n, d, seed, time_budget)
        results.append(coloring)
        total_colors += count
        remaining_time = deadline - time.time()
    
    return results

# EVOLVE-BLOCK-END
