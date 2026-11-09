"""Strong edge colouring of bounded-degree graphs via simulated annealing with swap moves."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

DEADLINE_MARGIN = 10  # seconds to finish before 150s wall-clock

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    def get_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
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

    def precompute_conflicts(edges, adj):
        n_edges = len(edges)
        conflicts = [set() for _ in range(n_edges)]
        
        edges_at_vertex = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            edges_at_vertex[u].append(i)
            edges_at_vertex[v].append(i)
        
        for i, (u1, v1) in enumerate(edges):
            for e2_idx in edges_at_vertex[u1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
                    conflicts[e2_idx].add(i)
            for e2_idx in edges_at_vertex[v1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
                    conflicts[e2_idx].add(i)
            
            for neighbor in adj[u1]:
                if neighbor == v1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx != i:
                        u2, v2 = edges[e2_idx]
                        if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                            conflicts[i].add(e2_idx)
                            conflicts[e2_idx].add(i)
            for neighbor in adj[v1]:
                if neighbor == u1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx != i:
                        u2, v2 = edges[e2_idx]
                        if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                            conflicts[i].add(e2_idx)
                            conflicts[e2_idx].add(i)
        
        return conflicts

    def greedy_coloring(edges, conflicts, order):
        n_edges = len(edges)
        color = [-1] * n_edges
        color_classes = [set() for _ in range(n_edges)]
        
        for e_idx in order:
            taken = set()
            for c_idx in conflicts[e_idx]:
                if color[c_idx] != -1:
                    taken.add(color[c_idx])
            
            c = 0
            while c in taken:
                c += 1
            color[e_idx] = c
            color_classes[c].add(e_idx)
            
        return color, color_classes

    def simulated_annealing(edges, conflicts, initial_color, initial_classes, time_budget, rng):
        """
        Simulated annealing to minimize the number of colors used.
        Move: swap colors of two non-conflicting edges, or move an edge to a different color.
        Objective: number of colors used (max color index + 1).
        """
        n_edges = len(edges)
        if n_edges == 0:
            return initial_color, initial_classes, 0
        
        cur_color = list(initial_color)
        cur_classes = [set(c) for c in initial_classes]
        cur_max = max(cur_color) + 1 if cur_color else 0
        
        best_color = list(cur_color)
        best_classes = [set(c) for c in cur_classes]
        best_max = cur_max
        
        sa_start = time.time()
        T = 1.0
        T_min = 0.001
        cooling_rate = 0.9995
        max_iter = int(time_budget * 5000)  # ~5000 iterations per second
        
        for iteration in range(max_iter):
            if time.time() > deadline or time.time() - sa_start > time_budget:
                break
            
            # Generate a random move
            move_type = rng.random()
            
            if move_type < 0.3:
                # Move an edge to a random different color
                e_idx = rng.randrange(n_edges)
                old_c = cur_color[e_idx]
                new_c = rng.randrange(max(cur_max, 1))
                if new_c == old_c:
                    continue
                
                # Check if move is valid (no conflicts in new color class)
                valid = True
                for c_idx in conflicts[e_idx]:
                    if cur_color[c_idx] == new_c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Apply move
                cur_classes[old_c].remove(e_idx)
                cur_classes[new_c].add(e_idx)
                cur_color[e_idx] = new_c
                
                new_max = max(cur_color) + 1
                delta = new_max - cur_max
                cur_max = new_max
                
                # Acceptance
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    if new_max < best_max:
                        best_max = new_max
                        best_color = list(cur_color)
                        best_classes = [set(c) for c in cur_classes]
                else:
                    # Reject: undo
                    cur_classes[new_c].remove(e_idx)
                    cur_classes[old_c].add(e_idx)
                    cur_color[e_idx] = old_c
                    cur_max = old_c if old_c == max(cur_color) else max(cur_color) + 1
            
            elif move_type < 0.6:
                # Swap colors of two non-conflicting edges
                e1 = rng.randrange(n_edges)
                e2 = rng.randrange(n_edges)
                if e1 == e2:
                    continue
                if e1 in conflicts[e2]:
                    continue
                
                c1 = cur_color[e1]
                c2 = cur_color[e2]
                if c1 == c2:
                    continue
                
                # Check validity: after swap, e1 gets c2 and e2 gets c1
                # e1 with c2: no conflict with any edge currently in c2 (except e2 which is moving)
                valid1 = True
                for c_idx in conflicts[e1]:
                    if c_idx != e2 and cur_color[c_idx] == c2:
                        valid1 = False
                        break
                
                valid2 = True
                for c_idx in conflicts[e2]:
                    if c_idx != e1 and cur_color[c_idx] == c1:
                        valid2 = False
                        break
                
                if not valid1 or not valid2:
                    continue
                
                # Apply swap
                cur_classes[c1].remove(e1)
                cur_classes[c2].remove(e2)
                cur_classes[c1].add(e2)
                cur_classes[c2].add(e1)
                cur_color[e1] = c2
                cur_color[e2] = c1
                
                new_max = max(cur_color) + 1
                delta = new_max - cur_max
                cur_max = new_max
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    if new_max < best_max:
                        best_max = new_max
                        best_color = list(cur_color)
                        best_classes = [set(c) for c in cur_classes]
                else:
                    # Undo swap
                    cur_classes[c2].remove(e1)
                    cur_classes[c1].remove(e2)
                    cur_classes[c1].add(e1)
                    cur_classes[c2].add(e2)
                    cur_color[e1] = c1
                    cur_color[e2] = c2
                    cur_max = old_c if False else max(cur_color) + 1
            
            else:
                # Try to move an edge to a lower color (greedy improvement)
                e_idx = rng.randrange(n_edges)
                old_c = cur_color[e_idx]
                
                for new_c in range(old_c):
                    valid = True
                    for c_idx in conflicts[e_idx]:
                        if cur_color[c_idx] == new_c:
                            valid = False
                            break
                    
                    if valid:
                        cur_classes[old_c].remove(e_idx)
                        cur_classes[new_c].add(e_idx)
                        cur_color[e_idx] = new_c
                        new_max = max(cur_color) + 1
                        cur_max = new_max
                        if new_max < best_max:
                            best_max = new_max
                            best_color = list(cur_color)
                            best_classes = [set(c) for c in cur_classes]
                        break
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return best_color, best_classes, best_max

    def optimize_instance(n, d, seed, time_budget):
        edges, degree = get_instance(n, d, seed)
        adj = build_adjacency(edges)
        conflicts = precompute_conflicts(edges, adj)
        n_edges = len(edges)
        
        best_color_list = None
        best_max_color = float('inf')
        
        elapsed_start = time.time()
        
        # Try multiple initial colorings
        initial_orderings = []
        
        # Natural order
        initial_orderings.append(list(range(n_edges)))
        
        # Reverse
        initial_orderings.append(list(reversed(range(n_edges))))
        
        # Degree-based
        edge_degrees = [degree[u] + degree[v] for u, v in edges]
        initial_orderings.append(sorted(range(n_edges), key=lambda i: edge_degrees[i], reverse=True))
        
        # Conflict-based
        conflict_counts = [len(conflicts[i]) for i in range(n_edges)]
        initial_orderings.append(sorted(range(n_edges), key=lambda i: conflict_counts[i], reverse=True))
        
        # Random
        for s in range(10):
            rng_init = random.Random(seed + 2000 + s)
            shuffled = list(range(n_edges))
            rng_init.shuffle(shuffled)
            initial_orderings.append(shuffled)
        
        for ordering in initial_orderings:
            if time.time() > deadline or time.time() - elapsed_start > time_budget:
                break
            
            color, color_classes = greedy_coloring(edges, conflicts, ordering)
            current_max = max(color) + 1 if color else 0
            
            if current_max < best_max_color:
                best_max_color = current_max
                best_color_list = list(color)
                best_classes = [set(c) for c in color_classes]
            
            # Run simulated annealing on this initial solution
            remaining = time_budget - (time.time() - elapsed_start)
            if remaining > 2.0:
                sa_budget = min(remaining * 0.4, 10.0)
                rng_sa = random.Random(seed + 3000 + len(ordering))
                
                sa_color, sa_classes, sa_max = simulated_annealing(
                    edges, conflicts, color, color_classes, sa_budget, rng_sa
                )
                
                if sa_max < best_max_color:
                    best_max_color = sa_max
                    best_color_list = list(sa_color)
                    best_classes = [set(c) for c in sa_classes]
        
        # Final polish: run SA on the best solution found
        if best_color_list is not None:
            remaining = time_budget - (time.time() - elapsed_start)
            if remaining > 3.0:
                best_classes = [set() for _ in range(best_max_color)]
                for i, c in enumerate(best_color_list):
                    best_classes[c].add(i)
                
                rng_final = random.Random(seed + 5000)
                final_color, final_classes, final_max = simulated_annealing(
                    edges, conflicts, best_color_list, best_classes, remaining * 0.5, rng_final
                )
                
                if final_max < best_max_color:
                    best_max_color = final_max
                    best_color_list = list(final_color)
        
        # Fallback
        if best_color_list is None:
            color, _ = greedy_coloring(edges, conflicts, list(range(n_edges)))
            best_color_list = list(color)
            best_max_color = max(best_color_list) + 1 if best_color_list else 0
        
        return best_color_list, best_max_color

    results = []
    
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    remaining_time = deadline - time.time()
    
    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline:
            edges, _ = get_instance(n, d, seed)
            adj = build_adjacency(edges)
            conflicts = precompute_conflicts(edges, adj)
            color, _ = greedy_coloring(edges, conflicts, list(range(len(edges))))
            results.append(color)
            continue
        
        if idx == 0:
            time_budget = min(remaining_time * 0.25, 15.0)
        elif idx == 1:
            time_budget = min(remaining_time * 0.35, 25.0)
        else:
            time_budget = min(remaining_time * 0.40, 35.0)
        
        coloring, _ = optimize_instance(n, d, seed, time_budget)
        results.append(coloring)
        remaining_time = deadline - time.time()
    
    return results

# EVOLVE-BLOCK-END
