import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    n = 40
    target_edges = 401
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def calc_book(adj, u, v):
        common = adj[u] & adj[v]
        return common.bit_count()

    def run_sa(seed, init_strategy, time_budget):
        random.seed(seed)

        adj = [0] * n
        edges = set()
        book_sizes = {}

        def add_edge(u, v):
            if u == v: return False
            if u > v: u, v = v, u
            if (u, v) in edges: return False
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True

        def remove_edge(u, v):
            if u == v: return False
            if u > v: u, v = v, u
            if (u, v) not in edges: return False
            edges.remove((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return True

        def calc_book(u, v):
            common = adj[u] & adj[v]
            return common.bit_count()

        def update_books_for_vertex(v):
            for w in range(n):
                if (1 << w) & adj[v]:
                    u, x = (v, w) if v < w else (w, v)
                    book_sizes[(u, x)] = calc_book(u, x)

        def get_max_book():
            if not book_sizes: return 0
            return max(book_sizes.values())

        def get_max_edges():
            m = get_max_book()
            return [(u, v) for (u, v), s in book_sizes.items() if s == m]

        # Initialize based on strategy
        if init_strategy == 0:
            # K20,20 + 1 edge
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            add_edge(0, 1)
        elif init_strategy == 1:
            # K19,21 + 2 edges
            for i in range(19):
                for j in range(19, 40):
                    add_edge(i, j)
            add_edge(19, 39)
            add_edge(0, 1)
        elif init_strategy == 2:
            # K20,20 + 1 edge, then perturb randomly
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            add_edge(0, 1)
            for _ in range(15):
                u, v = random.choice(list(edges))
                remove_edge(u, v)
                for _ in range(50):
                    x, y = random.choice(all_pairs)
                    if (x, y) not in edges:
                        add_edge(x, y)
                        break
        elif init_strategy == 3:
            # Random graph with 401 edges
            shuffled = all_pairs[:]
            random.shuffle(shuffled)
            for i in range(target_edges):
                add_edge(shuffled[i][0], shuffled[i][1])
        elif init_strategy == 4:
            # K20,20 + 1 edge, then do local improvement
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            add_edge(0, 1)
            # Quick local improvement pass
            for _ in range(50):
                if time.time() - start_time > deadline - 5:
                    break
                max_edges = get_max_edges()
                if not max_edges:
                    break
                u, v = random.choice(max_edges)
                remove_edge(u, v)
                best_add = None
                best_book = 999
                for _ in range(30):
                    x, y = random.choice(all_pairs)
                    if (x, y) in edges:
                        continue
                    pot = calc_book(x, y)
                    if pot < best_book:
                        best_book = pot
                        best_add = (x, y)
                if best_add:
                    add_edge(best_add[0], best_add[1])
                else:
                    add_edge(u, v)
        elif init_strategy == 5:
            # K20,20 + 1 edge, then swap some cross edges with low-book intra edges
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            add_edge(0, 1)
            # Try to replace some high-book cross edges with intra edges
            for _ in range(10):
                if time.time() - start_time > deadline - 5:
                    break
                max_edges = get_max_edges()
                if not max_edges:
                    break
                u, v = random.choice(max_edges)
                # Check if u and v are in same partition
                same_partition = (u < 20 and v < 20) or (u >= 20 and v >= 20)
                if same_partition:
                    # This is already an intra edge, skip
                    continue
                remove_edge(u, v)
                # Try to add an intra edge
                best_add = None
                best_book = 999
                for _ in range(30):
                    x, y = random.choice(all_pairs)
                    if (x, y) in edges:
                        continue
                    # Prefer intra-partition edges
                    if not ((x < 20 and y < 20) or (x >= 20 and y >= 20)):
                        continue
                    pot = calc_book(x, y)
                    if pot < best_book:
                        best_book = pot
                        best_add = (x, y)
                if best_add:
                    add_edge(best_add[0], best_add[1])
                else:
                    add_edge(u, v)
        elif init_strategy == 6:
            # Star topology in one partition
            # Start with K20,20
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            # Remove edges from vertex 0 to B
            for j in range(20, 40):
                remove_edge(0, j)
            # Add star edges from 0 to other A vertices
            for i in range(1, 20):
                add_edge(0, i)
            # Now we have 380 + 19 = 399 edges. Need 2 more.
            # Add random intra-B edges
            b_pairs = [(i, j) for i in range(20, 40) for j in range(i+1, 40)]
            random.shuffle(b_pairs)
            for u, v in b_pairs:
                if len(edges) >= target_edges:
                    break
                add_edge(u, v)
        elif init_strategy == 7:
            # Deterministic greedy: Start with K20,20, remove edges to make room for intra edges
            # that have low book size.
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            
            # We need to add 1 edge.
            # The best single edge to add to K20,20 is an intra-edge.
            # Any intra-edge (u,v) in A has book size 0.
            # Any cross-edge is already present.
            # So just add (0,1).
            add_edge(0, 1)
        else:
            # K20,20 + 1 edge (default)
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            add_edge(0, 1)
        
        # Initialize book sizes
        for u, v in edges:
            book_sizes[(u, v)] = calc_book(u, v)
        
        best_max = get_max_book()
        best_edges = set(edges)
        
        T = 3.0
        T_min = 0.05
        T_decay = 0.995
        
        phase = 1  # 1 = hill climb, 2 = SA
        hill_climb_end = start_time + time_budget * 0.4
        
        while time.time() - start_time < deadline and time.time() < start_time + time_budget:
            current_max = get_max_book()
            
            if current_max < best_max:
                best_max = current_max
                best_edges = set(edges)
                if best_max <= 6:
                    break
            
            if phase == 1 and time.time() > hill_climb_end:
                phase = 2
            
            max_edges = get_max_edges()
            if not max_edges:
                break
                
            u, v = random.choice(max_edges)
            
            candidate = None
            candidate_book = 999
            
            # Try multiple random candidates, pick the one with lowest potential book
            for _ in range(20):
                x, y = random.choice(all_pairs)
                if (x, y) in edges:
                    continue
                
                pot_book = calc_book(x, y)
                
                if phase == 1:
                    # In hill climb phase, only accept if it doesn't increase max
                    if pot_book >= current_max:
                        continue
                
                if pot_book < candidate_book:
                    candidate = (x, y)
                    candidate_book = pot_book
                    
            if candidate is None:
                continue
                
            x, y = candidate
            
            remove_edge(u, v)
            update_books_for_vertex(u)
            update_books_for_vertex(v)
            
            add_edge(x, y)
            update_books_for_vertex(x)
            update_books_for_vertex(y)
            
            new_max = get_max_book()
            
            if phase == 1:
                # Hill climb: only accept improvements or equal
                if new_max <= current_max:
                    if new_max < current_max:
                        T = max(T_min, T * 0.9)
                    else:
                        T = max(T_min, T * 0.99)
                else:
                    # Revert
                    remove_edge(x, y)
                    update_books_for_vertex(x)
                    update_books_for_vertex(y)
                    add_edge(u, v)
                    update_books_for_vertex(u)
                    update_books_for_vertex(v)
                    T = max(T_min, T * 0.95)
            else:
                # SA phase
                if new_max < current_max:
                    T = max(T_min, T * 0.9)
                elif new_max == current_max:
                    T = max(T_min, T * 0.99)
                else:
                    delta = new_max - current_max
                    accept_prob = math.exp(-delta / T)
                    if random.random() < accept_prob:
                        T = max(T_min, T * 0.99)
                    else:
                        remove_edge(x, y)
                        update_books_for_vertex(x)
                        update_books_for_vertex(y)
                        add_edge(u, v)
                        update_books_for_vertex(u)
                        update_books_for_vertex(v)
                        T = max(T_min, T * 0.95)
                
                # Temperature restart
                if T < T_min:
                    T = 3.0
        
        return best_max, best_edges
    
    # Run multiple SA instances with different seeds and strategies
    best_overall_max = float('inf')
    best_overall_edges = None
    
    # Allocate time across diverse starts
    strategies = [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
        (6, 6), (7, 7), (0, 8), (1, 9), (2, 10), (3, 11),
    ]
    
    num_runs = min(len(strategies), 10)
    time_per_run = 12.0  # ~10 * 12 = 120s, leaving buffer
    
    for i in range(num_runs):
        if time.time() - start_time > deadline - 5:
            break
            
        strategy = strategies[i][0]
        seed = strategies[i][1]
        
        cur_max, cur_edges = run_sa(seed, strategy, time_per_run)
        
        if cur_max < best_overall_max:
            best_overall_max = cur_max
            best_overall_edges = cur_edges
            
        # If we found book size 6 or less, we can stop early
        if best_overall_max <= 6:
            break
    
    if best_overall_edges is None:
        # Fallback: return K20,20 + 1
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return [[u, v] for u, v in best_overall_edges]
