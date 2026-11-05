"""Strong edge colourings via WalkSAT-style stochastic local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, adj):
        """Build conflict sets: for each edge, the set of edges that conflict with it."""
        n_edges = len(edges)
        # conflict[i] = set of j != i such that edges i and j cannot share a color
        conflict = [set() for _ in range(n_edges)]
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                # Share a vertex?
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i].add(j)
                    conflict[j].add(i)
                # Joined by an edge?
                elif ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict

    def greedy_dsatur(edges, conflict):
        """DSATUR-style greedy: at each step, pick the edge with most constrained color set."""
        n_edges = len(edges)
        colour = [-1] * n_edges
        # Track which colors are forbidden for each uncolored edge
        # forbidden[i] = set of colors that edge i cannot take
        forbidden = [set() for _ in range(n_edges)]
        uncolored = set(range(n_edges))
        
        for _ in range(n_edges):
            # Pick edge with most forbidden colors (DSATUR heuristic)
            best_e = None
            best_count = -1
            for e in uncolored:
                cnt = len(forbidden[e])
                if cnt > best_count:
                    best_count = cnt
                    best_e = e
            
            if best_e is None:
                break
            
            # Assign smallest available color
            c = 0
            while c in forbidden[best_e]:
                c += 1
            colour[best_e] = c
            uncolored.remove(best_e)
            
            # Update forbidden sets for neighbors
            for nb in conflict[best_e]:
                if nb in uncolored:
                    forbidden[nb].add(c)
        
        max_color = max(colour) if colour else 0
        return colour, max_color + 1

    def count_colors(colour):
        """Return the number of distinct colors used."""
        return len(set(colour))

    def is_valid(colour, conflict):
        """Check if coloring is valid (no two conflicting edges share a color)."""
        n = len(colour)
        for i in range(n):
            for j in conflict[i]:
                if j > i and colour[i] == colour[j]:
                    return False
        return True

    def walk_sat_coloring(edges, conflict, initial_colour, initial_k, time_budget=30.0, seed=42):
        """
        WalkSAT-style local search to minimize the number of colors.
        
        State: colour assignment + set of active colors.
        Objective: minimize number of active colors.
        Moves: reassign an edge to a different color (possibly a new color).
        
        We try to reduce the number of colors by:
        1. Identifying edges with the highest color.
        2. Trying to move them to lower colors.
        3. If stuck, do a random 'kick' (reassign a random edge to a random color).
        """
        rng = random.Random(seed)
        n_edges = len(edges)
        colour = initial_colour[:]
        current_k = initial_k
        
        # Precompute: for each edge, list of conflicting edges
        conflict_list = [list(conflict[i]) for i in range(n_edges)]
        
        def get_taken_colors(edge_idx):
            """Get set of colors used by conflicting edges of edge_idx."""
            taken = set()
            for nb in conflict_list[edge_idx]:
                taken.add(colour[nb])
            return taken

        best_colour = colour[:]
        best_k = current_k
        start = time.time()
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() - start < time_budget:
            # Check if we can reduce k
            # Find edges with the highest color (k-1)
            high_edges = [i for i in range(n_edges) if colour[i] == current_k - 1]
            
            if not high_edges:
                # No edges with highest color, reduce k
                current_k -= 1
                if current_k < best_k:
                    best_k = current_k
                    best_colour = colour[:]
                no_improve = 0
                continue
            
            # Try to move each high edge to a lower color
            moved = False
            rng.shuffle(high_edges)
            
            for e_idx in high_edges:
                taken = get_taken_colors(e_idx)
                # Find smallest available color below current_k
                new_c = 0
                while new_c < current_k and new_c in taken:
                    new_c += 1
                
                if new_c < current_k:
                    colour[e_idx] = new_c
                    moved = True
                    break
            
            if moved:
                # Check if we can reduce k now
                if not any(colour[i] == current_k - 1 for i in range(n_edges)):
                    current_k -= 1
                    if current_k < best_k:
                        best_k = current_k
                        best_colour = colour[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            # If stuck for too long, do a random kick
            if no_improve >= max_no_improve:
                # Random kick: reassign a random edge to a random color
                kick_edge = rng.randint(0, n_edges - 1)
                # Try a color that might help
                kick_color = rng.randint(0, current_k)
                colour[kick_edge] = kick_color
                no_improve = 0
        
        # Verify best solution
        if not is_valid(best_colour, conflict):
            # Fallback to initial
            return initial_colour, initial_k
        
        return best_colour, best_k

    def tabu_search_coloring(edges, conflict, initial_colour, initial_k, time_budget=30.0, seed=42):
        """
        Tabu search to minimize number of colors.
        """
        rng = random.Random(seed)
        n_edges = len(edges)
        colour = initial_colour[:]
        current_k = initial_k
        
        conflict_list = [list(conflict[i]) for i in range(n_edges)]
        
        def get_taken_colors(edge_idx):
            taken = set()
            for nb in conflict_list[edge_idx]:
                taken.add(colour[nb])
            return taken

        best_colour = colour[:]
        best_k = current_k
        start = time.time()
        
        # Tabu list: (edge, color) pairs that were recently changed
        tabu = {}
        tabu_tenure = 20
        tabu_counter = 0
        
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() - start < time_budget:
            # Try to reduce k by moving high-color edges
            high_color = current_k - 1
            high_edges = [i for i in range(n_edges) if colour[i] == high_color]
            
            if not high_edges:
                current_k -= 1
                if current_k < best_k:
                    best_k = current_k
                    best_colour = colour[:]
                no_improve = 0
                continue
            
            # Find best move: move a high edge to the lowest available color
            best_move = None
            best_target_color = None
            
            for e_idx in high_edges:
                if (e_idx, high_color) in tabu and tabu[(e_idx, high_color)] > tabu_counter:
                    continue
                taken = get_taken_colors(e_idx)
                new_c = 0
                while new_c < current_k and new_c in taken:
                    new_c += 1
                if new_c < current_k:
                    if best_move is None or new_c < best_target_color:
                        best_move = e_idx
                        best_target_color = new_c
            
            if best_move is not None:
                # Make the move
                tabu[(best_move, high_color)] = tabu_counter + tabu_tenure
                colour[best_move] = best_target_color
                tabu_counter += 1
                
                # Check if k can be reduced
                if not any(colour[i] == high_color for i in range(n_edges)):
                    current_k -= 1
                    if current_k < best_k:
                        best_k = current_k
                        best_colour = colour[:]
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
                # If stuck, do a diversification move
                if no_improve >= max_no_improve:
                    # Pick a random edge and move it to a random valid color
                    e_idx = rng.randint(0, n_edges - 1)
                    taken = get_taken_colors(e_idx)
                    available = [c for c in range(current_k) if c not in taken]
                    if available:
                        new_c = rng.choice(available)
                        tabu[(e_idx, colour[e_idx])] = tabu_counter + tabu_tenure
                        colour[e_idx] = new_c
                        tabu_counter += 1
                    no_improve = 0
        
        if not is_valid(best_colour, conflict):
            return initial_colour, initial_k
        
        return best_colour, best_k

    def multi_start_search(edges, conflict, time_budget=60.0):
        """
        Multiple starts with different greedy orderings, followed by local search.
        """
        n_edges = len(edges)
        rng = random.Random(12345)
        
        # Compute constraint degrees
        constraint_deg = [len(conflict[i]) for i in range(n_edges)]
        
        best_colour = None
        best_k = float('inf')
        start = time.time()
        
        # Start 1: DSATUR greedy
        colour, k = greedy_dsatur(edges, conflict)
        if k < best_k:
            best_k = k
            best_colour = colour[:]
        
        # Local search on best so far
        if time.time() - start < time_budget:
            remaining = time_budget - (time.time() - start)
            if remaining > 10:
                colour2, k2 = walk_sat_coloring(edges, conflict, best_colour, best_k, 
                                                 time_budget=min(remaining * 0.5, 25.0), seed=1)
                if k2 < best_k:
                    best_k = k2
                    best_colour = colour2[:]
        
        # More random starts if time permits
        if time.time() - start < time_budget - 15:
            for trial in range(5):
                if time.time() - start > time_budget - 15:
                    break
                
                # Random ordering biased toward high-constraint edges
                order = list(range(n_edges))
                # Partially sort by constraint degree with noise
                for i in range(len(order) - 1, 0, -1):
                    j = rng.randint(0, i)
                    if constraint_deg[order[i]] > constraint_deg[order[j]] and rng.random() < 0.6:
                        order[i], order[j] = order[j], order[i]
                
                # Greedy in this order
                colour_trial = [-1] * n_edges
                forbidden = [set() for _ in range(n_edges)]
                uncolored = set(range(n_edges))
                
                for e in order:
                    c = 0
                    while c in forbidden[e]:
                        c += 1
                    colour_trial[e] = c
                    uncolored.remove(e)
                    for nb in conflict[e]:
                        if nb in uncolored:
                            forbidden[nb].add(c)
                
                k_trial = max(colour_trial) + 1 if colour_trial else 0
                
                if k_trial < best_k:
                    best_k = k_trial
                    best_colour = colour_trial[:]
                
                # Quick local search
                if time.time() - start < time_budget - 10:
                    remaining = time_budget - (time.time() - start)
                    if remaining > 8:
                        colour_ls, k_ls = walk_sat_coloring(edges, conflict, colour_trial, k_trial,
                                                             time_budget=min(remaining * 0.4, 12.0), seed=trial+10)
                        if k_ls < best_k:
                            best_k = k_ls
                            best_colour = colour_ls[:]
        
        # Final polish with tabu search if time permits
        if time.time() - start < time_budget - 5 and best_k > 1:
            remaining = time_budget - (time.time() - start)
            if remaining > 5:
                colour_tabu, k_tabu = tabu_search_coloring(edges, conflict, best_colour, best_k,
                                                            time_budget=min(remaining - 2, 20.0), seed=999)
                if k_tabu < best_k:
                    best_k = k_tabu
                    best_colour = colour_tabu[:]
        
        return best_colour, best_k

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    # Allocate time budget per instance
    total_budget = deadline - start_time - 10  # Leave 10s safety margin
    per_instance = total_budget / 3.0
    
    for idx, spec in enumerate(specs):
        if time.time() > deadline - 5:
            break
        
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict(edges, adj)
        n_edges = len(edges)
        
        # Time budget for this instance
        instance_start = time.time()
        instance_budget = min(per_instance, deadline - instance_start - 5)
        
        if instance_budget < 5:
            # Very little time left, just do quick greedy
            colour, k = greedy_dsatur(edges, conflict)
            result = [colour[i] for i in range(n_edges)]
            out.append(result)
            continue
        
        colour, k = multi_start_search(edges, conflict, time_budget=instance_budget)
        
        result = [colour[i] for i in range(n_edges)]
        out.append(result)
    
    # Fill remaining with simple greedy if we ran out of time
    while len(out) < 3:
        spec = specs[len(out)]
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        conflict = build_conflict(edges, adj)
        colour, _ = greedy_dsatur(edges, conflict)
        result = [colour[i] for i in range(len(edges))]
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
