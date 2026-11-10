import time
import random
import sys

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X):
        if not P and not X:
            if len(R) >= 2:
                yield tuple(sorted(R))
            return
        
        if P or X:
            # Pivot selection
            u = max(P | X, key=lambda v: len(P & adj[v]))
            candidates = P - adj[u]
        else:
            candidates = set()
            
        for v in candidates:
            yield from bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)

    def get_maximal_cliques(adj):
        n = len(adj)
        return list(bron_kerbosch(adj, set(), set(range(n)), set()))

    def is_valid_transversal(trans, cliques):
        for c in cliques:
            hit = False
            for v in c:
                if v in trans:
                    hit = True
                    break
            if not hit:
                return False
        return True

    def local_search(trans, cliques, rng, max_steps=500):
        trans = set(trans)
        n = max(max(v for c in cliques for v in c) + 1, 1)
        all_vertices = set(range(n))
        
        # First, try to remove redundant vertices
        changed = True
        while changed:
            changed = False
            for v in list(trans):
                new_trans = trans - {v}
                if is_valid_transversal(new_trans, cliques):
                    trans = new_trans
                    changed = True
        
        # Then, try to swap vertices to reduce size
        for _ in range(max_steps):
            # Try removing a random vertex
            if not trans:
                break
            v = rng.choice(list(trans))
            new_trans = trans - {v}
            if is_valid_transversal(new_trans, cliques):
                trans = new_trans
                continue
            
            # Try swapping v with a random other vertex
            others = all_vertices - trans
            if not others:
                continue
            w = rng.choice(list(others))
            new_trans = (trans - {v}) | {w}
            if is_valid_transversal(new_trans, cliques):
                trans = new_trans
                continue
                
            # Try adding a vertex? No, we want to minimize size.
            # But maybe adding one allows removing two?
            # That's a 1-in, 2-out move.
            # Let's stick to 0-in, 1-out and 1-in, 1-out for now.
            
        return trans

    def greedy_set_cover(cliques, n, rng):
        uncovered = set(range(len(cliques)))
        selected = set()
        
        # Pre-compute incidence: for each vertex, which cliques it hits
        # cliques is a list of tuples
        # Let's use indices for cliques
        cliques_list = list(cliques)
        
        # Map vertex -> set of clique indices
        v_to_cliques = {v: set() for v in range(n)}
        for i, c in enumerate(cliques_list):
            for v in c:
                v_to_cliques[v].add(i)
                
        while uncovered:
            best_v = -1
            best_count = -1
            ties = []
            for v in range(n):
                # Count how many uncovered cliques v hits
                count = len(v_to_cliques[v] & uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
                    ties = [v]
                elif count == best_count:
                    ties.append(v)
            
            if best_count == 0:
                break # Should not happen if uncovered is not empty
                
            if len(ties) > 1:
                best_v = rng.choice(ties)
                
            selected.add(best_v)
            
            # Remove covered cliques
            uncovered -= v_to_cliques[best_v]
            
        return selected

    def simulated_annealing(trans, cliques, rng, start_temp=1.0, end_temp=0.1, steps=1000):
        trans = set(trans)
        current_cost = len(trans)
        best_trans = set(trans)
        best_cost = current_cost
        
        n = max(max(v for c in cliques for v in c) + 1, 1)
        all_vertices = set(range(n))
        
        for i in range(steps):
            temp = start_temp * (end_temp / start_temp) ** (i / steps)
            
            # Generate a neighbor
            # Option 1: Remove a random vertex
            # Option 2: Add a random vertex
            # Option 3: Swap a random vertex
            r = rng.random()
            if r < 0.3 and trans:
                v = rng.choice(list(trans))
                new_trans = trans - {v}
            elif r < 0.6 and all_vertices - trans:
                v = rng.choice(list(all_vertices - trans))
                new_trans = trans | {v}
            else:
                if not trans:
                    continue
                v = rng.choice(list(trans))
                others = all_vertices - trans
                if not others:
                    continue
                w = rng.choice(list(others))
                new_trans = (trans - {v}) | {w}
                
            if not is_valid_transversal(new_trans, cliques):
                continue
                
            new_cost = len(new_trans)
            delta = new_cost - current_cost
            
            if delta <= 0 or rng.random() < pow(2.718, -delta / temp):
                trans = new_trans
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_trans = set(trans)
                    
        return best_trans

    out = []
    start_time = time.time()
    total_budget = 140.0 # Leave 10s margin for 150s limit
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    for idx, (n, p, seed) in enumerate(instances):
        inst_start = time.time()
        # Check if we have enough time for this instance
        # Assume roughly equal time per instance, but adjust based on remaining
        remaining_time = total_budget - (time.time() - start_time)
        if remaining_time <= 0:
            # Fallback: return best possible (empty or all vertices) immediately
            # But we need to generate the graph at least? No, we can just return [] if we know it's invalid?
            # No, we must return a valid transversal.
            # If we run out of time, we should return a known valid transversal.
            # The set of all vertices is always a valid transversal.
            # So we can just return list(range(n)) as a fallback.
            out.append(list(range(n)))
            continue
            
        # Allocate time for this instance
        # If it's the first instance, give it more time?
        # Let's just use a fixed fraction of remaining time.
        # But we need to be careful.
        # Let's say we have T remaining, and K instances left.
        # Give T/K to each?
        # But some instances might be harder.
        # Let's just give 35s per instance as before, but capped by remaining time.
        time_limit = min(35.0, remaining_time)
        
        adj = random_graph(n, p, seed)
        cliques = get_maximal_cliques(adj)
        
        if not cliques:
            out.append([])
            continue
            
        best_set = None
        best_size = n # Upper bound
        
        # Initial greedy solution
        rng_init = random.Random(seed)
        greedy_sol = greedy_set_cover(cliques, n, rng_init)
        greedy_sol = local_search(greedy_sol, cliques, rng_init, max_steps=100)
        
        if len(greedy_sol) < best_size:
            best_size = len(greedy_sol)
            best_set = set(greedy_sol)
            
        # Iterative improvement
        rng = random.Random(12345)
        iteration = 0
        while time.time() - inst_start < time_limit:
            # New greedy start with different seed
            rng_iter = random.Random(seed + iteration)
            candidate = greedy_set_cover(cliques, n, rng_iter)
            
            # Local search
            candidate = local_search(candidate, cliques, rng_iter, max_steps=200)
            
            if len(candidate) < best_size:
                best_size = len(candidate)
                best_set = set(candidate)
                
                # If we found a very small set, we can stop early?
                # Or continue to see if we can do better.
                # Let's continue.
                
            # Simulated annealing on the current best
            if best_set is not None:
                sa_sol = simulated_annealing(best_set, cliques, rng_iter, start_temp=1.0, end_temp=0.1, steps=500)
                if len(sa_sol) < best_size:
                    best_size = len(sa_sol)
                    best_set = sa_sol
                    
            iteration += 1
            
            # Check time again
            if time.time() - inst_start > time_limit:
                break
                
        if best_set is None:
            best_set = set(range(n))
            
        out.append(sorted(best_set))
        
    return out
