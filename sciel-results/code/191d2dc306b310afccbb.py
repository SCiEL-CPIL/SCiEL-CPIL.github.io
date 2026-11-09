import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    N = 20

    def is_valid_addition(G, u, v):
        """
        Check if adding edge (u, v) to graph G keeps it K3,3-free.
        G is a list of sets, where G[i] is the set of neighbors of i.
        Assumes (u, v) is not currently an edge.
        """
        Nu = G[u]
        Nv = G[v]
        
        # We need to check triples {a, b, c} such that |N(a) \cap N(b) \cap N(c)| <= 2.
        # Adding (u, v) affects triples containing u or v.
        # Case 1: Triple is {u, x, y}.
        # New N(u) = Nu U {v}.
        # Condition: |(Nu U {v}) \cap G[x] \cap G[y]| <= 2.
        # This is | (Nu \cap G[x] \cap G[y]) \cup ({v} \cap G[x] \cap G[y]) | <= 2.
        # Since v not in Nu, these are disjoint.
        # Size = |Nu \cap G[x] \cap G[y]| + (1 if v in G[x] and v in G[y] else 0).
        
        for x in range(N):
            if x == u or x == v:
                continue
            Nx = G[x]
            for y in range(x + 1, N):
                if y == u or y == v:
                    continue
                Ny = G[y]
                
                # Calculate intersection of Nu, Nx, Ny
                # Optimization: iterate over smallest set?
                # Nu, Nx, Ny are sets.
                inter = Nu & Nx & Ny
                count_inter = len(inter)
                
                if v in Nx and v in Ny:
                    if count_inter > 1:
                        return False
                else:
                    if count_inter > 2:
                        return False
                        
        # Case 2: Triple is {u, v, x}.
        # New N(u) = Nu U {v}, New N(v) = Nv U {u}.
        # N(u) \cap N(v) \cap N(x) = (Nu \cup {v}) \cap (Nv \cup {u}) \cap Nx.
        # v is in Nu \cup {v} but not in Nv \cup {u} (since v not in Nv, and v!=u). So v not in intersection.
        # u is in Nv \cup {u} but not in Nu \cup {v} (since u not in Nu, and u!=v). So u not in intersection.
        # So intersection is Nu \cap Nv \cap Nx.
        # This was already checked in the old graph for triple {u, v, x} (which was valid).
        # So no new violations here.
        
        # Case 3: Triple is {x, y, z} not containing u or v.
        # Unaffected.
        
        return True

    def greedy_init():
        G = [set() for _ in range(N)]
        edge_set = set()
        possible = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(possible)
        
        for u, v in possible:
            if is_valid_addition(G, u, v):
                G[u].add(v)
                G[v].add(u)
                edge_set.add((u, v))
                
        return G, edge_set, len(edge_set)

    def local_search(G, edge_set, count):
        # G is list of sets, edge_set is set of tuples
        # We modify G and edge_set in place
        last_improvement = time.time()
        
        while time.time() < deadline:
            improved = False
            # Try to add a random non-edge
            # Sample a few times to find a valid one
            for _ in range(50):
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in edge_set:
                    continue
                
                if is_valid_addition(G, u, v):
                    G[u].add(v)
                    G[v].add(u)
                    edge_set.add((u, v))
                    count += 1
                    improved = True
                    last_improvement = time.time()
                    break
            
            if not improved:
                # Try to remove a random edge to unblock
                # Only do this if we haven't improved for a while?
                # Or just do it probabilistically
                if edge_set and random.random() < 0.5:
                    e = random.choice(list(edge_set))
                    u, v = e
                    G[u].remove(v)
                    G[v].remove(u)
                    edge_set.remove(e)
                    count -= 1
            else:
                # If improved, we might want to try adding more immediately
                # The loop continues anyway
                pass
                
        return count

    best_count = 0
    best_edge_set = set()
    best_G = None

    # Run multiple restarts
    # Each restart: Greedy + Local Search
    # Limit time per restart? No, just check global deadline.
    
    # First, do a few quick greedy inits to get a baseline
    for i in range(20):
        if time.time() > deadline:
            break
        G, edge_set, count = greedy_init()
        if count > best_count:
            best_count = count
            best_edge_set = set(edge_set)
            best_G = [set(s) for s in G]
            
    # Now, run local search on the best one found so far, and maybe others
    # It's better to run local search on the current best, and restart if stuck?
    # Or run local search on each greedy result?
    # Local search can be expensive if it runs for a long time.
    # Let's allocate time for local search.
    
    # Strategy:
    # 1. Find best greedy.
    # 2. Run local search on it until time runs out or no improvement for a while.
    # 3. If time remains, restart greedy and compare.
    
    # Let's refine:
    # Run 5 restarts.
    # For each restart:
    #   Greedy init.
    #   Local search for a fixed time slice (e.g., 20s) or until no improvement.
    #   Keep best.
    
    # Since deadline is global, we just run local search until deadline.
    # But we might want to restart if local search gets stuck.
    # Simple approach:
    # Run local search on the best greedy.
    # If it stops improving for 5s, break and try a new greedy?
    
    # Let's just run local search on the best greedy found initially.
    # And periodically check if a new greedy is better? No, greedy is fast.
    
    # Improved Plan:
    # 1. Generate 5 greedy inits. Keep the best one (G_best, E_best, C_best).
    # 2. Run local search on G_best, E_best, C_best.
    #    Inside local search, if no improvement for 5s, break.
    # 3. If time remains, generate new greedy, compare to current best.
    #    If new greedy > current best, switch to it and continue local search.
    
    # Let's implement this.
    
    # Step 1: Greedy inits
    for _ in range(10):
        if time.time() > deadline:
            break
        G, edge_set, count = greedy_init()
        if count > best_count:
            best_count = count
            best_edge_set = set(edge_set)
            best_G = [set(s) for s in G]

    # Step 2: Local Search on best
    # We need to pass G and edge_set to be modified.
    # We use best_G and best_edge_set.
    
    # To allow restarts within local search if stuck:
    stuck_time = time.time()
    
    while time.time() < deadline:
        # Run local search step
        # We'll do a batch of moves
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in best_edge_set:
                continue
            
            if is_valid_addition(best_G, u, v):
                best_G[u].add(v)
                best_G[v].add(u)
                best_edge_set.add((u, v))
                best_count += 1
                improved = True
                stuck_time = time.time()
                break
        
        if not improved:
            # Try removing an edge
            if best_edge_set:
                e = random.choice(list(best_edge_set))
                u, v = e
                best_G[u].remove(v)
                best_G[v].remove(u)
                best_edge_set.remove(e)
                best_count -= 1
            
            # Check if stuck for too long
            if time.time() - stuck_time > 5.0:
                # Restart with new greedy
                G, edge_set, count = greedy_init()
                if count > best_count:
                    best_count = count
                    best_edge_set = set(edge_set)
                    best_G = [set(s) for s in G]
                    stuck_time = time.time()
                else:
                    # Maybe try a different random seed?
                    # Or just continue with current, maybe remove more edges?
                    # Let's just continue
                    pass

    # Convert best_edge_set to list of lists
    return [[u, v] for u, v in best_edge_set]

if __name__ == "__main__":
    # Test
    edges = construct()
    print(len(edges))
