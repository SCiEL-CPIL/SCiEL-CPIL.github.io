"""Find dense triangle-free 4-chromatic graph on 16 vertices."""
import time
import random

def construct():
    deadline = time.time() + 100  # safety margin
    N = 16
    
    def is_triangle_free(adj):
        for u in range(N):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        return True
    
    def is_3_colorable(adj):
        color = [-1] * N
        # Order by degree descending for better pruning
        order = sorted(range(N), key=lambda x: -len(adj[x]))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == N:
                return True
            v = order[idx]
            # Try colors 0,1,2
            for c in range(3):
                conflict = False
                for u in adj[v]:
                    if color[u] == c:
                        conflict = True
                        break
                if not conflict:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    def get_edge_count(adj):
        return sum(len(adj[u]) for u in range(N)) // 2
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle."""
        if v in adj[u]:
            return False
        # Check if adding (u,v) creates any triangle
        for w in adj[u]:
            if w in adj[v]:
                return False
        for w in adj[v]:
            if w in adj[u]:
                return False
        # Safe to add
        adj[u].add(v)
        adj[v].add(u)
        return True
    
    def greedy_complete(adj):
        """Try to add all possible edges in random order."""
        random.shuffle(range(N))
        order = list(range(N))
        random.shuffle(order)
        added = 0
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    i, j = j, i
                if try_add_edge(adj, i, j):
                    added += 1
        return added
    
    def local_search(adj, max_iter=5000):
        """Local search: try to add edges, verify 3-colorability."""
        best_edges = get_edge_count(adj)
        if not is_3_colorable(adj):
            return adj, best_edges
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to add a random edge
            u, v = random.sample(range(N), 2)
            if v not in adj[u]:
                # Check triangle-free condition
                can_add = True
                for w in adj[u]:
                    if w in adj[v]:
                        can_add = False
                        break
                if can_add:
                    # Tentatively add
                    adj[u].add(v)
                    adj[v].add(u)
                    # Check 3-colorability
                    if not is_3_colorable(adj):
                        new_count = get_edge_count(adj)
                        if new_count > best_edges:
                            best_edges = new_count
                    else:
                        # Remove edge
                        adj[u].discard(v)
                        adj[v].discard(u)
        
        return adj, best_edges
    
    def create_mycielski_c7():
        """Create Mycielski of C7 (11 vertices) + 5 extra."""
        adj = [set() for _ in range(N)]
        n_orig = 7
        # C7 edges
        for i in range(n_orig):
            adj[i].add((i + 1) % n_orig)
            adj[(i + 1) % n_orig].add(i)
        # Mycielski w-vertices
        for i in range(n_orig):
            j = (i + 1) % n_orig
            adj[i].add(n_orig + j)
            adj[n_orig + j].add(i)
            adj[j].add(n_orig + i)
            adj[n_orig + i].add(j)
        # Apex
        apex = 14
        for i in range(n_orig):
            adj[n_orig + i].add(apex)
            adj[apex].add(n_orig + i)
        # Vertex 15 connected to w-vertices
        for i in range(n_orig):
            adj[15].add(n_orig + i)
            adj[n_orig + i].add(15)
        return adj
    
    def random_4_chromatic_seed():
        """Generate a random triangle-free 4-chromatic graph."""
        # Start with odd cycle C5 or C7
        n_cycle = random.choice([5, 7])
        adj = [set() for _ in range(N)]
        for i in range(n_cycle):
            adj[i].add((i + 1) % n_cycle)
            adj[(i + 1) % n_cycle].add(i)
        
        # Add Mycielski transformation to get 4-chromatic
        # This creates a 4-chromatic triangle-free graph
        new_n = 2 * n_cycle + 1
        new_adj = [set() for _ in range(new_n)]
        # Copy original
        for i in range(n_cycle):
            for j in adj[i]:
                new_adj[i].add(j)
        # Add shadow vertices
        for i in range(n_cycle):
            for j in adj[i]:
                if j > i:  # avoid double
                    new_adj[i].add(n_cycle + j)
                    new_adj[n_cycle + j].add(i)
                    new_adj[j].add(n_cycle + i)
                    new_adj[n_cycle + i].add(j)
        # Apex
        apex = 2 * n_cycle
        for i in range(n_cycle):
            new_adj[n_cycle + i].add(apex)
            new_adj[apex].add(n_cycle + i)
        
        # If we have more than 16 vertices, we need to handle that
        # For n_cycle=7, we get 15 vertices, add one more
        if new_n < N:
            # Add remaining vertices connected to independent set
            for i in range(new_n, N):
                # Connect to some shadow vertices
                for j in range(n_cycle):
                    if random.random() < 0.5:
                        new_adj[i].add(n_cycle + j)
                        new_adj[n_cycle + j].add(i)
            new_adj = new_adj[:N]
        elif new_n > N:
            # Trim - this shouldn't happen with our choices
            new_adj = new_adj[:N]
        
        # Pad if needed
        while len(new_adj) < N:
            new_adj.append(set())
        
        return new_adj[:N]
    
    best_adj = None
    best_count = 0
    
    # Strategy 1: Start from Mycielski C7
    adj1 = create_mycielski_c7()
    if not is_3_colorable(adj1):
        count1 = get_edge_count(adj1)
        if count1 > best_count:
            best_count = count1
            best_adj = [set(s) for s in adj1]
        # Try local search
        adj1, count1 = local_search(adj1, max_iter=2000)
        if not is_3_colorable(adj1):
            if count1 > best_count:
                best_count = count1
                best_adj = [set(s) for s in adj1]
    
    # Strategy 2: Multiple random seeds
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        adj = random_4_chromatic_seed()
        
        # Verify it's triangle-free and not 3-colorable
        if not is_triangle_free(adj):
            continue
        if is_3_colorable(adj):
            continue
        
        count = get_edge_count(adj)
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]
        
        # Local search
        adj, count = local_search(adj, max_iter=1000)
        if not is_3_colorable(adj):
            if count > best_count:
                best_count = count
                best_adj = [set(s) for s in adj]
    
    # Strategy 3: Random initialization + greedy completion
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        adj = [set() for _ in range(N)]
        # Random sparse initialization
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.3:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Ensure triangle-free
        for u in range(N):
            to_remove = []
            for v in list(adj[u]):
                for w in adj[v]:
                    if w in adj[u] and w > v:
                        to_remove.append(v)
                        break
            for v in to_remove:
                adj[u].discard(v)
                adj[v].discard(u)
        
        # Greedy complete
        greedy_complete(adj)
        
        # Check 3-colorability
        if not is_3_colorable(adj) and is_triangle_free(adj):
            count = get_edge_count(adj)
            if count > best_count:
                best_count = count
                best_adj = [set(s) for s in adj]
            
            # Local search
            adj, count = local_search(adj, max_iter=1000)
            if not is_3_colorable(adj):
                if count > best_count:
                    best_count = count
                    best_adj = [set(s) for s in adj]
    
    # Final fallback: return Mycielski C7 if nothing better found
    if best_adj is None:
        best_adj = create_mycielski_c7()
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if v > u:
                edges.append([u, v])
    
    return edges
